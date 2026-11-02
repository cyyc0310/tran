"""Shared resource experiments with declared strategies and explicit source-information budgets."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from importlib import import_module
import re
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from collect_weather_transfer_pilot import ROOT, digest
from run_weather_transfer_pilot import joined_data
from run_resource_amplitudes import past_power
from transcif.research import multicountry as mc
from transcif.research.capacity_transfer import load_capacity
from transcif.research.coarse_trajectories import crps
from transcif.research.conditional_risk import interval_metrics
from transcif.research.monthly_adaptation import weighted_quantile


def validate_observed_source_rows(frame: pd.DataFrame, raw: dict) -> None:
    """Bind expanded source rows to their canonical country and absolute label clock."""
    if (frame.empty or not frame.split.eq('train').all() or not frame.clock.dt.year.eq(2022).all()
            or frame.clock.dt.month.gt(9).any() or frame.valid_at.max() >= pd.Timestamp('2022-10-02',tz='UTC')
            or frame.duplicated(['region','valid_at']).any()):
        raise ValueError('proper unique source history only, ending September 2022')
    for region,positions in frame.groupby('region').indices.items():
        selected=frame.iloc[positions]; labels=raw[region].loc[selected.label_row.astype(int)]
        for actual,declared in [(pd.to_datetime(labels.time,utc=True).to_numpy(),selected.valid_at.to_numpy()),
                                (labels.clock.to_numpy(),selected.clock.to_numpy()),
                                (labels.country.to_numpy(),selected.country.to_numpy())]:
            if not np.array_equal(actual,declared):
                raise ValueError('source country or clock differs from canonical generation')
        origins=pd.to_datetime(selected.forecast_origin_local)
        if not np.array_equal(origins.to_numpy(),selected.clock.dt.floor('D').to_numpy()):
            raise ValueError('source public origin must be its canonical local day start')
        expected_period=(selected.clock.dt.to_period('M')-2).astype(str)
        if not np.array_equal(expected_period.to_numpy(),selected.latest_allowed_period.astype(str).to_numpy()):
            raise ValueError('source reference month must be its canonical t-2 month')
        expected=mc.public_features(mc.public_months(raw[region]),selected.clock)
        if not np.allclose(expected[list(mc.HISTORY)].to_numpy(float),selected[list(mc.HISTORY)].to_numpy(float),atol=1e-10,rtol=0.):
            raise ValueError('source public monthly inputs differ from canonical past-only features')
        for _,day in selected.groupby(selected.clock.dt.floor('D')):
            first=day.loc[day.clock.dt.hour.eq(0)]
            if len(first)!=1 or not pd.to_datetime(day.origin,utc=True).eq(first.valid_at.iloc[0]).all():
                raise ValueError('source UTC origin differs from its canonical local day start')


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--weather-cache', type=Path, required=True)
    parser.add_argument('--climate-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--reference-results', type=Path, required=True)
    parser.add_argument('--source-training', type=Path)
    parser.add_argument('--test-file', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]*', args.strategy):
        raise ValueError('strategy must be a local research module identifier')
    amp = import_module('transcif.research.'+args.strategy)
    parent = args.reference_results.resolve()
    inherited = json.loads((parent/'input_code_seal.json').read_text())
    for name, expected in inherited.items():
        if digest(Path(name)) != expected:
            raise ValueError('previous input/code seal changed')
    for seal in ('prediction_seal.json', 'model_seal.json'):
        for name, expected in json.loads((parent/seal).read_text()).items():
            if digest(parent/name) != expected:
                raise ValueError('previous source/prediction artifacts changed')
    climate_cache = args.climate_cache.resolve()
    manifest = json.loads((climate_cache/'manifest.json').read_text())
    if (not manifest['complete']
            or manifest['eligible_for_issued_public'] is not False or manifest['available_at'] is not None):
        raise ValueError('fixed historical climate incomplete or falsely certified')
    for name, expected in manifest['files'].items():
        if digest(climate_cache/name) != expected:
            raise ValueError('climate checkpoint changed')
    for name, expected in {**manifest.get('code', {}), **manifest.get('source_files', {})}.items():
        if digest(Path(name)) != expected:
            raise ValueError('climate collector code changed since verification')
    climate = pd.read_csv(climate_cache/'climatology.csv')
    data, weather_hashes = joined_data(args.weather_cache.resolve())
    raw, _ = mc.load_countries(ROOT)
    training_data = data
    extra_training_seals = {}
    if args.source_training is not None:
        folder = args.source_training.resolve()
        training_manifest = json.loads((folder/'manifest.json').read_text())
        if (not training_manifest['complete'] or training_manifest['role'] != 'source_observed_weather_training'
                or training_manifest['eligible_for_issued_public'] is not False
                or training_manifest['available_at'] is not None):
            raise ValueError('explicit source-observed information budget required')
        for name, expected in training_manifest['files'].items():
            if digest(folder/name) != expected: raise ValueError('source cohort changed')
            extra_training_seals[str(folder/name)] = expected
        for name, expected in training_manifest['source_files'].items():
            if digest(Path(name)) != expected: raise ValueError('source data/producer changed')
            extra_training_seals[name] = expected
        extra_training_seals[str(folder/'manifest.json')] = digest(folder/'manifest.json')
        training_data = pd.read_csv(folder/'cohort.csv', parse_dates=['clock', 'valid_at'])
        validate_observed_source_rows(training_data,raw)
    capacity_path = ROOT/'results/public-capacity-input-r1/capacity.json'
    capacity = load_capacity(capacity_path); fuel = np.empty((len(data), len(mc.FINE)))
    for region, group in data.groupby('region'):
        label = raw[region].loc[group.label_row.astype(int)]
        np.testing.assert_array_equal(pd.to_datetime(label.time, utc=True).to_numpy(), group.valid_at.to_numpy())
        fuel[group.index] = label[list(mc.FINE)].to_numpy()
        c = capacity['regions'][region]
        data.loc[group.index, 'capacity_factor'] = 400.+600.*c['coal_mw']/(c['coal_mw']+c['gas_mw'])
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(amp.__file__), ROOT/'scripts/run_resource_amplitudes.py',
             *args.test_file, args.protocol.resolve(),
             ROOT/'src/transcif/research/coarse_weather_shapes.py', capacity_path,
             climate_cache/'manifest.json', parent/'predictions.csv', parent/'calibration.json']
    paths += [climate_cache/name for name in manifest['files']]
    paths += [Path(name) for name in {**manifest.get('code', {}), **manifest.get('source_files', {})}]
    seals = {**inherited, **weather_hashes, **extra_training_seals, **{str(p.resolve()): digest(p) for p in paths}}
    (out/'input_code_seal.json').write_text(json.dumps(seals, indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(),
        **{name: version(name) for name in ('numpy','pandas','scikit-learn','joblib')}), indent=2)+'\n')
    (out/'execution_plan.json').write_text(json.dumps(dict(methods=amp.METHODS, tracks=['matched','full_source_extra_labels'],
        budgets=['monthly','capacity'], strategy=args.strategy, ratio_strength=1.,
        source_information='observed_weather_expanded' if args.source_training else 'matched_NWP_pilot',
        target_model_selection=False, original_weather_public_certification=False,
        reference_years=list(range(2016,2021)), adaptive_development=True), indent=2)+'\n')
    prior = pd.read_csv(parent/'predictions.csv')
    old_cal = {(x['configuration'],x['track'],x['budget'],x['method']):x['expansion'] for x in
               json.loads((parent/'calibration.json').read_text())}
    identities = getattr(amp, 'IDENTITIES', {'joint_original':'joint_original', 'weather_solar':'weather_solar',
                                          'cube_weather_wind_solar':'weather_wind_solar'})
    predictions, calibration, particles, diagnostics, model_seals = [], [], [], [], {}
    for target in ('DE','GB','US'):
        others = [c for c in ('DE','GB','US') if c != target]
        for source in [(others[0],),(others[1],),tuple(others)]:
            tag = '+'.join(source)+'_to_'+target
            train = training_data.loc[training_data.country.isin(source)&training_data.split.eq('train')].copy()
            cal = data.loc[data.country.isin(source)&data.split.eq('calibration')].copy()
            query = data.loc[data.country.eq(target)&data.split.eq('development')].copy()
            train_fuel = np.empty((len(train),len(mc.FINE)))
            for region, positions in train.groupby('region').indices.items():
                labels = raw[region].loc[train.iloc[positions].label_row.astype(int)]
                np.testing.assert_array_equal(pd.to_datetime(labels.time,utc=True).to_numpy(),train.iloc[positions].valid_at.to_numpy())
                np.testing.assert_array_equal(labels.country.to_numpy(),train.iloc[positions].country.to_numpy())
                np.testing.assert_array_equal(labels.clock.to_numpy(),train.iloc[positions].clock.to_numpy())
                train_fuel[positions] = labels[list(mc.FINE)].to_numpy()
            coarse = mc.coarse_values(pd.DataFrame(train_fuel, columns=mc.FINE))
            power, power_audit = past_power(train, raw)
            fitted = amp.fit(train, coarse, power, climate, allowed_countries=source)
            fitted['audit']['past_power_receipts'] = power_audit
            path = out/f'{tag}_heads.joblib'; joblib.dump(fitted,path)
            restored = joblib.load(path)  # own newly written source model
            model_seals[path.name] = digest(path)
            (out/f'{tag}_training_audit.json').write_text(json.dumps(fitted['audit'],indent=2)+'\n')
            for track in ('matched','full_source_extra_labels'):
                if track == 'matched':
                    bank_path = ROOT/f'results/weather-transfer-seasonal-models-r1/{tag}_matched_bank.joblib'
                else:
                    bank_path = ROOT/f'results/three-country-coarse-trajectories-r2/{tag}_bank.joblib'

                seals[str(bank_path)] = digest(bank_path)
                bank = joblib.load(bank_path)  # own sealed source artifact only
                source_members,_ = amp.predict_methods(bank,fitted,cal,climate)
                query_members,diag = amp.predict_methods(bank,fitted,query,climate)
                if track == 'matched':
                    reloaded, _ = amp.predict_methods(bank,restored,query,climate)
                    for method in amp.METHODS: np.testing.assert_array_equal(query_members[method],reloaded[method])
                diag['configuration'],diag['track'] = tag,track; diagnostics.append(diag)
                truth = fuel[cal.index]@mc.FACTORS/fuel[cal.index].sum(axis=1); w = mc.weights(cal)
                for budget in ('monthly','capacity'):
                    cs = cal.capacity_factor.to_numpy() if budget=='capacity' else np.full(len(cal),bank['macro_ef'])
                    cq = query.capacity_factor.to_numpy() if budget=='capacity' else np.full(len(query),bank['macro_ef'])
                    for method in amp.METHODS:
                        source_array = source_members[method]*cs[:,None]
                        array = query_members[method]*cq[:,None]
                        lo,hi = np.quantile(source_array,[.05,.95],axis=1)
                        radius = weighted_quantile(np.maximum.reduce([lo-truth,truth-hi,np.zeros(len(cal))]),w,.9)
                        if method in identities:
                            np.testing.assert_allclose(radius,old_cal[tag,track,budget,identities[method]],atol=1e-10,rtol=0.)
                        calibration.append(dict(configuration=tag,track=track,budget=budget,method=method,
                                                expansion=radius,source_hours=len(cal)))
                        lo,hi = np.quantile(array,[.05,.95],axis=1)
                        for risk,expansion in [('raw',0.),('calibrated',radius)]:
                            rows = query[['country','region','origin','valid_at','label_row']].copy()
                            rows['configuration'],rows['track'] = tag,track
                            rows['budget'],rows['method'],rows['risk'] = budget,method,risk
                            rows['point'] = array.mean(axis=1)
                            rows['lower'],rows['upper'] = np.maximum(lo-expansion,0.),np.minimum(hi+expansion,1000.)
                            if method in identities:
                                old = prior.loc[prior.configuration.eq(tag)&prior.track.eq(track)&prior.budget.eq(budget)&prior.method.eq(identities[method])&prior.risk.eq(risk)]
                                joined = rows.assign(valid_at=rows.valid_at.astype(str)).merge(old,on=['region','valid_at'],suffixes=('_new','_old'),validate='one_to_one')
                                assert len(joined)==len(rows)
                                for field in ('point','lower','upper'):
                                    np.testing.assert_allclose(joined[field+'_new'],joined[field+'_old'],atol=1e-10,rtol=0.)
                            predictions.append(rows)
                        p = out/f'{tag}_{track}_{budget}_{method}_members.npz'
                        np.savez_compressed(p,members=array,row=query.index.to_numpy())
                        particles.append(dict(path=p.name,sha256=digest(p),configuration=tag,track=track,budget=budget,method=method))
                print(tag,track,'all predictions and old-baseline checks complete',flush=True)
    # No candidate has been scored on target labels before all predictions/radii are written and sealed.
    (out/'input_code_seal.json').write_text(json.dumps(seals,indent=2)+'\n')
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    pd.concat(diagnostics,ignore_index=True).to_csv(out/'amplitude_diagnostics.csv',index=False)
    (out/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    (out/'model_seal.json').write_text(json.dumps(model_seals,indent=2)+'\n')
    pred_seals = {p['path']:p['sha256'] for p in particles}
    pred_seals.update({name:digest(out/name) for name in ('predictions.csv','calibration.json','amplitude_diagnostics.csv','execution_plan.json')})
    pred_seals.update({p.name:digest(p) for p in [out/'runtime.json',*sorted(out.glob('*_training_audit.json'))]})
    (out/'prediction_seal.json').write_text(json.dumps(pred_seals,indent=2)+'\n')
    pred = pd.read_csv(out/'predictions.csv'); scores=[]
    for item in particles:
        with np.load(out/item['path'],allow_pickle=False) as a: members,ix = a['members'],a['row']
        query = data.loc[ix]; truth = fuel[ix]@mc.FACTORS/fuel[ix].sum(axis=1); raw_crps = crps(members,truth)
        selected = pred.loc[pred.configuration.eq(item['configuration'])&pred.track.eq(item['track'])&pred.budget.eq(item['budget'])&pred.method.eq(item['method'])]
        for region,positions in query.groupby('region').indices.items():
            for risk in ('raw','calibrated'):
                rows = selected.loc[selected.region.eq(region)&selected.risk.eq(risk)]
                np.testing.assert_array_equal(rows.valid_at,query.iloc[positions].valid_at.astype(str))
                y = truth[positions]; point = rows.point.to_numpy()
                day = pd.DataFrame(dict(day=rows.origin.to_numpy(),y=y,point=point))
                centered = day[['y','point']]-day.groupby('day')[['y','point']].transform('mean')
                scores.append(dict(configuration=item['configuration'],country=query.country.iloc[0],region=region,
                    track=item['track'],budget=item['budget'],method=item['method'],risk=risk,hours=len(rows),
                    mae=float(np.abs(point-y).mean()),shape_mae=float(np.abs(centered.point-centered.y).mean()),
                    raw_crps=float(raw_crps[positions].mean()),**interval_metrics(y,rows.lower.to_numpy(),rows.upper.to_numpy())))
    metrics=pd.DataFrame(scores);metrics.to_csv(out/'region_metrics.csv',index=False)
    configs=metrics.groupby(['configuration','country','track','budget','method','risk']).mean(numeric_only=True).reset_index()
    configs=configs.rename(columns={'hours':'mean_hours_per_region'});configs.to_csv(out/'configuration_metrics.csv',index=False)
    configs.groupby(['track','budget','method','risk']).mean(numeric_only=True).reset_index().to_csv(out/'summary.csv',index=False)
    for name,expected in seals.items():
        if digest(Path(name))!=expected: raise RuntimeError('input/code changed during run')
    for name,expected in {**pred_seals,**model_seals}.items():
        if digest(out/name)!=expected: raise RuntimeError('sealed output changed during run')
    (out/'completion.json').write_text(json.dumps(dict(complete=True,adaptive_development=True,
        unique_target_hours=int(data.split.eq('development').sum()),source_train_hours=int(training_data.split.eq('train').sum()),
        source_information='observed_weather_expanded' if args.source_training else 'matched_NWP_pilot',
        source_calibration_hours=int(data.split.eq('calibration').sum()),target_calibration_used=False,
        new_country_opened=False,novelty_established=False,identity_verified=True),indent=2)+'\n')


if __name__=='__main__':
    main()
