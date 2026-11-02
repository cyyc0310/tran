"""Fixed climate/weather amplitude controls; predictions sealed before target scores."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from collect_weather_transfer_pilot import ROOT, digest
from run_weather_transfer_pilot import joined_data
from transcif.research import resource_amplitudes as amp
from transcif.research import multicountry as mc
from transcif.research.capacity_transfer import load_capacity
from transcif.research.coarse_trajectories import crps
from transcif.research.conditional_risk import interval_metrics
from transcif.research.monthly_adaptation import weighted_quantile


def past_power(train: pd.DataFrame, raw: dict) -> tuple[np.ndarray, list]:
    """Use source history only, before its own simulated forecasting origin."""
    months = {region: mc.public_months(raw[region]) for region in train.region.unique()}
    values, audit = [], []
    for ix in amp.day_groups(train):
        row = train.iloc[ix[0]]; period = pd.Period(row.latest_allowed_period, freq='M')
        frame = raw[row.region]; rows = frame.loc[frame.clock.dt.to_period('M').eq(period)]
        if period > pd.Period(pd.Timestamp(row.forecast_origin_local),freq='M')-2:
            raise ValueError('source monthly power exceeds origin information cutoff')
        hours = rows.expected_hours.iloc[0]; energy = months[row.region].loc[period].to_numpy(float)
        if hours <= 0 or len(rows)/hours < .99:
            raise ValueError('source monthly energy coverage incomplete')
        power = energy/hours
        np.testing.assert_allclose(power*hours,energy,rtol=1e-12,atol=1e-6)
        values.append(power)
        audit.append(dict(region=row.region,source_day=str(row.clock.floor('D')),reference_period=str(period),
            origin_local=str(row.forecast_origin_local),expected_month_hours=int(hours),observed_hours=len(rows),
            energy=energy.tolist(),mean_power=power.tolist(),
            support='canonical source hourly MW-equivalent sum gives MWh; divide by expected local-month hours'))
    return np.asarray(values), audit


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--weather-cache', type=Path, required=True)
    parser.add_argument('--climate-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent = ROOT/'results/weather-transfer-seasonal-models-r1'
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
    if (not manifest['complete'] or manifest['years'] != list(range(2016, 2021))
            or manifest['eligible_for_issued_public'] is not False or manifest['available_at'] is not None):
        raise ValueError('fixed historical climate incomplete or falsely certified')
    for name, expected in manifest['files'].items():
        if digest(climate_cache/name) != expected:
            raise ValueError('climate checkpoint changed')
    for name, expected in manifest['code'].items():
        if digest(Path(name)) != expected:
            raise ValueError('climate collector code changed since verification')
    climate = pd.read_csv(climate_cache/'climatology.csv')
    data, weather_hashes = joined_data(args.weather_cache.resolve())
    raw, _ = mc.load_countries(ROOT)
    capacity_path = ROOT/'results/public-capacity-input-r1/capacity.json'
    capacity = load_capacity(capacity_path); fuel = np.empty((len(data), len(mc.FINE)))
    for region, group in data.groupby('region'):
        label = raw[region].loc[group.label_row.astype(int)]
        np.testing.assert_array_equal(pd.to_datetime(label.time, utc=True).to_numpy(), group.valid_at.to_numpy())
        fuel[group.index] = label[list(mc.FINE)].to_numpy()
        c = capacity['regions'][region]
        data.loc[group.index, 'capacity_factor'] = 400.+600.*c['coal_mw']/(c['coal_mw']+c['gas_mw'])
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT/'src/transcif/research/resource_amplitudes.py',
             ROOT/'tests/test_resource_amplitudes.py', ROOT/'tests/test_weather_reference.py',
             ROOT/'docs/RESOURCE_REFERENCE_DEVELOPMENT_PROTOCOL.md',
             ROOT/'src/transcif/research/coarse_weather_shapes.py', capacity_path,
             climate_cache/'manifest.json', parent/'predictions.csv', parent/'calibration.json']
    paths += [climate_cache/name for name in manifest['files']]
    paths += [Path(name) for name in manifest['code']]
    seals = {**inherited, **weather_hashes, **{str(p.resolve()): digest(p) for p in paths}}
    (out/'input_code_seal.json').write_text(json.dumps(seals, indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(),
        **{name: version(name) for name in ('numpy','pandas','scikit-learn','joblib')}), indent=2)+'\n')
    (out/'execution_plan.json').write_text(json.dumps(dict(methods=amp.METHODS, tracks=['matched','full_source_extra_labels'],
        budgets=['monthly','capacity'], ridge_alpha=10., ratio_strength=1.,
        target_model_selection=False, original_weather_public_certification=False,
        reference_years=list(range(2016,2021)), adaptive_development=True), indent=2)+'\n')
    prior = pd.read_csv(parent/'predictions.csv')
    old_cal = {(x['configuration'],x['budget'],x['method']):x['cqr_expansion'] for x in
               json.loads((parent/'calibration.json').read_text()) if x['method'].startswith('joint')}
    predictions, calibration, particles, diagnostics, model_seals = [], [], [], [], {}
    for target in ('DE','GB','US'):
        others = [c for c in ('DE','GB','US') if c != target]
        for source in [(others[0],),(others[1],),tuple(others)]:
            tag = '+'.join(source)+'_to_'+target
            train = data.loc[data.country.isin(source)&data.split.eq('train')].copy()
            cal = data.loc[data.country.isin(source)&data.split.eq('calibration')].copy()
            query = data.loc[data.country.eq(target)&data.split.eq('development')].copy()
            coarse = mc.coarse_values(pd.DataFrame(fuel[train.index], columns=mc.FINE))
            power, power_audit = past_power(train, raw)
            fitted = amp.fit(train, coarse, power, climate, allowed_countries=source)
            fitted['audit']['past_power_receipts'] = power_audit
            path = out/f'{tag}_heads.joblib'; joblib.dump(fitted,path)
            restored = joblib.load(path)  # own newly written source model
            daily,_,_ = amp.day_inputs(query,climate)
            for kind in ('B','W'):
                np.testing.assert_array_equal(amp.head_amplitudes(fitted,daily,kind),amp.head_amplitudes(restored,daily,kind))
            model_seals[path.name] = digest(path)
            (out/f'{tag}_training_audit.json').write_text(json.dumps(fitted['audit'],indent=2)+'\n')
            for track in ('matched','full_source_extra_labels'):
                if track == 'matched':
                    bank_path = parent/f'{tag}_matched_bank.joblib'; old_method = 'joint_matched'
                else:
                    bank_path = ROOT/f'results/three-country-coarse-trajectories-r2/{tag}_bank.joblib'
                    old_method = 'joint_full_source_extra_labels'
                seals[str(bank_path)] = digest(bank_path)
                bank = joblib.load(bank_path)  # own sealed source artifact only
                source_members,_ = amp.predict_methods(bank,fitted,cal,climate)
                query_members,diag = amp.predict_methods(bank,fitted,query,climate)
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
                        if method=='joint_original':
                            np.testing.assert_allclose(radius,old_cal[tag,budget,old_method],atol=1e-10,rtol=0.)
                        calibration.append(dict(configuration=tag,track=track,budget=budget,method=method,
                                                expansion=radius,source_hours=len(cal)))
                        lo,hi = np.quantile(array,[.05,.95],axis=1)
                        for risk,expansion in [('raw',0.),('calibrated',radius)]:
                            rows = query[['country','region','origin','valid_at','label_row']].copy()
                            rows['configuration'],rows['track'] = tag,track
                            rows['budget'],rows['method'],rows['risk'] = budget,method,risk
                            rows['point'] = array.mean(axis=1)
                            rows['lower'],rows['upper'] = np.maximum(lo-expansion,0.),np.minimum(hi+expansion,1000.)
                            if method=='joint_original':
                                old = prior.loc[prior.configuration.eq(tag)&prior.budget.eq(budget)&prior.method.eq(old_method)&prior.risk.eq(risk)]
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
        unique_target_hours=int(data.split.eq('development').sum()),source_train_hours=int(data.split.eq('train').sum()),
        source_calibration_hours=int(data.split.eq('calibration').sum()),target_calibration_used=False,
        new_country_opened=False,novelty_established=False,identity_verified=True),indent=2)+'\n')


if __name__=='__main__':
    main()
