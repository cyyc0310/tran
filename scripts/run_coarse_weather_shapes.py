"""Adaptive development of identity-preserving coarse weather shape references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from collect_weather_transfer_pilot import ROOT, digest
from run_weather_transfer_pilot import joined_data
from transcif.research import coarse_weather_shapes as shapes
from transcif.research import multicountry as mc
from transcif.research.coarse_trajectories import crps
from transcif.research.capacity_transfer import load_capacity
from transcif.research.conditional_risk import interval_metrics
from transcif.research.monthly_adaptation import weighted_quantile


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent = ROOT/'results/weather-transfer-seasonal-models-r1'
    prior_seal = json.loads((parent/'input_code_seal.json').read_text())
    for name, expected in prior_seal.items():
        if digest(Path(name)) != expected:
            raise ValueError('previous frozen input/code changed')
    for name, expected in json.loads((parent/'model_seal.json').read_text()).items():
        if digest(parent/name) != expected:
            raise ValueError('previous fitted source model changed')
    for name, expected in json.loads((parent/'prediction_seal.json').read_text()).items():
        if digest(parent/name) != expected:
            raise ValueError('previous sealed predictions changed')
    data, weather_hashes = joined_data(args.cache.resolve()); raw, _ = mc.load_countries(ROOT)
    capacity_path = ROOT/'results/public-capacity-input-r1/capacity.json'
    capacity = load_capacity(capacity_path)
    fuel = np.empty((len(data), len(mc.FINE)))
    for region, group in data.groupby('region'):
        frame = raw[region].loc[group.label_row.astype(int)]
        if not np.array_equal(pd.to_datetime(frame.time, utc=True).to_numpy(), group.valid_at.to_numpy()):
            raise ValueError('absolute label/weather clocks differ')
        fuel[group.index] = frame[list(mc.FINE)].to_numpy()
        c = capacity['regions'][region]
        data.loc[group.index, 'capacity_factor'] = 400.+600.*c['coal_mw']/(c['coal_mw']+c['gas_mw'])
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT/'src/transcif/research/coarse_weather_shapes.py',
             ROOT/'tests/test_coarse_weather_shapes.py', ROOT/'docs/COARSE_WEATHER_SHAPE_PROTOCOL.md',
             parent/'input_code_seal.json', parent/'model_seal.json', parent/'prediction_seal.json',
             parent/'predictions.csv', parent/'summary.csv', parent/'calibration.json', capacity_path]
    source_hashes = {**prior_seal, **weather_hashes, **{str(p.resolve()): digest(p) for p in paths}}
    (out/'input_code_seal.json').write_text(json.dumps(source_hashes, indent=2))
    references = pd.read_csv(parent/'predictions.csv')
    reference_rows = references.loc[references.method.eq('joint_matched')]
    old_radii = {(row['configuration'], row['budget']): row['cqr_expansion']
                 for row in json.loads((parent/'calibration.json').read_text()) if row['method'] == 'joint_matched'}
    calibration, predictions, particles, model_seals = [], [], [], {}
    for target in ('DE', 'GB', 'US'):
        others = [c for c in ('DE', 'GB', 'US') if c != target]
        for source in [(others[0],), (others[1],), tuple(others)]:
            tag = '+'.join(source)+'_to_'+target
            train = data.loc[data.country.isin(source) & data.split.eq('train')].copy()
            cal = data.loc[data.country.isin(source) & data.split.eq('calibration')].copy()
            query = data.loc[data.country.eq(target) & data.split.eq('development')].copy()
            bank_path = parent/f'{tag}_matched_bank.joblib'
            bank = joblib.load(bank_path)  # own sealed source artifact only
            coarse = mc.coarse_values(pd.DataFrame(fuel[train.index], columns=mc.FINE))
            fitted = shapes.fit_crossfit(train, coarse, allowed_countries=source)
            daily = np.stack([coarse[ix] for ix in shapes.day_groups(train)])
            observed_shape = np.divide(daily, daily.mean(axis=1, keepdims=True),
                out=np.ones_like(daily), where=daily.mean(axis=1, keepdims=True) > 0)
            bank_shape = np.divide(bank['q'], bank['q'].mean(axis=1, keepdims=True),
                out=np.ones_like(bank['q']), where=bank['q'].mean(axis=1, keepdims=True) > 0)
            np.testing.assert_allclose(observed_shape, bank_shape, atol=1e-10, rtol=0.)
            p = out/f'{tag}_heads.joblib'; joblib.dump(fitted, p)
            restored = joblib.load(p)
            for a, b in zip(shapes.predict_heads(fitted['final'], query), shapes.predict_heads(restored['final'], query)):
                np.testing.assert_array_equal(a, b)
            model_seals[p.name] = digest(p)
            (out/f'{tag}_crossfit_audit.json').write_text(json.dumps(fitted['audit'], indent=2))
            source_members = shapes.predict_methods(bank, fitted, cal)
            query_members = shapes.predict_methods(bank, fitted, query)
            source_truth = fuel[cal.index]@mc.FACTORS/fuel[cal.index].sum(axis=1)
            w = mc.weights(cal)
            for budget in ('monthly', 'capacity'):
                cs = cal.capacity_factor.to_numpy() if budget == 'capacity' else np.full(len(cal), bank['macro_ef'])
                cq = query.capacity_factor.to_numpy() if budget == 'capacity' else np.full(len(query), bank['macro_ef'])
                for method in shapes.METHODS:
                    source_array = source_members[method]*cs[:, None]
                    array = query_members[method]*cq[:, None]
                    lo, hi = np.quantile(source_array, [.05, .95], axis=1)
                    radius = weighted_quantile(np.maximum.reduce([lo-source_truth, source_truth-hi, np.zeros(len(cal))]), w, .9)
                    if method in ('joint_original', 'learned_zero'):
                        np.testing.assert_allclose(radius, old_radii[tag, budget], atol=1e-10, rtol=0.)
                    calibration.append(dict(configuration=tag, budget=budget, method=method,
                                            expansion=radius, source_hours=len(cal)))
                    lo, hi = np.quantile(array, [.05, .95], axis=1)
                    for risk, expansion in [('raw', 0.), ('calibrated', radius)]:
                        rows = query[['country', 'region', 'origin', 'valid_at', 'label_row']].copy()
                        rows['configuration'], rows['budget'], rows['method'], rows['risk'] = tag, budget, method, risk
                        rows['point'] = array.mean(axis=1)
                        rows['lower'], rows['upper'] = np.maximum(lo-expansion, 0.), np.minimum(hi+expansion, 1000.)
                        if method in ('joint_original', 'learned_zero'):
                            old = reference_rows.loc[reference_rows.configuration.eq(tag) & reference_rows.budget.eq(budget) & reference_rows.risk.eq(risk)]
                            matched = rows.assign(valid_at=rows.valid_at.astype(str)).merge(old,
                                on=['region', 'valid_at'], suffixes=('_new', '_old'), validate='one_to_one')
                            if len(matched) != len(rows):
                                raise ValueError('old/new identity cohort differs')
                            for field in ('point', 'lower', 'upper'):
                                np.testing.assert_allclose(matched[field+'_new'], matched[field+'_old'], atol=1e-10, rtol=0.)
                        predictions.append(rows)
                    path = out/f'{tag}_{budget}_{method}_members.npz'
                    np.savez_compressed(path, members=array, row=query.index.to_numpy())
                    particles.append({'path': path.name, 'sha256': digest(path), 'method': method,
                                      'configuration': tag, 'budget': budget})
            print(tag, 'all shape references predicted; identity checked', flush=True)
    prediction_path = out/'predictions.csv'
    pd.concat(predictions, ignore_index=True).to_csv(prediction_path, index=False)
    (out/'calibration.json').write_text(json.dumps(calibration, indent=2))
    (out/'model_seal.json').write_text(json.dumps(model_seals, indent=2))
    prediction_seal = {'predictions.csv': digest(prediction_path), 'calibration.json': digest(out/'calibration.json'),
                       **{r['path']: r['sha256'] for r in particles}}
    (out/'prediction_seal.json').write_text(json.dumps(prediction_seal, indent=2))
    # All configurations are now sealed. Target labels enter scoring only here.
    scores = []
    pred = pd.read_csv(prediction_path)
    for record in particles:
        if digest(out/record['path']) != record['sha256']:
            raise ValueError('sealed particle file changed')
        with np.load(out/record['path'], allow_pickle=False) as archive:
            members, ix = archive['members'], archive['row']
        query = data.loc[ix]; truth = fuel[ix]@mc.FACTORS/fuel[ix].sum(axis=1)
        raw_crps = crps(members, truth)
        subset = pred.loc[pred.configuration.eq(record['configuration']) & pred.budget.eq(record['budget']) & pred.method.eq(record['method'])]
        for region, positions in query.groupby('region').indices.items():
            for risk in ('raw', 'calibrated'):
                rows = subset.loc[subset.region.eq(region) & subset.risk.eq(risk)]
                np.testing.assert_array_equal(rows.valid_at, query.iloc[positions].valid_at.astype(str))
                y = truth[positions]; point = rows.point.to_numpy()
                day = pd.DataFrame({'day': rows.origin.to_numpy(), 'y': y, 'point': point})
                centered = day[['y', 'point']]-day.groupby('day')[['y', 'point']].transform('mean')
                scores.append(dict(configuration=record['configuration'], country=query.country.iloc[0], region=region,
                    budget=record['budget'], method=record['method'], risk=risk, hours=len(rows),
                    mae=float(np.abs(point-y).mean()), shape_mae=float(np.abs(centered.point-centered.y).mean()),
                    raw_crps=float(raw_crps[positions].mean()),
                    **interval_metrics(y, rows.lower.to_numpy(), rows.upper.to_numpy())))
    metrics = pd.DataFrame(scores); metrics.to_csv(out/'region_metrics.csv', index=False)
    configs = metrics.groupby(['configuration', 'country', 'budget', 'method', 'risk']).mean(numeric_only=True).reset_index()
    configs = configs.rename(columns={'hours': 'mean_hours_per_region'})
    configs.to_csv(out/'configuration_metrics.csv', index=False)
    configs.groupby(['budget', 'method', 'risk']).mean(numeric_only=True).reset_index().to_csv(out/'summary.csv', index=False)
    pd.read_csv(parent/'summary.csv').to_csv(out/'prior_frozen_references.csv', index=False)
    for name, expected in source_hashes.items():
        if digest(Path(name)) != expected:
            raise RuntimeError('input/code changed during run')
    for name, expected in {**prediction_seal, **model_seals}.items():
        if digest(out/name) != expected:
            raise RuntimeError('sealed output changed during run')
    (out/'completion.json').write_text(json.dumps({'complete': True, 'adaptive_development': True,
        'identity_verified': True, 'target_calibration_used': False, 'new_country_opened': False,
        'novelty_established': False, 'unique_target_hours': int(data.split.eq('development').sum())}, indent=2))


if __name__ == '__main__':
    main()
