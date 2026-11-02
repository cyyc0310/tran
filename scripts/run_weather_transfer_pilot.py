"""Matched-label weather information references, sealed before target scoring."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from collect_weather_transfer_pilot import ROOT, digest
from transcif.data.gfs_compact import load_compact_window
from transcif.research import coarse_trajectories as ct
from transcif.research import multicountry as mc
from transcif.research.capacity_transfer import load_capacity
from transcif.research.conditional_risk import interval_metrics
from transcif.research.monthly_adaptation import weighted_quantile

WEATHER = ('temperature_c', 'wind_speed_ms', 'shortwave_w_per_m2')


def inputs(frame: pd.DataFrame, weather: bool, capacity: bool) -> np.ndarray:
    """Public inputs only; fixed weather transforms shared by all learners."""
    columns = list(mc.HISTORY)+(['capacity_factor'] if capacity else [])
    x = frame[columns].to_numpy(float)
    if weather:
        t, w, s = frame[list(WEATHER)].to_numpy(float).T
        v = np.column_stack([t/20., w/10., s/500., np.maximum(18.-t, 0)/20.,
                             np.maximum(t-22., 0)/20., np.minimum(w, 25.)**3/1000.])
        interactions = v[:, :, None]*frame[list(mc.GROUPS)].to_numpy()[:, None, :]
        x = np.column_stack([x, v, interactions.reshape(len(frame), -1)])
    if not np.isfinite(x).all():
        raise ValueError('nonfinite public model inputs')
    return x


def point_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, kind: str,
              quantile: float | None = None) -> tuple:
    scaler = StandardScaler().fit(x, sample_weight=w)
    if kind == 'ridge':
        model = Ridge(alpha=10.)
    elif kind == 'hgb':
        model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15,
            min_samples_leaf=24, l2_regularization=10., random_state=416, early_stopping=False,
            **({'loss': 'quantile', 'quantile': quantile} if quantile is not None else {}))
    else:
        raise ValueError(kind)
    model.fit(scaler.transform(x), y, sample_weight=w*len(w))
    return scaler, model


def evaluate_model(model: tuple, x: np.ndarray) -> np.ndarray:
    scaler, fitted = model
    return fitted.predict(scaler.transform(x))


def save_checked(model: tuple, path: Path, x: np.ndarray) -> str:
    """Persist source-trained parameters and verify reload on public query rows."""
    joblib.dump(model, path)
    # These are our own just-written artifacts, never an external pickle.
    restored = joblib.load(path)
    np.testing.assert_array_equal(evaluate_model(model, x[:24]), evaluate_model(restored, x[:24]))
    return digest(path)


def shape_anchor(bank: dict, public: pd.DataFrame, factor: np.ndarray) -> np.ndarray:
    profile = np.einsum('d,dhf->hf', bank['prior'], bank['q'])
    generation = public[list(mc.GROUPS)].to_numpy()*profile[public.clock.dt.hour.to_numpy()]
    return factor*generation[:, 0]/generation.sum(axis=1)


def joined_data(cache: Path) -> tuple[pd.DataFrame, dict]:
    receipt_path = cache/'pre_model_source_receipt.json'
    for name, expected in json.loads(receipt_path.read_text())['files'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('pre-model source/producer receipt changed')
    protocol = json.loads((cache/'plan.json').read_text())
    for name, expected in json.loads((cache/'cohort_seal.json').read_text()).items():
        if digest(cache/name) != expected:
            raise ValueError('cohort seal changed')
    selected = pd.read_csv(cache/'cohort.csv', parse_dates=['clock', 'valid_at'])
    records, hashes = [], {str(receipt_path): digest(receipt_path)}
    for split, dates in protocol['dates'].items():
        for date in dates:
            directory = cache/f'window-{date}'
            weather = load_compact_window(directory, protocol['points'])
            origin = pd.Timestamp(date, tz='UTC')
            if (pd.Timestamp(weather['initialization_at']) != origin-pd.Timedelta(days=1)
                    or weather['first_lead'] != 24 or weather['end_lead'] != 72):
                raise ValueError('weather window differs from fixed plan')
            rows = pd.DataFrame(weather['rows']).rename(columns={'point': 'region'})
            rows['valid_at'] = pd.to_datetime(rows.valid_at, utc=True)
            rows['origin'] = str(origin)
            records.append(rows)
            hashes[str(directory/'manifest.json')] = digest(directory/'manifest.json')
            manifest = json.loads((directory/'manifest.json').read_text())
            hashes.update({str(directory/record['file']): record['sha256'] for record in manifest['files']})
    weather = pd.concat(records, ignore_index=True)
    merged = selected.merge(weather, on=['origin', 'region', 'valid_at'], how='left', validate='one_to_one')
    if len(merged) != len(selected) or merged[list(WEATHER)].isna().any().any():
        raise ValueError('missing or ambiguous matched weather')
    return merged, hashes


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Fail before creating an output if collection is incomplete.
    data, weather_hashes = joined_data(args.cache.resolve())
    raw, audit = mc.load_countries(ROOT)
    capacity_path = ROOT/'results/public-capacity-input-r1/capacity.json'
    capacity = load_capacity(capacity_path)
    for name, expected in capacity['input_hashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('capacity source changed')
    fuel = np.empty((len(data), len(mc.FINE)))
    for region, group in data.groupby('region'):
        label = raw[region].loc[group.label_row.astype(int)]
        actual_time = pd.to_datetime(label.time, utc=True)
        if not np.array_equal(actual_time.to_numpy(), group.valid_at.to_numpy()):
            raise ValueError('weather/label absolute clock mismatch')
        fuel[group.index] = label[list(mc.FINE)].to_numpy()
        c = capacity['regions'][region]
        data.loc[group.index, 'capacity_factor'] = 400.+600.*c['coal_mw']/(c['coal_mw']+c['gas_mw'])
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'models').mkdir()
    input_paths = [Path(__file__), ROOT/'scripts/collect_weather_transfer_pilot.py',
        ROOT/'docs/WEATHER_TRANSFER_SEASONAL_PROTOCOL.md', capacity_path,
        args.cache.resolve()/'cohort.csv', args.cache.resolve()/'plan.json',
        args.cache.resolve()/'model_plan.json',
        ROOT/'src/transcif/research/multicountry.py', ROOT/'src/transcif/research/coarse_trajectories.py',
        ROOT/'src/transcif/research/public_horizon.py']
    input_paths.extend(ROOT/'src/transcif/research'/name for name in
                       ('capacity_transfer.py', 'conditional_risk.py', 'monthly_adaptation.py', 'cross_country.py'))
    input_paths.append(ROOT/'src/transcif/data/smard.py')
    input_paths.extend(ROOT/'results/three-country-coarse-trajectories-r2'/f'{tag}_bank.joblib'
                       for tag in ['DE_to_GB', 'US_to_GB', 'DE+US_to_GB',
                                   'GB_to_DE', 'US_to_DE', 'GB+US_to_DE',
                                   'DE_to_US', 'GB_to_US', 'DE+GB_to_US'])
    source_hashes = {str(p): digest(p) for p in input_paths}
    runtime = {'python': platform.python_version(), **{name: version(name)
               for name in ('numpy', 'pandas', 'scikit-learn', 'joblib', 'eccodes')}}
    (out/'runtime.json').write_text(json.dumps(runtime, indent=2))
    for info in audit.values():
        if 'path' in info:
            p = ROOT/info['path']; source_hashes[str(p)] = digest(p)
    for year in (2022, 2023):
        p = ROOT/f'results/smard-de-{year}-full.json'; source_hashes[str(p)] = digest(p)
        for record in json.loads(p.read_text())['records']:
            p = ROOT/record['path']; source_hashes[str(p)] = digest(p)
    (out/'input_code_seal.json').write_text(json.dumps({**source_hashes, **weather_hashes}, indent=2))
    predictions, sources, calibration_records, model_hashes = [], [], [], {}
    for target in ('DE', 'GB', 'US'):
        others = [c for c in ('DE', 'GB', 'US') if c != target]
        for source in [(others[0],), (others[1],), tuple(others)]:
            tag = '+'.join(source)+'_to_'+target
            train = data.loc[data.country.isin(source) & data.split.eq('train')].copy()
            cal = data.loc[data.country.isin(source) & data.split.eq('calibration')].copy()
            query = data.loc[data.country.eq(target) & data.split.eq('development')].copy()
            if any(frame.empty for frame in (train, cal, query)):
                raise ValueError('empty fixed-cohort split')
            bank = ct.fit(train, fuel[train.index])
            bank_path = out/f'{tag}_matched_bank.joblib'
            joblib.dump(bank, bank_path)
            model_hashes[bank_path.name] = digest(bank_path)
            sources.append(dict(configuration=tag, training=len(train), calibration=len(cal),
                                development=len(query), training_regions=sorted(train.region.unique())))
            for budget in ('monthly', 'capacity'):
                fitted_predictions = {}
                for split, frame in [('train', train), ('cal', cal), ('query', query)]:
                    factor = (frame.capacity_factor.to_numpy() if budget == 'capacity'
                              else np.full(len(frame), bank['macro_ef']))
                    fitted_predictions[split] = {'flat': factor*frame.thermal.to_numpy(),
                        'shape_anchor': shape_anchor(bank, frame, factor)}
                w, wc = mc.weights(train), mc.weights(cal)
                ytrain = fuel[train.index]@mc.FACTORS/fuel[train.index].sum(axis=1)
                ycal = fuel[cal.index]@mc.FACTORS/fuel[cal.index].sum(axis=1)
                for kind in ('ridge', 'hgb', 'residual_ridge'):
                    for weather in (False, True):
                        name = kind+('_weather' if weather else '_no_weather')
                        xs = {s: inputs(f, weather, budget == 'capacity')
                              for s, f in [('train', train), ('cal', cal), ('query', query)]}
                        target_train = ytrain.copy()
                        if kind == 'residual_ridge':
                            target_train -= fitted_predictions['train']['shape_anchor']
                        model = point_fit(xs['train'], target_train, w, 'hgb' if kind == 'hgb' else 'ridge')
                        path = out/'models'/f'{tag}_{budget}_{name}_point.joblib'
                        model_hashes[str(path.relative_to(out))] = save_checked(model, path, xs['query'])
                        for split in xs:
                            point = evaluate_model(model, xs[split])
                            if kind == 'residual_ridge':
                                point += fitted_predictions[split]['shape_anchor']
                            fitted_predictions[split][name] = np.clip(point, 0., 1000.)
                for name, pcal in fitted_predictions['cal'].items():
                    error = ycal-pcal
                    radius = weighted_quantile(np.abs(error), wc, .9)
                    # Standard heteroscedastic reference: source training errors
                    # learn scale; disjoint later source days calibrate it.
                    weather = name.endswith('_weather') and not name.endswith('_no_weather')
                    xt = inputs(train, weather, budget == 'capacity')
                    xc = inputs(cal, weather, budget == 'capacity')
                    xq = inputs(query, weather, budget == 'capacity')
                    absolute = np.abs(ytrain-fitted_predictions['train'][name])
                    scale = point_fit(xt, np.log1p(absolute), w, 'ridge')
                    path = out/'models'/f'{tag}_{budget}_{name}_scale.joblib'
                    model_hashes[str(path.relative_to(out))] = save_checked(scale, path, xq)
                    sc = np.maximum(np.expm1(np.clip(evaluate_model(scale, xc), 0., 8.)), 1.)
                    sq = np.maximum(np.expm1(np.clip(evaluate_model(scale, xq), 0., 8.)), 1.)
                    normalized = weighted_quantile(np.abs(error)/sc, wc, .9)
                    calibration_records.append(dict(configuration=tag, budget=budget, method=name,
                        global_radius=radius, normalized_radius=normalized, source_hours=len(cal)))
                    point = fitted_predictions['query'][name]
                    for risk, spread in [('global', radius), ('hetero', normalized*sq)]:
                        rows = query[['region', 'country', 'valid_at', 'origin', 'label_row']].copy()
                        rows['configuration'], rows['budget'], rows['method'], rows['risk'] = tag, budget, name, risk
                        rows['point'], rows['lower'], rows['upper'] = point, np.maximum(point-spread, 0.), np.minimum(point+spread, 1000.)
                        predictions.append(rows)
                    if name.startswith('hgb_'):
                        quantiles = []
                        for quantile in (.05, .95):
                            model = point_fit(xt, ytrain, w, 'hgb', quantile)
                            path = out/'models'/f'{tag}_{budget}_{name}_quantile{quantile}.joblib'
                            model_hashes[str(path.relative_to(out))] = save_checked(model, path, xq)
                            quantiles.append((evaluate_model(model, xc), evaluate_model(model, xq)))
                        ca, cb = quantiles[0][0], quantiles[1][0]
                        cqr = weighted_quantile(np.maximum.reduce([
                            np.minimum(ca, cb)-ycal, ycal-np.maximum(ca, cb), np.zeros(len(cal))]), wc, .9)
                        calibration_records[-1]['cqr_expansion'] = cqr
                        a, b = quantiles[0][1], quantiles[1][1]
                        rows = query[['region', 'country', 'valid_at', 'origin', 'label_row']].copy()
                        rows['configuration'], rows['budget'], rows['method'], rows['risk'] = tag, budget, name, 'cqr'
                        rows['point'] = point
                        rows['lower'], rows['upper'] = np.clip(np.minimum(a, b)-cqr, 0., 1000.), np.clip(np.maximum(a, b)+cqr, 0., 1000.)
                        predictions.append(rows)
                # Matched bank and frozen full-source reference, visibly distinct
                # label budgets. Both get this pilot's identical calibration days.
                for name, reference in [('joint_matched', bank),
                        ('joint_full_source_extra_labels', joblib.load(ROOT/'results/three-country-coarse-trajectories-r2'/f'{tag}_bank.joblib'))]:
                    arrays = []
                    for frame in (cal, query):
                        values, _ = ct.predict(reference, frame)
                        factor = frame.capacity_factor.to_numpy() if budget == 'capacity' else np.full(len(frame), reference['macro_ef'])
                        arrays.append(values['joint']*factor[:, None])
                    lo, hi = np.quantile(arrays[0], [.05, .95], axis=1)
                    radius = weighted_quantile(np.maximum.reduce([lo-ycal, ycal-hi, np.zeros(len(ycal))]), wc, .9)
                    calibration_records.append(dict(configuration=tag, budget=budget, method=name,
                        cqr_expansion=radius, source_hours=len(cal)))
                    lo, hi = np.quantile(arrays[1], [.05, .95], axis=1)
                    for risk, expansion in [('raw', 0.), ('calibrated', radius)]:
                        rows = query[['region', 'country', 'valid_at', 'origin', 'label_row']].copy()
                        rows['configuration'], rows['budget'], rows['method'], rows['risk'] = tag, budget, name, risk
                        rows['point'] = arrays[1].mean(axis=1)
                        rows['lower'], rows['upper'] = np.maximum(lo-expansion, 0.), np.minimum(hi+expansion, 1000.)
                        predictions.append(rows)
            print(tag, 'predicted', flush=True)
    prediction_path = out/'predictions.csv'
    pd.concat(predictions, ignore_index=True).to_csv(prediction_path, index=False)
    seal = {'predictions.csv': digest(prediction_path)}
    (out/'prediction_seal.json').write_text(json.dumps(seal, indent=2))
    (out/'source_cohorts.json').write_text(json.dumps(sources, indent=2))
    (out/'calibration.json').write_text(json.dumps(calibration_records, indent=2))
    (out/'model_seal.json').write_text(json.dumps(model_hashes, indent=2))
    # Target labels enter the scorer only after every configuration is sealed.
    pred = pd.read_csv(prediction_path)
    if digest(prediction_path) != seal['predictions.csv']:
        raise ValueError('predictions changed')
    metrics = []
    for keys, group in pred.groupby(['configuration', 'country', 'region', 'budget', 'method', 'risk']):
        region = keys[2]; f = raw[region].loc[group.label_row.astype(int), list(mc.FINE)].to_numpy()
        y = f@mc.FACTORS/f.sum(axis=1); point = group.point.to_numpy()
        daily = pd.DataFrame({'day': group.origin.to_numpy(), 'y': y, 'point': point})
        centered = daily[['y', 'point']]-daily.groupby('day')[['y', 'point']].transform('mean')
        metrics.append(dict(zip(['configuration', 'country', 'region', 'budget', 'method', 'risk'], keys)) |
            {'hours': len(group), 'mae': float(np.abs(point-y).mean()),
             'shape_mae': float(np.abs(centered.point-centered.y).mean()),
             **interval_metrics(y, group.lower.to_numpy(), group.upper.to_numpy())})
    result = pd.DataFrame(metrics); result.to_csv(out/'region_metrics.csv', index=False)
    config = result.groupby(['configuration', 'budget', 'method', 'risk']).mean(numeric_only=True).reset_index()
    config = config.rename(columns={'hours': 'mean_hours_per_region'})
    config.to_csv(out/'configuration_metrics.csv', index=False)
    summary = config.groupby(['budget', 'method', 'risk']).mean(numeric_only=True).reset_index()
    summary.to_csv(out/'summary.csv', index=False)
    for name, expected in {**source_hashes, **weather_hashes}.items():
        if digest(Path(name)) != expected:
            raise RuntimeError('inputs/code changed during experiment')
    for name, expected in model_hashes.items():
        if digest(out/name) != expected:
            raise RuntimeError('saved model changed during experiment')


if __name__ == '__main__':
    main()
