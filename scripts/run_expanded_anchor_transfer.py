"""Evaluate frozen ordinary models on the expanded, already-seen countries."""
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
from run_hourly_transfer_baselines import canonical_fuel, paired_recomposition
from run_weather_transfer_pilot import joined_data
from transcif.research import coarse_trajectories as ct
from transcif.research import hourly_anchor_baselines as anchored
from transcif.research import hourly_transfer_baselines as direct
from transcif.research import multicountry as mc
from transcif.research import resource_controls_daily as daily
from transcif.research import resource_controls as controls
from transcif.research.capacity_transfer import load_capacity
from transcif.research.monthly_adaptation import weighted_quantile
from transcif.research.resource_amplitudes import scale_paths

KEYS = ['configuration', 'country', 'region', 'track', 'budget', 'method', 'risk']


def recomposition(bank: dict, fitted: dict, public: pd.DataFrame, climate: pd.DataFrame) -> dict:
    """Add a label-matched cube reference; keep all three R5 paths unchanged."""
    result = paired_recomposition(bank, fitted, public, climate)
    _, cube, _ = daily.day_inputs(public, climate)
    result['paired_cube_weather'] = np.empty((len(public), 128))
    for day, positions in enumerate(daily.day_groups(public)):
        modified = {**bank, 'q': scale_paths(bank['q'], bank['support'], cube[day])}
        paths, _ = ct.predict(modified, public.iloc[positions])
        result['paired_cube_weather'][positions] = paths['joint']
    return result


def without_issued_recomposition(bank: dict, fitted: dict, public: pd.DataFrame, climate: pd.DataFrame) -> dict:
    """Never consume issued values; neutral placeholders are discarded by B heads."""
    neutral = public.copy()
    neutral.loc[:, ['temperature_c', 'wind_speed_ms', 'shortwave_w_per_m2']] = 0.
    features, _, _ = daily.day_inputs(neutral, climate)
    factors = controls.predict_amplitudes(fitted, features, 'hgb', 'B')
    result = {method: np.empty((len(public), 128)) for method in ['paired_joint', 'paired_hgb_no_weather']}
    for day, positions in enumerate(daily.day_groups(public)):
        part = public.iloc[positions]
        paths, _ = ct.predict(bank, part)
        result['paired_joint'][positions] = paths['joint']
        modified = {**bank, 'q': scale_paths(bank['q'], bank['support'], factors[day])}
        paths, _ = ct.predict(modified, part)
        result['paired_hgb_no_weather'][positions] = paths['joint']
    return result


def require_source(audit: dict, source: tuple, hours: int) -> None:
    if audit['source_countries'] != sorted(source) or audit['source_hours'] != hours:
        raise ValueError('model differs from proper paired source cohort')


def overlap_check(current: pd.DataFrame, previous: pd.DataFrame) -> dict:
    """Only point/median are invariant when the source calibration set expands."""
    retained = ((previous.method.isin(['paired_joint', 'paired_hgb_no_weather', 'paired_hgb_weather'])
                 & previous.risk.eq('raw'))
                | (previous.method.isin(['cif_no_weather', 'cif_weather', 'thermal_no_weather', 'thermal_weather'])
                   & previous.risk.isin(['global', 'cqr_nonnegative'])))
    old = previous.loc[previous.track.eq('paired_labels') & retained & previous.method.isin(current.method.unique())]
    if old.empty:
        raise ValueError('at least one frozen reference must have shared hours')
    keys = KEYS + ['valid_at', 'label_row']
    matched = old.merge(current[keys + ['point', 'median']], on=keys, how='left',
                        validate='one_to_one', suffixes=('_old', '_new'))
    if matched[['point_new', 'median_new']].isna().any().any():
        raise ValueError('old reference hours missing from expanded cohort')
    error = 0.
    for field in ('point', 'median'):
        np.testing.assert_allclose(matched[field + '_old'], matched[field + '_new'], rtol=0., atol=1e-10)
        error = max(error, float(np.abs(matched[field + '_old'] - matched[field + '_new']).max()))
    return dict(rows=len(matched), point_and_median_max_error=error,
                intervals_not_compared='source calibration deliberately expanded')


def score_group(group: pd.DataFrame) -> dict:
    centered = group[['truth', 'point']] - group.groupby('origin')[['truth', 'point']].transform('mean')
    return dict(hours=len(group), shape_mae=float(np.abs(centered.truth - centered.point).mean()),
        **direct.score_intervals(group.truth.to_numpy(), group.point.to_numpy(), group['median'].to_numpy(),
                                 group.lower.to_numpy(), group.upper.to_numpy()))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--weather-cache', type=Path, required=True)
    parser.add_argument('--anchor-models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--without-issued-weather', action='store_true',
                        help='Separate shared-climate budget; evaluate sealed cohort without reading GFS values.')
    args = parser.parse_args()
    cache, anchor_models = args.weather_cache.resolve(), args.anchor_models.resolve()
    if not args.without_issued_weather:
        weather_complete = json.loads((cache / 'collection_complete.json').read_text())
        protocol = json.loads((cache / 'plan.json').read_text())
        planned_dates = {date for dates in protocol['dates'].values() for date in dates}
        if not weather_complete['complete'] or len(planned_dates) != 24 or set(weather_complete['windows']) != planned_dates:
            raise ValueError('all 24 fixed weather windows must complete before issued-weather evaluation')
        for date, expected in weather_complete['windows'].items():
            if digest(cache / f'window-{date}/manifest.json') != expected:
                raise ValueError('expanded weather completion manifest changed')
    complete = json.loads((anchor_models / 'completion.json').read_text())
    if not complete['complete'] or complete['model_families'] != 48 or not complete['source_labels_unchanged']:
        raise ValueError('completed same-label anchor source models required')
    parent = ROOT / 'results/hourly-direct-baselines-r5'
    inputs = json.loads((anchor_models / 'input_code_seal.json').read_text())
    for name, expected in json.loads((anchor_models / 'model_seal.json').read_text()).items():
        inputs[str(anchor_models / name)] = expected
    for name, expected in json.loads((parent / 'prediction_seal.json').read_text()).items():
        inputs[str(parent / name)] = expected
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise ValueError('source model/input/metadata seal changed')
    if args.without_issued_weather:
        receipt_path = cache / 'pre_model_source_receipt.json'
        for name, expected in json.loads(receipt_path.read_text())['files'].items():
            if digest(ROOT / name) != expected:
                raise ValueError('sealed public cohort/source changed')
        data = pd.read_csv(cache / 'cohort.csv', parse_dates=['clock', 'valid_at'])
        weather_inputs = {str(receipt_path): digest(receipt_path)}
    else:
        data, weather_inputs = joined_data(cache)
    inputs.update(weather_inputs)
    for name, expected in json.loads((cache / 'pre_model_source_receipt.json').read_text())['files'].items():
        absolute = str(ROOT / name)
        if absolute in inputs and inputs[absolute] != expected:
            raise ValueError('expanded weather source conflicts with inherited inputs')
        inputs[absolute] = expected
    climate_path = ROOT / 'results/resource-saturated-reference-r1/climatology.csv'
    capacity_path = ROOT / 'results/public-capacity-input-r1/capacity.json'
    climate = pd.read_csv(climate_path); capacity = load_capacity(capacity_path)
    for region, part in data.groupby('region'):
        stock = capacity['regions'][region]
        data.loc[part.index, 'capacity_factor'] = 400. + 600. * stock['coal_mw'] / (stock['coal_mw'] + stock['gas_mw'])
    raw, _ = mc.load_countries(ROOT)
    files = [Path(__file__), ROOT / 'tests/test_expanded_anchor_transfer.py',
        ROOT / 'docs/EXPANDED_ANCHOR_TRANSFER_PROTOCOL.md', anchor_models / 'completion.json',
        anchor_models / 'input_code_seal.json', anchor_models / 'model_seal.json',
        parent / 'predictions.csv', parent / 'prediction_seal.json',
        climate_path, capacity_path]
    if not args.without_issued_weather:
        files.append(cache / 'collection_complete.json')
    for path in files:
        name, current = str(path.resolve()), digest(path)
        if name in inputs and inputs[name] != current:
            raise ValueError('frozen input changed before evaluation')
        inputs[name] = current
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    (output / 'input_code_seal.json').write_text(json.dumps(inputs, indent=2) + '\n')
    runtime = dict(python=platform.python_version(), **{
        name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib']})
    if runtime != json.loads((parent / 'runtime.json').read_text()):
        raise ValueError('runtime must match the source absolute-response reference')
    (output / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    (output / 'execution_plan.json').write_text(json.dumps(dict(
        source_candidate_hours=35448, source_labels_unchanged=True, adaptive_development=True,
        issued_weather_used=not args.without_issued_weather, shared_historical_climate=True,
        target_calibration=False, new_country_opened=False, probability_metric='WIS90',
        fixed_anchor_rule='visible thermal share times common public EF; source-global radius only',
        reference_tracks='paired labels only; no extra-full-year reference', target_model_selection=False), indent=2) + '\n')
    predictions, calibrations, member_seals = [], [], {}

    def append(query: pd.DataFrame, tag: str, budget: str, method: str, risk: str, forecast: dict) -> None:
        rows = query[['country', 'region', 'origin', 'valid_at', 'label_row']].copy()
        rows['configuration'], rows['track'], rows['budget'] = tag, 'paired_labels', budget
        rows['method'], rows['risk'] = method, risk
        for field in ('point', 'median', 'lower', 'upper'):
            rows[field] = forecast[field]
        predictions.append(rows)

    for target in ('DE', 'GB', 'US'):
        others = [country for country in ('DE', 'GB', 'US') if country != target]
        for source in [(others[0],), (others[1],), tuple(others)]:
            tag = '+'.join(source) + '_to_' + target; key = '+'.join(sorted(source))
            cal = data.loc[data.country.isin(source) & data.split.eq('calibration')].copy()
            query = data.loc[data.country.eq(target) & data.split.eq('development')].copy()
            fuel = canonical_fuel(cal, raw)
            ycal = fuel @ mc.FACTORS / fuel.sum(axis=1); wcal = mc.weights(cal)
            bank = joblib.load(parent / 'models' / f'{key}_paired_bank.joblib')
            if bank['audit']['countries'] != sorted(source):
                raise ValueError('paired bank includes a non-source country')
            hours = bank['audit']['eligible_hours']
            amplitude_path = ROOT / f'results/resource-daily-clean-r4/{tag}_heads.joblib'
            amplitude = joblib.load(amplitude_path)
            require_source(amplitude['audit'], source, hours)
            provider = without_issued_recomposition if args.without_issued_weather else recomposition
            references_cal = provider(bank, amplitude, cal, climate)
            references_query = provider(bank, amplitude, query, climate)
            for budget in ('monthly', 'capacity'):
                cs = cal.capacity_factor.to_numpy() if budget == 'capacity' else np.full(len(cal), bank['macro_ef'])
                cq = query.capacity_factor.to_numpy() if budget == 'capacity' else np.full(len(query), bank['macro_ef'])
                for method, normalized in references_query.items():
                    members = normalized * cq[:, None]; source_members = references_cal[method] * cs[:, None]
                    lo, hi = np.quantile(source_members, [.05, .95], axis=1)
                    signed = weighted_quantile(np.maximum(lo - ycal, ycal - hi), wcal, .9)
                    calibrations.append(dict(configuration=tag, budget=budget, method=method, source_hours=len(cal),
                        signed_expansion=signed, nonnegative_expansion=max(signed, 0.)))
                    lower, median, upper = np.quantile(members, [.05, .5, .95], axis=1)
                    for risk, radius in [('raw', 0.), ('calibrated', max(signed, 0.)), ('calibrated_signed', signed)]:
                        append(query, tag, budget, method, risk, dict(point=members.mean(axis=1), median=median,
                            lower=np.clip(np.minimum(lower-radius, median), 0., 1000.),
                            upper=np.clip(np.maximum(upper+radius, median), 0., 1000.)))
                    path = output / f'{tag}_{budget}_{method}_members.npz'
                    np.savez_compressed(path, members=members, row=query.index.to_numpy())
                    member_seals[path.name] = digest(path)
                for weather in ((False,) if args.without_issued_weather else (False, True)):
                    suffix = 'weather' if weather else 'no_weather'
                    xc = direct.design(cal, climate, weather=weather, capacity=budget == 'capacity')
                    xq = direct.design(query, climate, weather=weather, capacity=budget == 'capacity')
                    for kind in ('cif', 'thermal', *anchored.KINDS):
                        method = kind + '_' + suffix
                        directory = parent if kind in ('cif', 'thermal') else anchor_models
                        fitted = joblib.load(directory / 'models' / f'{key}_{budget}_{method}.joblib')
                        require_source(fitted['audit'], source, hours)
                        if fitted['audit']['budget'] != budget or fitted['audit']['weather'] != weather:
                            raise ValueError('source feature budget differs')
                        if kind in anchored.KINDS:
                            if fitted['kind'] != kind:
                                raise ValueError('anchored response differs')
                            pc = anchored.forecast(fitted, xc, cal.thermal.to_numpy(), cs/1000.)
                            pq = anchored.forecast(fitted, xq, query.thermal.to_numpy(), cq/1000.)
                        else:
                            if fitted['audit']['response'] != kind:
                                raise ValueError('absolute response differs')
                            pc = direct.forecast(fitted, xc, np.ones(len(cal)) if kind == 'cif' else cs/1000.)
                            pq = direct.forecast(fitted, xq, np.ones(len(query)) if kind == 'cif' else cq/1000.)
                        calibration = direct.calibrate(pc, ycal, wcal)
                        calibrations.append(dict(configuration=tag, budget=budget, method=method,
                            source_hours=len(cal), **calibration))
                        components = output / f'{tag}_{budget}_{method}_components.npz'
                        np.savez_compressed(components, row=query.index.to_numpy(), source_row=cal.index.to_numpy(),
                            **pq, **{'source_' + field: values for field, values in pc.items()})
                        member_seals[components.name] = digest(components)
                        for risk, forecast in direct.intervals(pq, calibration).items():
                            append(query, tag, budget, method, risk, forecast)
                point_cal = cal.thermal.to_numpy() * cs; point = query.thermal.to_numpy() * cq
                radius = weighted_quantile(np.abs(point_cal-ycal), wcal, .9)
                calibrations.append(dict(configuration=tag, budget=budget, method='fixed_monthly_thermal',
                    source_hours=len(cal), global_radius=radius))
                append(query, tag, budget, 'fixed_monthly_thermal', 'global', dict(point=point, median=point,
                    lower=np.clip(point-radius, 0., 1000.), upper=np.clip(point+radius, 0., 1000.)))
            print(tag, 'all frozen-model predictions and source calibration complete', flush=True)
    pd.concat(predictions, ignore_index=True).to_csv(output / 'predictions.csv', index=False)
    (output / 'calibration.json').write_text(json.dumps(calibrations, indent=2) + '\n')
    pred = pd.read_csv(output / 'predictions.csv')
    overlap = overlap_check(pred, pd.read_csv(parent / 'predictions.csv'))
    (output / 'overlap_reference_check.json').write_text(json.dumps(overlap, indent=2) + '\n')
    seals = {**member_seals, **{name: digest(output / name) for name in [
        'predictions.csv', 'calibration.json', 'runtime.json', 'execution_plan.json', 'overlap_reference_check.json']}}
    (output / 'prediction_seal.json').write_text(json.dumps(seals, indent=2) + '\n')
    # Target labels are consumed for scoring only after every model forecast is sealed.
    fuel = canonical_fuel(data, raw)
    labels = data.copy(); labels['truth'] = fuel @ mc.FACTORS / fuel.sum(axis=1)
    for column, name in enumerate(mc.FINE):
        labels['label_generation_' + name] = fuel[:, column]
    labels.to_csv(output / 'cohort_with_labels.csv.gz', index=False, compression='gzip')
    identity = ['country', 'region', 'origin', 'valid_at', 'label_row']
    label_values = labels[identity + ['truth']].copy(); label_values['valid_at'] = label_values.valid_at.astype(str)
    scored = pred.merge(label_values, on=identity, how='left', validate='many_to_one')
    if scored.truth.isna().any() or len(scored) != len(pred):
        raise ValueError('scoring labels do not bind every sealed prediction')
    for dimensions, filename in [(KEYS, 'region_metrics.csv'), (KEYS + ['origin'], 'day_metrics.csv')]:
        rows = [dict(zip(dimensions, key)) | score_group(group)
                for key, group in scored.groupby(dimensions, sort=False)]
        pd.DataFrame(rows).to_csv(output / filename, index=False)
    metrics = pd.read_csv(output / 'region_metrics.csv')
    configs = metrics.groupby(['configuration', 'country', 'track', 'budget', 'method', 'risk']).mean(numeric_only=True).reset_index()
    configs = configs.rename(columns={'hours': 'mean_hours_per_region'})
    configs.to_csv(output / 'configuration_metrics.csv', index=False)
    configs.groupby(['track', 'budget', 'method', 'risk']).mean(numeric_only=True).reset_index().to_csv(output / 'summary.csv', index=False)
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise RuntimeError('source/model/weather/code input changed during evaluation')
    for name, expected in seals.items():
        if digest(output / name) != expected:
            raise RuntimeError('sealed forecast changed')
    (output / 'completion.json').write_text(json.dumps(dict(complete=True, adaptive_development=True,
        source_candidate_hours=35448, unique_target_hours=int(data.split.eq('development').sum()),
        unique_calibration_hours=int(data.split.eq('calibration').sum()), source_labels_unchanged=True,
        issued_weather_used=not args.without_issued_weather, shared_historical_climate=True,
        target_calibration=False, new_country_opened=False, novelty_established=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
