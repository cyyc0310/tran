"""Replay old-calendar source calibration on the same sealed R6a forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from collect_weather_transfer_pilot import ROOT, digest
from run_expanded_anchor_transfer import KEYS, score_group
from transcif.research import hourly_transfer_baselines as direct
from transcif.research import multicountry as mc
from transcif.research.monthly_adaptation import weighted_quantile

IDENTITY = ['country', 'region', 'origin', 'valid_at', 'label_row']
FIELDS = ('point', 'scale', 'lower', 'median', 'upper')


def identity(frame: pd.DataFrame) -> pd.MultiIndex:
    """Normalize metadata only; realized CIF cannot affect calendar membership."""
    keys = frame[IDENTITY].copy()
    for name in ('origin', 'valid_at'):
        keys[name] = pd.to_datetime(keys[name], utc=True)
    keys['label_row'] = keys.label_row.astype(int)
    result = pd.MultiIndex.from_frame(keys)
    if result.has_duplicates:
        raise ValueError('ambiguous cohort identity')
    return result


def legacy_indices(labels: pd.DataFrame, old: pd.DataFrame, source: tuple, target: str) -> np.ndarray:
    """All and only old proper-source H1 rows must be present in the new cohort."""
    if not source or target in source:
        raise ValueError('target cannot enter source calibration')
    source_rows = labels.loc[labels.split.eq('calibration') & labels.country.isin(source)]
    declared = old.loc[old.split.eq('calibration') & old.country.isin(source)]
    if declared.empty or set(declared.country) != set(source):
        raise ValueError('every proper source country needs legacy calibration rows')
    old_keys = identity(declared)
    matched = source_rows.loc[identity(source_rows).isin(old_keys)]
    if len(matched) != len(declared) or set(identity(matched)) != set(old_keys):
        raise ValueError('legacy source calendar not fully preserved')
    return matched.index.to_numpy(dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT / 'results/expanded-anchor-no-issued-r6a')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent.resolve()
    completed = json.loads((parent / 'completion.json').read_text())
    if not completed['complete'] or completed['issued_weather_used']:
        raise ValueError('completed no-issued R6a parent required')
    inputs = json.loads((parent / 'input_code_seal.json').read_text())
    for name, expected in json.loads((parent / 'prediction_seal.json').read_text()).items():
        inputs[str(parent / name)] = expected
    old_root = ROOT / 'results/hourly-direct-baselines-r5'
    old_cohort = ROOT / 'results/weather-transfer-seasonal-r1/cohort.csv'
    paths = [Path(__file__), ROOT / 'tests/test_expanded_calibration_budget.py',
        ROOT / 'docs/EXPANDED_CALIBRATION_BUDGET_PROTOCOL.md', parent / 'input_code_seal.json',
        parent / 'prediction_seal.json', parent / 'completion.json', parent / 'cohort_with_labels.csv.gz',
        parent / 'summary.csv', old_root / 'calibration.json', old_cohort]
    for path in paths:
        name, current = str(path.resolve()), digest(path)
        if name in inputs and inputs[name] != current:
            raise ValueError('inherited source/forecast changed')
        inputs[name] = current
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise ValueError('parent input or sealed component changed')
    labels = pd.read_csv(parent / 'cohort_with_labels.csv.gz')
    old = pd.read_csv(old_cohort)
    old_calibration = {(r['configuration'], r['budget'], r['method']): r
                       for r in json.loads((old_root / 'calibration.json').read_text())}
    records = json.loads((parent / 'calibration.json').read_text())
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    (output / 'input_code_seal.json').write_text(json.dumps(inputs, indent=2) + '\n')
    (output / 'execution_plan.json').write_text(json.dumps(dict(
        posthoc_development=True, target_dataset_already_viewed=True, new_model_fit=False,
        legacy_calendar=['2023-01-15', '2023-03-15', '2023-05-15'],
        expanded_calendar='2023 H1 each month 15th and 25th; original eligibility retained',
        identical_target_hours=2400, target_calibration=False, new_country_opened=False,
        source_calibration_formula='unchanged source weighted empirical 90th percentile'), indent=2) + '\n')
    predictions, calibration = [], []
    legacy_cache = {}; scalar_checks = []

    def append(query: pd.DataFrame, record: dict, risk: str, forecast: dict) -> None:
        rows = query[IDENTITY].copy()
        for name in ('configuration', 'budget', 'method'):
            rows[name] = record[name]
        rows['track'], rows['risk'] = 'paired_labels', risk
        for field in ('point', 'median', 'lower', 'upper'):
            rows[field] = forecast[field]
        predictions.append(rows)

    for record in records:
        tag, budget, method = (record[name] for name in ('configuration', 'budget', 'method'))
        source_name, target = tag.split('_to_'); source = tuple(source_name.split('+'))
        if tag not in legacy_cache:
            legacy_cache[tag] = legacy_indices(labels, old, source, target)
        ix = legacy_cache[tag]; cal = labels.iloc[ix]; y = cal.truth.to_numpy(); w = mc.weights(cal)
        expected_hours = old_calibration[tag, budget, 'paired_joint']['source_hours']
        if len(cal) != expected_hours:
            raise ValueError('legacy source fine-label budget differs from R5')
        if method.startswith('paired_'):
            old_record = old_calibration[tag, budget, method]
            signed = old_record['signed_expansion']
            with np.load(parent / f'{tag}_{budget}_{method}_members.npz', allow_pickle=False) as archive:
                query = labels.iloc[archive['row']]; members = archive['members']
            lower, median, upper = np.quantile(members, [.05, .5, .95], axis=1)
            for risk, radius in [('raw', 0.), ('calibrated', max(signed, 0.)), ('calibrated_signed', signed)]:
                append(query, record, risk, dict(point=members.mean(axis=1), median=median,
                    lower=np.clip(np.minimum(lower-radius, median), 0., 1000.),
                    upper=np.clip(np.maximum(upper+radius, median), 0., 1000.)))
            calibration.append(dict(configuration=tag, budget=budget, method=method, source_hours=len(ix),
                signed_expansion=signed, nonnegative_expansion=max(signed, 0.), provenance='R5 frozen source radii'))
        elif method == 'fixed_monthly_thermal':
            query = labels.loc[labels.country.eq(target) & labels.split.eq('development')]
            bank = joblib.load(old_root / 'models' / f'{"+".join(sorted(source))}_paired_bank.joblib')
            ef_cal = cal.capacity_factor.to_numpy() if budget == 'capacity' else np.full(len(cal), bank['macro_ef'])
            ef_query = query.capacity_factor.to_numpy() if budget == 'capacity' else np.full(len(query), bank['macro_ef'])
            point = query.thermal.to_numpy() * ef_query
            radius = weighted_quantile(np.abs(cal.thermal.to_numpy()*ef_cal-y), w, .9)
            append(query, record, 'global', dict(point=point, median=point,
                lower=np.clip(point-radius, 0., 1000.), upper=np.clip(point+radius, 0., 1000.)))
            calibration.append(dict(configuration=tag, budget=budget, method=method, source_hours=len(ix),
                                    global_radius=radius, provenance='same old source rows; no new fit'))
        else:
            with np.load(parent / f'{tag}_{budget}_{method}_components.npz', allow_pickle=False) as archive:
                rows = archive['source_row']; mask = np.isin(rows, ix)
                if not np.array_equal(rows[mask], ix):
                    raise ValueError('component source row order differs from legacy calibration')
                pc = {field: archive['source_'+field][mask] for field in FIELDS}
                pq = {field: archive[field] for field in FIELDS}
                query = labels.iloc[archive['row']]
            value = direct.calibrate(pc, y, w)
            if (tag, budget, method) in old_calibration:
                for field, actual in value.items():
                    expected = old_calibration[tag, budget, method][field]
                    np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-9)
                    scalar_checks.append(abs(actual-expected))
            for risk, forecast in direct.intervals(pq, value).items():
                append(query, record, risk, forecast)
            calibration.append(dict(configuration=tag, budget=budget, method=method, source_hours=len(ix),
                                    provenance='same old source rows and fixed components', **value))
    prediction = pd.concat(predictions, ignore_index=True)
    current = pd.read_csv(parent / 'predictions.csv')
    keys = KEYS + ['origin', 'valid_at', 'label_row']
    joined = prediction.merge(current[keys + ['point', 'median']], on=keys,
                               validate='one_to_one', suffixes=('_legacy', '_expanded'))
    if len(joined) != len(prediction) or len(joined) != len(current):
        raise ValueError('counterfactual must use exactly the same forecast cohort')
    point_errors = []
    for field in ('point', 'median'):
        error = np.abs(joined[field+'_legacy']-joined[field+'_expanded']).max()
        if error > 1e-10:
            raise ValueError('source calibration must not change point or median')
        point_errors.append(float(error))
    prediction.to_csv(output / 'predictions.csv.gz', index=False, compression='gzip')
    (output / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    (output / 'invariance_check.json').write_text(json.dumps(dict(
        prediction_rows=len(prediction), point_median_max_error=max(point_errors),
        old_direct_scalar_checks=len(scalar_checks), old_direct_scalar_max_error=max(scalar_checks)), indent=2) + '\n')
    sealed = {name: digest(output / name) for name in [
        'predictions.csv.gz', 'calibration.json', 'invariance_check.json', 'execution_plan.json']}
    (output / 'prediction_seal.json').write_text(json.dumps(sealed, indent=2) + '\n')
    # Same already-viewed target labels are only used after the counterfactual is sealed.
    scored = prediction.merge(labels[IDENTITY+['truth']], on=IDENTITY, how='left', validate='many_to_one')
    if scored.truth.isna().any():
        raise ValueError('missing scoring truth')
    region = pd.DataFrame([dict(zip(KEYS, key)) | score_group(group)
                          for key, group in scored.groupby(KEYS, sort=False)])
    region.to_csv(output / 'region_metrics.csv', index=False)
    configs = region.groupby(['configuration', 'country', 'track', 'budget', 'method', 'risk']).mean(numeric_only=True).reset_index()
    configs = configs.rename(columns={'hours': 'mean_hours_per_region'})
    configs.to_csv(output / 'configuration_metrics.csv', index=False)
    summary = configs.groupby(['track', 'budget', 'method', 'risk']).mean(numeric_only=True).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    comparison = summary.merge(pd.read_csv(parent / 'summary.csv'), on=['track', 'budget', 'method', 'risk'],
                               validate='one_to_one', suffixes=('_legacy', '_expanded'))
    for field in ('mae', 'shape_mae', 'median_mae', 'wis90', 'coverage90', 'width90', 'interval_score90'):
        comparison[field+'_delta_expanded_minus_legacy'] = comparison[field+'_expanded']-comparison[field+'_legacy']
    comparison.to_csv(output / 'paired_comparison.csv', index=False)
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise RuntimeError('frozen replay inputs changed')
    for name, expected in sealed.items():
        if digest(output / name) != expected:
            raise RuntimeError('counterfactual forecast changed')
    (output / 'completion.json').write_text(json.dumps(dict(complete=True, posthoc_development=True,
        source_training_unchanged=True, target_points_unchanged=True, identical_target_hours=2400,
        source_legacy_unique_hours=int(old.split.eq('calibration').sum()), source_expanded_unique_hours=2424,
        new_country_opened=False, novelty_established=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
