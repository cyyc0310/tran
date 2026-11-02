#!/usr/bin/env python3
"""Paired source-feature budget experiment for US-to-GB transfer."""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import run_official_intensity_transfer as official
import run_official_pseudotarget_ba_backoff as pseudo
from run_cross_country import ROOT

PROTOCOL = ROOT / 'docs/EIA923_PSEUDOTARGET_MONTHLY_TRANSFER_PROTOCOL_20261101.md'
EVIDENCE = ROOT / 'docs/evidence/eia923-monthly-pseudotarget-r1'
BA_PROTOCOL = ROOT / 'docs/OFFICIAL_PSEUDOTARGET_BA_BACKOFF_PROTOCOL_20261101.md'
MIX_FIELDS = ('coal', 'gas', 'oil', 'other', 'nuclear', 'hydro', 'wind', 'solar')
FUEL_MAP = {'COL': 'coal', 'NG': 'gas', 'DFO': 'oil', 'RFO': 'oil', 'PC': 'oil',
            'NUC': 'nuclear', 'WAT': 'hydro', 'HYC': 'hydro', 'HPS': 'hydro',
            'WND': 'wind', 'SUN': 'solar'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eia_archive(year: int) -> Path:
    if year in (2020, 2021):
        return EVIDENCE / 'input' / f'f923_{year}.zip'
    return ROOT / 'docs/evidence/eia923-ba-mapping-r1' / f'f923_{year}.zip'


def parse_monthly(years: tuple[int, ...], bas: tuple[str, ...]) -> pd.DataFrame:
    """Aggregate official monthly net generation using the schema for each year."""
    output = []
    for year in years:
        path = eia_archive(year)
        with ZipFile(path) as archive:
            member = next(n for n in archive.namelist()
                          if n.startswith('EIA923_Schedules_2_3_4_5_M_') and n.endswith('.xlsx'))
            workbook = load_workbook(BytesIO(archive.read(member)), read_only=True, data_only=True)
            try:
                rows = workbook['Page 1 Generation and Fuel Data'].iter_rows(min_row=6, values_only=True)
                header = next(rows)
                columns = {str(v).replace('\n', ' ').strip(): i for i, v in enumerate(header) if v is not None}
                ba_field = 'Balancing Authority Code'
                fuel_field = next((f for f in ('MER Fuel Type Code', 'Reported Fuel Type Code') if f in columns), None)
                legacy_prime_mover = columns.get('Reported Prime Mover') if fuel_field == 'Reported Fuel Type Code' else None
                if ba_field not in columns or fuel_field is None:
                    raise ValueError(f'EIA-923 BA/fuel columns missing in {year}: {columns.keys()}')
                month_columns = [columns[f'Netgen {calendar.month_name[m]}'] for m in range(1, 13)]
                totals: dict[tuple[str, int, str], float] = defaultdict(float)
                for row in rows:
                    if not row or row[0] is None:
                        continue
                    ba = str(row[columns[ba_field]] or '').strip()
                    if ba not in bas:
                        continue
                    fuel = str(row[columns[fuel_field]] or '').strip()
                    prime_mover = str(row[legacy_prime_mover] or '').strip() if legacy_prime_mover is not None else ''
                    category = FUEL_MAP.get(fuel, 'other')
                    for month, col in enumerate(month_columns, 1):
                        try:
                            value = float(row[col])
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(value):
                            # Legacy reports encoded pumped storage as WAT plus
                            # prime mover PS. Treat negative net generation as
                            # storage consumption, not hydro output.
                            if fuel == 'WAT' and prime_mover == 'PS':
                                value = max(value, 0.)
                            totals[(ba, month, category)] += value
                for ba in bas:
                    for month in range(1, 13):
                        record = {'ba': ba, 'observed_month': f'{year}-{month:02d}'}
                        record.update({field: totals[(ba, month, field)] for field in MIX_FIELDS})
                        total = sum(record[field] for field in MIX_FIELDS)
                        if total <= 0:
                            raise ValueError(f'nonpositive EIA-923 monthly generation: {ba} {year}-{month:02d}')
                        record.update({
                            'thermal': sum(record[f] for f in ('coal', 'gas', 'oil', 'other')) / total,
                            'hydro': record['hydro'] / total,
                            'nuclear': record['nuclear'] / total,
                            'wind': record['wind'] / total,
                            'solar': record['solar'] / total,
                        })
                        output.append(record)
            finally:
                workbook.close()
    return pd.DataFrame(output)


def replace_source_features(prepared: dict[str, pd.DataFrame], monthly: pd.DataFrame) -> tuple[dict, dict]:
    """Create EIA-923 M-budget source features and common-support baseline frames."""
    eia_prepared, support = {}, {}
    shares = monthly.set_index(['ba', 'observed_month'])
    for region in official.US_REGIONS:
        base = prepared[region].copy()
        lookup = shares.reindex(pd.MultiIndex.from_arrays(
            [np.repeat(region, len(base)), base.observed_month.astype(str)], names=['ba', 'observed_month']))
        valid = lookup[list(official.COARSE)].notna().all(axis=1).to_numpy()
        base = base.loc[valid].reset_index(drop=True)
        lookup = lookup.loc[valid].reset_index(drop=True)
        for field in official.COARSE:
            base[field] = lookup[field].to_numpy(float)
        for field in official.COARSE:
            for suffix in ('sin', 'cos'):
                base[f'{field}_hour_{suffix}'] = base[field] * base[f'hour_{suffix}']
        eia_prepared[region] = base
        support[region] = set(pd.to_datetime(base.time, utc=True).astype(str))
    common = {}
    for region in official.US_REGIONS:
        base = prepared[region].copy()
        keep = pd.to_datetime(base.time, utc=True).astype(str).isin(support[region])
        common[region] = base.loc[keep].reset_index(drop=True)
    common['GB'] = prepared['GB'].copy()
    eia_prepared['GB'] = prepared['GB'].copy()
    return {'hourly_reconstruction': common, 'eia923_monthly': eia_prepared}, {
        region: {'baseline_rows': len(common[region]), 'eia923_rows': len(eia_prepared[region]),
                 'common_support_rows': len(common[region])}
        for region in official.US_REGIONS
    }


def collect_for(regions, prepared, labels, year, months=None):
    return pseudo.collect(regions, prepared, labels, year, months)


def process_arm(name: str, prepared: dict, raw: dict, labels: dict, out: Path,
                code_hashes: dict, data_hashes: dict) -> tuple[Path, dict]:
    source = list(official.US_REGIONS)
    profile = pseudo.prototypes(prepared, source)
    rows, records = [], []
    alpha_values = (0., .25, .5, .75, 1.)
    candidate_grid = [(0., np.inf)] + [(a, t) for a in alpha_values[1:] for t in pseudo.TAUS]
    candidate_keys = [f'alpha={a:.2f}|tau={"inf" if np.isinf(t) else str(t)}' for a, t in candidate_grid]
    candidate_values = dict(zip(candidate_keys, candidate_grid))
    for pseudo_ba in source:
        valid = prepared[pseudo_ba].loc[prepared[pseudo_ba].year.eq(2023)
                                         & prepared[pseudo_ba].clock.dt.month.le(6)].copy()
        valid, truth = pseudo.aligned(pseudo_ba, valid, labels)
        experts = [r for r in source if r != pseudo_ba]
        pooled_train, pooled_y = collect_for(experts, prepared, labels, 2022)
        pooled = pseudo.fit_booster(pooled_train, pooled_y)
        pooled_pred = pooled.predict(valid[list(pseudo.FEATURES)])
        inner = {}
        for excluded in experts:
            train_regions = [r for r in experts if r != excluded]
            train, y = collect_for(train_regions, prepared, labels, 2022)
            inner[excluded] = pseudo.fit_booster(train, y)
        cache = {('inf' if np.isinf(t) else str(t)): pseudo.ensemble_prediction(
            valid, experts, inner, profile, t) for t in pseudo.TAUS}
        for (alpha, tau), candidate in zip(candidate_grid, candidate_keys):
            expert_pred, weights = cache['inf' if np.isinf(tau) else str(tau)]
            prediction = pooled_pred if alpha == 0 else (1-alpha)*pooled_pred + alpha*expert_pred
            rows.append({'arm': name, 'pseudo_target': pseudo_ba, 'candidate': candidate,
                         'alpha': alpha, 'tau': 'inf' if np.isinf(tau) else tau,
                         'mae': float(np.abs(truth-prediction).mean()),
                         'shape_mae': pseudo.shape_mae(valid.clock, truth, prediction),
                         'hours': len(valid), 'pooled_mae': float(np.abs(truth-pooled_pred).mean()),
                         'expert_mae': float(np.abs(truth-expert_pred).mean()),
                         'mean_max_expert_weight': float(weights.max(axis=1).mean())})
        records.append({'pseudo_target': pseudo_ba, 'validation_hours': len(valid), 'experts': experts})
        print(name, 'pseudo-target', pseudo_ba, 'validated', len(valid), flush=True)
    validation = pd.DataFrame(rows)
    selection = validation.groupby('candidate', sort=False).mae.mean().sort_values(kind='mergesort')
    selected = str(selection.index[0]); alpha, tau = candidate_values[selected]
    validation.to_csv(out / f'{name}_pseudo_target_validation.csv', index=False)
    (out / f'{name}_selection.json').write_text(json.dumps({
        'selected_candidate': selected, 'selected_alpha': alpha,
        'selected_tau': 'inf' if np.isinf(tau) else tau,
        'equal_ba_macro_mae': {str(k): float(v) for k, v in selection.items()},
        'folds': records,
    }, indent=2))

    train, y_train = collect_for(source, prepared, labels, 2022)
    pooled = official.train_official(train, y_train)
    fuel_parts = []
    for region in source:
        q = prepared[region].loc[prepared[region].year.eq(2022)].copy()
        q, _ = pseudo.aligned(region, q, labels)
        fuel_parts.append(raw[region].iloc[q.raw_row.to_numpy()][list(official.FUELS)].to_numpy(float))
    proxy = official.fit_models(train.reset_index(drop=True), np.concatenate(fuel_parts))
    experts = {}
    for excluded in source:
        q, y = collect_for([r for r in source if r != excluded], prepared, labels, 2022)
        experts[excluded] = pseudo.fit_booster(q, y)

    method_names = list(pseudo.SCORED_METHODS) + ['capacity_ridge', 'capacity_boosting']
    residual_parts = {m: [] for m in method_names}
    for region in source:
        cal = prepared[region].loc[prepared[region].year.eq(2023)
                                   & prepared[region].clock.dt.month.between(7, 12)].copy()
        cal, truth = pseudo.aligned(region, cal, labels)
        pred = official.point_predictions(pooled, proxy, cal)
        pu, _ = pseudo.ensemble_prediction(cal, source, experts, profile, np.inf)
        pg, _ = pseudo.ensemble_prediction(cal, source, experts, profile, tau)
        pred.update({'pseudo_uniform': pu, 'pseudo_gated': pg,
                     'pseudo_backoff': (1-alpha)*pred['coarse_boosting']+alpha*pg})
        for method in method_names:
            residual_parts[method].append(truth-pred[method])
    calibration = pseudo.calibrate_residuals(residual_parts)

    query = pseudo.query_for(prepared, 'GB', 2024)
    pred = official.point_predictions(pooled, proxy, query)
    pu, _ = pseudo.ensemble_prediction(query, source, experts, profile, np.inf)
    pg, weights = pseudo.ensemble_prediction(query, source, experts, profile, tau)
    pred.update({'pseudo_uniform': pu, 'pseudo_gated': pg,
                 'pseudo_backoff': (1-alpha)*pred['coarse_boosting']+alpha*pg})
    path = out / f'{name}_predictions.csv.gz'
    pd.DataFrame({'time_utc': pd.to_datetime(query.time, utc=True).astype(str).to_numpy(),
                  'observed_month': query.observed_month.to_numpy(),
                  'gated_max_expert_weight': weights.max(axis=1), **pred}).to_csv(
                      path, index=False, compression='gzip')
    seal = {'arm': name, 'source': 'US', 'target': 'GB', 'path': path.name, 'sha256': sha(path),
            'hours_predicted': len(query), 'selected_alpha': alpha,
            'selected_tau': 'inf' if np.isinf(tau) else tau}
    (out / f'{name}_prediction_seal.json').write_text(json.dumps(seal, indent=2))
    if any(sha(Path(p)) != digest for p, digest in code_hashes.items()):
        raise RuntimeError('code/protocol changed during prediction')
    if any(sha(Path(p)) != digest for p, digest in data_hashes.items()):
        raise RuntimeError('input changed during prediction')
    (out / f'{name}_calibration.json').write_text(json.dumps(calibration, indent=2))
    return path, {'query': query, 'calibration': calibration, 'methods': method_names}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output
    out.mkdir(parents=True, exist_ok=False)
    table = official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    for name, digest in table['input_hashes'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'annual capacity input changed: {name}')
    source_assets = [official.SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz'
                     for r in official.US_REGIONS]
    source_assets += [official.SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz',
                      official.EIA_ROOT/'feature_inputs/GB_generation_2022_2024.csv.gz']
    if not all(p.is_file() for p in source_assets):
        subprocess.run([sys.executable, str(official.PREPARE_LABELS)], check=True)
    raw, baseline, labels, audit = official.prepare(ROOT, table)
    monthly = parse_monthly((2020, 2021, 2022, 2023, 2024), official.US_REGIONS)
    arms, support = replace_source_features(baseline, monthly)
    eia_files = [eia_archive(y) for y in (2020, 2021, 2022, 2023, 2024)]
    code_files = [Path(__file__), Path(official.__file__), Path(pseudo.__file__),
                  official.PREPARE_LABELS, ROOT/'scripts/audit_eia923_ba_mix_alignment.py',
                  ROOT/'scripts/run_cross_country.py', ROOT/'src/transcif/research/cross_country.py',
                  ROOT/'src/transcif/research/capacity_transfer.py', PROTOCOL, BA_PROTOCOL,
                  official.PROTOCOL, official.CAPACITY_PROTOCOL,
                  ROOT/'scripts/prepare_official_annual_capacity_budget.py']
    data_files = [*eia_files, official.EIA_ROOT/'acquisition_manifest.json',
                  official.EIA_ROOT/'coverage_audit_2022_2024.json', official.CAPACITY_TABLE_PATH,
                  official.SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz',
                  official.EIA_ROOT/'feature_inputs/GB_generation_2022_2024.csv.gz',
                  *[official.SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz' for r in official.US_REGIONS],
                  *[ROOT/f'results/research/data/{r}_hourly.csv' for r in official.US_REGIONS],
                  *[ROOT/name for name in table['input_hashes']]]
    hashes = {'code': {str(p): sha(p) for p in dict.fromkeys(code_files)},
              'data': {str(p): sha(p) for p in dict.fromkeys(data_files)}}
    (out/'startup_hashes.json').write_text(json.dumps(hashes, indent=2))
    (out/'input_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    (out/'support_audit.json').write_text(json.dumps(support, indent=2))
    monthly.to_csv(out/'eia923_monthly_ba_mix.csv', index=False)
    seals = {}; run_info = {}
    for arm_name, prepared in arms.items():
        path, info = process_arm(arm_name, prepared, raw, labels, out, hashes['code'], hashes['data'])
        seals[arm_name] = json.loads((out/f'{arm_name}_prediction_seal.json').read_text())
        run_info[arm_name] = info
    (out/'paired_prediction_seal.json').write_text(json.dumps(seals, indent=2))
    # Target labels are loaded only after both prediction files and their seals exist.
    subprocess.run([sys.executable, str(official.PREPARE_LABELS), '--target-labels'], check=True)
    target_file = official.TARGET_LABEL_ROOT/'GB_consumption_2024_halfhour.csv.gz'
    target_hash = {str(target_file.relative_to(ROOT)): sha(target_file)}
    target_labels = official.source_labels('GB', years=(2024,))
    monthly_rows = []
    for arm_name, path in ((n, out/f'{n}_predictions.csv.gz') for n in arms):
        info = run_info[arm_name]; query = info['query']
        truth = official.aligned_labels(query, target_labels)
        scores = pd.read_csv(path)
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            for method in info['methods']:
                metric = official.score(truth[mask], scores.loc[mask, method].to_numpy(float),
                    info['calibration'][method]['bias'], info['calibration'][method]['sigma'],
                    query.loc[mask, 'clock'].reset_index(drop=True))
                monthly_rows.append({'arm': arm_name, 'source': 'US', 'target': 'GB',
                                     'method': method, 'month': int(month), **metric})
    monthly_metrics = pd.DataFrame(monthly_rows)
    monthly_metrics.to_csv(out/'monthly_metrics.csv', index=False)
    perf = ['mae', 'shape_mae', 'crps_gaussian', 'coverage90', 'width90', 'interval_score90']
    country = monthly_metrics.groupby(['arm', 'source', 'target', 'method'])[perf].mean().reset_index()
    country.to_csv(out/'country_metrics.csv', index=False)
    (out/'target_scoring_hashes.json').write_text(json.dumps(target_hash, indent=2))
    (out/'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_US_TO_GB_EIA923_MONTHLY_PSEUDOTARGET_COMPARISON',
        'target_label_used_before_both_prediction_seals': False,
        'target_gb_2024_previously_viewed_in_prior_experiment': True,
        'source_training_year': 2022, 'pseudo_target_selection_period': 'US BA 2023H1',
        'source_calibration_period': 'US BA 2023H2', 'target_score_year': 2024,
        'target_inputs': 'calendar and exact t-2-or-older coarse monthly shares',
        'eia923_vintage': 'current final files; lag simulated, not as-issued',
        'common_support': support, 'arms': list(arms), 'capacity_inputs_used': True,
        'actual_meta_learning': False,
    }, indent=2))
    print(country.to_string(index=False))


if __name__ == '__main__':
    main()
