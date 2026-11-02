"""Paired official-monthly versus hourly-reconstructed local adaptation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_eia923_pseudotarget_transfer as eia923
from run_cross_country import ROOT
from transcif.research.cross_country import COARSE, EF, FUELS, labels, load_inputs, monthly_inputs
from transcif.research.monthly_adaptation import (
    METHODS, fit_risks, fit_source, monthly_table, predict_hours, risk_bounds,
)

PROTOCOL = ROOT / 'docs/EIA923_MONTHLY_ADAPTATION_PROTOCOL_20261101.md'
TARGET_YEAR = 2024


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shares_from_eia923() -> dict[str, pd.DataFrame]:
    raw = eia923.parse_monthly((2020, 2021, 2022, 2023, 2024), tuple(eia923.official.US_REGIONS))
    result = {}
    for ba, group in raw.groupby('ba', sort=True):
        table = group.set_index('observed_month')[list(COARSE)].copy()
        table.index = pd.PeriodIndex(table.index, freq='M')
        result[ba] = table.sort_index()
    return result


def fit_native(source_regions: list[str], raw: dict[str, pd.DataFrame],
               views: dict[str, dict[int, pd.DataFrame]]) -> dict:
    """Fit only the fixed EF/profile actually consumed by the adaptation predictor."""
    source_frames = []
    source_fuels = []
    for region in source_regions:
        common = None
        for lag in (2, 3, 4):
            frame = views[region][lag]
            valid = frame.index[frame.year.eq(2022) & frame[list(COARSE)].notna().all(axis=1)]
            common = valid if common is None else common.intersection(valid)
        if common is None or not len(common):
            raise ValueError(f'no common source training hours: {region}')
        frame = raw[region].loc[common].copy()
        source_frames.append(frame)
        source_fuels.append(frame[list(FUELS)].to_numpy(float))
    train = pd.concat(source_frames, ignore_index=True)
    fuel = np.concatenate(source_fuels)
    counts = train.groupby('region').region.transform('size').to_numpy(float)
    weights = 1. / counts
    weights *= len(weights) / weights.sum()
    thermal = fuel[:, :4].sum(axis=1)
    thermal_ef = float(np.sum(weights * (fuel @ EF)) / np.sum(weights * thermal))
    coarse = np.column_stack([thermal, fuel[:, 5], fuel[:, 4], fuel[:, 6], fuel[:, 7]])
    profiles = []
    for region in source_regions:
        mask = train.region.to_numpy() == region
        group = pd.DataFrame(coarse[mask], columns=COARSE)
        group['hour'] = train.loc[mask, 'clock'].dt.hour.to_numpy()
        hourly = group.groupby('hour')[list(COARSE)].mean().reindex(range(24)).fillna(0.)
        mean = hourly.mean()
        profiles.append(hourly.div(mean.replace(0., np.nan)).fillna(1.).to_numpy())
    return {'thermal_ef': thermal_ef, 'profile': np.mean(profiles, axis=0)}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    raw, audit = load_inputs(ROOT)
    views = {r: {2: monthly_inputs(f, 2), 3: monthly_inputs(f, 3), 4: monthly_inputs(f, 4)}
             for r, f in raw.items()}
    hourly_tables = {r: monthly_table(f) for r, f in raw.items()}
    eia_tables = {**hourly_tables, **shares_from_eia923()}
    arms = {'hourly_reconstructed': hourly_tables, 'eia923_us': eia_tables}
    all_frames = {r: v[2].dropna(subset=list(COARSE)) for r, v in views.items()}

    input_files = [ROOT/a['path'] for a in audit.values()]
    eia_files = [eia923.eia_archive(y) for y in (2020, 2021, 2022, 2023, 2024)]
    code_files = [Path(__file__), Path(eia923.__file__), ROOT/'src/transcif/research/monthly_adaptation.py',
                  ROOT/'src/transcif/research/cross_country.py', ROOT/'src/transcif/research/budget_transfer.py',
                  ROOT/'scripts/run_cross_country.py', ROOT/'scripts/run_monthly_adaptation.py', PROTOCOL]
    (out/'startup_hashes.json').write_text(json.dumps({
        'code': {str(p): sha(p) for p in dict.fromkeys(code_files)},
        'data': {str(p): sha(p) for p in dict.fromkeys(input_files+eia_files)},
    }, indent=2))
    (out/'input_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    (out/'eia923_monthly_shares.csv').write_text(pd.concat(
        [table.assign(ba=ba) for ba, table in shares_from_eia923().items()]
    ).reset_index(names='period').to_csv(index=False))

    records, seals, forecasts = [], [], []
    for arm, tables in arms.items():
        for source, target_country in [('US', 'GB'), ('GB', 'US')]:
            sources = [r for r in raw if raw[r].country.iloc[0] == source and r != 'BPAT']
            targets = [r for r in raw if raw[r].country.iloc[0] == target_country]
            native = fit_native(sources, raw, views)
            source_model = fit_source({r: tables[r] for r in sources})
            for method in METHODS:
                chunks = {half: [] for half in (1, 2)}
                for region in sources:
                    frame = all_frames[region].loc[all_frames[region].year.eq(2023)]
                    point, x = predict_hours(frame, tables[region], source_model, method, native)
                    error = labels(raw[region].loc[frame.index])-point
                    for half in (1, 2):
                        mask = (frame.clock.dt.month.le(6) if half == 1 else frame.clock.dt.month.gt(6)).to_numpy()
                        chunks[half].append((x[mask], error[mask], np.repeat(region, mask.sum())))
                parts = []
                for half in (1, 2):
                    parts.extend(np.concatenate([row[j] for row in chunks[half]]) for j in range(3))
                fitted = fit_risks(*parts)
                for region in targets:
                    frame = all_frames[region].loc[all_frames[region].year.eq(TARGET_YEAR)]
                    point, x = predict_hours(frame, tables[region], source_model, method, native)
                    bounds = risk_bounds(fitted, x, point)
                    payload = {'time': frame.time.astype(str), 'point_raw': point, 'point': point+fitted['bias']}
                    for risk, (lo, hi) in bounds.items():
                        payload[f'{risk}_lo'], payload[f'{risk}_hi'] = lo, hi
                    path = out/f'{arm}_{source}_{method}_to_{region}.csv'
                    pd.DataFrame(payload).to_csv(path, index=False)
                    seals.append({'arm': arm, 'source': source, 'target': region,
                                  'path': path.name, 'sha256': sha(path), 'rows': len(frame)})
                    records.append((arm, source, target_country, region, method, frame, path))
                    for period in frame.period.unique():
                        coarse = x[frame.period.eq(period).to_numpy()][0, -5:]
                        forecasts.append({'arm': arm, 'source': source, 'target': region,
                                          'method': method, 'period': str(period),
                                          'latest_allowed_month': str(period-2),
                                          **dict(zip(COARSE, coarse))})
                print(arm, source, method, 'sealed', flush=True)
    (out/'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    pd.DataFrame(forecasts).to_csv(out/'coarse_forecasts.csv', index=False)

    # Score only after all arms and methods have prediction seals.
    rows = []
    for arm, source, target_country, region, method, frame, path in records:
        pred = pd.read_csv(path)
        truth = labels(raw[region].loc[frame.index])
        for month in sorted(frame.clock.dt.month.unique()):
            mask = frame.clock.dt.month.eq(month).to_numpy()
            selected = frame.loc[mask]
            a, b = truth[mask], pred.point.to_numpy()[mask]
            tmp = pd.DataFrame({'day': selected.clock.dt.strftime('%Y-%m-%d').to_numpy(), 'y': a, 'p': b})
            centered = tmp[['y', 'p']]-tmp.groupby('day')[['y', 'p']].transform('mean')
            for risk in ('global', 'hetero', 'cqr'):
                lo, hi = pred[f'{risk}_lo'].to_numpy()[mask], pred[f'{risk}_hi'].to_numpy()[mask]
                rows.append({'arm': arm, 'source': source, 'target': target_country, 'region': region,
                             'method': method, 'risk': risk, 'month': int(month),
                             'common_month': month in (2, 4, 6), 'hours': int(mask.sum()),
                             'mae': float(np.abs(a-b).mean()),
                             'raw_mae': float(np.abs(a-pred.point_raw.to_numpy()[mask]).mean()),
                             'shape_mae': float(np.abs(centered.y-centered.p).mean()),
                             'coverage90': float(((a>=lo)&(a<=hi)).mean()),
                             'width90': float(np.mean(hi-lo)),
                             'interval_score90': float(np.mean(hi-lo+20*np.maximum(lo-a, 0)+20*np.maximum(a-hi, 0)))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out/'monthly_metrics.csv', index=False)
    metric_names = ['mae', 'raw_mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    regional = monthly.loc[monthly.common_month].groupby(
        ['arm', 'source', 'target', 'region', 'method', 'risk'])[metric_names].mean().reset_index()
    regional.to_csv(out/'common_region_metrics.csv', index=False)
    country = regional.groupby(['arm', 'source', 'target', 'method', 'risk'])[metric_names].mean().reset_index()
    country.to_csv(out/'common_country_metrics.csv', index=False)
    hashes = json.loads((out/'startup_hashes.json').read_text())
    for group in hashes.values():
        if any(sha(Path(p)) != d for p, d in group.items()):
            raise RuntimeError('code or data changed during experiment')
    (out/'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_EIA923_MONTHLY_ADAPTATION',
        'arms': list(arms), 'directions': ['US_to_GB', 'GB_to_US'],
        'training_year': 2022, 'source_risk_fit': '2023H1', 'source_calibration': '2023H2',
        'target_year': TARGET_YEAR, 'prediction_rows': len(seals),
        'monthly_vintage': 'EIA-923 current final archives; exact t-2 lag simulated, not as-issued',
        'target_response': 'fixed-EF hourly generation proxy, scored post-seal',
    }, indent=2))
    print(country.to_string(index=False))


if __name__ == '__main__':
    main()
