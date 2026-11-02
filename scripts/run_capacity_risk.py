"""Compare source-only conditional intervals around sealed cross-country forecasts."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from transcif.research import capacity_transfer, conditional_risk, multicountry
from run_cross_country import ROOT, digest

POINTS = ('anchor700', 'macro_recomposition', 'capacity_anchor_shaped',
          'capacity_source_rate', 'capacity_factor')


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    parent = ROOT / 'results/three-country-capacity-r2'
    table_path = ROOT / 'results/public-capacity-input-r1/capacity.json'
    table = capacity_transfer.load_capacity(table_path)
    paths = [Path(__file__), ROOT / 'src/transcif/research/conditional_risk.py',
             ROOT / 'src/transcif/research/capacity_transfer.py',
             ROOT / 'src/transcif/research/multicountry.py',
             ROOT / 'src/transcif/research/monthly_adaptation.py',
             ROOT / 'src/transcif/research/cross_country.py', ROOT / 'src/transcif/data/smard.py',
             ROOT / 'scripts/run_cross_country.py', ROOT / 'docs/CAPACITY_CONDITIONAL_RISK_PROTOCOL.md',
             table_path, parent / 'input_audit.json', parent / 'prediction_seal.json']
    paths += list(parent.glob('*.joblib')) + list(parent.glob('*_metadata.json'))
    for item in json.loads((parent / 'prediction_seal.json').read_text()):
        path = parent / item['path']
        if digest(path) != item['sha256']:
            raise ValueError(f'parent point seal changed: {path}')
        paths.append(path)
    for name, expected in table['input_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'capacity provenance changed: {name}')
        paths.append(ROOT / name)
    for item in json.loads((parent / 'input_audit.json').read_text()).values():
        if 'path' in item:
            paths.append(ROOT / item['path'])
    for year in (2022, 2023):
        manifest = ROOT / f'results/smard-de-{year}-full.json'
        paths.append(manifest)
        paths.extend(ROOT / r['path'] for r in json.loads(manifest.read_text())['records'])
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    (out / 'input_code_seal.json').write_text(json.dumps(hashes, indent=2))
    raw, _ = multicountry.load_countries(ROOT)
    public = {}
    cohorts = []
    for region, frame in raw.items():
        f = multicountry.public_features(multicountry.public_months(frame), frame.clock)
        f['country'], f['region'], f['raw_row'] = frame.country.iloc[0], region, f.index
        f = f.dropna(subset=list(multicountry.HISTORY))
        f = capacity_transfer.add_capacity(f, table)
        public[region] = f
        h1 = f.loc[f.clock.dt.year.eq(2023) & f.clock.dt.month.le(6)]
        cohorts.extend({'region': region, 'country': frame.country.iloc[0], 'month': int(m), 'hours': int(n)}
                       for m, n in h1.groupby(h1.clock.dt.month).size().items())
    pd.DataFrame(cohorts).to_csv(out / 'source_period_cohorts.csv', index=False)
    records, seals, invariants, support = [], [], [], []
    for metadata_path in sorted(parent.glob('*_metadata.json')):
        tag = metadata_path.name.removesuffix('_metadata.json')
        metadata = json.loads(metadata_path.read_text())
        source_countries, target = metadata['source_countries'], metadata['target_country']
        source_regions = [r for r, f in raw.items() if f.country.iloc[0] in source_countries]
        model = joblib.load(parent / f'{tag}.joblib')
        cap_model = joblib.load(parent / f'{tag}_capacity.joblib')
        frames = [public[r].loc[public[r].clock.dt.year.eq(2023) & public[r].clock.dt.month.le(6)]
                  for r in source_regions]
        source = pd.concat(frames, ignore_index=True)
        fuel = np.concatenate([raw[r].loc[f.raw_row, list(multicountry.FINE)].to_numpy()
                               for r, f in zip(source_regions, frames)])
        y = fuel @ multicountry.FACTORS / fuel.sum(axis=1)
        predictions = multicountry.predict(model, source)
        predictions.update(capacity_transfer.predict(cap_model, source))
        train_mask = source.clock.dt.month.le(3).to_numpy()
        cal_mask = ~train_mask
        for mask in (train_mask, cal_mask):
            if set(source.loc[mask, 'country']) != set(source_countries):
                raise ValueError('a source country is absent from a risk period')
        risks, summary = {}, {}
        for method in POINTS:
            error = y - predictions[method]
            risks[method] = conditional_risk.fit(
                source.loc[train_mask], error[train_mask], predictions[method][train_mask],
                source.loc[cal_mask], error[cal_mask], predictions[method][cal_mask])
            summary[method] = {k: risks[method][k] for k in ('bias', *conditional_risk.KINDS)}
        joblib.dump(risks, out / f'{tag}_risk.joblib')
        (out / f'{tag}_risk.json').write_text(json.dumps({
            'source_countries': source_countries, 'target': target,
            'risk_fit_hours': int(train_mask.sum()), 'risk_cal_hours': int(cal_mask.sum()),
            'parameters': summary}, indent=2))
        for region, f in public.items():
            if raw[region].country.iloc[0] != target:
                continue
            query = f.loc[f.clock.dt.year.eq(2023) & f.clock.dt.month.gt(6)]
            old = pd.read_csv(parent / f'{tag}_{region}.csv')
            point = multicountry.predict(model, query)
            point.update(capacity_transfer.predict(cap_model, query))
            # Match the parent writer/parser before comparing binary float values.
            checked = pd.read_csv(io.StringIO(pd.DataFrame({
                'time': raw[region].loc[query.raw_row, 'time'].astype(str), **point}).to_csv(index=False)))
            pd.testing.assert_frame_equal(checked[['time', *POINTS]], old[['time', *POINTS]], check_exact=True)
            invariants.append({'configuration': tag, 'region': region,
                               'all_five_parent_points_exactly_equal': True})
            data = {'time': old.time}
            for method in POINTS:
                values = old[method].to_numpy()
                data[f'{method}_point'] = values
                intervals = conditional_risk.bounds(risks[method], query, values)
                radius = metadata['radii'][method]
                intervals['global_h1'] = (values - radius, values + radius)
                for kind, (lower, upper) in intervals.items():
                    data[f'{method}_{kind}_lo'] = lower
                    data[f'{method}_{kind}_hi'] = upper
            path = out / f'{tag}_{region}.csv'
            pd.DataFrame(data).to_csv(path, index=False)
            seals.append({'path': path.name, 'sha256': digest(path)})
            records.append((tag, target, region, query, path))
            for name in ('capacity_coal_share', 'thermal', 'thermal_change', 'nuclear'):
                support.append({'configuration': tag, 'region': region, 'feature': name,
                                'source_q1_min': float(source.loc[train_mask, name].min()),
                                'source_q1_max': float(source.loc[train_mask, name].max()),
                                'source_q2_min': float(source.loc[cal_mask, name].min()),
                                'source_q2_max': float(source.loc[cal_mask, name].max()),
                                'target_min': float(query[name].min()), 'target_max': float(query[name].max())})
        print(tag, 'conditional intervals sealed', flush=True)
    (out / 'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    (out / 'parent_point_invariance.json').write_text(json.dumps(invariants, indent=2))
    pd.DataFrame(support).to_csv(out / 'public_feature_support.csv', index=False)
    rows = []
    for tag, target, region, query, path in records:
        if digest(path) != next(s['sha256'] for s in seals if s['path'] == path.name):
            raise RuntimeError('sealed interval file changed before scoring')
        predicted = pd.read_csv(path, float_precision='round_trip')
        fuel = raw[region].loc[query.raw_row, list(multicountry.FINE)].to_numpy()
        y = fuel @ multicountry.FACTORS / fuel.sum(axis=1)
        if ((y < 0) | (y > 1000.)).any():
            raise ValueError('label outside the fixed-proxy support')
        for method in POINTS:
            point = predicted[f'{method}_point'].to_numpy()
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                day = query.loc[mask].clock.dt.strftime('%Y-%m-%d').to_numpy()
                temp = pd.DataFrame({'y': y[mask], 'p': point[mask], 'day': day})
                centered = temp[['y', 'p']] - temp.groupby('day')[['y', 'p']].transform('mean')
                for kind in ('global_h1', *conditional_risk.KINDS):
                    a = predicted[f'{method}_{kind}_lo'].to_numpy()[mask]
                    b = predicted[f'{method}_{kind}_hi'].to_numpy()[mask]
                    for clipping in ('raw', 'common_support'):
                        lo, hi = (a, b) if clipping == 'raw' else (np.clip(a, 0., 1000.), np.clip(b, 0., 1000.))
                        rows.append({'configuration': tag, 'target': target, 'region': region,
                                     'method': method, 'risk': kind, 'bounds': clipping, 'month': int(month),
                                     'hours': int(mask.sum()), 'mae': float(np.abs(y[mask] - point[mask]).mean()),
                                     'shape_mae': float(np.abs(centered.y - centered.p).mean()),
                                     **conditional_risk.interval_metrics(y[mask], lo, hi)})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / 'monthly_metrics.csv', index=False)
    fields = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    regional = monthly.groupby(['configuration', 'target', 'region', 'method', 'risk', 'bounds'])[fields].mean().reset_index()
    regional.to_csv(out / 'region_metrics.csv', index=False)
    country = regional.groupby(['configuration', 'target', 'method', 'risk', 'bounds'])[fields].mean().reset_index()
    country.to_csv(out / 'country_metrics.csv', index=False)
    macro = country.groupby(['method', 'risk', 'bounds'])[fields].mean().reset_index()
    macro.to_csv(out / 'macro_metrics.csv', index=False)
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f'input/code changed during run: {name}')
    (out / 'terminal_verification.json').write_text(json.dumps({'input_code_files': len(hashes),
         'unchanged': True, 'sealed_target_files': len(seals), 'parent_point_files_equal': len(invariants)}, indent=2))
    print(macro.loc[macro.bounds.eq('common_support')].to_string(index=False))


if __name__ == '__main__':
    main()
