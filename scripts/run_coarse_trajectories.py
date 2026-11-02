"""Sealed country-held-out coarse trajectory references; source-only interval calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from transcif.research import coarse_trajectories as trajectory
from transcif.research import multicountry as mc
from transcif.research.capacity_transfer import load_capacity
from transcif.research.conditional_risk import interval_metrics
from transcif.research.monthly_adaptation import weighted_quantile
from run_cross_country import ROOT, digest


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--joint-response', action='store_true')
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    modes = (*trajectory.MODES, *trajectory.RESPONSE_MODES) if args.joint_response else trajectory.MODES
    parent = ROOT / 'results/three-country-capacity-r2'
    table_path = ROOT / 'results/public-capacity-input-r1/capacity.json'
    table = load_capacity(table_path)
    paths = [Path(__file__), ROOT/'src/transcif/research/coarse_trajectories.py',
             ROOT/'docs/COARSE_TRAJECTORY_PROTOCOL.md', ROOT/'src/transcif/research/multicountry.py',
             ROOT/'src/transcif/research/cross_country.py', ROOT/'src/transcif/data/smard.py',
             ROOT/'src/transcif/research/capacity_transfer.py',
             ROOT/'src/transcif/research/monthly_adaptation.py',
             ROOT/'src/transcif/research/conditional_risk.py', table_path, parent/'input_audit.json']
    if args.joint_response: paths.append(ROOT/'docs/COARSE_RESPONSE_PROTOCOL.md')
    for name, expected in table['input_hashes'].items():
        if digest(ROOT/name) != expected: raise ValueError('capacity provenance changed')
        paths.append(ROOT/name)
    for item in json.loads((parent/'input_audit.json').read_text()).values():
        if 'path' in item: paths.append(ROOT/item['path'])
    for year in (2022, 2023):
        p = ROOT/f'results/smard-de-{year}-full.json'; paths.append(p)
        paths.extend(ROOT/r['path'] for r in json.loads(p.read_text())['records'])
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    (out/'input_code_seal.json').write_text(json.dumps(hashes, indent=2))
    raw, _ = mc.load_countries(ROOT)
    public = {}
    for r, f in raw.items():
        v = mc.public_features(mc.public_months(f), f.clock)
        v['country'], v['region'], v['raw_row'] = f.country.iloc[0], r, v.index
        public[r] = v.dropna(subset=list(mc.HISTORY))
    records, seals = [], []
    for target in ('DE', 'GB', 'US'):
        other = [c for c in ('DE', 'GB', 'US') if c != target]
        for source in [(other[0],), (other[1],), tuple(other)]:
            tag = '+'.join(source)+'_to_'+target
            regions = [r for r, f in raw.items() if f.country.iloc[0] in source]
            train_frames = [public[r].loc[public[r].clock.dt.year.eq(2022)] for r in regions]
            train = pd.concat(train_frames, ignore_index=True)
            fuel = np.concatenate([raw[r].loc[v.raw_row, list(mc.FINE)].to_numpy()
                                   for r, v in zip(regions, train_frames)])
            bank = trajectory.fit(train, fuel, include_response=args.joint_response)
            joblib.dump(bank, out/f'{tag}_bank.joblib')
            (out/f'{tag}_bank_audit.json').write_text(json.dumps(bank['audit'], indent=2))
            cal_frames, scores = [], {(b, m): [] for b in ('monthly', 'capacity') for m in modes}
            for r in regions:
                v = public[r].loc[public[r].clock.dt.year.eq(2023) & public[r].clock.dt.month.le(6)]
                values, _ = trajectory.predict(bank, v)
                f = raw[r].loc[v.raw_row, list(mc.FINE)].to_numpy()
                y = f@mc.FACTORS/f.sum(axis=1); cal_frames.append(v)
                c = table['regions'][r]; cap = 400.+600.*c['coal_mw']/(c['coal_mw']+c['gas_mw'])
                for budget, factor in [('monthly', bank['macro_ef']), ('capacity', cap)]:
                    for mode, members in trajectory.members(values, factor).items():
                        lo, hi = np.quantile(members, [.05, .95], axis=1)
                        scores[(budget, mode)].append(np.maximum.reduce([lo-y, y-hi, np.zeros(len(y))]))
            w = mc.weights(pd.concat(cal_frames, ignore_index=True))
            radii = {(b, m): weighted_quantile(np.concatenate(v), w, .9) for (b, m), v in scores.items()}
            (out/f'{tag}_calibration.json').write_text(json.dumps({b: {m: radii[b, m] for m in modes}
                                                                 for b in ('monthly', 'capacity')}, indent=2))
            for r, v in public.items():
                if raw[r].country.iloc[0] != target: continue
                query = v.loc[v.clock.dt.year.eq(2023) & v.clock.dt.month.gt(6)]
                values, diagnostics = trajectory.predict(bank, query)
                diagnostics.to_csv(out/f'{tag}_{r}_support.csv', index=False)
                c = table['regions'][r]; cap = 400.+600.*c['coal_mw']/(c['coal_mw']+c['gas_mw'])
                for budget, factor in [('monthly', bank['macro_ef']), ('capacity', cap)]:
                    p = out/f'{tag}_{r}_{budget}.npz'
                    np.savez(p, time=raw[r].loc[query.raw_row, 'time'].astype(str).to_numpy(dtype=str),
                             **trajectory.members(values, factor))
                    seals.append({'path': p.name, 'sha256': digest(p)})
                    records.append((tag, target, r, budget, query, p, radii))
            print(tag, 'all trajectory members sealed', flush=True)
    (out/'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    rows = []
    for tag, target, r, budget, query, path, radii in records:
        if digest(path) != next(s['sha256'] for s in seals if s['path'] == path.name):
            raise RuntimeError('sealed members changed')
        fuel = raw[r].loc[query.raw_row, list(mc.FINE)].to_numpy(); y = fuel@mc.FACTORS/fuel.sum(axis=1)
        with np.load(path, allow_pickle=False) as members:
            np.testing.assert_array_equal(members['time'], raw[r].loc[query.raw_row, 'time'].astype(str).to_numpy())
            for mode in modes:
                e = members[mode]; point = e.mean(axis=1); crps = trajectory.crps(e, y)
                lo, hi = np.quantile(e, [.05, .95], axis=1)
                for month in sorted(query.clock.dt.month.unique()):
                    mask = query.clock.dt.month.eq(month).to_numpy()
                    tmp = pd.DataFrame({'y': y[mask], 'p': point[mask],
                                        'day': query.loc[mask].clock.dt.strftime('%Y-%m-%d').to_numpy()})
                    centered = tmp[['y', 'p']]-tmp.groupby('day')[['y', 'p']].transform('mean')
                    for risk in ('raw', 'calibrated'):
                        radius = 0. if risk == 'raw' else radii[budget, mode]
                        rows.append({'configuration': tag, 'target': target, 'region': r, 'budget': budget,
                                     'mode': mode, 'risk': risk, 'month': int(month), 'hours': int(mask.sum()),
                                     'mae': float(np.abs(y[mask]-point[mask]).mean()),
                                     'shape_mae': float(np.abs(centered.y-centered.p).mean()),
                                     'crps': float(crps[mask].mean()),
                                     **interval_metrics(y[mask], lo[mask]-radius, hi[mask]+radius)})
    x = pd.DataFrame(rows); x.to_csv(out/'monthly_metrics.csv', index=False)
    fields = ['mae', 'shape_mae', 'crps', 'coverage90', 'width90', 'interval_score90']
    groups = ['configuration', 'target', 'region', 'budget', 'mode', 'risk']
    regional = x.groupby(groups)[fields].mean().reset_index(); regional.to_csv(out/'region_metrics.csv', index=False)
    country = regional.groupby([g for g in groups if g != 'region'])[fields].mean().reset_index()
    country.to_csv(out/'country_metrics.csv', index=False)
    macro = country.groupby(['budget', 'mode', 'risk'])[fields].mean().reset_index()
    macro.to_csv(out/'macro_metrics.csv', index=False)
    for name, expected in hashes.items():
        if digest(ROOT/name) != expected: raise RuntimeError(f'input/code changed: {name}')
    (out/'terminal_verification.json').write_text(json.dumps({'unchanged_input_files': len(hashes),
                                                            'sealed_member_files': len(seals)}, indent=2))
    print(macro.loc[macro.risk.eq('calibrated')].to_string(index=False))


if __name__ == '__main__':
    main()
