"""Geographic source validation before deploying learned increments abroad."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.research.budget_transfer import budget_views
from transcif.research.cross_country import COARSE, FUELS, fit_models, labels, load_inputs, monthly_inputs
from transcif.research.monthly_adaptation import region_weights, weighted_quantile
from transcif.research.structured_transfer import predict_structured, save_structured, train_structured
from run_cross_country import ROOT, digest

REGIONS = ('CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')


def score(frame, y, p):
    tmp = pd.DataFrame({'day': frame.clock.dt.strftime('%Y-%m-%d').to_numpy(), 'y': y, 'p': p})
    centered = tmp[['y', 'p']]-tmp.groupby('day')[['y', 'p']].transform('mean')
    return float(np.abs(y-p).mean()), float(np.abs(centered.y-centered.p).mean())


def choose(table):
    regional = table.groupby(['region', 'head', 'strength'])[['mae', 'shape_mae']].mean().reset_index()
    native = regional.loc[regional['head'].eq('native')].set_index('region')
    baseline = native.mae.mean()
    eligible = []
    for (head, strength), rows in regional.groupby(['head', 'strength']):
        if head == 'native':
            continue
        paired = rows.set_index('region').join(native[['mae', 'shape_mae']], rsuffix='_base')
        if (paired.mae.mean() <= .95*baseline and (paired.mae < paired.mae_base).sum() >= 4
                and paired.shape_mae.mean() <= paired.shape_mae_base.mean()):
            eligible.append((float(paired.mae.mean()), head, float(strength)))
    if not eligible:
        return {'head': 'native', 'strength': 0., 'passed': False}
    _, head, strength = min(eligible)
    return {'head': head, 'strength': strength, 'passed': True}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT/'src/transcif/research/structured_transfer.py',
             ROOT/'src/transcif/research/cross_country.py', ROOT/'src/transcif/research/budget_transfer.py',
             ROOT/'src/transcif/research/monthly_adaptation.py', ROOT/'docs/CROSS_COUNTRY_SOURCE_HOLDOUT_PROTOCOL.md']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    (out/'startup_code_hashes.json').write_text(json.dumps(hashes, indent=2))
    raw, audit = load_inputs(ROOT)
    views = {r: {lag: monthly_inputs(f, lag) for lag in (2, 3, 4)} for r, f in raw.items()}
    train = {r: budget_views(views[r], 2022, False).drop_duplicates('raw_row') for r in (*REGIONS, 'GB')}
    valid = {r: views[r][2].dropna(subset=list(COARSE)).assign(age_months=2.) for r in raw}
    def fit_native(regions):
        frame = pd.concat([train[r] for r in regions], ignore_index=True)
        fuel = np.concatenate([raw[r].loc[train[r].raw_row, list(FUELS)].to_numpy() for r in regions])
        return frame, fuel, fit_models(frame, fuel)
    def native_predict(native, f):
        q = native['profile'][f.clock.dt.hour.to_numpy()]*f[list(COARSE)].to_numpy()
        return native['thermal_ef']*q[:, 0]/q.sum(axis=1)
    rows = []
    for held in REGIONS:
        frame, fuel, native = fit_native([r for r in REGIONS if r != held])
        target = valid[held].loc[valid[held].year.eq(2023)&valid[held].clock.dt.month.le(6)]
        base = native_predict(native, target)
        predictions = [('native', 0., base)]
        for head in ('residual', 'factor', 'joint'):
            fitted = train_structured(frame, fuel, head, 412, native)
            p = predict_structured(fitted, target)
            predictions += [(head, strength, base+strength*(p-base)) for strength in (.25, .5, 1.)]
        y = labels(raw[held].loc[target.index])
        for head, strength, p in predictions:
            for month in sorted(target.clock.dt.month.unique()):
                mask = target.clock.dt.month.eq(month).to_numpy()
                mae, shape = score(target.loc[mask], y[mask], p[mask])
                rows.append({'region': held, 'head': head, 'strength': strength,
                             'month': int(month), 'hours': int(mask.sum()), 'mae': mae, 'shape_mae': shape})
        print('source fold', held, 'completed', flush=True)
    validation = pd.DataFrame(rows)
    validation.to_csv(out/'source_validation.csv', index=False)
    selection = {'US': choose(validation), 'GB': {'head': 'native', 'strength': 0.,
                                               'passed': False, 'reason': 'one source region'}}
    (out/'source_selection.json').write_text(json.dumps(selection, indent=2))
    records, seals = [], []
    for source, sources, targets in [('US', REGIONS, ('GB',)), ('GB', ('GB',), tuple(r for r in raw if r != 'GB'))]:
        frame, fuel, native = fit_native(sources)
        policy = selection[source]
        fitted = None
        if policy['head'] != 'native':
            fitted = train_structured(frame, fuel, policy['head'], 412, native)
            save_structured(fitted, out/f'{source}_selected.pt')
        def predict(f):
            base = native_predict(native, f)
            return base if fitted is None else base+policy['strength']*(predict_structured(fitted, f)-base)
        residuals, group_ids = [], []
        for r in sources:
            f = valid[r].loc[valid[r].year.eq(2023)&valid[r].clock.dt.month.gt(6)]
            residuals.append(np.abs(labels(raw[r].loc[f.index])-predict(f)))
            group_ids.append(np.repeat(r, len(f)))
        errors, groups = np.concatenate(residuals), np.concatenate(group_ids)
        radius = weighted_quantile(errors, region_weights(groups), .9)
        (out/f'{source}_radius.json').write_text(json.dumps({'radius': radius}))
        for r in targets:
            f = valid[r].loc[valid[r].year.eq(2024)]
            p = predict(f)
            path = out/f'{source}_to_{r}.csv'
            pd.DataFrame({'time': f.time.astype(str), 'point': p, 'lo': p-radius, 'hi': p+radius}).to_csv(path,index=False)
            seals.append({'path': path.name, 'sha256': digest(path)})
            records.append((source,r,f,path))
    (out/'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    rows = []
    for source, r, f, path in records:
        pred = pd.read_csv(path)
        y = labels(raw[r].loc[f.index])
        for month in sorted(f.clock.dt.month.unique()):
            mask = f.clock.dt.month.eq(month).to_numpy()
            p, lo, hi = [pred[c].to_numpy()[mask] for c in ('point','lo','hi')]
            mae, shape = score(f.loc[mask],y[mask],p)
            rows.append({'source':source,'region':r,'month':int(month),'hours':int(mask.sum()),
                         'mae':mae,'shape_mae':shape,'coverage90':float(((y[mask]>=lo)&(y[mask]<=hi)).mean()),
                         'width90':float(np.mean(hi-lo))})
    pd.DataFrame(rows).to_csv(out/'target_monthly_metrics.csv',index=False)
    if any(digest(ROOT/k)!=v for k,v in hashes.items()):
        raise RuntimeError('code changed during run')
    for item in audit.values():
        hashes[item['path']] = digest(ROOT/item['path'])
    (out/'provenance.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(selection))


if __name__=='__main__':
    main()
