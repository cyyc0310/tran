"""Add an unmodified native reference under the source-holdout H2 calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.research.budget_transfer import budget_views
from transcif.research.cross_country import COARSE, FUELS, fit_models, labels, load_inputs, monthly_inputs
from transcif.research.monthly_adaptation import region_weights, weighted_quantile
from run_cross_country import ROOT, digest
from run_source_holdout import REGIONS, score


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    destination = args.run/'matched_native_monthly_metrics.csv'
    if destination.exists():
        raise FileExistsError(destination)
    raw, _ = load_inputs(ROOT)
    views = {r: {lag: monthly_inputs(f, lag) for lag in (2, 3, 4)} for r, f in raw.items()}
    valid = {r: v[2].dropna(subset=list(COARSE)) for r, v in views.items()}
    rows, radii = [], {}
    for source, sources, targets in [('US', REGIONS, ('GB',)), ('GB', ('GB',), tuple(r for r in raw if r!='GB'))]:
        frames = [budget_views(views[r], 2022, False).drop_duplicates('raw_row') for r in sources]
        fuel = np.concatenate([raw[r].loc[f.raw_row, list(FUELS)].to_numpy() for r, f in zip(sources, frames)])
        native = fit_models(pd.concat(frames, ignore_index=True), fuel)
        def predict(f):
            q = native['profile'][f.clock.dt.hour.to_numpy()]*f[list(COARSE)].to_numpy()
            return native['thermal_ef']*q[:, 0]/q.sum(axis=1)
        residuals, regions = [], []
        for r in sources:
            f = valid[r].loc[valid[r].year.eq(2023)&valid[r].clock.dt.month.gt(6)]
            residuals.append(np.abs(labels(raw[r].loc[f.index])-predict(f)))
            regions.append(np.repeat(r, len(f)))
        radius = weighted_quantile(np.concatenate(residuals), region_weights(np.concatenate(regions)), .9)
        radii[source] = radius
        for r in targets:
            f = valid[r].loc[valid[r].year.eq(2024)]
            y, p = labels(raw[r].loc[f.index]), predict(f)
            for m in sorted(f.clock.dt.month.unique()):
                mask = f.clock.dt.month.eq(m).to_numpy()
                mae, shape = score(f.loc[mask], y[mask], p[mask])
                rows.append({'source':source,'region':r,'month':int(m),'hours':int(mask.sum()),
                             'mae':mae,'shape_mae':shape,
                             'coverage90':float((np.abs(y[mask]-p[mask])<=radius).mean()),'width90':2*radius})
    pd.DataFrame(rows).to_csv(destination,index=False)
    (args.run/'matched_native_audit.json').write_text(json.dumps({'status':'POST_SELECTION_REFERENCE_ONLY',
        'radii':radii,'auditor_sha256':digest(Path(__file__))},indent=2))


if __name__=='__main__':
    main()
