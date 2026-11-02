"""Post-score diagnostics, never used for selecting the frozen pilot models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.research.cross_country import COARSE, EF, FUELS, load_inputs, monthly_inputs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--diagnostic-name', default='posthoc_support_diagnostic.json')
    args = parser.parse_args()
    if Path(args.diagnostic_name).name != args.diagnostic_name:
        raise ValueError('diagnostic-name must be a filename')
    output = args.run/args.diagnostic_name
    if output.exists():
        raise FileExistsError(output)
    raw, _ = load_inputs(ROOT)
    inputs = {r: monthly_inputs(f).dropna(subset=list(COARSE)) for r, f in raw.items()}
    rows = []
    for source, target in [('US', 'GB'), ('GB', 'US')]:
        source_inputs = pd.concat([f.loc[f.year.eq(2022)] for f in inputs.values()
                                   if f.country.iloc[0] == source])
        lo, hi = source_inputs[list(COARSE)].min(), source_inputs[list(COARSE)].max()
        source_fuels = np.concatenate([raw[r].loc[inputs[r].loc[inputs[r].year.eq(2022)].index,
                                                  list(FUELS)].to_numpy()
                                       for r in raw if raw[r].country.iloc[0] == source])
        for r, f in inputs.items():
            if f.country.iloc[0] != target:
                continue
            selected = f.loc[f.year.eq(2024)]
            target_fuels = raw[r].loc[selected.index, list(FUELS)].to_numpy()
            row = {'source': source, 'region': r,
                   'scored_hours': len(selected),
                   'fraction_of_2024_hours': len(selected)/8784,
                   'scored_hours_by_month': selected.groupby('period').size().rename(index=str).to_dict(),
                   'thermal_share_source_range': [float(lo.thermal), float(hi.thermal)],
                   'thermal_share_target_range': [float(selected.thermal.min()), float(selected.thermal.max())],
                   'outside_any_coarse_coordinate_fraction': float(
                       ((selected[list(COARSE)] < lo) | (selected[list(COARSE)] > hi)).any(axis=1).mean()),
                   'source_pooled_thermal_ef_diagnostic': float(
                       np.sum(source_fuels @ EF) / np.sum(source_fuels[:, :4])),
                   'target_actual_thermal_ef_diagnostic_NOT_INPUT': float(
                       np.sum(target_fuels @ EF) / np.sum(target_fuels[:, :4]))}
            rows.append(row)
    output.write_text(json.dumps({'status': 'POSTHOC_NO_MODEL_SELECTION', 'rows': rows}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
