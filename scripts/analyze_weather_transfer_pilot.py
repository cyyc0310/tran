"""Predeclared paired weather contrasts; descriptive, not confirmation evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from collect_weather_transfer_pilot import ROOT, digest
from transcif.research import multicountry as mc
from transcif.research.conditional_risk import interval_metrics

KEYS = ['configuration', 'country', 'region', 'budget', 'origin', 'valid_at']
GROUP = ['configuration', 'country', 'region', 'budget', 'origin']
METRICS = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']


def contrasts() -> list[tuple[str, str, str, str, str]]:
    """Candidate, its risk, reference, its risk, and information attribution."""
    pairs = []
    for kind in ('ridge', 'hgb', 'residual_ridge'):
        for risk in ('global', 'hetero', *(('cqr',) if kind == 'hgb' else ())):
            pairs.append((kind+'_weather', risk, kind+'_no_weather', risk, 'matched_weather_increment'))
        for risk in ('global', 'hetero'):
            pairs.append((kind+'_weather', risk, 'shape_anchor', risk, 'weather_model_vs_simple_shape'))
    pairs.extend([
        ('shape_anchor', 'global', 'flat', 'global', 'source_shape_increment'),
        ('joint_full_source_extra_labels', 'calibrated', 'joint_matched', 'calibrated', 'additional_source_labels'),
        ('hgb_weather', 'cqr', 'joint_full_source_extra_labels', 'calibrated', 'extra_label_strong_reference'),
    ])
    return pairs


def paired_days(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Require exactly equal unique forecast keys before any paired statistic."""
    if left.empty or right.empty or left.duplicated(KEYS).any() or right.duplicated(KEYS).any():
        raise ValueError('nonempty unique paired forecast keys required')
    merged = left.merge(right, on=KEYS, how='outer', suffixes=('_left', '_right'),
                        indicator=True, validate='one_to_one')
    if not merged['_merge'].eq('both').all():
        raise ValueError('paired forecasts have different cohorts')
    if not np.array_equal(merged.y_left.to_numpy(), merged.y_right.to_numpy()):
        raise ValueError('paired truth differs')
    rows = []
    for keys, group in merged.groupby(GROUP):
        row = dict(zip(GROUP, keys)); row['hours'] = len(group)
        y = group.y_left.to_numpy()
        for side in ('left', 'right'):
            point = group[f'point_{side}'].to_numpy()
            metrics = {'mae': float(np.abs(point-y).mean()),
                'shape_mae': float(np.abs((point-point.mean())-(y-y.mean())).mean()),
                **interval_metrics(y, group[f'lower_{side}'].to_numpy(), group[f'upper_{side}'].to_numpy())}
            row.update({side+'_'+key: value for key, value in metrics.items()})
        row.update({'delta_'+key: row['left_'+key]-row['right_'+key] for key in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(days: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Equal observed days per region, regions per country/configuration, then configs."""
    identity = ['comparison', 'attribution', 'budget']
    measures = [side+'_'+key for side in ('left', 'right', 'delta') for key in METRICS]
    regions = days.groupby(identity+['configuration', 'country', 'region'])[measures].mean().reset_index()
    configurations = regions.groupby(identity+['configuration', 'country'])[measures].mean().reset_index()
    macro = configurations.groupby(identity)[measures].mean().reset_index()
    # These are descriptive checks. Three development dates do not establish
    # nominal coverage or algorithm novelty, whatever these flags show.
    macro['point_improved'] = macro.delta_mae.lt(0.)
    macro['coverage_at_least_90pct'] = macro.left_coverage90.ge(.9)
    macro['coverage_not_lower'] = macro.delta_coverage90.ge(0.)
    macro['interval_narrower'] = macro.delta_width90.lt(0.)
    macro['interval_score_improved'] = macro.delta_interval_score90.lt(0.)
    return regions, configurations, macro


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); experiment = args.experiment.resolve()
    for seal in ('input_code_seal.json', 'prediction_seal.json', 'model_seal.json'):
        for name, expected in json.loads((experiment/seal).read_text()).items():
            path = Path(name) if Path(name).is_absolute() else experiment/name
            if digest(path) != expected:
                raise ValueError(f'changed sealed artifact: {path}')
    pred = pd.read_csv(experiment/'predictions.csv')
    raw, _ = mc.load_countries(ROOT)
    for region, group in pred.groupby('region'):
        f = raw[region].loc[group.label_row.astype(int)]
        if not np.array_equal(pd.to_datetime(f.time, utc=True).to_numpy(),
                              pd.to_datetime(group.valid_at, utc=True).to_numpy()):
            raise ValueError('label timestamp mismatch')
        fuel = f[list(mc.FINE)].to_numpy()
        pred.loc[group.index, 'y'] = fuel@mc.FACTORS/fuel.sum(axis=1)
    tables = []
    for left, left_risk, right, right_risk, attribution in contrasts():
        a = pred.loc[pred.method.eq(left) & pred.risk.eq(left_risk)]
        b = pred.loc[pred.method.eq(right) & pred.risk.eq(right_risk)]
        paired = paired_days(a, b)
        paired['comparison'] = f'{left}/{left_risk} minus {right}/{right_risk}'
        paired['attribution'] = attribution
        tables.append(paired)
    days = pd.concat(tables, ignore_index=True)
    regions, configurations, macro = aggregate(days)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    for name, frame in [('days', days), ('regions', regions), ('configurations', configurations), ('macro', macro)]:
        frame.to_csv(out/f'{name}.csv', index=False)
    limitations = {'confirmation_opened': False, 'novelty_established': False,
        'scope': 'three fixed development origins, already-opened countries; no bootstrap significance claim',
        'coverage': 'source calibration transfers without any target-hourly coverage guarantee',
        'weather': 'single-point forecast proxy, original public availability not certified',
        'selection': 'all contrasts fixed in code before new scores; no target-ranked primary',
        'source_labels': 'full-source bank is explicitly an extra-label reference',
        'analyzer_sha256': digest(Path(__file__)),
        'prediction_sha256': digest(experiment/'predictions.csv')}
    (out/'interpretation.json').write_text(json.dumps(limitations, indent=2)+'\n')
    print(macro.loc[macro.attribution.eq('matched_weather_increment'),
                    ['budget', 'comparison', *['delta_'+key for key in METRICS]]].to_string(index=False))


if __name__ == '__main__':
    main()
