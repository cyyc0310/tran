"""Posthoc zero-fit diagnostics of completed V49 source errors, never raw truth I/O."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE, OUT = ROOT/'results/source-risk-v49', ROOT/'results/source-risk-v49-diagnostics'
SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
ARMS = ('Scale-anchor', 'Scale-recompose', 'B0-SizeET-KeepA', 'B0-SizeET-ReplaceR',
        'B0-SizeET-AlignedKeep', 'B1-SizeCapRidge-KeepA', 'B1-SizeCapRidge-ReplaceR',
        'B1-SizeCapRidge-AlignedKeep')
BUDGETS = ('B0',)*5+('B1',)*3
PANELS = ('H1_geo_backcast', 'H2_after_bank')
DAYS = pd.date_range('2024-01-25', '2024-12-31', tz='UTC').as_unit('ns')
EXTRA_TABLES = ('source_case_metrics.csv', 'source_region_metrics.csv', 'source_outer_metrics.csv',
                'label_support.csv', 'source_month_support.csv')
MEAN_METRICS = ('hourly_mse', 'level_mse', 'shape_mse', 'signed_level_bias',
                'mean_abs_level_error', 'shape_mae')
CODE = (Path(__file__).resolve(), ROOT/'tests/test_analyze_source_risk_v49.py')


def sha(path: Path | str) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _verify(mapping: Mapping[str, str]) -> None:
    for name, expected in mapping.items():
        path = Path(name)
        if not path.is_file() or path.is_symlink() or sha(path) != expected:
            raise ValueError('Regular input hash mismatch: '+name)


def _real(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError('Real '+name+' required')
    array = np.asarray(value, float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('Finite matching '+name+' required')
    return array


def decompose(point: np.ndarray, truth: np.ndarray, hourly_error: np.ndarray,
              level_error: np.ndarray, shape_error: np.ndarray) -> dict[str, Any]:
    """Recompute from cached truth/point; independently check saved residual arrays."""
    y = np.asarray(truth)
    if y.ndim != 2 or y.shape[1] != 24 or not len(y):
        raise ValueError('Nonempty days by24 source truth required')
    n = len(y)
    y = _real(y, (n, 24), 'truth')
    p = _real(point, (8, n, 24), 'point')
    error = y[None]-p
    level = error.mean(-1)
    shape = error-level[..., None]
    for saved, expected, name in ((hourly_error, error, 'hourly_error'),
                                 (level_error, level, 'level_error'), (shape_error, shape, 'shape_error')):
        actual = _real(saved, expected.shape, name)
        if not np.allclose(actual, expected, atol=1e-9, rtol=0):
            raise ValueError('Cached '+name+' disagrees with independent truth-minus-point decomposition')
    try:
        with np.errstate(over='raise', invalid='raise'):
            h, l, q = (error**2).mean(-1), level**2, (shape**2).mean(-1)
    except FloatingPointError as exc:
        raise ValueError('Finite squared errors required') from exc
    if not np.allclose(h, l+q, atol=1e-8, rtol=1e-12):
        raise ValueError('Daily orthogonal squared-error identity H=L+Q failed')
    return dict(hourly_mse=h, level_mse=l, shape_mse=q, signed_level_bias=level,
                mean_abs_level_error=abs(level), shape_mae=abs(shape).mean(-1),
                identity_abs=abs(h-l-q), identity_max_abs=float(abs(h-l-q).max()))


def _derived(row: dict[str, Any]) -> dict[str, Any]:
    for prefix in ('hourly', 'level', 'shape'):
        row[prefix+'_rmse'] = float(np.sqrt(row[prefix+'_mse']))
    row['level_mse_fraction'] = (None if row['hourly_mse'] == 0
                                 else float(row['level_mse']/row['hourly_mse']))
    return row


def aggregate(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean six outer MSEs per held source, then seven held MSEs; derive ratios last."""
    keys = ['outer', 'held', 'panel', 'arm', 'budget']
    expected = {(o, h, p, a, b) for o in SOURCES for h in SOURCES if o != h
                for p in PANELS for a, b in zip(ARMS, BUDGETS)}
    if (not set(keys+['valid_days', 'identity_max_abs', *MEAN_METRICS]) <= set(tasks)
            or len(tasks) != 672 or tasks.duplicated(keys).any()
            or set(tasks[keys].itertuples(index=False, name=None)) != expected
            or not np.isfinite(tasks[[*MEAN_METRICS, 'valid_days', 'identity_max_abs']].to_numpy(float)).all()
            or np.any(tasks[['hourly_mse', 'level_mse', 'shape_mse', 'mean_abs_level_error', 'shape_mae']] < 0)
            or np.any(tasks.valid_days <= 0) or np.any(tasks.valid_days != np.floor(tasks.valid_days))
            or not np.allclose(tasks.hourly_mse, tasks.level_mse+tasks.shape_mse, atol=1e-8, rtol=1e-12)):
        raise ValueError('Complete finite orthogonal 42task by2panel by8base summaries required')
    held_rows = []
    for (held, panel, arm, budget), group in tasks.groupby(['held', 'panel', 'arm', 'budget'], sort=False):
        if len(group) != 6 or group.valid_days.nunique() != 1:
            raise ValueError('Six equally weighted outer tasks on identical held support required')
        held_rows.append(_derived(dict(held=held, panel=panel, arm=arm, budget=budget,
            valid_days=int(group.valid_days.iloc[0]), outer_count=6,
            identity_max_abs=float(group.identity_max_abs.max()),
            **{name: float(group[name].mean()) for name in MEAN_METRICS})))
    held_table = pd.DataFrame(held_rows)
    macro = []
    for (panel, arm, budget), group in held_table.groupby(['panel', 'arm', 'budget'], sort=False):
        if len(group) != 7:
            raise ValueError('Seven equally weighted held sources required')
        macro.append(_derived(dict(panel=panel, arm=arm, budget=budget, held_count=7,
            unique_source_days=int(group.valid_days.sum()), identity_max_abs=float(group.identity_max_abs.max()),
            **{name: float(group[name].mean()) for name in MEAN_METRICS})))
    return held_table, pd.DataFrame(macro)


def _inputs(source: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    completed = _read(source/'completed.json')
    if (completed['status'] != 'COMPLETE' or any(completed[name] is not False for name in
            ('source_2025_truth_read', 'target_values_opened', 'unseen_values_opened'))):
        raise ValueError('Completed source-only V49 output required')
    expected_paths = {str(source/'risk'/f'{o}__{h}.npz') for o in SOURCES for h in SOURCES if o != h}
    expected_paths |= {str(source/'labels'/f'{s}.npz') for s in SOURCES}
    expected_paths |= {str(source/name) for name in ('task_index.json', 'summary.json', *EXTRA_TABLES)}
    if set(completed['output_hashes']) != expected_paths:
        raise ValueError('Exact completed V49 output hash registry required')
    hashes = dict(completed['output_hashes'])
    hashes[str(source/'completed.json')] = sha(source/'completed.json')
    for name, field in (('freeze.json', 'freeze_sha256'), ('prediction_seal.json', 'prediction_seal_sha256'),
                        ('label_started.json', 'label_started_sha256')):
        hashes[str(source/name)] = completed[field]
    _verify(hashes)  # All current artifacts and index bytes pass before any npz opens.
    index, summary = _read(source/'task_index.json'), _read(source/'summary.json')
    if (summary['risk_models_fitted'] is not False or summary['source_2025_truth_read'] is not False
            or summary['directed_tasks'] != 42 or summary['bases'] != 8):
        raise ValueError('Zero-risk-fit eight-base source-only summary required')
    labels = index['labels']
    if len(labels) != 7 or {e['region'] for e in labels} != set(SOURCES):
        raise ValueError('Seven unique held-source label archives required')
    for e in labels:
        path = str(source/'labels'/f"{e['region']}.npz")
        if e['path'] != path or e['sha256'] != hashes[path]:
            raise ValueError('Current label-index hash/path mismatch')
    entries = index['entries']
    pairs = {(e['outer'], e['held']) for e in entries}
    if len(entries) != 42 or pairs != {(o, h) for o in SOURCES for h in SOURCES if o != h}:
        raise ValueError('Exact42 unique directed source tasks required')
    for e in entries:
        o, h = e['outer'], e['held']
        path = str(source/'risk'/f'{o}__{h}.npz')
        if (e['pair'] != sorted((o, h)) or e['label_region'] != h or e['risk_path'] != path
                or e['risk_sha256'] != hashes[path] or e['source_label_path'] != str(source/'labels'/f'{h}.npz')):
            raise ValueError('Current task-index risk hash, pair and held-label link required')
    if set(index['outer_tasks']) != set(SOURCES):
        raise ValueError('Seven source outer task registries required')
    for outer in SOURCES:
        paths = index['outer_tasks'][outer]
        if len(paths) != 6 or set(paths) != {e['risk_path'] for e in entries if e['outer'] == outer}:
            raise ValueError('Exact six directed risk archives per outer required')
    return index, summary, hashes


def _labels(entry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(entry['path'], allow_pickle=False) as archive:
        names = ('origins', 'truth', 'positions', 'keys', 'panels')
        if set(archive.files) != set(names):
            raise ValueError('Exact cached source-label schema required')
        result = {name: archive[name].copy() for name in names}
    pos, origins, source = result['positions'], result['origins'], entry['region']
    n = len(origins)
    if (not n or n != entry['days'] or pos.shape != (n,) or not np.issubdtype(pos.dtype, np.integer)
            or not np.issubdtype(origins.dtype, np.integer) or np.any(pos < 0) or np.any(pos >= len(DAYS))
            or np.any(np.diff(pos) <= 0) or not np.array_equal(origins, DAYS.asi8[pos])
            or result['keys'].tolist() != [f'{source}/{at:%Y-%m-%d}' for at in DAYS[pos]]
            or not np.array_equal(result['panels'], np.where(DAYS[pos].month <= 6, *PANELS))):
        raise ValueError('Canonical held-label support, keys and two-panel dates required')
    _real(result['truth'], (n, 24), 'cached source truth')
    if np.any(result['truth'] < 0) or np.any(result['truth'] > 1000):
        raise ValueError('Bounded source CIF truth required')
    return result


def _task_rows(entry: Mapping[str, Any], label: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    n = len(label['origins'])
    with np.load(entry['risk_path'], allow_pickle=False) as archive:
        if (archive['arms'].tolist() != list(ARMS) or archive['budgets'].tolist() != list(BUDGETS)
                or entry['valid_days'] != n or any(not np.array_equal(archive[name], label[name])
                                                 for name in ('origins', 'keys', 'panels'))):
            raise ValueError('Same held support and fixed risk base registry required')
        point = _real(archive['point'], (8, n, 24), 'cached point')
        if np.any(point < 0) or np.any(point > 1000):
            raise ValueError('Post-clip cached points must lie in[0,1000]')
        daily = decompose(point, label['truth'], archive['hourly_error'], archive['level_error'], archive['shape_error'])
    rows = []
    for panel in PANELS:
        take = label['panels'] == panel
        if not take.any():
            raise ValueError('Each held source requires observed support in both panels')
        for i, (arm, budget) in enumerate(zip(ARMS, BUDGETS)):
            rows.append(_derived(dict(outer=entry['outer'], held=entry['held'], panel=panel, arm=arm, budget=budget,
                valid_days=int(take.sum()), identity_max_abs=float(daily['identity_abs'][i, take].max()),
                **{name: float(daily[name][i, take].mean()) for name in MEAN_METRICS})))
    return rows


def run(source: Path = SOURCE, output: Path = OUT) -> None:
    """Analyze only completed cached V49 outputs, once; do not invoke producers."""
    source, output = Path(source).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve existing diagnostics attempt')
    index, prior_summary, hashes = _inputs(source)
    code = {str(path): sha(path) for path in CODE}
    output.mkdir()  # Exclusive even if another process raced the initial check.
    _write(output/'freeze.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), posthoc=True,
        source=str(source), input_hashes=hashes, code_hashes=code, new_fits=0, correlations_computed=False,
        identity_atol=1e-8, identity_rtol=1e-12, aggregation='equal outer within held, then equal held; MSE before sqrt/ratio'))
    labels = {e['region']: _labels(e) for e in index['labels']}
    tasks = pd.DataFrame([row for entry in index['entries'] for row in _task_rows(entry, labels[entry['held']])])
    held, macro = aggregate(tasks)
    counts = dict(source_label_days=sum(len(z['origins']) for z in labels.values()),
                  directed_valid_days=sum(e['valid_days'] for e in index['entries']))
    if (any(prior_summary[name] != value for name, value in counts.items())
            or prior_summary['base_day_examples'] != 8*counts['directed_valid_days']):
        raise ValueError('Completed summary and actual cached support counts disagree')
    for name, table in (('task_panel.csv', tasks), ('held_panel.csv', held), ('macro.csv', macro)):
        table.to_csv(output/name, index=False, mode='x')
    results = macro.astype(object).where(pd.notna(macro), None).to_dict('records')
    _write(output/'summary.json', dict(posthoc=True, new_fits=0, correlations_computed=False,
        rows=dict(task_panel=len(tasks), held_panel=len(held), macro=len(macro)), **counts,
        panels_separate=True, repeated_tasks_are_independent_samples=False, risk_calibration_claim=False,
        H1_is_geographic_backcast=True, authentic_first_vintage_online_backtest=False,
        identity_max_abs=float(tasks.identity_max_abs.max()), null_fraction='0/0: empty CSV cell and JSON null',
        macro_results=results, threshold_selection=False, source_2025_truth_read=False,
        target_values_opened=False, unseen_values_opened=False))
    _verify(hashes); _verify(code)
    artifacts = {str(output/name): sha(output/name) for name in
                 ('freeze.json', 'task_panel.csv', 'held_panel.csv', 'macro.csv', 'summary.json')}
    _write(output/'completed.json', dict(status='COMPLETE', posthoc=True, new_fits=0, output_hashes=artifacts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True, help='Analyze completed V49 cached outputs once')
    parser.parse_args()
    run()
