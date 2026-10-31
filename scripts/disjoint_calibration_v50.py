"""V50 metadata, prediction-only features and interval scores; no model fitting."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scripts import fleet_information_v34 as fleet

LEGACY = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
EXTERNAL = ('SWPP', 'AVA', 'BANC', 'JEA', 'SC', 'NWMT', 'SCL', 'WACM', 'SEC', 'SCEG')
BASES = ('Scale-anchor', 'B0-SizeET-ReplaceR', 'B0-SizeET-AlignedKeep',
         'B1-SizeCapRidge-ReplaceR', 'B1-SizeCapRidge-AlignedKeep')
BUDGETS = ('B0', 'B0', 'B0', 'B1', 'B1')
EXTRA_NAMES = ('point_hour', 'std_hour', 'point_day_mean', 'point_day_min', 'point_day_max',
               'mc_day_mean_std', 'mc_centered_shape_rms', 'hour_sin', 'hour_cos')
CAL_DAYS = pd.date_range('2024-07-01', '2024-12-24', tz='UTC').as_unit('ns')
TEST_DAYS = pd.date_range('2025-01-01', '2025-06-30', tz='UTC').as_unit('ns')
ASOF = pd.Timestamp('2025-01-01', tz='UTC')
LAGS, RULES, FACTORS = (90, 30, 7), ('equal_region', 'HCP'), (.75, 1., 1.25, 1.5, 2.)
FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
METRICS = ('coverage90', 'width90', 'interval_score90', 'lower_miss', 'upper_miss', 'path_coverage90')


def real(value, shape: tuple, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError('Real '+name+' required')
    x = np.asarray(value, float)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError('Finite matching '+name+' required')
    return x


def eligible(days: pd.DatetimeIndex, lag: int) -> np.ndarray:
    """One fixed Jan1 snapshot; a response is assumed known day-end plus lag."""
    if lag not in LAGS or type(lag) is not int:
        raise ValueError('Predeclared 90/30/7 day publication scenario required')
    dates = pd.DatetimeIndex(days).as_unit('ns')
    if (dates.tz is None or str(dates.tz) != 'UTC' or dates.hasnans or dates.has_duplicates
            or not dates.equals(dates.normalize()) or np.any(~dates.isin(CAL_DAYS))):
        raise ValueError('Unique registered external UTC days required')
    return np.asarray(dates+pd.Timedelta(days=lag+1) <= ASOF)


def metadata(path: str | Path, source: str) -> list[str]:
    """Inspect only time/region/valid on the fixed external prediction grid."""
    if source not in EXTERNAL or Path(path).name != source+'_hourly.csv':
        raise ValueError('Registered external source file required')
    with Path(path).open(newline='') as f:
        rows = [{k: r[k] for k in ('time', 'region', 'valid')} for r in csv.DictReader(f)
                if '2024-07-01' <= r['time'] < '2024-12-25']
    if not rows:
        raise ValueError('External calendar metadata required')
    table = pd.DataFrame(rows)
    dates = pd.DatetimeIndex(pd.to_datetime(table.time, utc=True, errors='raise')).as_unit('ns')
    if (dates.has_duplicates or not dates.equals(dates.floor('h')) or not table.region.eq(source).all()
            or not set(table.valid) <= {'True', 'False'}):
        raise ValueError('Unique UTC hourly source identity and valid flags required')
    answer = []
    for at in CAL_DAYS:
        ids = np.flatnonzero(dates.floor('D') == at)
        if len(ids) != 24 or not table.iloc[ids].valid.eq('True').all():
            continue
        if not np.array_equal(np.sort(dates[ids].asi8), pd.date_range(at, periods=24, freq='h').as_unit('ns').asi8):
            raise ValueError('Exact daily hour grid required')
        answer.append(f'{source}/{at:%Y-%m-%d}')
    return answer


def external_truth(path: str | Path, source: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Called only after prediction seal; never convert out-of-window rows."""
    if source not in EXTERNAL or Path(path).name != source+'_hourly.csv':
        raise ValueError('Registered calibration source required')
    expected = metadata(path, source)
    if keys != expected:
        raise ValueError('Prelocked calendar support required before fuel conversion')
    with Path(path).open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if '2024-07-01' <= r['time'] < '2024-12-25']
    table = pd.DataFrame(rows)
    dates = pd.DatetimeIndex(pd.to_datetime(table.time, utc=True)).as_unit('ns')
    positions, truths = [], []
    for key in keys:
        at = pd.Timestamp(key.split('/')[1], tz='UTC')
        ids = np.flatnonzero(dates.floor('D') == at)
        ids = ids[np.argsort(dates[ids].asi8)]
        g = table.iloc[ids][['generation_'+f for f in FUELS]].to_numpy(float)
        if g.shape != (24, 8) or not np.isfinite(g).all() or np.any(g < 0) or np.any(g.sum(1) <= 0):
            raise ValueError('Valid generation must agree with prelocked metadata; do not drop days')
        positions.append(CAL_DAYS.get_loc(at)); truths.append((g/g.sum(1)[:, None])@EF)
    if not positions:
        raise ValueError('Nonempty external truth support required')
    return np.asarray(positions, int), np.stack(truths)


def feature_packet(raw: np.ndarray, features: dict[str, np.ndarray]) -> dict:
    """Budget whitelist: coarse features plus quantities derived from this base only."""
    a = np.asarray(raw)
    if a.ndim != 4 or a.shape[0] not in (3, 5) or a.shape[2:] != (128, 24) or not a.shape[1]:
        raise ValueError('Three B0 or all five bases by days by128 by24 required')
    a = real(a, a.shape, 'base paths').clip(0, 1000)
    n = a.shape[1]
    if len(a) == 5:
        size = real(features['Size'], (n, 128, 31), 'Size')
        cap = real(features['SizeCap'], (n, 128, 36), 'SizeCap')
        if not np.array_equal(cap[..., :31], size):
            raise ValueError('Literal common coarse feature prefix across budgets required')
    point, std = a.mean(2), a.std(2, ddof=0)
    level = a.mean(-1).std(-1, ddof=0)
    centered = a-a.mean(-1, keepdims=True)
    shape = np.sqrt(centered.var(2, ddof=0).mean(-1))
    matrices, names = [], []
    phase = np.arange(24)*2*np.pi/24
    for i, budget in enumerate(BUDGETS[:len(a)]):
        family = 'Size' if budget == 'B0' else 'SizeCap'
        field_names = tuple(fleet.FEATURE_NAMES[family])
        visible = real(features[family], (n, 128, len(field_names)), family).mean(1)
        daily = [point[i].mean(-1), point[i].min(-1), point[i].max(-1), level[i], shape[i]]
        extra = np.stack([point[i], std[i], *[np.broadcast_to(z[:, None], (n, 24)) for z in daily],
                          np.broadcast_to(np.sin(phase), (n, 24)), np.broadcast_to(np.cos(phase), (n, 24))], -1)
        matrices.append(np.concatenate((np.broadcast_to(visible[:, None], (n, 24, len(field_names))), extra), -1))
        names.append(field_names+EXTRA_NAMES)
    return dict(X=matrices, feature_names=names, point=point, std_hour=std, std_level=level, std_shape=shape)


def interval_metrics(lower: np.ndarray, upper: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(truth)
    if y.ndim != 2 or y.shape[1] != 24 or not len(y):
        raise ValueError('Days by24 truth required')
    y = real(y, y.shape, 'truth')
    lo, hi = np.asarray(lower), np.asarray(upper)
    if lo.shape != hi.shape or lo.shape[-2:] != y.shape:
        raise ValueError('Matching interval array suffix days by24 required')
    lo, hi = real(lo, lo.shape, 'lower'), real(hi, hi.shape, 'upper')
    if np.any(lo > hi) or np.any(lo < 0) or np.any(hi > 1000) or np.any(y < 0) or np.any(y > 1000):
        raise ValueError('Ordered physical support intervals and truth required')
    left, right = y < lo, y > hi
    hit = ~left & ~right
    return dict(coverage90=hit.mean(-1), width90=(hi-lo).mean(-1),
                interval_score90=((hi-lo)+20*np.maximum(lo-y, 0)+20*np.maximum(y-hi, 0)).mean(-1),
                lower_miss=left.mean(-1), upper_miss=right.mean(-1), path_coverage90=hit.all(-1).astype(float))


def raw_intervals(raw: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(raw)
    if a.ndim != 4 or a.shape[0] not in (3, 5) or a.shape[2:] != (128, 24) or factor not in FACTORS:
        raise ValueError('Three B0 or five raw bases and fixed global spread factor required')
    a = real(a, a.shape, 'raw reference').clip(0, 1000)
    center = a.mean(2, keepdims=True)
    ordered = np.sort(np.clip(center+factor*(a-center), 0, 1000), axis=2)
    return ordered[:, :, 6], ordered[:, :, 121]
