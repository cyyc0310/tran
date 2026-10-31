"""Fixed ordinary center projections on frozen native uncertainty paths.

Source-only information diagnostic; no new model or calibration is fitted.
"""
from __future__ import annotations

from typing import Any

import numpy as np

SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
PACKETS = ('D2', 'D8', 'T8', 'G5')
ALPHAS = (0., .5, 1.)
FACTORS = (.75, 1., 1.25, 1.5, 2.)
METHODS = ('ET_day', 'ET_block', 'ET_full', 'raw_day')
SAMPLES = 128


def arms(packet: str) -> tuple[tuple[str, float], ...]:
    """The exact registered grid, including intentionally repeated alpha-zero arms."""
    if packet not in PACKETS:
        raise ValueError('Registered packet required')
    methods = METHODS + (('raw_block',) if packet == 'T8' else ())
    return tuple((method, alpha) for method in methods for alpha in ALPHAS)


def project(delta: np.ndarray, resolution: str) -> np.ndarray:
    """Orthogonal equal-hour day/UTC-six-hour/full projections of a center delta."""
    x = np.asarray(delta, float)
    if x.ndim != 2 or x.shape[1] != 24 or not np.isfinite(x).all():
        raise ValueError('Finite day-by-24 center deltas required')
    if resolution == 'full':
        return x.copy()
    if resolution == 'day':
        return np.repeat(x.mean(1, keepdims=True), 24, axis=1)
    if resolution == 'block':
        return np.repeat(x.reshape(-1, 4, 6).mean(2), 6, axis=1)
    raise ValueError('Registered center projection required')


def extract_et_center(archive: Any) -> np.ndarray:
    """Take the fitted point, never the ensemble mean containing Q4 residuals."""
    names = np.asarray(archive['models']).tolist()
    expected = ['ridge_global', 'ET_global', 'ET_hetero', 'ET_QRF']
    centers = np.asarray(archive['centers'])
    if names != expected or centers.ndim != 3 or centers.shape[0] != 4 or centers.shape[2] != 24:
        raise ValueError('Registered V17 fitted centers required')
    et = centers[1]
    if (not np.isfinite(centers).all() or np.any(centers < 0) or np.any(centers > 1000)
            or not np.array_equal(et, centers[2]) or not np.array_equal(et, centers[3])):
        raise ValueError('V17 ET fitted points must be identical across risk models')
    return et.copy()


def shifted_paths(native: np.ndarray, delta: np.ndarray, alpha: float) -> np.ndarray:
    """Translate all 128 particles together; alpha zero is an exact byte-value copy."""
    n, d = np.asarray(native), np.asarray(delta, float)
    if (n.ndim != 3 or n.shape[1:] != (SAMPLES, 24) or d.shape != (len(n), 24)
            or not np.isfinite(n).all() or not np.isfinite(d).all()
            or np.any(n < 0) or np.any(n > 1000) or alpha not in ALPHAS):
        raise ValueError('Bounded native paths, aligned delta and fixed alpha required')
    return n.copy() if alpha == 0 else n + alpha*d[:, None, :]


def summarize_paths(raw: np.ndarray, native: np.ndarray, requested: np.ndarray) -> dict[str, np.ndarray]:
    """Separate requested, actual unclipped and realized clipped centers."""
    if raw.shape != native.shape or requested.shape != (len(raw), 24) or not np.isfinite(raw).all():
        raise ValueError('Finite matching paths and requested center required')
    clipped = np.clip(raw, 0, 1000)
    uncenter = raw.mean(1)
    native_center = native.mean(1)
    noise_error = np.abs((raw-uncenter[:, None])-(native-native_center[:, None])).max((1, 2))
    if not np.isfinite(requested).all() or np.max(abs(uncenter-requested), initial=0) > 1e-9 or noise_error.max(initial=0) > 1e-9:
        raise ValueError('Center translation or native pre-clip uncertainty changed')
    return dict(requested_center=requested, unclipped_center=uncenter,
                realized_center=clipped.mean(1), new_clipping_rate=((raw < 0) | (raw > 1000)).mean((1, 2)),
                native_noise_max_abs_error=noise_error)


def build(native: np.ndarray, donor_ids: np.ndarray, et_center: np.ndarray,
          day_anchor: np.ndarray, block_anchor: np.ndarray | None, packet: str) -> dict[str, np.ndarray]:
    """Construct all projections without refitting or changing particle identity."""
    n = np.asarray(native)
    if n.ndim != 3 or n.shape[1:] != (128, 24) or not len(n):
        raise ValueError('Native day-by-128-by-24 paths required')
    days = len(n)
    ids, center, anchor = np.asarray(donor_ids), np.asarray(et_center, float), np.asarray(day_anchor, float)
    if (ids.shape != (days, 128) or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0)
            or center.shape != (days, 24) or anchor.shape != (days,)
            or not np.isfinite(center).all() or not np.isfinite(anchor).all()
            or np.any(center < 0) or np.any(center > 1000) or np.any(anchor < 0) or np.any(anchor > 1000+1e-9)):
        raise ValueError('Aligned finite fitted centers, daily anchors and donor IDs required')
    if packet == 'T8':
        block = np.asarray(block_anchor, float)
        if block.shape != (days, 4) or not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1000+1e-9):
            raise ValueError('T8 requires four finite released EWMA H/G block anchors')
    elif block_anchor is not None:
        raise ValueError('Block H/G anchors are not available to this packet')
    grid = arms(packet)
    mean = n.mean(1)
    delta = center-mean
    offsets = {f'ET_{resolution}': project(delta, resolution) for resolution in ('day', 'block', 'full')}
    offsets['raw_day'] = anchor[:, None]-project(mean, 'day')
    if packet == 'T8':
        offsets['raw_block'] = np.repeat(block, 6, axis=1)-project(mean, 'block')
    rows, summaries, shifts = [], [], []
    for method, alpha in grid:
        shifted = shifted_paths(n, offsets[method], alpha)
        requested = mean.copy() if alpha == 0 else mean+alpha*offsets[method]
        rows.append(shifted)
        summaries.append(summarize_paths(shifted, n, requested))
        shifts.append(np.zeros_like(mean) if alpha == 0 else alpha*offsets[method])
    return dict(raw=np.stack(rows), methods=np.array([g[0] for g in grid]), alphas=np.array([g[1] for g in grid]),
                donor_ids=ids.copy(), native_mean=mean, et_center=center.copy(), day_anchor=anchor.copy(),
                block_anchor=block.copy() if packet == 'T8' else np.empty((days, 0)), shift=np.stack(shifts),
                native_boundary_rate=((n <= 0) | (n >= 1000)).mean((1, 2)),
                **{key: np.stack([s[key] for s in summaries]) for key in summaries[0]})


def scaled_paths(raw: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve the frozen score convention and expose final spread clipping."""
    if factor not in FACTORS or raw.ndim != 3 or raw.shape[1:] != (128, 24) or not np.isfinite(raw).all():
        raise ValueError('Registered spread and finite raw paths required')
    paths = np.clip(raw, 0, 1000)
    point = paths.mean(1)
    expanded = point[:, None]+factor*(paths-point[:, None])
    return point, np.clip(expanded, 0, 1000), ((expanded < 0) | (expanded > 1000)).mean((1, 2))
