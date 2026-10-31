"""Fixed scalar-level hourly bridge controls; pure arrays, no fitted models or I/O."""
from __future__ import annotations

from typing import Any

import numpy as np


POS, ZERO = (0, 1, 2, 7), (3, 4, 5, 6)
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
EF.setflags(write=False)
LEVEL_ARMS = (
    tuple(f'{scope}-{family}-{method}' for scope in ('Legacy', 'Expanded')
          for family in ('Size', 'SizeCap', 'DirectEF') for method in ('Ridge', 'ET'))
    + ('Capacity-OP', 'CoalOilUpper', 'Mix-Router-DirectEFET-w0.25')
    + tuple(f'Gain-{candidate}-{target}-{method}'
            for candidate in ('SizeCapET', 'DirectEFET') for target in ('EF', 'CIF', 'Joint')
            for method in ('Mean', 'HistoryForecast-Ridge', 'HistoryForecast-ET',
                           'HistoryForecastCap-Ridge', 'HistoryForecastCap-ET')))
MODES = ('Replace', 'Keep')
CONTROLS = ('CoalOilUpper', 'Mix-Router-DirectEFET-w0.25',
            'Gain-SizeCapET-CIF-HistoryForecast-ET')
ARMS = (('Scale-anchor', 'Scale-recompose', 'M8-reference')
        + tuple(f'{mode}-{level}' for mode in MODES for level in LEVEL_ARMS)
        + tuple(f'{mode}-{level}' for mode in ('CenterOnly', 'FlatKeep') for level in CONTROLS))
_B0_LEVELS = {f'{scope}-Size-{method}' for scope in ('Legacy', 'Expanded')
              for method in ('Ridge', 'ET')}
BUDGET_BY_ARM = {arm: ('M8' if arm == 'M8-reference' else 'B0'
                      if arm in ARMS[:2] or any(arm == f'{mode}-{level}'
                                               for mode in MODES for level in _B0_LEVELS)
                      else 'B1') for arm in ARMS}
ATOL = 1e-9


def _real(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f'Real {name} required')
    try:
        result = np.asarray(raw, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Real numeric {name} required') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'Finite {name} with shape {shape} required')
    return result


def _simplex(value: np.ndarray, name: str) -> None:
    if np.any(value < 0) or not np.allclose(value.sum(-1), 1., atol=1e-12, rtol=0):
        raise ValueError(f'Nonnegative simplex {name} required')


def build(scale_raw: np.ndarray, scale_fine: np.ndarray, query_mean: np.ndarray,
          MG: np.ndarray, levels: np.ndarray, scale_recompose: np.ndarray,
          m8_raw: np.ndarray) -> dict[str, Any]:
    """Return raw99 paths and 96-arm diagnostics; leave clipping/scoring to callers.

    Date/source/particle identity is a caller metadata obligation. Centered donor
    trajectories are retained algebraically, not a monthly energy constraint or
    an uncertainty calibration. FlatKeep deliberately removes within-day shape.
    """
    a = _real(scale_raw, (6, 128, 24), 'scale_raw')
    fine = _real(scale_fine, (6, 128, 8), 'scale_fine')
    mean = _real(query_mean, (6, 128, 4), 'query_mean')
    mg = _real(MG, (6, 128, 5), 'MG')
    s = _real(levels, (45, 6, 128), 'levels')
    recompose = _real(scale_recompose, (6, 128, 24), 'scale_recompose')
    m8 = _real(m8_raw, (6, 128, 24), 'm8_raw')
    for value, name in ((fine, 'scale_fine'), (mean, 'query_mean'), (mg, 'MG')):
        _simplex(value, name)
    if (not np.allclose(fine[..., POS].sum(-1), mg[..., 0], atol=1e-12, rtol=0)
            or not np.allclose(fine[..., ZERO], mg[..., 1:], atol=1e-12, rtol=0)):
        raise ValueError('scale_fine POS and zero-carbon components must match MG')
    if np.any(s < 400 * mg[None, ..., 0] - ATOL) or np.any(s > 1000 * mg[None, ..., 0] + ATOL):
        raise ValueError('Monthly levels must lie in [400*MG_POS,1000*MG_POS]')
    try:
        with np.errstate(over='raise', invalid='raise'):
            return _paths(a, fine, mean, mg, s, recompose, m8)
    except FloatingPointError as exc:
        raise ValueError('Finite hourly bridge arithmetic required') from exc


def _paths(a: np.ndarray, fine: np.ndarray, mean: np.ndarray, mg: np.ndarray,
           s: np.ndarray, recompose: np.ndarray, m8: np.ndarray) -> dict[str, Any]:
    u = fine @ EF
    p = mg[..., 0] * (mean @ EF[list(POS)])
    replace = a[None] - u[None, ..., None] + s[..., None]
    keep = a[None] - p[None, ..., None] + s[..., None]
    ids = [LEVEL_ARMS.index(name) for name in CONTROLS]
    center = a[None] + (s[ids] - p[None]).mean(-1)[..., None, None]
    flat = np.repeat(keep[ids].mean(-1, keepdims=True), 24, axis=-1)
    new = np.concatenate((replace, keep, center, flat))
    raw = np.concatenate((a[None], recompose[None], m8[None], new))
    centered = new - new.mean(-1, keepdims=True)
    shape_change = abs(centered - (a - a.mean(-1, keepdims=True))[None]).max((2, 3))
    inherited = u - p
    hour_minus_scale = a.mean(-1) - u
    zero = mg[..., 0] == 0
    selected_levels = np.concatenate((s, s, s[ids], s[ids]))

    def repeat(value: np.ndarray) -> np.ndarray:
        return np.broadcast_to(value, (96, 6)).copy()

    diagnostics = dict(
        raw_clip_fraction=((new < 0) | (new > 1000)).mean((2, 3)),
        zero_mg_pos_draw_count=repeat(zero.sum(-1)),
        zero_mg_pos_nonzero_raw_hour_count=((abs(new) > ATOL) & zero[None, ..., None]).sum((2, 3)),
        zero_mg_pos_positive_clipped_hour_count=((new > 0) & zero[None, ..., None]).sum((2, 3)),
        new_level_mean=selected_levels.mean(-1), raw_hour_mean=new.mean((2, 3)),
        inherited_level_residual_mean=repeat(inherited.mean(-1)),
        inherited_level_residual_std=repeat(inherited.std(-1, ddof=0)),
        old_hour_minus_scale_mean=repeat(hour_minus_scale.mean(-1)),
        old_hour_minus_scale_std=repeat(hour_minus_scale.std(-1, ddof=0)),
        raw_centered_shape_max_abs_change=shape_change)
    invariants = dict(
        replace_self_reconstruction=float(abs(a - u[..., None] + u[..., None] - a).max()),
        keep_self_reconstruction=float(abs(a - p[..., None] + p[..., None] - a).max()),
        keep_minus_replace=float(abs(keep - replace - inherited[None, ..., None]).max()),
        center_keep_ensemble_mean=float(abs(center.mean(2) - keep[ids].mean(2)).max()),
        flat_keep_particle_mean=float(abs(flat.mean(-1) - keep[ids].mean(-1)).max()),
        shifted_centered_shape=float(shape_change[:93].max()))
    if (not np.isfinite(raw).all() or any(not np.isfinite(v).all() for v in diagnostics.values())
            or any(not np.isfinite(v) or v > ATOL for v in invariants.values())):
        raise ValueError('Finite bridge diagnostics and algebraic invariants within 1e-9 required')
    return dict(raw=raw, old_scale_level=u, old_point_level=p, diagnostics=diagnostics,
                audit=dict(invariant_max_abs=invariants,
                           diagnostic_arms=list(ARMS[3:]), raw_clipped=False,
                           scalar_level_does_not_identify_r4=True,
                           monthly_energy_constraint=False, risk_calibration_claim=False))
