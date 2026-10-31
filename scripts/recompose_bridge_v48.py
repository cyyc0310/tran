"""Fixed recompose-base controls; reuse the frozen bridge algebra, no fitting."""
from __future__ import annotations

import numpy as np
from scripts import daily_bridge_v47 as prior
from scripts import hourly_bridge_v46 as validation

LEVEL_ARMS = prior.LEVEL_ARMS
NEW_ARMS = (tuple('Recompose-' + name for name in prior.ARMS[3:])
            + tuple('Recompose-AlignedKeep-' + name for name in LEVEL_ARMS))
ARMS = prior.ARMS + NEW_ARMS
BUDGET_BY_ARM = dict(prior.BUDGET_BY_ARM) | {
    new: prior.BUDGET_BY_ARM[old] for new, old in zip(NEW_ARMS[:60], prior.ARMS[3:])} | {
    name: prior.BUDGET_BY_ARM['Keep-' + level] for name, level in zip(NEW_ARMS[60:], LEVEL_ARMS)}
EF = prior.EF


def build(recompose_raw: np.ndarray, anchor_raw: np.ndarray, recompose_fine: np.ndarray, anchor_fine: np.ndarray,
          query_mean: np.ndarray, MG: np.ndarray, levels: np.ndarray) -> dict:
    """Return 75 controls; discard the helper's three temporary reference slots.

    Filling its reference slots with R avoids any dependence on a higher-budget
    M8 reference. Those slots are discarded, never relabeled as actual references.
    Date, key and exact particle alignment remain caller obligations.
    """
    fine = validation._real(recompose_fine, (181, 128, 8), 'recompose fine')
    old = validation._real(anchor_fine, (181, 128, 8), 'anchor fine')
    if not np.array_equal(fine, old):
        raise ValueError('Literal inherited Scale anchor/recompose fine pairing required')
    a = validation._real(anchor_raw, (181, 128, 24), 'anchor raw')
    result = prior.build_paths(recompose_raw, fine, query_mean, MG, levels,
                               recompose_raw, recompose_raw)
    raw = result['raw'][3:]
    u, p = result['old_scale_level'], result['old_point_level']
    r = np.asarray(recompose_raw, float)
    mg = np.asarray(MG, float)
    aligned = raw[15:30] + (a.mean(-1) - r.mean(-1))[None, ..., None]
    expected_mean = (a.mean(-1)[None] - p[None] + levels).copy()
    centered = aligned - aligned.mean(-1, keepdims=True)
    centered_r = r - r.mean(-1, keepdims=True)
    change = abs(centered - centered_r[None]).max((2, 3))
    extra = {k: v[15:30].copy() for k, v in result['diagnostics'].items()}
    zero = mg[..., 0] == 0
    extra.update(raw_clip_fraction=((aligned < 0) | (aligned > 1000)).mean((2, 3)),
        zero_mg_pos_nonzero_raw_hour_count=((abs(aligned) > 1e-9) & zero[None, ..., None]).sum((2, 3)),
        zero_mg_pos_positive_clipped_hour_count=((aligned > 0) & zero[None, ..., None]).sum((2, 3)),
        raw_hour_mean=aligned.mean((2, 3)), raw_centered_shape_max_abs_change=change,
        old_hour_minus_scale_mean=np.broadcast_to((a.mean(-1)-u).mean(-1), (15, 181)),
        old_hour_minus_scale_std=np.broadcast_to((a.mean(-1)-u).std(-1), (15, 181)))
    invariants = result['audit']['invariant_max_abs']
    invariants.update(aligned_keep_particle_mean=float(abs(aligned.mean(-1)-expected_mean).max()),
                      aligned_recompose_shape=float(change.max()))
    if (not np.isfinite(aligned).all() or any(not np.isfinite(v).all() for v in extra.values())
            or any(not np.isfinite(v) or v > 1e-9 for v in invariants.values())):
        raise ValueError('Finite mean-aligned paths/diagnostics and invariants within 1e-9 required')
    result['raw'] = np.concatenate((raw, aligned))
    result['diagnostics'] = {k: np.concatenate((v, extra[k])) for k, v in result['diagnostics'].items()}
    result['audit']['diagnostic_arms'] = list(NEW_ARMS)
    result['audit'].update(base='Scale-recompose', temporary_reference_slots_discarded=True,
                           aligned_base_particle_mean='Scale-anchor', M8_input_required=False, new_fits=0)
    return result
