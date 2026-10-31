"""Finite column permutations preserving empirical margins and nonempty fuel rows."""
from __future__ import annotations

import hashlib
import math

import numpy as np


def feasible_indices(activity: np.ndarray, key: str, seed: int, minimum: int = 256, sweeps: int = 16):
    activity = np.asarray(activity, float)
    if (activity.ndim != 2 or activity.shape[1] != 8 or not len(activity)
            or not np.isfinite(activity).all() or (activity < 0).any()
            or (np.sum(activity > 0, axis=1) == 0).any()):
        raise ValueError('positive-generation source activity required')
    if minimum < 1 or sweeps < 0:
        raise ValueError('invalid finite shuffle settings')
    n = len(activity)
    copies = max(1, math.ceil(minimum / n))
    row_ids = np.tile(np.arange(n), copies)
    indices = np.tile(row_ids[:, None], (1, activity.shape[1]))
    positive = activity > 0
    counts = positive[row_ids].sum(axis=1)
    varying = np.flatnonzero(np.ptp(activity, axis=0) > 0)
    digest = hashlib.sha256(f'v169/{key}/{seed}'.encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    attempted, accepted, changed = 0, 0, 0
    if len(varying) >= 2:
        for _ in range(sweeps):
            for column in rng.permutation(varying):
                order = rng.permutation(len(indices))
                left = order[:len(order) // 2 * 2:2]
                right = order[1:len(order) // 2 * 2:2]
                left_ids, right_ids = indices[left, column].copy(), indices[right, column].copy()
                left_positive, right_positive = positive[left_ids, column], positive[right_ids, column]
                new_left = counts[left] - left_positive.astype(int) + right_positive.astype(int)
                new_right = counts[right] - right_positive.astype(int) + left_positive.astype(int)
                feasible = (new_left > 0) & (new_right > 0)
                attempted += len(left)
                accepted += int(feasible.sum())
                changed += int((feasible & (activity[left_ids, column] != activity[right_ids, column])).sum())
                indices[left[feasible], column] = right_ids[feasible]
                indices[right[feasible], column] = left_ids[feasible]
                counts[left[feasible]], counts[right[feasible]] = new_left[feasible], new_right[feasible]
    for column in range(8):
        if not np.array_equal(np.bincount(indices[:, column], minlength=n), np.full(n, copies)):
            raise ValueError('shuffle changed empirical fuel margin')
    values = activity[indices, np.arange(8)]
    if (np.sum(values > 0, axis=1) == 0).any() or not np.array_equal(counts, (values > 0).sum(axis=1)):
        raise ValueError('shuffle violated positive-generation support')
    return indices, {'source_rows': n, 'copies': copies, 'particles': len(indices),
                     'attempted_swaps': attempted, 'accepted_swaps': accepted, 'changed_swaps': changed,
                     'index_sha256': hashlib.sha256(indices.tobytes()).hexdigest()}


class FeasibleCopulaCache:
    """Cache by source support pattern; positive column rescaling changes no swaps."""

    def __init__(self):
        self.indices = {}

    def transform(self, activity, profile: int, hour: int, seed: int):
        activity = np.asarray(activity, float)
        if (activity.ndim != 2 or activity.shape[1] != 8 or not len(activity)
                or not np.isfinite(activity).all() or (activity < 0).any()
                or (np.sum(activity > 0, axis=1) == 0).any()):
            raise ValueError('positive-generation source activity required')
        support = hashlib.sha256((activity > 0).tobytes()).hexdigest()
        # Constancy of a column also controls which swaps are attempted.
        varying = tuple(np.ptp(activity, axis=0) > 0)
        equality = np.column_stack([np.unique(activity[:, j], return_inverse=True)[1]
                                    for j in range(activity.shape[1])]).astype(np.int32)
        equality_hash = hashlib.sha256(equality.tobytes()).hexdigest()
        key = (profile, hour, seed, activity.shape, support, varying, equality_hash)
        if key not in self.indices:
            self.indices[key] = feasible_indices(activity, f'{profile}/{hour}', seed)
        indices, metadata = self.indices[key]
        values = activity[indices, np.arange(8)]
        return values, metadata
