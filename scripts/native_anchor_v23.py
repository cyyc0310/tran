"""DRAFT/PAUSED: legacy daily-packet anchor controls, not the monthly/yearly route.

Pure synthetic helpers remain available; no formal stage is authorized. The user
main scenario permits only public monthly or annual fuel statistics. This draft
has no completed protocol, independent approval or actual experiment output.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scripts.local_sketch_v17 import PointModel, SOURCES
from scripts.native_center_v18 import project, shifted_paths, summarize_paths, scaled_paths

INPUTS = ('native_plus_D2_anchor', 'packet_only_T8', 'packet_only_G5', 'joint_T8', 'joint_G5')
WIDTHS = (55, 51, 35, 99, 83)
PROJECTIONS, ALPHAS, FACTORS = ('day', 'block', 'full'), (.5, 1.), (.75, 1., 1.25, 1.5, 2.)
SEED, TREES, LEAF = 17023, 150, 10


class AnchorModel(PointModel):
    """Use the frozen V17 ET/scaler/weights/y-A/clip implementation verbatim."""
    def __init__(self) -> None:
        super().__init__('ET')


def arms() -> tuple[tuple[str, str, float], ...]:
    return (('native', 'none', 0.),) + tuple((name, p, a) for name in INPUTS for p in PROJECTIONS for a in ALPHAS)


def input_arrays(parent: dict[str, np.ndarray], phase: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if phase not in ('train', 'query'):
        raise ValueError('Registered training/query phase required')
    anchors = parent[phase + '_anchors']
    native = parent[phase + '_x__native_only']
    n = len(native)
    if (native.shape != (n, 54) or anchors.shape != (4, n) or not np.isfinite(native).all()
            or not np.isfinite(anchors).all() or np.any(anchors < 0) or np.any(anchors > 1000 + 1e-9)):
        raise ValueError('Complete inherited native features and four-packet anchors required')
    x = {'native_plus_D2_anchor': np.column_stack((native, anchors[0]))}
    a = {'native_plus_D2_anchor': anchors[0].copy()}
    for name in INPUTS[1:]:
        x[name] = parent[phase + '_x__' + name].copy()
        a[name] = anchors[2 if name.endswith('T8') else 3].copy()
    for name, width in zip(INPUTS, WIDTHS):
        if x[name].shape != (n, width) or not np.isfinite(x[name]).all():
            raise ValueError('Five complete inherited input matrices required')
    return x, a


def deltas_from_centers(native: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if (native.ndim != 3 or native.shape[1:] != (128, 24) or centers.shape != (5, len(native), 24)
            or not np.isfinite(native).all() or not np.isfinite(centers).all()
            or np.any(native < 0) or np.any(native > 1000) or np.any(centers < 0) or np.any(centers > 1000)):
        raise ValueError('Native128 paths and five clipped absolute desired centers required')
    return centers - native.mean(1)[None]


def reconstruct(native: np.ndarray, deltas: np.ndarray, arm: tuple[str, str, float]) -> tuple[np.ndarray, np.ndarray]:
    if deltas.shape != (5, len(native), 24) or not np.isfinite(deltas).all() or arm not in arms():
        raise ValueError('Five finite deltas and a registered31-arm setting required')
    name, resolution, alpha = arm
    shift = np.zeros((len(native), 24)) if name == 'native' else project(deltas[INPUTS.index(name)], resolution)
    raw = shifted_paths(native, shift, alpha)
    point = native.mean(1) if alpha == 0 else native.mean(1) + alpha * shift
    return raw, point


def diagnostics(native: np.ndarray, deltas: np.ndarray) -> dict[str, np.ndarray]:
    records = []
    for arm in arms():
        raw, requested = reconstruct(native, deltas, arm)
        record = summarize_paths(raw, native, requested)
        ordered, clipped = np.sort(raw, axis=1), np.sort(np.clip(raw, 0, 1000), axis=1)
        record.update(unclipped_width90=(ordered[:, 121] - ordered[:, 6]).mean(1),
                      initial_clipped_width90=(clipped[:, 121] - clipped[:, 6]).mean(1))
        records.append(record)
    return {key: np.stack([row[key] for row in records]) for key in records[0]}


def exact_reproduction(native: np.ndarray, centers: np.ndarray, deltas: np.ndarray,
                       references: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Hard equality gate: do not silently accept numerical or training changes."""
    if set(references) != {'T8', 'G5'}:
        raise ValueError('Both historical packet-only references required')
    expected_deltas = deltas_from_centers(native, centers)
    if not np.array_equal(deltas, expected_deltas):
        raise ValueError('Stored deltas differ from clipped desired point minus native mean')
    checked = 0
    for packet, reference in references.items():
        name = 'packet_only_' + packet;i = INPUTS.index(name)
        if not np.array_equal(centers[i], reference['center']):
            raise ValueError('Exact V17 packet-only desired-center reproduction failed: ' + packet)
        grid = list(zip(reference['methods'].tolist(), reference['alphas'].tolist()))
        for resolution in PROJECTIONS:
            for alpha in ALPHAS:
                historical = ('ET_' + resolution, alpha)
                if grid.count(historical) != 1:
                    raise ValueError('Registered historical V18 ET projection grid required')
                raw, _ = reconstruct(native, deltas, (name, resolution, alpha))
                if not np.array_equal(raw, reference['raw'][grid.index(historical)]):
                    raise ValueError('Exact V18 packet-only path reproduction failed: ' + str((packet, resolution, alpha)))
                checked += 1
    return dict(packet_centers=2, projection_alpha_paths=checked)
