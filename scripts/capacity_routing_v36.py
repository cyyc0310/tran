"""Ten exploratory range-fallback baselines; pure arrays, no fitting or truth."""
from __future__ import annotations

import numpy as np


RECIPES = ('Global0', 'Global0.25', 'Global0.5', 'Global0.75', 'Global1',
           'Range4', 'CoalUpper', 'OilUpper', 'CoalOilUpper', 'AnchorEFUpper')
FUELS = ('coal', 'gas', 'oil', 'other')
TOLERANCE = 1e-12


def _array(value, shape=None):
    if np.iscomplexobj(value):
        raise ValueError('Real finite arrays required')
    result = np.asarray(value, dtype=float)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError('Finite arrays with registered shapes required')
    return result


def _simplex(value, dimension, shape=None):
    result = _array(value, shape)
    if (result.ndim < 2 or result.shape[-1] != dimension or not result.size or np.any(result < 0)
            or not np.allclose(result.sum(-1), 1., atol=TOLERANCE, rtol=0)):
        raise ValueError('Nonempty nonnegative simplex rows required')
    return result


def weights(train_q: np.ndarray, query_q: np.ndarray) -> np.ndarray:
    """Return (10,6) anchor weights from training extrema, not statistical support."""
    train = _simplex(train_q, 4)
    query = _simplex(query_q, 4, (6, 4))
    if train.ndim != 2:
        raise ValueError('Training capacity rows must have shape (N,4)')
    low, high = train.min(0), train.max(0)
    upper = query > high+TOLERANCE
    anchor_train = 400+600*train[:, 0]+400*train[:, 2]
    anchor_query = 400+600*query[:, 0]+400*query[:, 2]
    return np.vstack([np.repeat(np.arange(5)[:, None]/4, 6, axis=1),
        np.any((query < low-TOLERANCE) | upper, axis=1), upper[:, 0], upper[:, 2],
        upper[:, 0] | upper[:, 2], anchor_query > anchor_train.max()+TOLERANCE]).astype(float)


def blend(weight: np.ndarray, baseEF: np.ndarray, anchorEF: np.ndarray,
          MG: np.ndarray) -> dict[str, np.ndarray]:
    """Mix aligned EF particles before multiplying MG; never mix median points.

    The runner restores archived CIF particles and mean points at every binary
    endpoint. The anchor must be constant within each origin's 128 particles.
    """
    w = _array(weight, (10, 6))
    base, anchor = _array(baseEF, (6, 128)), _array(anchorEF, (6, 128))
    mg = _simplex(MG, 5, (6, 128, 5))
    if np.any((w < 0) | (w > 1)) or np.any(base < 0) or np.any(anchor < 0):
        raise ValueError('Convex weights and nonnegative EF required')
    if not np.array_equal(anchor, np.broadcast_to(anchor[:, :1], anchor.shape)):
        raise ValueError('Capacity anchor must be particle-constant within each origin')
    ef = (1-w[..., None])*base[None]+w[..., None]*anchor[None]
    for i in range(10):
        ef[i, w[i] == 0] = base[w[i] == 0]
        ef[i, w[i] == 1] = anchor[w[i] == 1]
    cif = ef*mg[None, ..., 0]
    if not np.isfinite(ef).all() or not np.isfinite(cif).all():
        raise ValueError('Finite blended particles required')
    return dict(EF_draws=ef, CIF_draws=cif)
