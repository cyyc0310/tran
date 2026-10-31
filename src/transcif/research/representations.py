"""Fixed orthonormal trajectory coordinates; no target-label fitted weights.

Haar supports non-power-of-two horizons by stopping at the first odd coarse
length. The remaining coarse block receives an orthonormal DCT. Scale weights
are an explicit exploratory utility choice, not extra observational evidence.
"""

from functools import lru_cache

import numpy as np
from scipy.fft import dct


@lru_cache(maxsize=32)
def basis_matrix(length: int, kind: str) -> np.ndarray:
    """Return a matrix W with coefficients = trajectories @ W.T."""
    if not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError("length must be a positive integer.")
    if kind == "time":
        basis = np.eye(length)
    elif kind == "dct":
        basis = dct(np.eye(length), type=2, norm="ortho", axis=0)
    elif kind == "haar":
        coarse = np.eye(length)
        details = []
        while len(coarse) > 1 and len(coarse) % 2 == 0:
            details.append((coarse[::2] - coarse[1::2]) / np.sqrt(2.0))
            coarse = (coarse[::2] + coarse[1::2]) / np.sqrt(2.0)
        coarse = dct(coarse, type=2, norm="ortho", axis=0)
        basis = np.concatenate([coarse, *reversed(details)], axis=0)
    else:
        raise ValueError(f"Unknown representation: {kind}")
    basis.setflags(write=False)
    return basis


def scale_weights(length: int, kind: str) -> np.ndarray:
    """Fixed squared-distance weights; equal weights are the invariance control."""
    basis_matrix(length, kind)
    if kind == "time":
        return np.ones(length)
    if kind == "dct":
        return 1.0 / (1.0 + np.arange(length) / 3.0)
    sizes = []
    remaining = length
    while remaining > 1 and remaining % 2 == 0:
        remaining //= 2
        sizes.append(remaining)
    weights = [np.ones(remaining)]
    weights.extend(np.full(size, 2.0 ** (-level)) for level, size in enumerate(reversed(sizes), start=1))
    return np.concatenate(weights)


def transform_predictions(predictions: np.ndarray, kind: str, *, weighted: bool = True) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("predictions must be a finite (particles, nonempty horizon) matrix.")
    transformed = values @ basis_matrix(values.shape[1], kind).T
    if weighted:
        transformed = transformed * np.sqrt(scale_weights(values.shape[1], kind))
    return transformed


def prediction_metric(length: int, kind: str, *, weighted: bool = True) -> np.ndarray:
    """Time-coordinate quadratic kernel exactly equivalent to the transform.

    Weighted Haar utility can always be expressed as this time-domain kernel;
    improvement does not establish an intrinsic advantage of wavelets.
    """
    basis = basis_matrix(length, kind)
    weights = scale_weights(length, kind) if weighted else np.ones(length)
    return basis.T @ (weights[:, None] * basis)
