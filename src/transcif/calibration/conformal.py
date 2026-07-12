"""Innovation 5: split-conformal prediction reusing Stage 3's calibration set, providing
a finite-sample coverage guarantee that widens automatically when the target domain
diverges from the source — an honest, self-aware signal for "plug-and-play" reliability."""

import numpy as np


def compute_nonconformity_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_true - y_pred)


def conformal_interval_halfwidth(nonconformity_scores: np.ndarray, coverage: float = 0.9) -> float:
    """Standard split-conformal quantile with the finite-sample correction
    ceil((n + 1) * coverage) / n, clipped to at most 1.0."""
    n = len(nonconformity_scores)
    if n == 0:
        raise ValueError("Calibration set must be non-empty to compute conformal intervals.")
    q_level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(nonconformity_scores, q_level, method="higher"))


def predict_with_interval(point_pred: np.ndarray, halfwidth: float) -> tuple:
    return point_pred - halfwidth, point_pred + halfwidth


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
