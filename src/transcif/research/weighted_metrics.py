"""Metrics for a finite weighted empirical predictive distribution."""

import numpy as np


def _validate(predictions: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions, weights = np.asarray(predictions, dtype=float), np.asarray(weights, dtype=float)
    if predictions.ndim != 2 or not predictions.size or weights.shape != (len(predictions),):
        raise ValueError("Expected nonempty predictions (particles, horizon) and weights (particles,).")
    if not np.isfinite(predictions).all() or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Predictions must be finite; weights must be finite, nonnegative, with positive total.")
    return predictions, weights / weights.sum()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray | float) -> np.ndarray:
    """Inverse empirical CDF (first value whose cumulative mass reaches q).

    Supports values (particles,) or (particles, horizon). The quantile axis
    comes first for an array of requested quantiles; a scalar removes it.
    """
    values = np.asarray(values, dtype=float)
    one_dimensional = values.ndim == 1
    predictions, weights = _validate(values[:, None] if one_dimensional else values, weights)
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.ndim > 1 or not np.isfinite(quantiles).all() or np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError("Quantiles must lie in [0, 1] and be scalar or one-dimensional.")
    scalar = quantiles.ndim == 0
    quantiles = np.atleast_1d(quantiles)
    result = np.empty((len(quantiles), predictions.shape[1]))
    for hour in range(predictions.shape[1]):
        order = np.argsort(predictions[:, hour], kind="stable")
        # Zero-weight particles do not belong to the empirical distribution.
        order = order[weights[order] > 0]
        cumulative = np.cumsum(weights[order])
        cumulative[-1] = 1.0
        indexes = np.minimum(np.searchsorted(cumulative, quantiles, side="left"), len(order) - 1)
        result[:, hour] = predictions[order[indexes], hour]
    if one_dimensional:
        result = result[:, 0]
    return result[0] if scalar else result


class PreparedDistribution:
    """Cache truth-independent terms when scoring one distribution repeatedly.

    Inputs are copied so later caller mutations cannot invalidate the cached
    means, quantiles, or weighted pair-distance term used in CRPS.
    """

    def __init__(self, predictions: np.ndarray, weights: np.ndarray):
        predictions, weights = _validate(predictions, weights)
        self.predictions = predictions.copy()
        self.weights = weights.copy()
        self.predictions.setflags(write=False)
        self.weights.setflags(write=False)
        self.mean = self.weights @ self.predictions
        order = np.argsort(self.predictions, axis=0)
        sorted_values = np.take_along_axis(self.predictions, order, axis=0)
        sorted_weights = self.weights[order]
        mass_before = np.cumsum(sorted_weights, axis=0) - sorted_weights
        values_before = np.cumsum(sorted_weights * sorted_values, axis=0) - sorted_weights * sorted_values
        self.half_pair_distance = np.sum(sorted_weights * (sorted_values * mass_before - values_before), axis=0)
        self.lower, self.upper = weighted_quantile(self.predictions, self.weights, np.array([0.05, 0.95]))
        self.width90 = float(np.mean(self.upper - self.lower))

    def score(self, truth: np.ndarray) -> dict[str, float]:
        truth = np.asarray(truth, dtype=float)
        if truth.shape != (self.predictions.shape[1],) or not np.isfinite(truth).all():
            raise ValueError("Truth must be finite with shape (horizon,).")
        residual = self.mean - truth
        crps = np.sum(self.weights[:, None] * np.abs(self.predictions - truth), axis=0) - self.half_pair_distance
        return {
            "mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.square(residual).mean())),
            "crps": float(crps.mean()),
            "coverage90": float(np.mean((truth >= self.lower) & (truth <= self.upper))),
            "width90": self.width90,
            "shape_mae": float(np.abs((self.mean - self.mean.mean()) - (truth - truth.mean())).mean()),
            "level_mae": float(abs(self.mean.mean() - truth.mean())),
        }


def distribution_metrics(predictions: np.ndarray, weights: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return PreparedDistribution(predictions, weights).score(truth)
