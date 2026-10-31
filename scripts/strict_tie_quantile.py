"""Explicit pinball subgradient at zero for a version-pinned baseline control."""
from __future__ import annotations

import numpy as np
import sklearn
from sklearn._loss.loss import PinballLoss
from sklearn.ensemble import GradientBoostingRegressor


class StrictTiePinballLoss(PinballLoss):
    def __init__(self, sample_weight=None, quantile=.5):
        super().__init__(sample_weight=sample_weight, quantile=quantile)
        self.quantile_value = quantile

    def gradient(self, y_true, raw_prediction, sample_weight=None, gradient_out=None, n_threads=1):
        # The derivative at raw == y may be any value in [-q, 1-q].
        # Choose 1-q so tied zeros differ from strictly positive residuals.
        values = super().gradient(y_true, raw_prediction, sample_weight, gradient_out, n_threads)
        ties = np.asarray(y_true).reshape(-1) == np.asarray(raw_prediction).reshape(-1)
        if sample_weight is None:
            values[ties] = 1 - self.quantile_value
        else:
            values[ties] = (1 - self.quantile_value) * np.asarray(sample_weight).reshape(-1)[ties]
        return values


class StrictTieQuantileRegressor(GradientBoostingRegressor):
    def _get_loss(self, sample_weight):
        if sklearn.__version__ != "1.8.0" or self.loss != "quantile":
            raise RuntimeError("strict-tie baseline requires sklearn 1.8.0 and quantile loss")
        return StrictTiePinballLoss(sample_weight=sample_weight, quantile=self.alpha)


def audit_gradient_contract() -> None:
    for q in (.05, .95):
        loss = StrictTiePinballLoss(quantile=q)
        y, raw = np.asarray([1., -1., 0.]), np.zeros((3, 1))
        expected = np.asarray([-q, 1 - q, 1 - q])
        if not np.array_equal(loss.gradient(y, raw), expected):
            raise ValueError("pinball derivative convention mismatch")
        output = np.empty((3, 1))
        weights = np.asarray([2., 3., 4.])
        result = loss.gradient(y, raw, weights, output)
        native = PinballLoss(quantile=q).gradient(y, raw, weights, np.empty_like(output))
        if (result.shape != native.shape or not np.shares_memory(result, output)
                or not np.array_equal(output[:, 0], expected * weights)):
            raise ValueError("gradient output-buffer/weight contract mismatch")
        for dtype in (np.float32, np.float64):
            actual = loss.gradient(y.astype(dtype), raw.astype(dtype))
            reference = PinballLoss(quantile=q).gradient(y.astype(dtype), raw.astype(dtype))
            if actual.dtype != reference.dtype or actual.shape != reference.shape:
                raise ValueError("native gradient dtype/shape contract mismatch")
