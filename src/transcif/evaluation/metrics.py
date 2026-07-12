"""Point-forecast metrics (RMSE/MAE/sMAPE, matching both source papers) plus the
cross-domain degradation rate metric introduced for TransCIF's transfer evaluation."""

import numpy as np

SMAPE_ZERO_DENOMINATOR_EPSILON = 1e-8


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, SMAPE_ZERO_DENOMINATOR_EPSILON, denominator)
    return float(np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100)


def cross_domain_degradation_rate(in_domain_metric: float, cross_domain_metric: float) -> float:
    """Relative performance loss of a transferred model vs. a region-trained model on the
    same (lower-is-better) metric, expressed as a percentage."""
    if in_domain_metric == 0:
        raise ValueError("in_domain_metric must be nonzero to compute a relative degradation rate.")
    return float((cross_domain_metric - in_domain_metric) / in_domain_metric * 100)
