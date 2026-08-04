"""Evaluation metrics for CIF forecasting (MAE, RMSE, sMAPE)."""

import numpy as np


def compute_metrics(pred, true):
    """Return MAE, RMSE, sMAPE between predictions and ground truth."""
    mae = float(np.abs(pred - true).mean())
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    denom = (np.abs(pred) + np.abs(true)) / 2 + 1e-8
    smape = float(np.mean(np.abs(pred - true) / denom) * 100)
    return {"mae": mae, "rmse": rmse, "smape": smape}
