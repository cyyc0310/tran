"""Stage 0: scale-invariant reparameterization of raw grid signals (Innovation 1)."""

import numpy as np
import pandas as pd

MIN_ROLLING_PERIODS = 24


def compute_renew_share(renew_out: np.ndarray, nonrenew_out: np.ndarray) -> np.ndarray:
    """RenewShare_t = RenewOut_t / (RenewOut_t + NonRenewOut_t). NaN when total output is zero."""
    total = renew_out + nonrenew_out
    return np.divide(
        renew_out,
        total,
        out=np.full_like(renew_out, np.nan, dtype=float),
        where=total > 0,
    )


def compute_load_norm(load: np.ndarray, window: int = 720, quantile: float = 0.95) -> np.ndarray:
    """LoadNorm_t = Load_t / rolling_quantile(Load, window, quantile).

    Scale-invariant: multiplying `load` by any positive constant leaves the ratio unchanged
    once the rolling window is fully populated (the first `window` samples use a partially
    filled window and are backfilled from the first fully-populated value).
    """
    load_series = pd.Series(load, dtype=float)
    rolling_q = load_series.rolling(window=window, min_periods=MIN_ROLLING_PERIODS).quantile(quantile)
    rolling_q = rolling_q.bfill()
    return (load_series / rolling_q).to_numpy()


def compute_temp_anomaly(temp: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """TempAnomaly_t = Temp_t - mean(Temp | same day_of_year), i.e. climate-baseline anomaly."""
    frame = pd.DataFrame({"temp": temp, "doy": day_of_year})
    baseline = frame.groupby("doy")["temp"].transform("mean")
    return (frame["temp"] - baseline).to_numpy()
