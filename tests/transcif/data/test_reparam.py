import numpy as np
import pandas as pd
from transcif.data.reparam import (
    compute_renew_share,
    compute_load_norm,
    compute_temp_anomaly,
)


def test_compute_renew_share_basic_ratio():
    renew = np.array([30.0, 50.0, 0.0])
    nonrenew = np.array([70.0, 50.0, 100.0])
    result = compute_renew_share(renew, nonrenew)
    np.testing.assert_allclose(result, [0.3, 0.5, 0.0])


def test_compute_renew_share_zero_total_is_nan():
    renew = np.array([0.0])
    nonrenew = np.array([0.0])
    result = compute_renew_share(renew, nonrenew)
    assert np.isnan(result[0])


def test_compute_load_norm_scale_invariance():
    rng = np.random.default_rng(42)
    base_load = 100 + 20 * np.sin(np.linspace(0, 40 * np.pi, 2000)) + rng.normal(0, 2, 2000)
    base_load = np.clip(base_load, 1.0, None)
    scaled_load = base_load * 5.0

    norm_base = compute_load_norm(base_load, window=240, quantile=0.95)
    norm_scaled = compute_load_norm(scaled_load, window=240, quantile=0.95)

    valid = slice(240, 2000)
    np.testing.assert_allclose(norm_base[valid], norm_scaled[valid], rtol=1e-6)


def test_compute_temp_anomaly_centers_each_day_of_year():
    day_of_year = np.tile(np.arange(1, 4), 100)
    rng = np.random.default_rng(0)
    baseline_by_doy = {1: 10.0, 2: 15.0, 3: 20.0}
    temp = np.array([baseline_by_doy[d] for d in day_of_year]) + rng.normal(0, 0.01, len(day_of_year))

    anomaly = compute_temp_anomaly(temp, day_of_year)

    df = pd.DataFrame({"doy": day_of_year, "anomaly": anomaly})
    per_doy_mean = df.groupby("doy")["anomaly"].mean()
    np.testing.assert_allclose(per_doy_mean.to_numpy(), [0.0, 0.0, 0.0], atol=1e-2)
