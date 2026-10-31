import json

import numpy as np
import pandas as pd
import pytest

from scripts.coarse_dynamics_v9 import METHODS, _one_origin, _simplex, _simplex_rows, forecast_monthly
from transcif.research.coarse_forecast_v7 import COLUMNS, released_forecasts


def monthly(months=48):
    starts = pd.date_range("2021-01-01", periods=months, freq="MS", tz="UTC")
    t = np.arange(months)
    coal = .25 + .06 * np.sin(2 * np.pi * t / 12) + .001 * t
    gas = .55 - .03 * np.sin(2 * np.pi * t / 12) - .0005 * t
    frame = pd.DataFrame({"region": "R", "period_start": starts,
                          "period_end": starts + pd.offsets.MonthBegin(1),
                          "available_at": starts + pd.offsets.MonthBegin(1) + pd.Timedelta(days=25)})
    for col in COLUMNS:
        frame[col] = 0.
    frame[COLUMNS[0]] = coal
    frame[COLUMNS[1]] = gas
    frame[COLUMNS[3]] = 1 - coal - gas
    return frame


def compare(a, b):
    for method in METHODS:
        np.testing.assert_array_equal(a[method]["mix"], b[method]["mix"])
        np.testing.assert_array_equal(a[method]["draws"], b[method]["draws"])
        assert a[method]["audit"] == b[method]["audit"]


def test_methods_are_simplex_joint_and_json_serializable():
    result = forecast_monthly(monthly(), "R", ["2024-03-15", "2024-08-15"], n_samples=16)
    assert tuple(result) == METHODS
    for method in METHODS:
        point, draws = result[method]["mix"], result[method]["draws"]
        assert point.shape == (2, 8) and draws.shape == (2, 16, 8)
        assert np.isfinite(point).all() and np.isfinite(draws).all()
        assert (point >= 0).all() and (draws >= 0).all()
        np.testing.assert_allclose(point.sum(axis=-1), 1)
        np.testing.assert_allclose(draws.sum(axis=-1), 1)
        np.testing.assert_array_equal(point[:, [2, 4, 5, 6, 7]], 0)
        np.testing.assert_array_equal(draws[:, :, [2, 4, 5, 6, 7]], 0)
        json.dumps(result[method]["audit"])


def test_vectorized_joint_projection_matches_single_vector_projection():
    rng = np.random.default_rng(123)
    values = rng.normal(size=(50, 8))
    support = np.array([True, True, False, True, False, False, False, False])
    expected = np.stack([_simplex(row, support) for row in values])
    np.testing.assert_allclose(_simplex_rows(values, support), expected, atol=1e-15)


def test_unreleased_future_values_including_invalid_values_are_invisible():
    x = monthly()
    origin = ["2024-02-15"]
    expected = forecast_monthly(x, "R", origin, n_samples=8)
    mask = x.available_at > pd.Timestamp(origin[0], tz="UTC")
    changed = x.copy()
    changed.loc[mask, COLUMNS] = -999.
    compare(expected, forecast_monthly(changed, "R", origin, n_samples=8))


def test_delay_and_period_end_both_gate_a_month():
    x = monthly()
    origin = ["2024-02-15"]
    early = forecast_monthly(x, "R", origin, n_samples=8)
    delayed = forecast_monthly(x, "R", origin, extra_delay=90, n_samples=8)
    assert delayed["latest"]["audit"]["released_months"][0] < early["latest"]["audit"]["released_months"][0]
    assert not np.array_equal(delayed["latest"]["mix"], early["latest"]["mix"])
    # Even a manifest with a prematurely stamped availability is not legal
    # until the month it describes has ended.
    changed = x.copy()
    current = changed.period_start.eq(pd.Timestamp("2024-02-01", tz="UTC"))
    changed.loc[current, "available_at"] = pd.Timestamp("2024-02-02", tz="UTC")
    changed.loc[current, COLUMNS] = np.eye(8)[-1]
    compare(early, forecast_monthly(changed, "R", origin, n_samples=8))


def test_pseudo_forecast_cannot_use_its_own_label():
    x = monthly()
    origin = pd.Timestamp("2024-09-15", tz="UTC")
    part = x.loc[x.available_at.le(origin) & x.period_end.le(origin)].copy()
    part["delayed_at"] = part.available_at
    a = _one_origin(part, "R", origin, origin.day, 0, {})
    changed = part.copy()
    label = changed.period_start.eq(pd.Timestamp("2024-07-01", tz="UTC"))
    changed.loc[label, COLUMNS] = np.eye(8)[-1]
    b = _one_origin(changed, "R", origin, origin.day, 0, {})
    row_a = next(r for r in a.records if r["period_end"] == pd.Timestamp("2024-08-01", tz="UTC"))
    row_b = next(r for r in b.records if r["period_end"] == pd.Timestamp("2024-08-01", tz="UTC"))
    for method in METHODS:
        np.testing.assert_array_equal(row_a[method], row_b[method])
    assert not np.array_equal(row_a["truth"], row_b["truth"])
    # Once July is released, its changed outcome may legally update the bridge.
    assert not np.array_equal(a.points["release_bridge"], b.points["release_bridge"])


def test_v7_mixture_is_exact_temperature_point_mean():
    x = monthly()
    origin = ["2024-04-15"]
    a = forecast_monthly(x, "R", origin, n_samples=4)["v7_mixture"]["mix"]
    v7 = released_forecasts(x, "R", origin)
    expected = np.einsum("ne,nef->nf", v7["weights"]["0.03"], v7["experts"])
    np.testing.assert_allclose(a, expected, atol=1e-15)


def test_missing_months_and_batch_time_units_keep_predictions_stable():
    x = monthly().drop(index=[4, 5, 18, 31]).reset_index(drop=True)
    dates = pd.to_datetime(["2024-07-15", "2024-09-15"], utc=True)
    batch = forecast_monthly(x, "R", dates, n_samples=8)
    swapped = forecast_monthly(x, "R", dates[::-1], n_samples=8)
    scalar = forecast_monthly(x, "R", pd.DatetimeIndex([dates[0].as_unit("s")]), n_samples=8)
    for method in METHODS:
        np.testing.assert_array_equal(batch[method]["mix"][0], scalar[method]["mix"][0])
        np.testing.assert_array_equal(batch[method]["draws"][0], scalar[method]["draws"][0])
        np.testing.assert_array_equal(batch[method]["mix"], swapped[method]["mix"][::-1])
        np.testing.assert_array_equal(batch[method]["draws"], swapped[method]["draws"][::-1])


def test_cross_year_same_prefix_batch_matches_scalar_calls():
    # Both origins have the same final released month and month/day. Trend and
    # horizon must still advance by twelve months in the second prediction.
    x = monthly(48)
    dates = ["2025-03-15", "2026-03-15"]
    batch = forecast_monthly(x, "R", dates, n_samples=8)
    for i, date in enumerate(dates):
        single = forecast_monthly(x, "R", [date], n_samples=8)
        for method in METHODS:
            np.testing.assert_array_equal(batch[method]["mix"][i], single[method]["mix"][0])
            np.testing.assert_array_equal(batch[method]["draws"][i], single[method]["draws"][0])
            assert batch[method]["audit"]["horizon_months"][i] == single[method]["audit"]["horizon_months"][0]
    assert not np.array_equal(batch["harmonic_ridge"]["mix"][0], batch["harmonic_ridge"]["mix"][1])
    assert batch["latest"]["audit"]["horizon_months"] == [3, 15]


@pytest.mark.parametrize("field,value", [
    ("period_start", "2024-01-02"),
    ("period_end", "2024-03-01"),
    ("period_end", "2024-01-01"),
    ("available_at", pd.NaT),
])
def test_invalid_monthly_interval_or_release_timestamp_rejected(field, value):
    x = monthly()
    x.loc[36, field] = value
    with pytest.raises(ValueError):
        forecast_monthly(x, "R", ["2024-07-15"], n_samples=2)


def test_cold_start_and_bad_arguments():
    result = forecast_monthly(monthly(), "unknown", ["2024-01-15"], n_samples=3)
    for method in METHODS:
        np.testing.assert_allclose(result[method]["mix"], 1 / 8)
        np.testing.assert_allclose(result[method]["draws"], 1 / 8)
        assert result[method]["audit"]["fallback"] == ["cold_start"]
    with pytest.raises(ValueError):
        forecast_monthly(monthly(), "R", ["2024-01-15"], extra_delay=-1)
    with pytest.raises(ValueError):
        forecast_monthly(monthly(), "R", ["2024-01-15"], n_samples=0)
