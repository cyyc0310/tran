import numpy as np
import pandas as pd
import pytest

from transcif.research.coarse_forecast_v7 import COLUMNS, released_forecasts


def history():
    starts = pd.date_range("2022-01-01", periods=48, freq="MS", tz="UTC")
    x = pd.DataFrame({"region": "R", "period_start": starts,
        "period_end": starts + pd.offsets.MonthBegin(1)})
    x["available_at"] = x.period_end + pd.Timedelta(days=30)
    for i, col in enumerate(COLUMNS):
        x[col] = (.6 + .2*np.sin(np.arange(48)/6)) if i == 0 else 0.
    x[COLUMNS[1]] = 1 - x[COLUMNS[0]]
    return x


def test_unreleased_values_cannot_change_forecasts_or_expert_weights():
    x = history(); origins = pd.date_range("2025-01-01", periods=15, tz="UTC")
    a = released_forecasts(x, "R", origins, 90)
    mask = x.available_at + pd.Timedelta(days=90) > origins.max()
    x.loc[mask, COLUMNS] = np.eye(8)[-1]
    b = released_forecasts(x, "R", origins, 90)
    np.testing.assert_array_equal(a["experts"], b["experts"])
    np.testing.assert_array_equal(a["features"], b["features"])


def test_positive_delay_changes_information_set_and_simplex_is_preserved():
    x = history(); origin = [pd.Timestamp("2025-03-15", tz="UTC")]
    a = released_forecasts(x, "R", origin, 0); b = released_forecasts(x, "R", origin, 90)
    assert not np.array_equal(a["experts"], b["experts"])
    assert np.all(a["experts"] >= 0)
    np.testing.assert_allclose(a["experts"].sum(-1), 1)
    for w in a["weights"].values(): np.testing.assert_allclose(w.sum(-1), 1)


def test_delayed_hindcasts_do_not_reuse_final_released_prefix():
    x = history(); origin = [pd.Timestamp("2025-04-15", tz="UTC")]
    a = released_forecasts(x, "R", origin)
    # A month that is available now was not known at its own mid-month origin.
    x.loc[x.period_start.eq(pd.Timestamp("2025-02-01", tz="UTC")), COLUMNS] = np.eye(8)[-1]
    b = released_forecasts(x, "R", origin)
    assert not np.array_equal(a["losses"], b["losses"])
    assert np.isfinite(b["features"]).all()


def test_invalid_input_rejected_and_cold_start_is_explicit():
    x = history()
    with pytest.raises(ValueError): released_forecasts(x, "R", ["2025-01-01"], -1)
    out = released_forecasts(x, "missing", ["2025-01-01"])
    np.testing.assert_allclose(out["experts"], 1/8)
    assert out["features"][0, -6] == 0  # no historical pseudo-forecast support
