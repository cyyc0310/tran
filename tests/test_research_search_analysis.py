import numpy as np
import pandas as pd
import pytest

from transcif.research.search_analysis import paired_month_interval, input_regimes


def test_synchronized_month_estimate_keeps_equal_regions_and_day_counts():
    rows = []
    for region, dates, difference in (("A", ["2025-01-01", "2025-01-02", "2025-02-01"], [1, 1, 9]),
                                      ("B", ["2025-01-01", "2025-02-01"], [3, 3])):
        for date, value in zip(dates, difference):
            rows.extend([dict(region=region, origin=date, method="latest", crps=10),
                         dict(region=region, origin=date, method="candidate", crps=10+value)])
    score = paired_month_interval(pd.DataFrame(rows), "candidate", repeats=2000)
    assert score["difference"] == pytest.approx(((1+1+9)/3 + 3)/2)
    assert score["regions"] == 2 and score["region_days"] == 5


def test_pairing_rejects_unmatched_scoring_populations():
    frame = pd.DataFrame([dict(region="A", origin="2025-01-01", method="latest", crps=1),
                          dict(region="A", origin="2025-01-02", method="candidate", crps=2)])
    with pytest.raises(ValueError, match="identical"):
        paired_month_interval(frame, "candidate")


def test_input_regime_is_release_prefix_and_uses_old_same_month_only():
    history = pd.DataFrame([dict(region="A", period_start="2024-02-01", period_end="2024-03-01",
                                 available_at="2024-03-31", fuelshare_coal=.2),
                            dict(region="A", period_start="2025-01-01", period_end="2025-02-01",
                                 available_at="2025-03-03", fuelshare_coal=.9)])
    origins = pd.DatetimeIndex(["2025-02-01T00:00Z"])
    first = input_regimes(history, origins, "A", 0)
    assert first.age_regime.iloc[0] == ">240d"
    assert first.coal_regime.iloc[0] == "10-40% coal"
    assert first.seasonal_history.iloc[0] == "same-month history"
    history.loc[1, "fuelshare_coal"] = np.nan
    pd.testing.assert_frame_equal(first, input_regimes(history, origins, "A", 0))
