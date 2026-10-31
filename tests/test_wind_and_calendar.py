"""Wind-power and civil-calendar feature contracts."""

import numpy as np
import pandas as pd
import pytest

from transcif.physics.wind import wind_capacity_factor
from transcif.data.calendar import calendar_features


class TestWindCurve:
    def test_cut_in_and_rated(self):
        v = np.array([0.0, 3.0, 7.5, 12.0, 18.0, 25.0, 26.0])
        cf = wind_capacity_factor(v)
        assert cf[0] == 0.0
        assert cf[1] == pytest.approx(0.0)
        assert 0.0 < cf[2] < 1.0
        assert cf[3] == 1.0
        assert cf[4] == 1.0
        assert cf[6] == 0.0  # above cut-out

    def test_monotone_ramp(self):
        v = np.linspace(3.0, 12.0, 20)
        cf = wind_capacity_factor(v)
        assert (np.diff(cf) > 0).all()


class TestCalendar:
    def test_shape_and_range(self):
        hours = pd.date_range("2023-01-01", periods=72, freq="h", tz="UTC")
        c = calendar_features(hours)
        assert c.shape == (72, 6)
        assert (np.abs(c) <= 1.0 + 1e-6).all()

    def test_tz_shift_rotates_hour_phase(self):
        # Offset -8 means local = UTC-8: 00:00 UTC is 16:00 local, so the
        # hour phase at index 0 with offset -8 equals index 16 with offset 0.
        base = pd.date_range("2023-05-01", periods=24, freq="h", tz="UTC")
        c0 = calendar_features(base)
        c8 = calendar_features(base, timezone="Etc/GMT+8")
        np.testing.assert_allclose(c8[0, :2], c0[16, :2], atol=1e-6)

    def test_hour_channel_daily_periodicity(self):
        # Only the hour-of-day channels repeat every 24 h (day-of-week /
        # day-of-year channels drift by construction).
        hours = pd.date_range("2023-05-01", periods=48, freq="h", tz="UTC")
        c = calendar_features(hours)
        np.testing.assert_allclose(c[0, :2], c[24, :2], atol=1e-6)
