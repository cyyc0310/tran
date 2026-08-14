"""Tests for astronomy / calendar physics features (TransCIF-FD Phase FD-0)."""

import numpy as np
import pandas as pd
import pytest

from transcif.physics.astro import (
    sin_solar_elevation, clearsky_ghi, wind_capacity_factor, astro_features,
)
from transcif.data.calendar import calendar_features


class TestSolarElevation:
    def test_solar_noon_peak(self):
        # One full UTC day at longitude 0: elevation should peak near 12:00 UTC.
        hours = pd.date_range("2023-06-21", periods=24, freq="h", tz="UTC")
        s = sin_solar_elevation(hours, lat=51.5, lon=0.0)
        assert 11 <= int(np.argmax(s)) <= 13

    def test_longitude_shifts_solar_noon(self):
        # At lon -120 (PST), solar noon is ~20:00 UTC.
        hours = pd.date_range("2023-06-21", periods=24, freq="h", tz="UTC")
        s = sin_solar_elevation(hours, lat=36.8, lon=-119.5)
        assert 19 <= int(np.argmax(s)) <= 21

    def test_nighttime_nonpositive_summer(self):
        # Summer solstice at London: sun below horizon outside ~04:00-20:00 UTC.
        hours = pd.date_range("2023-06-21", periods=24, freq="h", tz="UTC")
        s = sin_solar_elevation(hours, lat=51.5, lon=0.0)
        assert (s[:4] <= 0).all()
        assert s.max() > 0.85  # max elevation ~62 deg in June at lat 51.5

    def test_winter_lower_than_summer(self):
        # Noon elevation sine in London: December < June.
        winter = pd.DatetimeIndex(["2023-12-21 12:00"], tz="UTC")
        summer = pd.DatetimeIndex(["2023-06-21 12:00"], tz="UTC")
        s_w = sin_solar_elevation(winter, 51.5, 0.0)[0]
        s_s = sin_solar_elevation(summer, 51.5, 0.0)[0]
        assert s_w < s_s

    def test_southern_hemisphere_inverted_season(self):
        # Sydney (lat -33.9): December noon sun higher than June noon.
        dec = pd.DatetimeIndex(["2023-12-21 02:00"], tz="UTC")  # ~noon AEST
        jun = pd.DatetimeIndex(["2023-06-21 02:00"], tz="UTC")
        s_dec = sin_solar_elevation(dec, -33.9, 151.2)[0]
        s_jun = sin_solar_elevation(jun, -33.9, 151.2)[0]
        assert s_dec > s_jun


class TestClearsky:
    def test_zero_at_night(self):
        assert clearsky_ghi(np.array([-0.2, 0.0]))[0] == 0.0

    def test_positive_and_bounded(self):
        s = np.linspace(0.01, 1.0, 50)
        g = clearsky_ghi(s)
        assert (g > 0).all()
        assert g.max() < 1100.0  # Haurwitz peak ~ 1050 W/m^2

    def test_monotone_in_elevation(self):
        s = np.linspace(0.05, 1.0, 30)
        g = clearsky_ghi(s)
        assert (np.diff(g) > 0).all()


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

    def test_astro_features_stack(self):
        hours = pd.date_range("2023-03-01", periods=48, freq="h", tz="UTC")
        a = astro_features(hours, 40.0, -5.0)
        assert a.shape == (48, 2)


class TestCalendar:
    def test_shape_and_range(self):
        hours = pd.date_range("2023-01-01", periods=72, freq="h", tz="UTC")
        c = calendar_features(hours, tz_offset=0.0)
        assert c.shape == (72, 6)
        assert (np.abs(c) <= 1.0 + 1e-6).all()

    def test_tz_shift_rotates_hour_phase(self):
        # Offset -8 means local = UTC-8: 00:00 UTC is 16:00 local, so the
        # hour phase at index 0 with offset -8 equals index 16 with offset 0.
        base = pd.date_range("2023-05-01", periods=24, freq="h", tz="UTC")
        c0 = calendar_features(base, tz_offset=0.0)
        c8 = calendar_features(base, tz_offset=-8.0)
        np.testing.assert_allclose(c8[0, :2], c0[16, :2], atol=1e-6)

    def test_hour_channel_daily_periodicity(self):
        # Only the hour-of-day channels repeat every 24 h (day-of-week /
        # day-of-year channels drift by construction).
        hours = pd.date_range("2023-05-01", periods=48, freq="h", tz="UTC")
        c = calendar_features(hours)
        np.testing.assert_allclose(c[0, :2], c[24, :2], atol=1e-6)
