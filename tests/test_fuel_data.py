"""Tests for the fuel-decomposed data layer (TransCIF-FD Phase FD-0).

Integration tests against the real ``data_2023`` dataset (skipped when the
data directory is absent) plus pure-unit tests of the EF rescaling logic.
"""

import statistics

import numpy as np
import pandas as pd
import pytest

from transcif.config.region_meta import REGION_META, get_region_meta
from transcif.data.fuel import (
    CANONICAL_FUELS, FUEL_INDEX, THERMAL_FUELS, canonical_fuel_efs,
    load_fuel_shares, attach_fuel_and_exog, region_fuel_efs, fuel_cif,
    build_fd_windows, jurisdiction_renewable_fuels,
)
from transcif.data.loaders import all_region_configs, load_region_data

HAS_DATA = None


def _has_data():
    global HAS_DATA
    if HAS_DATA is None:
        from transcif.config import DATA_DIR
        HAS_DATA = (DATA_DIR / "US_CISO_2023_hourly.csv").exists()
    return HAS_DATA


class TestRegionMeta:
    def test_all_29_regions_have_metadata(self):
        if not _has_data():
            pytest.skip("data_2023 not present")
        regions = all_region_configs()
        missing = [r for r in regions if r not in REGION_META]
        assert not missing, f"regions missing from REGION_META: {missing}"

    def test_lat_lon_ranges(self):
        for name, (lat, lon, tz) in REGION_META.items():
            assert -60 < lat < 60, name
            assert -180 <= lon <= 180, name
            assert -12 <= tz <= 14, name

    def test_unknown_region_falls_back(self):
        lat, lon, tz = get_region_meta("CN_Beijing")
        assert lat == 45.0 and tz == 0.0


class TestFuelSeries:
    @pytest.mark.skipif(not _has_data(), reason="data_2023 not present")
    def test_us_region_loads(self):
        cfgs = all_region_configs()
        hours, shares = load_fuel_shares("US_CISO", cfgs)
        assert hours is not None
        assert shares.shape[1] == len(CANONICAL_FUELS)
        row_sums = shares.sum(axis=1)
        assert 0.9 < row_sums.mean() <= 1.05

    @pytest.mark.skipif(not _has_data(), reason="data_2023 not present")
    def test_uk_region_loads(self):
        cfgs = all_region_configs()
        hours, shares = load_fuel_shares("UK_13_London", cfgs)
        assert hours is not None
        # UK perc columns sum to ~100 -> shares sum to ~1
        assert 0.95 < shares.sum(axis=1).mean() <= 1.05

    @pytest.mark.skipif(not _has_data(), reason="data_2023 not present")
    def test_au_region_has_no_fuel(self):
        cfgs = all_region_configs()
        hours, shares = load_fuel_shares("QLD1", cfgs)
        assert hours is None and shares is None

    @pytest.mark.skipif(not _has_data(), reason="data_2023 not present")
    def test_solar_share_peaks_midday_utc_offset(self):
        cfgs = all_region_configs()
        hours, shares = load_fuel_shares("US_CISO", cfgs)
        solar = shares[:, FUEL_INDEX["solar"]]
        prof = pd.Series(solar, index=hours).groupby(hours.hour).mean()
        assert 18 <= int(prof.idxmax()) <= 22  # CISO noon PST == 20 UTC


class TestAttachAndWindows:
    @pytest.mark.skipif(not _has_data(), reason="data_2023 not present")
    def test_attach_us(self):
        cfgs = all_region_configs()
        data = load_region_data("US_CISO", cfgs)
        attach_fuel_and_exog(data, "US_CISO", cfgs)
        T = len(data["rs"])
        assert data["has_fuel"] is True
        assert data["fuel_shares"].shape == (T, len(CANONICAL_FUELS))
        assert data["ef_vec"].shape == (len(CANONICAL_FUELS),)
        for key in ("weather", "astro", "calendar", "wind_cf", "clearsky_index"):
            assert data["exog"][key].shape[0] == T

    @pytest.mark.skipif(not _has_data(), reason="data_2023 not present")
    def test_attach_au_zeros(self):
        cfgs = all_region_configs()
        data = load_region_data("QLD1", cfgs)
        attach_fuel_and_exog(data, "QLD1", cfgs)
        assert data["has_fuel"] is False
        assert data["fuel_shares"].shape == (len(data["rs"]), len(CANONICAL_FUELS))
        assert (data["fuel_shares"] == 0).all()
        # AU thermal EFs collapse to ef_nr -> coal/gas split has no CIF effect
        thermal = data["ef_vec"][[FUEL_INDEX[f] for f in THERMAL_FUELS]]
        assert np.allclose(thermal, data["ef_nr"])

    @pytest.mark.skipif(not _has_data(), reason="data_2023 not present")
    def test_ef_vec_matches_region_efnr(self):
        cfgs = all_region_configs()
        for name in ("US_CISO", "UK_13_London"):
            data = load_region_data(name, cfgs)
            attach_fuel_and_exog(data, name, cfgs)
            annual = data["fuel_shares"][: int(len(data["fuel_shares"]) * 0.8)].mean(axis=0)
            juris = "us" if name.startswith("US_") else "uk"
            renewable = jurisdiction_renewable_fuels()[juris]
            nonren = [FUEL_INDEX[f] for f in CANONICAL_FUELS if f not in renewable]
            implied = float(np.dot(data["ef_vec"], annual)) / annual.sum()
            implied_nonren = float(
                np.dot(data["ef_vec"][nonren], annual[nonren])) / annual[nonren].sum()
            assert implied_nonren == pytest.approx(data["ef_nr"], rel=0.15)

    @pytest.mark.skipif(not _has_data(), reason="data_2023 not present")
    def test_physics_cif_consistent_with_truth(self):
        # Fuel shares + EF vector must reconstruct the reported CIF well.
        # Median across fuel regions is ~9 gCO2/kWh; the South-East England
        # cluster (UK_12/13/14, import-heavy) carries a known ~35-60
        # mismatch because the UK API computes CIF with its own internal
        # methodology rather than the static per-fuel EF convention.
        cfgs = all_region_configs()
        maes = {}
        for name in cfgs:
            if name.startswith(("QLD", "NSW", "VIC", "SA")):
                continue
            data = load_region_data(name, cfgs)
            attach_fuel_and_exog(data, name, cfgs)
            cif_hat = fuel_cif(data["fuel_shares"], data["ef_vec"])
            maes[name] = np.abs(cif_hat - data["cif"]).mean()
        assert statistics.median(maes.values()) < 15.0
        assert maes["US_CISO"] < 15.0
        assert maes["UK_13_London"] < 45.0  # documented API accounting gap

    @pytest.mark.skipif(not _has_data(), reason="data_2023 not present")
    def test_windows_shape_and_alignment(self):
        cfgs = all_region_configs()
        data = load_region_data("US_CISO", cfgs)
        attach_fuel_and_exog(data, "US_CISO", cfgs)
        w = build_fd_windows(data, seq_len=336, horizon=24, stride=24)
        assert w["x_rs"].shape[1] == 336
        assert w["y_cif"].shape[1] == 24
        assert w["x_fuel"].shape == (len(w["x_rs"]), 336, len(CANONICAL_FUELS))
        assert w["fut_exog"].shape == (len(w["x_rs"]), 24, 10)
        # y_fuel at window i must correspond to hours origin+i.. (calendar
        # consistency: future astro daytime iff y_fuel solar can be > 0)
        i = len(w["x_rs"]) // 2
        astro_day = w["fut_exog"][i, :, 0] > 0
        assert astro_day.sum() in range(8, 17)  # plausible day length

    @pytest.mark.skipif(not _has_data(), reason="data_2023 not present")
    def test_max_windows_subsample(self):
        cfgs = all_region_configs()
        data = load_region_data("US_CISO", cfgs)
        attach_fuel_and_exog(data, "US_CISO", cfgs)
        w = build_fd_windows(data, seq_len=336, horizon=24, stride=6,
                             max_windows=50, rng=np.random.default_rng(1))
        assert len(w["x_rs"]) == 50


class TestUnitEFs:
    def test_canonical_efs_values(self):
        efs = canonical_fuel_efs()
        d = dict(zip(CANONICAL_FUELS, efs))
        assert d["coal"] == 980.0
        assert d["solar"] == 0.0

    def test_fuel_cif_vectorised(self):
        efs = np.zeros(len(CANONICAL_FUELS))
        efs[FUEL_INDEX["coal"]] = 980.0
        shares = np.zeros((5, len(CANONICAL_FUELS)))
        shares[:, FUEL_INDEX["coal"]] = 0.5
        np.testing.assert_allclose(fuel_cif(shares, efs), 490.0)

    def test_region_efs_au_collapse(self):
        data = {"ef_nr": 841.59, "fuel_shares": None, "has_fuel": False}
        ef_vec = region_fuel_efs(data, "QLD1")
        assert ef_vec[FUEL_INDEX["gas"]] == 841.59
