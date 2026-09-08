"""Unit tests for the China deployment layer (cn_deploy.py).

Fast tests only — no model training, no dataset loading.  The calendar
fact tables are cross-checked against the official 2026 State-Council
notice (国办发明电〔2025〕7号) and Taiyuan's heating regulation.
"""

import numpy as np
import pandas as pd
import pytest

from transcif.data.cn_deploy import (
    CN_HOLIDAYS, HEATING_WINDOWS, cn_calendar_features,
    heating_season_mask, monthly_cif_from_gen, import_ef_from_senders,
    cn_monthly_table, proxy_anchor_trust, blend_monthly_table,
    build_cn_windows,
)
from transcif.data.fuel import FUEL_INDEX, CANONICAL_FUELS


def test_cn_holidays_2026_official():
    """2026 facts vs 国办发明电〔2025〕7号: CNY 2/15-2/23, workdays 2/14+2/28."""
    h26 = CN_HOLIDAYS[2026]
    assert (2, 15) in h26["holiday"] and (2, 23) in h26["holiday"]
    assert len([d for d in h26["holiday"] if d[0] == 2]) == 9
    assert (2, 14) in h26["workday"] and (2, 28) in h26["workday"]
    assert len([d for d in h26["holiday"] if d[0] == 10]) == 7


def test_cn_calendar_remap_and_flags():
    """Holiday weekdays carry Sunday dow channels; 调休 weekends Monday."""
    # 2026-02-16 (Mon, CNY) -> weekend channels; 2026-02-14 (Sat, 调休上班)
    # -> Monday channels; 2026-02-18 (Thu, CNY holiday, weekday) -> weekend.
    hours = pd.date_range("2026-02-13 00:00", periods=24 * 8, freq="h",
                          tz="UTC")
    cal, flags = cn_calendar_features(hours, tz_offset=8.0)
    local = hours + pd.Timedelta(hours=8)
    for i, ts in enumerate(local):
        md = (ts.month, ts.day)
        hol = md in CN_HOLIDAYS[2026]["holiday"]
        spec = md in CN_HOLIDAYS[2026]["workday"]
        assert flags[i, 0] == float(hol), f"flag mismatch {ts}"
        assert flags[i, 1] == float(spec), f"spec flag mismatch {ts}"
        if hol and ts.dayofweek < 5:
            np.testing.assert_allclose(cal[i, 2:4],
                                       [np.sin(2*np.pi*6/7), np.cos(2*np.pi*6/7)],
                                       atol=1e-6)
        if spec and ts.dayofweek >= 5:
            np.testing.assert_allclose(cal[i, 2:4], [0.0, 1.0], atol=1e-6)


def test_calendar_plain_days_unchanged():
    """Non-holiday weekdays keep their own day-of-week channels."""
    # local Monday 2026-07-06 00:00 .. 23:00 (= UTC 07-05 16:00 .. 07-06 15:00)
    hours = pd.date_range("2026-07-05 16:00", periods=24, freq="h", tz="UTC")
    cal, flags = cn_calendar_features(hours, tz_offset=8.0)
    assert not flags.any()
    np.testing.assert_allclose(cal[:, 2], 0.0, atol=1e-7)  # Monday sin(dow)
    np.testing.assert_allclose(cal[:, 3], 1.0, atol=1e-7)  # Monday cos(dow)


def test_heating_mask_taiyuan():
    """Taiyuan window 11/1 - 3/31 (verified 2026-09), wraps the year end."""
    hours2 = pd.date_range("2026-10-30 00:00", periods=24 * 6, freq="h", tz="UTC")
    m2 = heating_season_mask(hours2, tz_offset=8.0, province="shanxi")
    # local: 10-30 08:00 .. 11-05 07:00 -> True from local 11-01 00:00 = idx 40
    assert not m2[:40].any() and m2[40:].all()
    hours3 = pd.date_range("2026-03-30 00:00", periods=24 * 2, freq="h", tz="UTC")
    m3 = heating_season_mask(hours3, tz_offset=8.0, province="shanxi")
    # local: 03-30 08:00 .. 04-01 07:00 -> True until local 04-01 00:00 (idx 40)
    assert m3[:40].all() and not m3[40:].any()
    hours4 = pd.date_range("2026-03-31 16:00", periods=24, freq="h", tz="UTC")
    m4 = heating_season_mask(hours4, tz_offset=8.0, province="shanxi")
    # local: 04-01 00:00 .. 04-01 23:00 -> outside the heating window
    assert not m4.any()


def test_heating_mask_southern_province():
    hours = pd.date_range("2026-01-10", periods=24, freq="h", tz="UTC")
    assert not heating_season_mask(hours, 8.0, province=None).any()


def test_monthly_cif_from_gen_hydro_seasonality():
    """Sichuan-type sender: wet-season CIF << dry-season CIF."""
    sc = {
        "hydro": [3.5, 3.4, 3.8, 5.0, 6.5, 7.5, 8.0, 7.8, 6.8, 5.5, 4.2, 3.6],
        "coal":  [2.5, 2.4, 2.2, 1.5, 1.0, 0.8, 0.7, 0.7, 0.9, 1.4, 2.0, 2.4],
    }
    cif = monthly_cif_from_gen(sc)
    assert cif[6] < 0.6 * cif[0], f"wet {cif[6]:.0f} should be << dry {cif[0]:.0f}"


def test_import_ef_flow_weighted():
    """Flow-weighted average between senders; seasonal when flows vary."""
    a = np.full(12, 400.0)
    b = np.full(12, 100.0)
    ef = import_ef_from_senders({"A": a, "B": b}, {"A": 0.5, "B": 0.5})
    np.testing.assert_allclose(ef, 250.0, atol=1e-6)
    flows = {  # winter leans hydro-sender B, summer coal-sender A
        "A": np.array([0.2] * 6 + [0.8] * 6),
        "B": np.array([0.8] * 6 + [0.2] * 6),
    }
    ef2 = import_ef_from_senders({"A": a, "B": b}, flows)
    assert ef2[0] < 200 < ef2[6], "seasonal flows must move the imports EF"


def test_import_ef_empty():
    assert import_ef_from_senders({}, {}) == pytest.approx(np.full(12, 250.0))


def test_cn_monthly_table_layout():
    """(12,16) table, energy-weighted annual shares, CN renewable def."""
    gen = {
        "coal":  [10.0] * 12,
        "gas":   [2.0] * 12,
        "hydro": [2.0] * 12,
        "solar": [1.0] * 12,
    }
    out = cn_monthly_table(gen, lat=35.0)
    assert out["table"].shape == (12, 16)
    assert out["fd_config"].shape == (16,)
    assert out["ef_vec"].shape == (len(CANONICAL_FUELS),)
    # energy-weighted renewable share: (2+1)/15
    assert out["mean_rs"] == pytest.approx(3.0 / 15.0, abs=1e-6)
    # monthly rows carry the monthly renewable share
    np.testing.assert_allclose(out["table"][:, 0], 3.0 / 15.0, atol=1e-5)
    # annual non-ren share (coal+gas)/15; ef_nr = (10*950 + 2*400)/12 = 858.3
    expected_efnr = (10 * 950 + 2 * 400) / 12
    assert out["ef_nr"] == pytest.approx(expected_efnr, rel=1e-3)
    # imports EF remains the 250 fallback when no senders given
    assert out["imports_ef_monthly"] is None
    assert out["ef_vec"][FUEL_INDEX["imports"]] == pytest.approx(250.0)


def test_cn_monthly_table_imports_path():
    """Imports share enters the axis; sender CIFs weight the imports EF."""
    gen = {
        "coal": [6.0] * 12, "gas": [3.0] * 12, "solar": [1.0] * 12,
    }
    sc_cif = np.linspace(500, 100, 12)  # hydro sender: dry -> wet
    out = cn_monthly_table(
        gen, imports_gwh=np.full(12, 4.0),
        sender_cif_monthly={"SC": sc_cif},
        flow_shares={"SC": 1.0}, lat=31.0)
    assert out["imports_ef_monthly"] is not None
    assert out["imports_ef_monthly"][0] > out["imports_ef_monthly"][6]
    assert out["ef_vec"][FUEL_INDEX["imports"]] == \
        pytest.approx(float(sc_cif.mean()), rel=1e-6)
    # annual imports share 4/14
    assert out["fd_config"][2 + FUEL_INDEX["imports"]] == \
        pytest.approx(4 / 14, abs=1e-4)


def test_proxy_anchor_trust_gates():
    """No proxy -> 0.0 (annual fallback); good proxy -> high trust; a
    table whose implied level is biased vs an annual anchor is punished."""
    gen = {"coal": [10.0] * 12, "gas": [2.0] * 12}
    out = cn_monthly_table(gen, lat=35.0)
    assert proxy_anchor_trust(out["table"], thermal_proxy=None) == 0.0

    # proxy tracking the implied level: use a seasonal gen table so the
    # implied series has variance, then feed its own implied series back
    gen_seas = {"coal": [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 7.0, 7.5, 8.5,
                         9.5, 11.0, 12.5],
                "gas": [2.0] * 12,
                "solar": [1.0, 1.2, 1.6, 2.0, 2.3, 2.5, 2.4, 2.1, 1.8,
                          1.4, 1.1, 0.9]}
    out_seas = cn_monthly_table(gen_seas, lat=35.0)
    implied_seas = (1 - out_seas["table"][:, 0].astype(float)) \
        * out_seas["table"][:, 1].astype(float) * 1000
    assert implied_seas.std() > 1.0, "table must carry seasonal variance"
    assert proxy_anchor_trust(out_seas["table"], thermal_proxy=implied_seas) > 0.9

    # biased implied level vs the annual anchor -> penalised
    table_biased = out_seas["table"].copy()
    table_biased[:, 1] += 0.1  # +100 gCO2/kWh level bias
    t_biased = proxy_anchor_trust(
        table_biased, thermal_proxy=implied_seas,
        proxy_cif_annual=implied_seas.mean())
    assert t_biased < 0.3


def test_blend_monthly_table():
    gen = {"coal": [10.0] * 12, "gas": [2.0] * 12}
    out = cn_monthly_table(gen, lat=35.0)
    blended = blend_monthly_table(out["table"], out["fd_config"], 0.5)
    np.testing.assert_allclose(
        blended, 0.5 * out["table"] + 0.5 * out["fd_config"][None, :],
        atol=1e-6)
    np.testing.assert_allclose(
        blend_monthly_table(out["table"], out["fd_config"], 0.0),
        np.tile(out["fd_config"], (12, 1)), atol=1e-6)


def test_build_cn_windows_contract():
    """18-channel fut_exog, 10-channel weather, config row with lag."""
    gen = {"coal": [10.0] * 12, "gas": [2.0] * 12}
    out = cn_monthly_table(gen, lat=38.0)
    hours = pd.date_range("2023-07-10", periods=336 + 24, freq="h", tz="UTC")
    raw_wx = np.tile(np.array([[25.0, 500.0, 6.0]], np.float32),
                     (360, 1))
    w, flags, heating = build_cn_windows(
        hours, raw_wx, lat=38.0, lon=115.0, table=out["table"],
        province="shanxi")
    assert w["fut_exog"].shape == (1, 24, 18)
    assert w["fut_weather"].shape == (1, 24, 10)
    assert w["x_weather"].shape == (1, 336, 10)
    assert "config" in w and w["config"].shape == (1, 16)
    # July window, publication-lag 2 (at day-ahead D-1 = June mid-month the
    # June row is not yet published) -> May row (table[4])
    np.testing.assert_allclose(w["config"][0], out["table"][4], atol=1e-6)
    assert flags.shape == (24, 2) and heating.shape == (24,)
    # all-zero past history (cold mode) is the I_cfg contract
    assert not w["x_rs"].any() and not w["x_fuel"].any()


def test_build_cn_windows_holiday_advisory():
    """A CNY window flags holiday hours and keeps advisory honesty."""
    gen = {"coal": [10.0] * 12, "gas": [2.0] * 12}
    out = cn_monthly_table(gen, lat=31.0)
    # horizon = local 2026-02-15 08:00 .. 02-16 07:00 (all inside CNY)
    # -> UTC start = 02-14 16:00 - 336h
    start_utc = pd.Timestamp("2026-02-15 08:00", tz="UTC") - pd.Timedelta(hours=336)
    hours = pd.date_range(start_utc, periods=336 + 24, freq="h")
    raw_wx = np.tile(np.array([[5.0, 200.0, 4.0]], np.float32), (360, 1))
    w, flags, heating = build_cn_windows(
        hours, raw_wx, lat=31.0, lon=121.0, table=out["table"],
        province=None)
    assert flags[:, 0].sum() == 24  # all 24 horizon hours on CNY days


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
