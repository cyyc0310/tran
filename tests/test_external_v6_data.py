"""V6 QA selection, native provenance, release prefixes, and label boundaries."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transcif.data.eia930 import MODERN_FUEL_COLUMNS
from transcif.research.eia_proxy import latest_observation


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_external_v6.py"
SPEC = importlib.util.spec_from_file_location("prepare_external_v6", SCRIPT)
external = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(external)


@pytest.fixture
def native_inputs(tmp_path, monkeypatch):
    # Exercise an entire history year and a shorter final interval; the fixture
    # changes only the interval size, never the 80% selection threshold.
    monkeypatch.setattr(external, "YEARS", (2025, 2026))
    monkeypatch.setattr(external, "START", pd.Timestamp("2025-01-01T00:00Z"))
    monkeypatch.setattr(external, "STOP", pd.Timestamp("2026-02-01T00:00Z"))
    frames = []
    for year, stop in ((2025, "2026-01-01T00:00Z"), (2026, "2026-02-01T00:00Z")):
        ending = pd.date_range(f"{year}-01-01T01:00Z", stop, freq="h")
        for region in external.CANDIDATES:
            frame = pd.DataFrame({"Balancing Authority": region,
                "UTC Time at End of Hour": ending, "Demand (MW)": 68.,
                "Demand Forecast (MW)": 999., "Net Generation (MW)": 68.,
                "Total Interchange (MW)": 0.})
            for fuel, aliases in MODERN_FUEL_COLUMNS.items():
                frame["Net Generation (MW) from " + aliases[0]] = {
                    "coal": 50., "gas": 20., "battery_storage": -2.}.get(fuel, 0.)
            oil = "Net Generation (MW) from All Petroleum Products"
            if year == 2025:
                frame[oil] = np.nan
            else:
                frame[oil] = 1.
                frame["Net Generation (MW)"] = 69.
            if region == "PACW" and year == 2026:
                frame.loc[:8 * 24 - 1, "Net Generation (MW) from Natural Gas"] = np.nan
            frames.append(frame)
    source = tmp_path / "native.csv"
    pd.concat(frames, ignore_index=True).to_csv(source, index=False)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Frozen external V6 data protocol.\n")
    return source, protocol


def test_fixed_first_six_quality_selection_and_immutable_evidence(tmp_path, native_inputs):
    source, protocol = native_inputs
    assert external.CANDIDATES == ("AZPS", "DUK", "LGEE", "PACW", "PACE", "PSCO", "SRP", "TVA")
    assert external.REGIONS == ("AZPS", "DUK", "LGEE", "PACE", "PSCO", "SRP")
    assert not set(external.CANDIDATES) & set(external.PRIOR_REGIONS)
    out = tmp_path / "out"
    report = external.prepare_external([source], out, protocol)
    selection = report["candidate_selection"]
    assert selection["selected_regions"] == list(external.REGIONS)
    assert selection["qualifying_regions"][-1] == "TVA"
    assert not selection["candidates"]["PACW"]["qualifies"]
    assert report["final_external_year"] == 2026
    assert report["confirmation_stop_exclusive"] == external.STOP.isoformat()
    monthly = pd.read_csv(out / "monthly.csv")
    assert len(monthly) == len(external.REGIONS) * 13
    old = monthly[(monthly.region == "AZPS") & monthly.period_start.str.startswith("2025-01")].iloc[0]
    assert old.fuelshare_coal == pytest.approx(5 / 7)
    assert old.total_mwh == 744 * 70
    assert pd.Timestamp(old.available_at) == pd.Timestamp("2025-03-03T00:00Z")
    for region in external.REGIONS:
        hourly = pd.read_csv(out / f"{region}_hourly.csv")
        times = pd.to_datetime(hourly.time, utc=True)
        assert hourly.valid.all()
        assert times.min() == external.START
        assert times.max() == external.STOP - pd.Timedelta(hours=1)
        assert "native_demand_forecast_mw" not in hourly
        assert not any("cif" in column.lower() for column in hourly)
        assert report["regions"][region]["2026"]["valid_full_utc_days"] == 31
    assert report["protocol_sha256_before"] == report["protocol_sha256_after"] == external.sha(protocol)
    assert report["native_sources"][0]["sha256"] == external.sha(source)
    assert all(external.sha(out / name) == value for name, value in report["derived_hashes"].items())
    assert not report["cif_scores_inspected"]
    assert not report["cif_summaries_computed"]
    assert not report["target_hourly_used_for_fit_or_calibration"]
    assert json.loads((out / "audit.json").read_text())["environment"]["pandas"]
    with pytest.raises(FileExistsError, match="fresh output"):
        external.prepare_external([source], out, protocol)


def test_confirmation_month_cannot_be_used_before_release(tmp_path, native_inputs):
    source, protocol = native_inputs
    out = tmp_path / "out"
    external.prepare_external([source], out, protocol)
    monthly = pd.read_csv(out / "monthly.csv")
    at_end = latest_observation(monthly, "AZPS", pd.Timestamp("2026-01-31T00:00Z"))
    assert pd.Timestamp(at_end.period_start) == pd.Timestamp("2025-12-01T00:00Z")
    later = latest_observation(monthly, "AZPS", pd.Timestamp("2026-03-03T00:00Z"))
    assert pd.Timestamp(later.period_start) == pd.Timestamp("2026-01-01T00:00Z")


def test_history_inherits_monthly_gate_instead_of_annual_label_gate(tmp_path, native_inputs):
    source, protocol = native_inputs
    native = pd.read_csv(source)
    ending = pd.to_datetime(native["UTC Time at End of Hour"], utc=True)
    selected = (native["Balancing Authority"].eq("AZPS")
                & ending.ge("2025-01-01T01:00Z") & ending.lt("2025-05-01T01:00Z"))
    native.loc[selected, "Net Generation (MW) from Natural Gas"] = np.nan
    native.to_csv(source, index=False)
    out = tmp_path / "out"
    report = external.prepare_external([source], out, protocol)
    history = report["regions"]["AZPS"]["2025"]
    assert history["coverage"] < .8
    assert history["eligible_months"] == 8
    assert report["candidate_selection"]["candidates"]["AZPS"]["qualifies"]
    hourly = pd.read_csv(out / "AZPS_hourly.csv")
    assert hourly.loc[~hourly.valid, "generation_gas"].isna().all()


def test_changed_quality_does_not_silently_replace_frozen_region(tmp_path, native_inputs):
    source, protocol = native_inputs
    native = pd.read_csv(source)
    ending = pd.to_datetime(native["UTC Time at End of Hour"], utc=True)
    native.loc[native["Balancing Authority"].eq("AZPS") & ending.gt("2026-01-01T00:00Z"),
               "Net Generation (MW) from Natural Gas"] = np.nan
    native.to_csv(source, index=False)
    with pytest.raises(ValueError, match="first six quality-eligible"):
        external.prepare_external([source], tmp_path / "out", protocol)
    assert not (tmp_path / "out").exists()


def test_duplicate_native_hours_are_rejected_before_output(tmp_path, native_inputs):
    source, protocol = native_inputs
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="Duplicate native"):
        external.prepare_external([source, duplicate], tmp_path / "out", protocol)
    assert not (tmp_path / "out").exists()


def test_protocol_change_during_preparation_is_rejected(tmp_path, native_inputs, monkeypatch):
    source, protocol = native_inputs
    original = external._monthly_at_release

    def changed_protocol(*args, **kwargs):
        protocol.write_text("Changed after preparation started.\n")
        return original(*args, **kwargs)

    monkeypatch.setattr(external, "_monthly_at_release", changed_protocol)
    with pytest.raises(RuntimeError, match="Frozen protocol changed"):
        external.prepare_external([source], tmp_path / "out", protocol)
    assert not (tmp_path / "out").exists()
