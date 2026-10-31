"""External preparation must respect year boundaries and immutable evidence."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transcif.data.eia930 import MODERN_FUEL_COLUMNS


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_external_v4.py"
SPEC = importlib.util.spec_from_file_location("prepare_external_v4", SCRIPT)
external = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(external)


def native_inputs(tmp_path):
    frames = []
    for year in external.YEARS:
        end = pd.date_range(f"{year}-01-01T01:00Z", f"{year}-02-01T00:00Z", freq="h")
        for region in external.REGIONS:
            frame = pd.DataFrame({"Balancing Authority": region,
                "UTC Time at End of Hour": end, "Demand (MW)": 58.,
                "Demand Forecast (MW)": 999., "Net Generation (MW)": 58.,
                "Total Interchange (MW)": 0.})
            for fuel, aliases in MODERN_FUEL_COLUMNS.items():
                frame["Net Generation (MW) from " + aliases[0]] = {
                    "coal": 50., "gas": 10., "battery_storage": -2.}.get(fuel, 0.)
            # Same snapshot first reports oil in 2025; earlier yearly releases
            # cannot see it, even though some earlier hourly oil fields are NaN.
            oil = "Net Generation (MW) from All Petroleum Products"
            if year < 2025:
                frame[oil] = np.nan
            else:
                frame[oil] = 1.
                frame["Net Generation (MW)"] = 59.
            frames.append(frame)
    source = tmp_path / "native.csv"
    pd.concat(frames, ignore_index=True).to_csv(source, index=False)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Frozen external evaluation protocol.\n")
    return source, protocol


def test_yearly_prefix_excludes_next_year_fields_and_preserves_evidence(tmp_path):
    source, protocol = native_inputs(tmp_path)
    out = tmp_path / "out"
    report = external.prepare_external([source], out, protocol)
    monthly = pd.read_csv(out / "monthly.csv")
    assert len(monthly) == len(external.REGIONS) * len(external.YEARS)
    old = monthly[(monthly.region == "SWPP") & monthly.period_start.str.startswith("2024")].iloc[0]
    assert old.fuelshare_coal == pytest.approx(5 / 6)
    assert old.total_mwh == 744 * 60
    assert pd.Timestamp(old.available_at) == pd.Timestamp("2024-03-02T00:00Z")
    for region in external.REGIONS:
        hourly = pd.read_csv(out / f"{region}_hourly.csv")
        assert hourly.valid.all()
        assert len(hourly) == 744 * len(external.YEARS)
        assert "native_demand_forecast_mw" not in hourly
        assert not any("cif" in column.lower() for column in hourly)
        assert report["regions"][region]["2024"]["valid_full_utc_days"] == 31
    assert report["protocol_sha256_before"] == report["protocol_sha256_after"] == external.sha(protocol)
    assert report["native_sources"][0]["sha256"] == external.sha(source)
    assert all(external.sha(out / name) == value for name, value in report["derived_hashes"].items())
    assert not report["cif_scores_inspected"]
    assert not report["cif_summaries_computed"]
    assert not report["target_hourly_used_for_fit_or_calibration"]
    assert json.loads((out / "audit.json").read_text())["environment"]["pandas"]
    with pytest.raises(FileExistsError, match="fresh output"):
        external.prepare_external([source], out, protocol)


def test_duplicate_native_hours_are_rejected_before_output(tmp_path):
    source, protocol = native_inputs(tmp_path)
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="Duplicate native"):
        external.prepare_external([source, duplicate], tmp_path / "out", protocol)
    assert not (tmp_path / "out").exists()


def test_protocol_change_during_preparation_is_rejected(tmp_path, monkeypatch):
    source, protocol = native_inputs(tmp_path)
    original = external._monthly_at_release

    def changed_protocol(*args, **kwargs):
        protocol.write_text("Changed after preparation started.\n")
        return original(*args, **kwargs)

    monkeypatch.setattr(external, "_monthly_at_release", changed_protocol)
    with pytest.raises(RuntimeError, match="Frozen protocol changed"):
        external.prepare_external([source], tmp_path / "out", protocol)
    assert not (tmp_path / "out").exists()
