import json
import numpy as np
import pandas as pd
import pytest

from transcif.data.eia930 import MODERN_FUEL_COLUMNS
from transcif.research.eia_proxy import FUELS
from transcif.research.heldout_data import prepare_year


def inputs(tmp_path):
    historical = tmp_path / "history"
    historical.mkdir()
    hourly = pd.DataFrame({"time": ["2024-12-01T00:00Z"], "region": ["MISO"], "valid": [False]})
    for fuel in FUELS:
        hourly[f"generation_{fuel}"] = np.nan
    hourly.to_csv(historical / "MISO_hourly.csv", index=False)
    observation = dict(region="MISO", period_start="2024-12-01T00:00Z", period_end="2025-01-01T00:00Z",
                       available_at="2025-01-31T00:00Z", observed_hours=744, expected_hours=744,
                       coverage=1., total_mwh=100.)
    observation.update({f"fuelshare_{fuel}": float(fuel == "coal") for fuel in FUELS})
    pd.DataFrame([observation]).to_csv(historical / "monthly.csv", index=False)
    end = pd.date_range("2025-01-01T01:00Z", "2025-02-01T00:00Z", freq="h")
    frame = pd.DataFrame({"Balancing Authority": "MISO", "UTC Time at End of Hour": end,
                          "Demand (MW)": 58., "Demand Forecast (MW)": 999.,
                          "Net Generation (MW)": 58., "Total Interchange (MW)": 0.})
    for fuel, aliases in MODERN_FUEL_COLUMNS.items():
        frame["Net Generation (MW) from " + aliases[0]] = {"coal": 50., "gas": 10., "battery_storage": -2.}.get(fuel, 0.)
    native = tmp_path / "EIA930_BALANCE_2025_Jan_Jun.csv"
    frame.to_csv(native, index=False)
    return historical, native


def test_new_year_keeps_historical_inputs_and_excludes_storage(tmp_path):
    historical, native = inputs(tmp_path)
    original = pd.read_csv(historical / "monthly.csv")
    out = tmp_path / "out"
    report = prepare_year([native], historical, out, regions=("MISO",))
    actual = pd.read_csv(out / "monthly.csv")
    # Serialization may normalize timestamps; compare the original logical row.
    for col in original:
        if col not in {"period_start", "period_end", "available_at"}:
            assert actual[col].iloc[0] == original[col].iloc[0]
        else:
            assert pd.Timestamp(actual[col].iloc[0]) == pd.Timestamp(original[col].iloc[0])
    latest = actual.iloc[1]
    assert latest.fuelshare_coal == pytest.approx(5/6)
    assert latest.total_mwh == 744*60
    assert pd.Timestamp(latest.available_at) == pd.Timestamp("2025-03-03T00:00Z")
    frame = pd.read_csv(out / "MISO_hourly.csv")
    assert frame.valid.sum() == 744
    assert "native_demand_forecast_mw" not in frame
    assert not report["issued_public"] and not report["cif_scores_inspected"]
    assert report["regions"]["MISO"]["valid_days"] == 31
    assert json.loads((out / "audit.json").read_text())["derived_hashes"]


def test_preparation_refuses_reused_output_and_duplicate_snapshots(tmp_path):
    historical, native = inputs(tmp_path)
    with pytest.raises(ValueError, match="Duplicate native"):
        prepare_year([native, native], historical, tmp_path / "duplicate", regions=("MISO",))
    assert not (tmp_path / "duplicate").exists()
    out = tmp_path / "out"
    prepare_year([native], historical, out, regions=("MISO",))
    with pytest.raises(FileExistsError):
        prepare_year([native], historical, out, regions=("MISO",))
