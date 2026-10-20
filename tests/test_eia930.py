import numpy as np
import pandas as pd
import pytest

from transcif.data.eia930 import FUEL_COLUMNS, MODERN_FUEL_COLUMNS, normalize_balance, stage_balance


def native_frame():
    frame = pd.DataFrame({"Balancing Authority": ["CISO", "CISO"],
        "UTC Time at End of Hour": ["2023-01-01 01:00", "2023-01-01 02:00"],
        "Demand (MW)": [100., 100.], "Demand Forecast (MW)": [110., 110.],
        "Net Generation (MW)": [95., 95.], "Total Interchange (MW)": [-5., -5.]})
    for fuel, aliases in FUEL_COLUMNS.items():
        frame["Net Generation (MW) from " + aliases[0]] = [-5., 0.] if fuel == "hydro" else [10., np.nan]
    return frame


def test_native_eia_keeps_unknown_fuels_missing_values_and_charging():
    normalized = normalize_balance(native_frame())
    assert normalized.valid_at.iloc[0] == pd.Timestamp("2023-01-01T00:00Z")
    assert normalized.generation_hydro_and_pumped_storage_mw.iloc[0] == -5
    assert normalized.generation_oil_mw.iloc[0] == 10
    assert normalized.generation_unknown_mw.isna().iloc[1]
    assert normalized.native_demand_forecast_mw.iloc[0] == 110
    assert not {"issued_at", "available_at", "cif_g_per_kwh"}.intersection(normalized.columns)


def test_native_eia_rejects_ambiguous_fuel_schema_and_duplicate_vintages():
    frame = native_frame()
    frame["Net Generation (MW) from Petroleum"] = 1
    with pytest.raises(ValueError, match="exactly one"):
        normalize_balance(frame)
    frame = native_frame()
    frame["Net Generation (MW) from Battery Storage"] = 1
    with pytest.raises(ValueError, match="unmapped native fuel"):
        normalize_balance(frame)
    with pytest.raises(ValueError, match="duplicate EIA"):
        normalize_balance(pd.concat([native_frame(), native_frame()]))


def test_separate_storage_and_integrated_batteries_are_not_silently_merged():
    frame = native_frame().iloc[:, :6].copy()
    for fuel, aliases in MODERN_FUEL_COLUMNS.items():
        frame["Net Generation (MW) from " + aliases[0]] = -2. if fuel == "battery_storage" else 3.
    result = normalize_balance(frame)
    assert result.generation_battery_storage_mw.iloc[0] == -2
    assert result.generation_solar_with_battery_mw.iloc[0] == 3
    assert result.generation_hydro_mw.iloc[0] == 3
    assert "generation_hydro_and_pumped_storage_mw" not in result


def test_staging_multiple_native_schemas_keeps_snapshot_and_absent_component_information(tmp_path):
    old, new = tmp_path / "legacy.csv", tmp_path / "modern.csv"
    native_frame().to_csv(old, index=False)
    frame = native_frame().iloc[:, :6].copy()
    frame["UTC Time at End of Hour"] = ["2024-01-01 01:00", "2024-01-01 02:00"]
    for fuel, aliases in MODERN_FUEL_COLUMNS.items():
        frame["Net Generation (MW) from " + aliases[0]] = -2. if fuel == "battery_storage" else 3.
    frame.to_csv(new, index=False)
    output = tmp_path / "staged"
    report = stage_balance([old, new], ["CISO"], output)
    rows = pd.read_csv(output / "CISO.csv")
    assert len(rows) == 4 and rows.source_snapshot.tolist() == [0, 0, 1, 1]
    assert rows.generation_hydro_and_pumped_storage_mw.iloc[0] == -5
    assert np.isnan(rows.generation_hydro_mw.iloc[0])
    assert rows.generation_battery_storage_mw.iloc[-1] == -2
    assert not report["eligible_for_issued_public"]
    assert len(report["native_schemas"]) == 2
    assert report["regions"]["CISO"]["complete_fuel_rows"] == 3
