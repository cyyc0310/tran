import json

import numpy as np
import pandas as pd
import pytest

from transcif.research.eia_proxy import FUELS, latest_observation, prepare


def _source(tmp_path, frame, schemas):
    path = tmp_path / "source"
    path.mkdir()
    frame.to_csv(path / "MISO.csv", index=False)
    (path / "metadata.json").write_text(json.dumps({"native_schemas": {
        str(key): {"fuel_columns": columns} for key, columns in schemas.items()}}))
    return path


def _frame(times, columns):
    frame = pd.DataFrame({"region": "MISO", "valid_at": times, "source_snapshot": 0,
        "net_generation_mw": 0.0})
    for column, values in columns.items():
        frame[column] = values
    frame["net_generation_mw"] = frame[list(columns)].sum(axis=1)
    return frame


def test_native_snapshot_selects_schema_without_duplicate_generation(tmp_path):
    times = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
    old = ["generation_coal_mw", "generation_solar_mw", "generation_hydro_and_pumped_storage_mw"]
    new = ["generation_coal_mw", "generation_solar_without_battery_mw",
        "generation_solar_with_battery_mw", "generation_hydro_mw",
        "generation_pumped_storage_mw", "generation_geothermal_mw"]
    frame = _frame(times, {column: 0.0 for column in set(old + new)})
    frame["generation_coal_mw"] = 50.0
    frame["generation_solar_mw"] = 10.0  # Stale populated columns must not be added.
    frame["generation_hydro_and_pumped_storage_mw"] = 20.0
    frame["generation_solar_without_battery_mw"] = 12.0
    frame["generation_solar_with_battery_mw"] = 3.0
    frame["generation_hydro_mw"] = 6.0
    frame["generation_pumped_storage_mw"] = -4.0
    frame["generation_geothermal_mw"] = 2.0
    frame.loc[24:, "source_snapshot"] = 1
    frame.loc[:23, "net_generation_mw"] = frame.loc[:23, old].sum(axis=1)
    frame.loc[24:, "net_generation_mw"] = frame.loc[24:, new].sum(axis=1)
    source = _source(tmp_path, frame, {0: old, 1: new})
    report = prepare(source, tmp_path / "out")
    actual = pd.read_csv(tmp_path / "out/MISO_hourly.csv")
    assert actual.valid.all()
    assert actual.generation_solar.iloc[0] == 10
    assert actual.generation_solar.iloc[-1] == 15
    assert actual.generation_hydro.iloc[0] == 20
    assert actual.generation_hydro.iloc[-1] == 6
    assert actual.generation_other.iloc[-1] == 2
    assert "generation_pumped_storage" not in actual
    assert report["regions"]["MISO"]["snapshots"]["0"]["legacy_hydro_includes_pumped_storage"]


def test_sporadic_missing_invalidates_day_and_is_never_zero_label(tmp_path):
    times = pd.date_range("2023-01-01", periods=72, freq="h", tz="UTC")
    columns = {"generation_coal_mw": np.full(72, 20.), "generation_gas_mw": np.full(72, 10.),
        "generation_unknown_mw": np.full(72, np.nan)}
    frame = _frame(times, columns)
    frame.loc[30, "generation_gas_mw"] = np.nan
    source = _source(tmp_path, frame, {0: list(columns)})
    report = prepare(source, tmp_path / "out")
    actual = pd.read_csv(tmp_path / "out/MISO_hourly.csv")
    assert actual.valid.sum() == 48
    assert actual.loc[24:47, [f"generation_{fuel}" for fuel in FUELS]].isna().all().all()
    assert actual.loc[:23, "generation_gas"].eq(10).all()
    audit = report["regions"]["MISO"]["snapshots"]["0"]
    assert audit["rows_with_sporadic_missing"] == 1
    assert audit["unobserved_columns_excluded_not_certified_zero"] == ["generation_unknown_mw"]


def test_monthly_shares_are_energy_weighted_and_release_is_asof(tmp_path):
    times = pd.date_range("2023-01-01", "2023-02-01", inclusive="left", freq="h", tz="UTC")
    count = len(times)
    coal = np.where(np.arange(count) % 2 == 0, 100., 0.)
    gas = np.where(np.arange(count) % 2 == 0, 0., 10.)
    frame = _frame(times, {"generation_coal_mw": coal, "generation_gas_mw": gas})
    source = _source(tmp_path, frame, {0: ["generation_coal_mw", "generation_gas_mw"]})
    prepare(source, tmp_path / "out")
    monthly = pd.read_csv(tmp_path / "out/monthly.csv")
    assert len(monthly) == 1
    assert monthly.fuelshare_coal.iloc[0] == pytest.approx(100 / 110)
    assert monthly.total_mwh.iloc[0] == 372 * 110
    assert monthly.observed_hours.iloc[0] == monthly.expected_hours.iloc[0] == 744
    assert latest_observation(monthly, "MISO", "2023-03-02T23:00Z") is None
    released = latest_observation(monthly, "MISO", "2023-03-03T00:00Z")
    assert released is not None
    assert released.period_end == pd.Timestamp("2023-02-01T00:00Z")
    assert latest_observation(monthly, "BPAT", "2023-04-01T00:00Z") is None
    with pytest.raises(ValueError, match="timezone-aware"):
        latest_observation(monthly, "MISO", "2023-04-01")


def test_negative_generation_proxy_and_native_balance_are_separate(tmp_path):
    times = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
    columns = {"generation_coal_mw": 50., "generation_solar_mw": -2.,
        "generation_battery_storage_mw": -8.}
    frame = _frame(times, columns)
    frame.loc[26, "net_generation_mw"] = 200.  # One bad native balance invalidates this UTC day.
    source = _source(tmp_path, frame, {0: list(columns)})
    report = prepare(source, tmp_path / "out")
    actual = pd.read_csv(tmp_path / "out/MISO_hourly.csv")
    assert actual.valid.sum() == 24
    assert actual.loc[:23, "generation_solar"].eq(0).all()
    assert actual.loc[:23, "generation_coal"].eq(50).all()
    audit = report["regions"]["MISO"]["snapshots"]["0"]
    assert audit["negative_native_energy_mwh_clipped"]["generation_solar_mw"] == 96
    assert audit["excluded_signed_energy_mwh"]["generation_battery_storage_mw"] == -384
    assert not report["full_grid_emissions_truth"]
    assert not report["eligible_for_issued_public"]
    assert pd.read_csv(tmp_path / "out/monthly.csv").empty


def test_duplicate_hours_rejected_before_output(tmp_path):
    times = [pd.Timestamp("2023-01-01T00:00Z")] * 2
    frame = _frame(times, {"generation_coal_mw": 100.})
    source = _source(tmp_path, frame, {0: ["generation_coal_mw"]})
    with pytest.raises(ValueError, match="unique UTC"):
        prepare(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_asof_rejects_future_period_early_release_and_low_coverage():
    monthly = pd.DataFrame([
        {"region": "MISO", "period_start": "2023-01-01T00:00Z", "period_end": "2023-02-01T00:00Z",
         "available_at": "2023-03-03T00:00Z", "coverage": 1.0},
        {"region": "MISO", "period_start": "2023-03-01T00:00Z", "period_end": "2023-04-01T00:00Z",
         "available_at": "2023-03-04T00:00Z", "coverage": 1.0},
        {"region": "MISO", "period_start": "2023-02-01T00:00Z", "period_end": "2023-03-01T00:00Z",
         "available_at": "2023-03-31T00:00Z", "coverage": 0.5},
    ])
    for origin in ("2023-03-05T00:00Z", "2023-04-02T00:00Z"):
        result = latest_observation(monthly, "MISO", origin)
        assert result.period_start == pd.Timestamp("2023-01-01T00:00Z")


@pytest.mark.parametrize("change", ["append", "mutate"])
def test_monthly_release_cannot_depend_on_future_fuel_reporting(tmp_path, change):
    release = pd.Timestamp("2023-03-03T00:00Z")
    times = pd.date_range("2023-01-01", "2023-03-07", inclusive="left", freq="h", tz="UTC")
    frame = _frame(times, {"generation_coal_mw": 50., "generation_battery_storage_mw": np.nan})
    future = frame.valid_at.ge(release)
    before = frame.loc[~future].copy() if change == "append" else frame.copy()
    after = frame.copy()
    # This newly active field used to make January's missing storage readings
    # retroactively fail the full-snapshot completeness test, deleting its release.
    after.loc[future, "generation_battery_storage_mw"] = 7.
    after.loc[future, "generation_coal_mw"] = 500.
    after.loc[future, "net_generation_mw"] = 507.
    schemas = {0: ["generation_coal_mw", "generation_battery_storage_mw"]}
    for name, data in (("before", before), ("after", after)):
        fixture = tmp_path / name
        fixture.mkdir()
        source = _source(fixture, data, schemas)
        report = prepare(source, fixture / "out")
        assert report["monthly_quality_method"] == "prefix_at_assumed_release"
    original = pd.read_csv(tmp_path / "before/out/monthly.csv")
    changed = pd.read_csv(tmp_path / "after/out/monthly.csv")
    original = original.loc[pd.to_datetime(original.available_at, utc=True).eq(release)].reset_index(drop=True)
    changed = changed.loc[pd.to_datetime(changed.available_at, utc=True).eq(release)].reset_index(drop=True)
    assert len(original) == len(changed) == 1
    pd.testing.assert_frame_equal(original, changed)
    assert original.observed_hours.iloc[0] == 744
    retrospective = pd.read_csv(tmp_path / "after/out/MISO_hourly.csv")
    january = pd.to_datetime(retrospective.time, utc=True).dt.month.eq(1)
    assert not retrospective.loc[january, "valid"].any()
