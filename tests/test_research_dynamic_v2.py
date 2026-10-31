"""Leakage, support, transform and physical contract regressions for V2."""
import numpy as np
import pandas as pd
import pytest

from transcif.research.dynamic_v2 import (
    DynamicModel, METHODS, basis, daily_profiles, observation_features, quarterly, read_year,
)
from transcif.research.eia_proxy import FUELS


def history():
    records = []
    for month in range(1, 7):
        start = pd.Timestamp(year=2022, month=month, day=1, tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        record = dict(region="BPAT", period_start=start, period_end=end,
                      available_at=end + pd.Timedelta(days=30), total_mwh=month * 100,
                      coverage=.9)
        record.update({f"fuelshare_{fuel}": float(i == month % 2) for i, fuel in enumerate(FUELS)})
        records.append(record)
    return pd.DataFrame(records)


@pytest.mark.parametrize("kind", ["time", "dct", "haar"])
def test_complete_orthogonal_basis(kind):
    transform = basis(kind)
    np.testing.assert_allclose(transform @ transform.T, np.eye(24), atol=1e-14)
    values = np.random.default_rng(81).normal(size=(4, 24))
    np.testing.assert_allclose((values @ transform.T) @ transform, values, atol=1e-14)


def test_availability_boundary_support_and_future_append():
    original = history()
    release = original.iloc[0].available_at
    origins = pd.DatetimeIndex([release - pd.Timedelta(seconds=1), release])
    seasonal = np.ones((12, 8)) / 8
    x, _, _, permitted, ends = observation_features(origins, "BPAT", original, seasonal)
    assert permitted.tolist() == [False, True]
    assert ends[1] < origins[1].tz_localize(None).to_datetime64()
    altered = original.copy()
    altered.loc[1:, [f"fuelshare_{f}" for f in FUELS]] = -999
    xx = observation_features(origins, "BPAT", altered, seasonal)
    np.testing.assert_array_equal(x, xx[0])
    # A dishonest future support with an earlier availability is still excluded.
    future = original.iloc[[5]].copy()
    future["available_at"] = release - pd.Timedelta(days=1)
    xxx = observation_features(origins, "BPAT", pd.concat([original, future]), seasonal)
    np.testing.assert_array_equal(x, xxx[0])


def test_quarterly_complete_support_energy_weights_and_release():
    data = history()
    quarter = quarterly(data).iloc[0]
    expected = np.average(data.iloc[:3][[f"fuelshare_{f}" for f in FUELS]],
                          weights=data.iloc[:3].total_mwh, axis=0)
    np.testing.assert_allclose(quarter[[f"fuelshare_{f}" for f in FUELS]].to_numpy(float), expected)
    assert quarter.available_at == data.iloc[2].available_at
    assert quarterly(data.iloc[[0, 2]]).empty


def test_prediction_interface_prefix_invariance_and_no_history():
    model = DynamicModel(np.full((12, 24), 200.), np.ones((12, 8)) / 8,
                         {"ridge": (np.zeros(14), np.ones(14), np.zeros((15, 24)))}, {})
    origins = pd.DatetimeIndex(["2022-03-03T00:00:00Z"])
    before = model.predict(origins, "BPAT", history())
    mutated = history()
    mutated.loc[1:, [f"fuelshare_{f}" for f in FUELS]] = -999
    after = model.predict(origins, "BPAT", mutated)
    for method in METHODS:
        np.testing.assert_array_equal(before[method], after[method])
        assert np.all((before[method] >= 0) & (before[method] <= 1000))
    no_history = model.predict(origins, "CISO", history())
    for method in METHODS[3:]:
        np.testing.assert_array_equal(no_history[method], np.full((1, 24), 200.))


def test_read_year_excludes_future_numeric_and_hourly_labels(tmp_path):
    path = tmp_path / "hourly.csv"
    path.write_text("time,region,generation_coal,valid\n2022-01-01 00:00:00+00:00,BPAT,1,True\n2024-01-01 00:00:00+00:00,BPAT,DO_NOT_PARSE,False\n")
    frame = read_year(path, 2022)
    assert len(frame) == 1
    assert frame.generation_coal.dtype.kind in "if"


def test_training_cutoff_does_not_mutate_shared_source_validity():
    times = pd.date_range("2022-06-30", periods=48, freq="h", tz="UTC")
    frame = pd.DataFrame({"time": times, "valid": True})
    for fuel in FUELS:
        frame[f"generation_{fuel}"] = 1.
    before = frame.copy(deep=True)
    days, _ = daily_profiles(frame, "2022-07-01T00:00:00Z")
    assert len(days) == 1
    pd.testing.assert_frame_equal(frame, before)
    assert len(daily_profiles(frame)[0]) == 2


def test_nonzero_dynamic_prediction_uses_age_but_not_unreleased_rows():
    correction = np.zeros((11, 24))
    correction[1, :] = 50.
    scalar_fits = [(np.zeros(10), np.ones(10), correction[:, j]) for j in range(24)]
    fits = {"ridge": (np.zeros(14), np.ones(14), np.zeros((15, 24))),
            "age_time": scalar_fits}
    model = DynamicModel(np.full((12, 24), 200.), np.ones((12, 8)) / 8,
                         fits, {"age_time": np.full(24, 30.)})
    history_prefix = history().iloc[:1]
    origins = pd.DatetimeIndex(["2022-03-03T00:00:00Z", "2022-03-20T00:00:00Z"])
    forecast = model.predict(origins, "BPAT", history_prefix)["age_time"]
    assert not np.array_equal(forecast[0], forecast[1])
    full = history()
    full.loc[full.available_at > origins.max(), "fuelshare_coal"] = -999
    np.testing.assert_array_equal(forecast, model.predict(origins, "BPAT", full)["age_time"])


def test_vectorized_matches_original_asof_algorithm_exactly(monkeypatch):
    import transcif.research.dynamic_v2 as module

    data = history()
    # Nonmonotonic releases, missing observations, same-support revision tie.
    data.loc[1, "available_at"] = data.loc[3, "available_at"]
    revision = data.iloc[[0]].copy()
    revision["available_at"] += pd.Timedelta(days=5)
    revision["fuelshare_coal"], revision["fuelshare_gas"] = .3, .7
    data = pd.concat([data.drop(index=2), revision], ignore_index=True)
    origins = pd.date_range("2022-02-28", "2022-08-01", freq="7D", tz="UTC")
    seasonal = np.arange(96).reshape(12, 8) / 100

    def original(times, region, table, expected_mix):
        # Same original pre-vectorization algorithm, one forecast at a time.
        features, mixes, ages, valid, ends = [], [], [], [], []
        for origin in times:
            eligible = table[(table.region == region) & (table.available_at <= origin)
                             & (table.period_end <= origin) & (table.period_start < table.period_end)]
            if eligible.empty:
                features.append(np.zeros(10)); mixes.append(np.zeros(8)); ages.append(0.)
                valid.append(False); ends.append(np.datetime64("NaT", "ns")); continue
            row = eligible.sort_values(["period_end", "available_at"]).iloc[-1]
            mix = row[[f"fuelshare_{f}" for f in FUELS]].to_numpy(float)
            months = pd.date_range(row.period_start, row.period_end, freq="MS", inclusive="left")
            features.append(np.r_[mix - expected_mix[months.month - 1].mean(axis=0),
                                  (row.period_end - row.period_start).total_seconds() / 86400 / 90, row.coverage])
            mixes.append(mix); ages.append((origin - row.period_end).total_seconds() / 86400)
            valid.append(True); ends.append(row.period_end.tz_localize(None).to_datetime64())
        return (np.asarray(features), np.asarray(mixes), np.asarray(ages),
                np.asarray(valid), np.asarray(ends, dtype="datetime64[ns]"))

    actual = observation_features(origins, "BPAT", data, seasonal)
    expected = original(origins, "BPAT", data, seasonal)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)
    coefficients = np.arange(11 * 24, dtype=float).reshape(11, 24) / 100
    model = DynamicModel(np.full((12, 24), 200.), seasonal,
                         {"ridge": (np.zeros(14), np.ones(14), np.zeros((15, 24))),
                          "age_time": [(np.zeros(10), np.ones(10), coefficients[:, j]) for j in range(24)]},
                         {"age_time": np.arange(24) + 30.})
    fast = model.predict(origins, "BPAT", data)
    monkeypatch.setattr(module, "observation_features", original)
    slow = model.predict(origins, "BPAT", data)
    for key in METHODS:
        np.testing.assert_array_equal(fast[key], slow[key])


def test_target_latency_stress_does_not_change_source_or_advance_release():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/run_dynamic_v2.py"
    spec = importlib.util.spec_from_file_location("dynamic_runner_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    target = history()
    source = target.copy()
    source["region"] = "CISO"
    original = pd.concat([target, source], ignore_index=True)
    frozen = original.copy(deep=True)
    delayed = runner.shifted_target_history(original, "BPAT", 30)
    pd.testing.assert_frame_equal(original, frozen)
    np.testing.assert_array_equal(delayed.loc[delayed.region == "CISO", "available_at"], source.available_at)
    np.testing.assert_array_equal(delayed.loc[delayed.region == "BPAT", "available_at"],
                                  target.available_at + pd.Timedelta(days=30))
    with pytest.raises(ValueError):
        runner.shifted_target_history(original, "BPAT", -1)
