"""Tests for real AEMO/NEMED hourly CSV loading and sliding-window construction.

Per the project's real-data-only mandate, these tests load an excerpt of actual
2023 AEMO-derived SA1 hourly data (tests/fixtures/real_aemo_sample_sa1.csv) rather
than synthetic series."""

import pandas as pd
import pytest

from transcif.data.loaders import (
    GENERATION_REQUIRED_COLUMNS,
    NUM_REAL_CHANNELS,
    NUM_REAL_CHANNELS_WITH_GENERATION,
    TEMPERATURE_REQUIRED_COLUMNS,
    build_sliding_windows,
    load_region_hourly_csv,
    load_region_temperature_csv,
    load_region_windows,
    merge_temperature,
)

FIXTURE_PATH = "tests/fixtures/real_aemo_sample_sa1.csv"
FIXTURE_ROWS = 300
TEMPERATURE_FIXTURE_PATH = "tests/fixtures/real_temperature_sample_sa1.csv"

SEQ_LEN = 48
HORIZON = 12
STRIDE = 6


def test_load_region_hourly_csv_returns_sorted_frame_with_filled_renew_share():
    df = load_region_hourly_csv(FIXTURE_PATH)

    assert len(df) == FIXTURE_ROWS
    assert list(df["hour"]) == sorted(df["hour"])
    assert df["renew_share"].isna().sum() == 0


def test_load_region_hourly_csv_raises_on_missing_required_columns(tmp_path):
    bad_csv = tmp_path / "missing_columns.csv"
    pd.DataFrame({"hour": ["2023-01-01"], "renew_share": [0.5]}).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_region_hourly_csv(str(bad_csv))


def test_build_sliding_windows_shapes_match_stride_and_channel_count():
    df = load_region_hourly_csv(FIXTURE_PATH)
    x, y = build_sliding_windows(df, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE)

    window = SEQ_LEN + HORIZON
    expected_windows = (FIXTURE_ROWS - window) // STRIDE + 1

    assert x.shape == (expected_windows, SEQ_LEN, NUM_REAL_CHANNELS)
    assert y.shape == (expected_windows, HORIZON)


def test_build_sliding_windows_first_channel_matches_renew_share():
    df = load_region_hourly_csv(FIXTURE_PATH)
    x, _ = build_sliding_windows(df, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE)

    renew_share = df["renew_share"].to_numpy()
    assert x[0, :, 0].numpy() == pytest.approx(renew_share[:SEQ_LEN])


def test_build_sliding_windows_raises_when_series_shorter_than_window():
    df = load_region_hourly_csv(FIXTURE_PATH).iloc[:10].reset_index(drop=True)

    with pytest.raises(ValueError, match="shorter than window"):
        build_sliding_windows(df, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE)


def test_load_region_windows_matches_separate_load_and_build_calls():
    df = load_region_hourly_csv(FIXTURE_PATH)
    x_expected, y_expected = build_sliding_windows(df, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE)

    x_actual, y_actual = load_region_windows(FIXTURE_PATH, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE)

    assert x_actual.numpy() == pytest.approx(x_expected.numpy())
    assert y_actual.numpy() == pytest.approx(y_expected.numpy())


def test_build_sliding_windows_with_generation_channels_has_four_channels():
    df = load_region_hourly_csv(FIXTURE_PATH)
    x, y = build_sliding_windows(
        df, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE, include_generation_channels=True
    )

    window = SEQ_LEN + HORIZON
    expected_windows = (FIXTURE_ROWS - window) // STRIDE + 1

    assert x.shape == (expected_windows, SEQ_LEN, NUM_REAL_CHANNELS_WITH_GENERATION)
    assert y.shape == (expected_windows, HORIZON)


def test_build_sliding_windows_generation_channels_are_scale_invariant_not_raw_mw():
    """RenewOutNorm/NonRenewOutNorm must be rolling-quantile-normalized like LoadNorm, not
    raw MW -- otherwise cross-region transfer breaks since regions differ by an order of
    magnitude in absolute generation (this is the same Innovation-1 scale-invariance
    reason LoadNorm exists instead of raw load)."""
    df = load_region_hourly_csv(FIXTURE_PATH)
    x, _ = build_sliding_windows(
        df, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE, include_generation_channels=True
    )

    renew_out = df["renew_out"].to_numpy()
    assert not (x[0, :, 2].numpy() == pytest.approx(renew_out[:SEQ_LEN]))
    assert x[0, :, 2].numpy().max() <= 2.0


def test_build_sliding_windows_raises_when_generation_columns_missing(tmp_path):
    df = load_region_hourly_csv(FIXTURE_PATH).drop(columns=["renew_out", "nonrenew_out"])

    with pytest.raises(ValueError, match="missing required columns"):
        build_sliding_windows(
            df, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE, include_generation_channels=True
        )


def test_generation_required_columns_are_renew_and_nonrenew_out():
    assert GENERATION_REQUIRED_COLUMNS == ("renew_out", "nonrenew_out")


def test_load_region_temperature_csv_returns_sorted_frame():
    temp_df = load_region_temperature_csv(TEMPERATURE_FIXTURE_PATH)

    assert len(temp_df) == FIXTURE_ROWS
    assert list(temp_df["hour"]) == sorted(temp_df["hour"])
    assert temp_df["temperature_c"].isna().sum() == 0


def test_load_region_temperature_csv_raises_on_missing_required_columns(tmp_path):
    bad_csv = tmp_path / "missing_temp_columns.csv"
    pd.DataFrame({"hour": ["2023-01-01"]}).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_region_temperature_csv(str(bad_csv))


def test_merge_temperature_aligns_by_exact_hour_timestamp():
    df = load_region_hourly_csv(FIXTURE_PATH)
    temp_df = load_region_temperature_csv(TEMPERATURE_FIXTURE_PATH)

    merged = merge_temperature(df, temp_df)

    assert len(merged) == len(df)
    assert "temperature_c" in merged.columns
    assert merged["temperature_c"].isna().sum() == 0


def test_build_sliding_windows_with_temperature_channel_appends_temp_anomaly():
    df = load_region_hourly_csv(FIXTURE_PATH)
    temp_df = load_region_temperature_csv(TEMPERATURE_FIXTURE_PATH)
    merged = merge_temperature(df, temp_df)

    x, y = build_sliding_windows(
        merged, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE, include_temperature_channel=True
    )

    window = SEQ_LEN + HORIZON
    expected_windows = (FIXTURE_ROWS - window) // STRIDE + 1
    assert x.shape == (expected_windows, SEQ_LEN, NUM_REAL_CHANNELS + 1)
    assert y.shape == (expected_windows, HORIZON)


def test_build_sliding_windows_temperature_channel_is_anomaly_not_raw_celsius():
    """TempAnomaly must be a same-day-of-year climate-baseline deviation, not raw Celsius --
    raw temperature differs by region in ways that aren't the physically meaningful signal
    and would break the scale-invariance the other channels are designed to preserve."""
    df = load_region_hourly_csv(FIXTURE_PATH)
    temp_df = load_region_temperature_csv(TEMPERATURE_FIXTURE_PATH)
    merged = merge_temperature(df, temp_df)

    x, _ = build_sliding_windows(
        merged, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE, include_temperature_channel=True
    )

    raw_temp = merged["temperature_c"].to_numpy()
    assert not (x[0, :, -1].numpy() == pytest.approx(raw_temp[:SEQ_LEN]))


def test_build_sliding_windows_raises_when_temperature_column_missing():
    df = load_region_hourly_csv(FIXTURE_PATH)

    with pytest.raises(ValueError, match="missing required columns"):
        build_sliding_windows(
            df, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE, include_temperature_channel=True
        )


def test_load_region_windows_with_temp_csv_path_matches_manual_merge():
    df = load_region_hourly_csv(FIXTURE_PATH)
    temp_df = load_region_temperature_csv(TEMPERATURE_FIXTURE_PATH)
    merged = merge_temperature(df, temp_df)
    x_expected, y_expected = build_sliding_windows(
        merged, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE, include_temperature_channel=True
    )

    x_actual, y_actual = load_region_windows(
        FIXTURE_PATH, seq_len=SEQ_LEN, horizon=HORIZON, stride=STRIDE, temp_csv_path=TEMPERATURE_FIXTURE_PATH
    )

    assert x_actual.numpy() == pytest.approx(x_expected.numpy())
    assert y_actual.numpy() == pytest.approx(y_expected.numpy())


def test_temperature_required_columns_are_hour_and_temperature_c():
    assert TEMPERATURE_REQUIRED_COLUMNS == ("hour", "temperature_c")
