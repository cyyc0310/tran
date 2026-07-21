"""Real AEMO/NEMED hourly CSV loading and sliding-window construction for Stage 1
training. Windows carry two measured channels (RenewShare, LoadNorm) by default, plus
optionally RenewOutNorm/NonRenewOutNorm (`include_generation_channels=True` -- the raw
REG/NEG generation magnitudes the original AAAI-26 paper models separately, normalized
the same scale-invariant way as LoadNorm rather than as absolute MW so cross-region
transfer is preserved) and/or a TempAnomaly channel (`temp_csv_path=...` -- real hourly
temperature merged in by timestamp and turned into a same-day-of-year climate anomaly via
`compute_temp_anomaly` in reparam.py, since anomaly-from-local-climate-baseline is far
closer to scale-invariant across regions than raw degrees Celsius). The temperature CSV
is NOT AEMO/NEMED data -- it comes from Open-Meteo's free historical weather archive for
one representative city per region, since no AEMO source in this project carries weather."""

import numpy as np
import pandas as pd
import torch

from transcif.data.reparam import compute_load_norm, compute_temp_anomaly

NUM_REAL_CHANNELS = 2
NUM_REAL_CHANNELS_WITH_GENERATION = 4
REQUIRED_COLUMNS = ("hour", "renew_share", "total_energy_so")
GENERATION_REQUIRED_COLUMNS = ("renew_out", "nonrenew_out")
TEMPERATURE_REQUIRED_COLUMNS = ("hour", "temperature_c")


def load_region_hourly_csv(csv_path: str) -> pd.DataFrame:
    """Load a real AEMO/NEMED hourly export, sorted by time with RenewShare gaps filled."""
    df = pd.read_csv(csv_path, parse_dates=["hour"])
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"real data CSV missing required columns: {missing_columns}")

    df = df.sort_values("hour").reset_index(drop=True)
    df["renew_share"] = df["renew_share"].ffill().bfill()
    return df


def load_region_temperature_csv(csv_path: str) -> pd.DataFrame:
    """Load a real hourly temperature export (hour, temperature_c), sorted by time."""
    df = pd.read_csv(csv_path, parse_dates=["hour"])
    missing_columns = [column for column in TEMPERATURE_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"temperature CSV missing required columns: {missing_columns}")
    return df.sort_values("hour").reset_index(drop=True)


def merge_temperature(df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge a temperature series onto the hourly frame by exact timestamp, filling any
    unmatched hours by nearest-neighbor fill so short gaps don't break the sliding window."""
    merged = df.merge(temp_df, on="hour", how="left")
    merged["temperature_c"] = merged["temperature_c"].ffill().bfill()
    return merged


def build_sliding_windows(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    stride: int,
    include_generation_channels: bool = False,
    include_temperature_channel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (x, y) tensors from a real hourly DataFrame produced by `load_region_hourly_csv`
    (optionally passed through `merge_temperature` first). x has NUM_REAL_CHANNELS channels
    (RenewShare, LoadNorm), plus RenewOutNorm/NonRenewOutNorm when
    `include_generation_channels=True`, plus TempAnomaly (last channel) when
    `include_temperature_channel=True`; y is the RenewShare horizon."""
    window = seq_len + horizon
    renew_share = df["renew_share"].to_numpy()
    if len(renew_share) < window:
        raise ValueError(f"series length {len(renew_share)} shorter than window {window}")

    load_norm = compute_load_norm(df["total_energy_so"].to_numpy())
    load_norm = pd.Series(load_norm).ffill().bfill().to_numpy()

    channel_series = [renew_share, load_norm]
    if include_generation_channels:
        missing_columns = [c for c in GENERATION_REQUIRED_COLUMNS if c not in df.columns]
        if missing_columns:
            raise ValueError(f"real data CSV missing required columns: {missing_columns}")
        renew_out_norm = compute_load_norm(df["renew_out"].to_numpy())
        renew_out_norm = pd.Series(renew_out_norm).ffill().bfill().to_numpy()
        nonrenew_out_norm = compute_load_norm(df["nonrenew_out"].to_numpy())
        nonrenew_out_norm = pd.Series(nonrenew_out_norm).ffill().bfill().to_numpy()
        channel_series += [renew_out_norm, nonrenew_out_norm]

    if include_temperature_channel:
        if "temperature_c" not in df.columns:
            raise ValueError("real data CSV missing required columns: ['temperature_c']")
        day_of_year = df["hour"].dt.dayofyear.to_numpy()
        temp_anomaly = compute_temp_anomaly(df["temperature_c"].to_numpy(), day_of_year)
        channel_series += [temp_anomaly]

    x_windows, y_windows = [], []
    for start in range(0, len(renew_share) - window + 1, stride):
        channel_windows = [series[start : start + window][:seq_len] for series in channel_series]
        x_windows.append(np.stack(channel_windows, axis=-1))
        y_windows.append(renew_share[start + seq_len : start + window])

    x = torch.tensor(np.stack(x_windows), dtype=torch.float32)
    y = torch.tensor(np.stack(y_windows), dtype=torch.float32)
    return x, y


def load_region_windows(
    csv_path: str,
    seq_len: int,
    horizon: int,
    stride: int,
    include_generation_channels: bool = False,
    temp_csv_path: str = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a real hourly CSV (and optionally a temperature CSV, merged by timestamp) and
    build sliding-window training tensors in one call."""
    df = load_region_hourly_csv(csv_path)
    if temp_csv_path is not None:
        temp_df = load_region_temperature_csv(temp_csv_path)
        df = merge_temperature(df, temp_df)
    return build_sliding_windows(
        df,
        seq_len=seq_len,
        horizon=horizon,
        stride=stride,
        include_generation_channels=include_generation_channels,
        include_temperature_channel=temp_csv_path is not None,
    )
