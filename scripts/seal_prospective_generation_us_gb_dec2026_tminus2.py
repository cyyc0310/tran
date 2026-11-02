"""Seal a public NESO t-2 input sensitivity for the December 2026 GB window."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_prospective_generation_us_gb_dec2026 as dec  # noqa: E402
import run_prospective_generation_us_gb_nov2026_remainder as frozen  # noqa: E402

SNAPSHOT = ROOT / "docs/evidence/neso-public-hourly-snapshot-20261102/neso_generation_mix_2026_10_localmonth_asof_20261102.json"
PROTOCOL = ROOT / "docs/PROSPECTIVE_GENERATION_US_GB_DEC2026_TMINUS2_PROTOCOL.md"
OUT = ROOT / "results/prospective-generation-us-gb-dec2026-tminus2-r2"
FORECAST_ORIGIN = pd.Timestamp("2026-11-02T03:23:29Z")
OBSERVED_MONTH = pd.Period("2026-10", freq="M")


def october_shares() -> tuple[dict[str, float], dict]:
    payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    records = payload["result"]["records"]
    frame = pd.DataFrame(records)
    expected_columns = ["DATETIME", "COAL", "GAS", "NUCLEAR", "WIND", "WIND_EMB",
                        "HYDRO", "BIOMASS", "OTHER", "SOLAR"]
    missing = sorted(set(expected_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"NESO snapshot missing columns: {missing}")
    stamp = pd.to_datetime(frame["DATETIME"], errors="raise", utc=True)
    if len(frame) != 1490 or stamp.duplicated().any():
        raise ValueError("expected 1,490 unique half-hour rows for October 2026 London local time")
    steps = stamp.sort_values().diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta(minutes=30):
        raise ValueError("NESO snapshot is not a complete half-hour series")

    fuels = pd.DataFrame(index=frame.index)
    fuels["coal"] = pd.to_numeric(frame["COAL"], errors="coerce")
    fuels["gas"] = pd.to_numeric(frame["GAS"], errors="coerce")
    fuels["oil"] = 0.0
    fuels["other"] = (pd.to_numeric(frame["BIOMASS"], errors="coerce")
                      + pd.to_numeric(frame["OTHER"], errors="coerce"))
    fuels["nuclear"] = pd.to_numeric(frame["NUCLEAR"], errors="coerce")
    fuels["hydro"] = pd.to_numeric(frame["HYDRO"], errors="coerce")
    fuels["wind"] = (pd.to_numeric(frame["WIND"], errors="coerce")
                     + pd.to_numeric(frame["WIND_EMB"], errors="coerce"))
    fuels["solar"] = pd.to_numeric(frame["SOLAR"], errors="coerce")
    values = fuels.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("NESO mapped fuel values must be finite and nonnegative")
    hour = stamp.dt.floor("h")
    hourly = fuels.groupby(hour).mean()
    counts = hour.groupby(hour).size()
    complete_hours = counts[counts.eq(2)].index
    hourly = hourly.reindex(complete_hours)
    local_month = hourly.index.tz_convert("Europe/London").tz_localize(None).to_period("M")
    monthly = hourly.loc[local_month == OBSERVED_MONTH].sum()
    expected_hours = int(((OBSERVED_MONTH + 1).start_time.tz_localize("Europe/London")
                          - OBSERVED_MONTH.start_time.tz_localize("Europe/London")).total_seconds() / 3600)
    observed_hours = int((local_month == OBSERVED_MONTH).sum())
    total = float(monthly.sum())
    if expected_hours != 745 or observed_hours != expected_hours or total <= 0:
        raise ValueError(f"October local-month support mismatch: {observed_hours}/{expected_hours}")
    shares = {
        "thermal": float(monthly[["coal", "gas", "oil", "other"]].sum() / total),
        "hydro": float(monthly.hydro / total),
        "nuclear": float(monthly.nuclear / total),
        "wind": float(monthly.wind / total),
        "solar": float(monthly.solar / total),
    }
    if abs(sum(shares.values()) - 1.0) > 1e-10:
        raise ValueError("five-category mapped shares do not close")
    audit = {
        "source_file": str(SNAPSHOT.relative_to(ROOT)),
        "snapshot_sha256": dec.sha(SNAPSHOT),
        "retrieval_completed_by_utc": FORECAST_ORIGIN.isoformat(),
        "observed_month": str(OBSERVED_MONTH),
        "input_vintage": "current NESO public snapshot; retrospectively revisable",
        "raw_half_hour_rows": len(frame), "complete_hour_rows_local_month": observed_hours,
        "expected_hour_rows_local_month": expected_hours,
        "columns_used": expected_columns,
        "carbon_intensity_column_read": False,
        "mapping": "thermal=coal+gas+zero_oil+biomass+OTHER; wind=WIND+WIND_EMB; hydro/nuclear/solar direct",
        "fuel_generation_sums": {str(k): float(v) for k, v in monthly.items()},
        "shares": shares,
    }
    return shares, audit


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if pd.Timestamp.now(tz="UTC") >= pd.Timestamp("2026-12-01T00:00:00Z"):
        raise RuntimeError("December target window has started; refuse retrospective sealing")
    if not SNAPSHOT.is_file() or not PROTOCOL.is_file():
        raise FileNotFoundError("frozen public snapshot or protocol missing")

    dec.OUT = OUT
    dec.PROTOCOL = PROTOCOL
    dec.FORECAST_ORIGIN = FORECAST_ORIGIN
    frozen.FORECAST_ORIGIN = FORECAST_ORIGIN
    frozen.OBSERVED_NESO_MONTH = OBSERVED_MONTH
    frozen.NESO = SNAPSHOT
    frozen._neso_september_shares = october_shares
    dec.main()

    # The shared December runner writes its original September-input metadata;
    # replace only that metadata for this separate October-input run and bind
    # the wrapper implementing the alternative input to the startup hashes.
    status_path = OUT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "SEALED_PROSPECTIVE_GENERATION_US_GB_DEC2026_TMINUS2"
    status["forecast_origin_utc"] = FORECAST_ORIGIN.isoformat()
    status["target_inputs"]["GB"] = {
        "source_month": str(OBSERVED_MONTH), "lag": "t-2",
        "vintage": "current NESO public snapshot; retrospectively revisable",
        "snapshot_sha256": dec.sha(SNAPSHOT),
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    hashes_path = OUT / "startup_hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes[str(Path(__file__).relative_to(ROOT))] = dec.sha(Path(__file__).resolve())
    hashes_path.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    if dec.sha(SNAPSHOT) != json.loads((OUT / "input_audit.json").read_text())[
            "neso_target_input"]["snapshot_sha256"]:
        raise RuntimeError("NESO target input snapshot hash mismatch")


if __name__ == "__main__":
    main()
