"""V233: audit within-month provincial shape inheritance in CM-China daily Power."""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path("/tmp/carbonmonitor_china_v233_live.csv")
PROTOCOL = ROOT / "docs/V233_CARBONMONITOR_CHINA_DAILY_SHAPE_AUDIT_PROTOCOL.md"
OUT = ROOT / "results/v233-carbonmonitor-china-daily-shape-audit-r2"
SOURCE_URL = "https://datas.carbonmonitor.org/API/downloadFullDataset.php?source=carbon_china"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    input_hash = sha256(INPUT)
    protocol_hash = sha256(PROTOCOL)
    raw = pd.read_csv(INPUT, usecols=["state", "date", "sector", "value"])
    raw = raw.loc[raw.sector.eq("Power")].copy()
    raw["date"] = pd.to_datetime(raw.date, format="%d/%m/%Y", errors="coerce")
    raw["value"] = pd.to_numeric(raw.value, errors="coerce")
    raw = raw.dropna(subset=["state", "date", "value"])
    raw = raw.loc[raw.value.ge(0)]
    raw["year"] = raw.date.dt.year
    raw["month"] = raw.date.dt.month
    if raw.duplicated(["state", "date"]).any():
        raise RuntimeError("duplicate state-date Power rows")

    month_rows = []
    coverage_rows = []
    for (year, month), block in raw.groupby(["year", "month"], sort=True):
        expected_dates = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month)
        pivot = block.pivot(index="state", columns="date", values="value").reindex(columns=expected_dates)
        coverage = pivot.notna().sum(axis=1) / len(expected_dates)
        for state, fraction in coverage.items():
            coverage_rows.append({"year": int(year), "month": int(month), "state": state,
                                  "expected_days": len(expected_dates), "observed_days": int(pivot.loc[state].notna().sum()),
                                  "coverage": float(fraction)})
        states = coverage.index[coverage.ge(.90)]
        if len(states) < 10:
            continue
        # Align on common days to avoid missingness creating artificial shape differences.
        common = pivot.loc[states].dropna(axis=1, how="any")
        if common.shape[1] < .90 * len(expected_dates):
            continue
        matrix = common.to_numpy(float)
        totals = matrix.sum(axis=1)
        keep = totals > 0
        matrix, states, totals = matrix[keep], states[keep], totals[keep]
        if len(states) < 10:
            continue
        profiles = matrix / totals[:, None]
        profile_mean = profiles.mean(axis=0)
        l1 = [float(np.abs(profiles[i] - profiles[j]).sum())
              for i, j in combinations(range(len(profiles)), 2)]
        correlations = []
        for i, j in combinations(range(len(profiles)), 2):
            a, b = profiles[i], profiles[j]
            if np.std(a) > 0 and np.std(b) > 0:
                correlations.append(float(np.corrcoef(a, b)[0, 1]))
        reconstructed = totals[:, None] * profile_mean[None, :]
        rmse = float(np.sqrt(np.mean((matrix - reconstructed) ** 2)))
        scale = float(np.sqrt(np.mean(matrix ** 2)))
        month_rows.append({"year": int(year), "month": int(month), "n_provinces": int(len(states)),
            "n_common_days": int(common.shape[1]), "mean_pairwise_profile_corr": float(np.mean(correlations)) if correlations else np.nan,
            "min_pairwise_profile_corr": float(np.min(correlations)) if correlations else np.nan,
            "mean_pairwise_normalized_l1": float(np.mean(l1)),
            "max_abs_profile_deviation_from_mean": float(np.max(np.abs(profiles - profile_mean))),
            "rank1_relative_rmse": rmse / scale if scale else 0.})
    month_df = pd.DataFrame(month_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    if len(month_df) < 12:
        raise RuntimeError(f"insufficient complete month blocks: {len(month_df)}")
    OUT.mkdir(parents=True)
    month_df.to_csv(OUT / "month_shape_metrics.csv", index=False)
    coverage_df.to_csv(OUT / "province_month_coverage.csv", index=False)
    by_year = month_df.groupby("year", as_index=False).agg(
        n_months=("month", "count"), mean_pairwise_profile_corr=("mean_pairwise_profile_corr", "mean"),
        min_pairwise_profile_corr=("min_pairwise_profile_corr", "min"),
        mean_pairwise_normalized_l1=("mean_pairwise_normalized_l1", "mean"),
        max_abs_profile_deviation_from_mean=("max_abs_profile_deviation_from_mean", "max"),
        rank1_relative_rmse=("rank1_relative_rmse", "mean"))
    by_year.to_csv(OUT / "year_shape_summary.csv", index=False)
    province_coverage = coverage_df.groupby("state", as_index=False).agg(
        mean_month_coverage=("coverage", "mean"), min_month_coverage=("coverage", "min"),
        complete_months=("coverage", lambda s: int(np.sum(s == 1.))))
    province_coverage.to_csv(OUT / "province_coverage_summary.csv", index=False)
    metadata = {"status": "snapshot_lineage_audit_not_CIF_validation", "source_snapshot": str(INPUT),
        "source_url": SOURCE_URL, "retrieved_date": "2026-10-30",
        "input_sha256": input_hash, "protocol_sha256": protocol_hash,
        "date_min": str(raw.date.min().date()), "date_max": str(raw.date.max().date()),
        "rows_power": int(len(raw)), "province_count": int(raw.state.nunique()),
        "month_blocks_analyzed": int(len(month_df)), "coverage_threshold": .90,
        "label_caveat": "Carbon Monitor China daily sector emissions estimate; revisable, not direct hourly CIF measurement"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(by_year.to_string(index=False))
    print(f"month_blocks={len(month_df)} rows={len(raw)} output={OUT}")


if __name__ == "__main__":
    main()
