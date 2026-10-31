"""Information-resolution ceiling for local positive-EF generation summaries.

This is a retrospective information-value experiment on already inspected
sources, not a forecast. It asks how well a coarse positive-EF group total per
time block constrains hourly CIF when hourly total generation is also known.
Fuel-specific monthly margins and dispatch constraints are deliberately
omitted, so the calculated intervals are sharp only for this reduced contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from identifiability_v15 import FUELS, read_evaluation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/research/data"
OUT = ROOT / "results/v51-coarse-positive-group-info"
REGIONS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
YEARS = (2022, 2023, 2024)
RESOLUTIONS = (0, 24, 12, 6, 3, 1)  # 0 denotes one monthly block.
POSITIVE_EF = (0, 1, 2, 7)
EF_MIN, EF_MAX = 400.0, 1000.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_month(region: str, year: int, month: pd.Period, times: pd.DatetimeIndex,
                generation: np.ndarray) -> list[dict]:
    keep = np.asarray(times.tz_localize(None).to_period("M") == month)
    ts, fuel = times[keep], generation[keep]
    if not len(ts) or len(ts) % 24:
        return []
    total = fuel.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("Expected positive hourly generation")
    positive = fuel[:, POSITIVE_EF].sum(axis=1)
    ef = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
    truth = fuel @ ef / total
    hours = ts.hour.to_numpy()
    day_key = ts.floor("D").asi8
    records = []
    for resolution in RESOLUTIONS:
        if resolution == 0:
            block = np.zeros(len(ts), dtype=np.int64)
        else:
            block = day_key + (hours // resolution).astype(np.int64)
        block_positive = pd.Series(positive).groupby(block, sort=False).transform("sum").to_numpy()
        block_total = pd.Series(total).groupby(block, sort=False).transform("sum").to_numpy()
        # Given block positive-group total P_b and hourly total G_h, the
        # feasible positive amount at h is [max(0,P_b-(G_b-G_h)), min(G_h,P_b)].
        p_lo = np.maximum(0., block_positive - (block_total - total))
        p_hi = np.minimum(total, block_positive)
        lower = EF_MIN * p_lo / total
        upper = EF_MAX * p_hi / total
        if np.any(lower > truth + 1e-8) or np.any(truth > upper + 1e-8):
            raise AssertionError("Observed truth escaped information-set bounds")
        for idx in range(len(ts)):
            records.append({
                "region": region, "year": year, "month": str(month),
                "resolution_hours": resolution, "hour": ts[idx].isoformat(),
                "total_generation": float(total[idx]),
                "positive_group_generation": float(positive[idx]),
                "truth_cif": float(truth[idx]), "lower": float(lower[idx]),
                "upper": float(upper[idx]), "width": float(upper[idx] - lower[idx]),
            })
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty result directory: {OUT}")
    records = []
    input_hashes = {}
    for region in REGIONS:
        path = DATA / f"{region}_hourly.csv"
        input_hashes[str(path)] = sha256(path)
        for year in YEARS:
            days, generation = read_evaluation(path, region, year)
            times = days.repeat(24) + pd.to_timedelta(
                np.tile(np.arange(24), len(days)), unit="h"
            )
            for month in times.tz_localize(None).to_period("M").unique():
                records.extend(score_month(region, year, month, times, generation))
    frame = pd.DataFrame(records)
    raw_path = OUT / "hourly_bounds.csv"
    frame.to_csv(raw_path, index=False)
    by_region = (
        frame.groupby(["resolution_hours", "region"], as_index=False)
        .agg(hours=("width", "size"), mean_width=("width", "mean"),
             median_width=("width", "median"), p90_width=("width", lambda x: x.quantile(.9)),
             truth_sd=("truth_cif", "std"), coverage=("truth_cif", lambda x: 1.0))
    )
    by_region["narrow_width_le100_share"] = frame.groupby(
        ["resolution_hours", "region"]
    )["width"].apply(lambda x: float((x <= 100).mean())).to_numpy()
    by_region.to_csv(OUT / "source_summary.csv", index=False)
    overall = (
        frame.groupby("resolution_hours", as_index=False)
        .agg(hours=("width", "size"), mean_width=("width", "mean"),
             median_width=("width", "median"), p90_width=("width", lambda x: x.quantile(.9)),
             truth_sd=("truth_cif", "std"), truth_coverage=("truth_cif", lambda x: 1.0))
    )
    overall.to_csv(OUT / "overall_summary.csv", index=False)
    summary = {
        "status": "RETROSPECTIVE_COARSE_GROUP_INFORMATION_CEILING",
        "contract": "Known hourly total generation plus exact positive-EF-group generation total per block; block widths 1,3,6,12,24h, or one whole month.",
        "sharpness": "Sharp for the reduced per-hour constraints only: 0<=positive-group output<=hourly total and exact block sum. Omits fuel-specific monthly margins, ramping, capacity and dispatch constraints.",
        "ef_support": [0.0, EF_MIN, EF_MAX],
        "sources": REGIONS,
        "years": YEARS,
        "input_sha256": input_hashes,
        "outputs_sha256": {p.name: sha256(p) for p in (raw_path, OUT / "source_summary.csv", OUT / "overall_summary.csv")},
        "rows": int(len(frame)),
        "interpretation": "Retrospective oracle value of a possible local coarse signal; not currently available Chinese data, a forecast, or an unseen-region evaluation.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(overall.to_string(index=False))
    print("\nBy-source mean interval width:")
    print(by_region.pivot(index="region", columns="resolution_hours", values="mean_width").to_string())


if __name__ == "__main__":
    main()
