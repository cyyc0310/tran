"""Joint LP bounds with monthly fuel inventories and block positive-group totals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from identifiability_v15 import FUELS, read_evaluation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/research/data"
OUT = ROOT / "results/v51-joint-month-hour-bounds"
REGIONS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
YEAR = 2024
MONTHS = (1, 2, 3, 4, 5, 6)
POSITIVE = np.array([0, 1, 2, 7])
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
BLOCK_HOURS = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lp_matrices(generation: np.ndarray) -> tuple:
    n, fuels = generation.shape
    nblocks = n // BLOCK_HOURS
    rows = n + fuels + nblocks
    A = lil_matrix((rows, n * fuels), dtype=float)
    b = np.r_[generation.sum(axis=1), generation.sum(axis=0),
              generation.reshape(nblocks, BLOCK_HOURS, fuels)[:, :, POSITIVE].sum(axis=(1, 2))]
    for h in range(n):
        A[h, h * fuels:(h + 1) * fuels] = 1.
    for f in range(fuels):
        A[n + f, f::fuels] = 1.
    for block in range(nblocks):
        start = block * BLOCK_HOURS
        for h in range(start, start + BLOCK_HOURS):
            for f in POSITIVE:
                A[n + fuels + block, h * fuels + f] = 1.
    return A.tocsr(), b


def relaxed_width(generation: np.ndarray, hour: int) -> float:
    total = generation.sum(axis=1)
    positive = generation[:, POSITIVE].sum(axis=1)
    block = hour // BLOCK_HOURS
    idx = np.arange(block * BLOCK_HOURS, block * BLOCK_HOURS + BLOCK_HOURS)
    p_block = float(positive[idx].sum())
    g_block = float(total[idx].sum())
    p_lo = max(0., p_block - (g_block - total[hour]))
    p_hi = min(total[hour], p_block)
    return 1000. * p_hi / total[hour] - 400. * p_lo / total[hour]


def main() -> None:
    global BLOCK_HOURS
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty result directory: {OUT}")
    records = []
    input_hashes = {}
    for region in REGIONS:
        path = DATA / f"{region}_hourly.csv"
        input_hashes[str(path)] = sha256(path)
        days, all_generation = read_evaluation(path, region, YEAR)
        all_times = days.repeat(24) + pd.to_timedelta(np.tile(np.arange(24), len(days)), unit="h")
        for month_no in MONTHS:
            period = pd.Period(f"{YEAR}-{month_no:02d}", freq="M")
            keep = np.asarray(all_times.tz_localize(None).to_period("M") == period)
            times = all_times[keep]
            month_generation = all_generation[keep]
            if len(times) == 0 or len(times) % 24:
                continue
            # Score one predeclared UTC day per month: the first complete day
            # whose calendar day is between 13 and 17 inclusive.
            dates = pd.DatetimeIndex(times.floor("D").unique())
            candidates = [d for d in dates if 13 <= d.day <= 17]
            if not candidates:
                continue
            eval_day = candidates[0]
            day_rows = np.flatnonzero(times.floor("D") == eval_day)
            day_rows = [h for h in day_rows if times[h].hour in (0, 6, 12, 18)]
            n, nf = month_generation.shape
            for block_hours in (6, 1):
                BLOCK_HOURS = block_hours
                A, b = lp_matrices(month_generation)
                for h in day_rows:
                    c = np.zeros(n * nf)
                    c[h * nf:(h + 1) * nf] = EF
                    low = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
                    high = linprog(-c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
                    if not low.success or not high.success:
                        raise RuntimeError(f"LP failure {region} {period} hour={times[h]}: {low.message}; {high.message}")
                    total = float(month_generation[h].sum())
                    truth = float(month_generation[h] @ EF / total)
                    lower, upper = float(low.fun / total), float(-high.fun / total)
                    if lower > truth + 1e-6 or truth > upper + 1e-6:
                        raise AssertionError("Truth outside joint LP bounds")
                    records.append({
                        "region": region, "month": str(period), "hour": times[h].isoformat(),
                        "block_hours": block_hours,
                        "truth": truth, "joint_lower": lower, "joint_upper": upper,
                        "joint_width": upper - lower,
                        "coarse_only_width": relaxed_width(month_generation, h),
                    })
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "sampled_hour_bounds.csv", index=False)
    per_region = frame.groupby(["block_hours", "region"], as_index=False).agg(
        hours=("joint_width", "size"), joint_width=("joint_width", "mean"),
        coarse_only_width=("coarse_only_width", "mean"),
        truth_sd=("truth", "std"),
        truth_inside=("truth", lambda x: 1.0),
    )
    per_region["joint_vs_coarse_reduction"] = 1. - per_region.joint_width / per_region.coarse_only_width
    per_region.to_csv(OUT / "source_summary.csv", index=False)
    overall = frame.groupby("block_hours", as_index=False).agg(
        hours=("joint_width", "size"), months=("month", "nunique"),
        sources=("region", "nunique"), joint_width=("joint_width", "mean"),
        coarse_only_width=("coarse_only_width", "mean"), truth_sd=("truth", "std"),
        truth_inside_fraction=("truth", lambda x: 1.0),
    )
    overall["joint_reduction"] = 1. - overall.joint_width / overall.coarse_only_width
    overall.to_csv(OUT / "overall_summary.csv", index=False)
    outputs = ("sampled_hour_bounds.csv", "source_summary.csv", "overall_summary.csv")
    summary = {
        "status": "RETROSPECTIVE_JOINT_MONTHLY_AND_BLOCK_IDENTIFICATION",
        "contract": "Exact monthly totals for eight fuels, hourly total generation, and exact positive-EF-group totals per 6-hour or 1-hour block. Score four paired clock hours (00, 06, 12, 18 UTC) on the first complete UTC date between day 13 and 17 in each source-month.",
        "sources": REGIONS, "year": YEAR, "months": MONTHS,
        "input_sha256": input_hashes,
        "outputs_sha256": {name: sha256(OUT / name) for name in outputs},
        "rows": len(frame),
        "interpretation": "Exact LP bounds for the stated finite information contract; target-month margins and six-hour totals are retrospective oracles, not forecast-time available inputs. No ramp, capacity, imports, or EF uncertainty.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(overall.to_string(index=False))
    print(per_region.to_string(index=False))


if __name__ == "__main__":
    main()
