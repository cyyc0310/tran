"""Block-bootstrap descriptive uncertainty for the five-country factor-head increment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/five-group-capacity-increment-r2/monthly_metrics.csv"
METRICS = ("mae", "shape_mae", "coverage90", "width90", "interval_score90")
BLOCK_LENGTHS = (1, 2, 3, 4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_matrix(frame: pd.DataFrame, metric: str) -> tuple[list[str], np.ndarray]:
    factor = frame.loc[frame.method.eq("M_factor")].pivot(
        index="target_country", columns="month", values=metric
    )
    recomposition = frame.loc[frame.method.eq("M_five_group_recomposition")].pivot(
        index="target_country", columns="month", values=metric
    )
    factor, recomposition = factor.sort_index().sort_index(axis=1), recomposition.sort_index().sort_index(axis=1)
    if factor.shape != (5, 12) or not factor.index.equals(recomposition.index) or not factor.columns.equals(recomposition.columns):
        raise ValueError("expected the same five countries and all 12 paired months in both arms")
    if factor.isna().any().any() or recomposition.isna().any().any():
        raise ValueError("paired metric grid contains missing cells")
    return list(factor.index), (factor - recomposition).to_numpy(float)


def block_indices(rng: np.random.Generator, n_months: int, length: int) -> np.ndarray:
    months: list[int] = []
    while len(months) < n_months:
        start = int(rng.integers(0, n_months))
        months.extend((start + offset) % n_months for offset in range(length))
    return np.asarray(months[:n_months], dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=17023)
    args = parser.parse_args()
    if args.replicates < 1000:
        raise ValueError("use at least 1,000 bootstrap replicates")
    frame = pd.read_csv(args.input)
    countries = sorted(frame.loc[frame.method.eq("M_factor"), "target_country"].unique().tolist())
    output_rows: list[dict] = []
    country_rows: list[dict] = []
    for metric_index, metric in enumerate(METRICS):
        country_names, delta = paired_matrix(frame, metric)
        if country_names != countries:
            raise ValueError("country roster changed during paired extraction")
        for country, value in zip(country_names, delta.mean(axis=1)):
            country_rows.append({"metric": metric, "target_country": country,
                                 "factor_minus_recomposition": float(value)})
        estimate = float(delta.mean())
        for block_length in BLOCK_LENGTHS:
            rng = np.random.default_rng(args.seed + metric_index * 101 + block_length)
            draws = np.empty(args.replicates, dtype=float)
            for replicate in range(args.replicates):
                country_ix = rng.integers(0, len(country_names), size=len(country_names))
                month_ix = block_indices(rng, delta.shape[1], block_length)
                draws[replicate] = delta[np.ix_(country_ix, month_ix)].mean()
            low, high = np.quantile(draws, [0.025, 0.975])
            output_rows.append({
                "metric": metric,
                "block_length_months": block_length,
                "factor_minus_recomposition": estimate,
                "bootstrap_2p5": float(low),
                "bootstrap_97p5": float(high),
                "fraction_draws_below_zero": float(np.mean(draws < 0)),
                "replicates": args.replicates,
                "seed": args.seed + metric_index * 101 + block_length,
                "unit_note": "coverage values are fractions; multiply differences by 100 for percentage points",
            })
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(output_rows).to_csv(args.output / "bootstrap_summary.csv", index=False)
    pd.DataFrame(country_rows).to_csv(args.output / "country_effects.csv", index=False)
    summary = {
        "analysis": "paired factor-head versus five-group recomposition block bootstrap",
        "source_metrics": str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else str(args.input),
        "source_metrics_sha256": sha256(args.input),
        "target_countries": countries,
        "target_period": "2024-01 through 2024-12",
        "resampling": "two-way country cluster resampling and circular contiguous month blocks",
        "block_lengths_months": list(BLOCK_LENGTHS),
        "replicates": args.replicates,
        "base_seed": args.seed,
        "posthoc_development_diagnostic": True,
        "does_not_retrain_or_select_model": True,
        "limits": [
            "only five previously viewed countries and one target year",
            "bootstrap intervals are descriptive and do not establish generalization to unseen countries",
            "monthly aggregates cannot fully model serial dependence within months or uncertainty in the model-fitting sample",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "provenance.json").write_text(json.dumps({
        str(args.input.resolve()): sha256(args.input),
        str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(output_rows).query("metric in ['mae', 'coverage90', 'interval_score90']")
          .to_string(index=False))


if __name__ == "__main__":
    main()
