#!/usr/bin/env python3
"""Nested country-held-out test of CN5 composition-weighted conformal intervals."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import run_cn5_factor_expert_transfer as base


ROOT = base.ROOT
PROTOCOL = ROOT / "docs/CN5_COMPOSITION_WEIGHTED_CONFORMAL_PROTOCOL_20261101.md"
BANDWIDTHS = (0.25, 0.5, 1.0, 2.0, math.inf)
ALPHA = 0.10


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def country_features(prepared: dict, countries: list[str], year: int) -> pd.DataFrame:
    return pd.concat([
        prepared[c][1].loc[prepared[c][1].year.eq(year)].assign(country=c)
        for c in countries
    ], ignore_index=True)


def fit_ratio(prepared: dict, countries: list[str], query: pd.DataFrame) -> np.ndarray:
    model = base.source_fit(prepared, countries)
    return np.asarray(base.MODEL.predict(model, query)["ratio_recomposition"], dtype=float)


def weighted_conformal_radius(
    calibration: pd.DataFrame,
    query_mix: np.ndarray,
    scales: np.ndarray,
    bandwidth: float,
) -> tuple[float, float]:
    """Equal-country, same-calendar-month weighted conformal radius."""
    if calibration.empty:
        raise ValueError("empty calibration set")
    mix = calibration[list(base.MODEL.GROUPS)].to_numpy(float)
    distance = np.mean(((mix - query_mix[None, :]) / scales[None, :]) ** 2, axis=1)
    if np.isinf(bandwidth):
        kernel = np.ones(len(calibration), dtype=float)
    else:
        kernel = np.exp(-distance / (2.0 * bandwidth * bandwidth))
    country_sizes = calibration.groupby("country").size().to_dict()
    n_countries = len(country_sizes)
    equal_country = np.array([
        1.0 / (n_countries * country_sizes[c]) for c in calibration.country
    ], dtype=float)
    weights = equal_country * kernel
    mean_weight = float(weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        return math.inf, 0.0
    weights /= mean_weight
    effective_n = float(weights.sum() ** 2 / np.square(weights).sum())

    residual = calibration.abs_error.to_numpy(float)
    order = np.argsort(residual, kind="mergesort")
    ordered_residual = residual[order]
    ordered_weights = weights[order]
    # The test point carries unit mass at +infinity, as in weighted conformal.
    cutoff = (1.0 - ALPHA) * (float(weights.sum()) + 1.0)
    cumulative = np.cumsum(ordered_weights)
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    if idx >= len(ordered_residual):
        return math.inf, effective_n
    return float(ordered_residual[idx]), effective_n


def calibration_for_outer(
    prepared: dict, sources: list[str], target: str,
) -> dict[str, pd.DataFrame]:
    residuals: dict[str, pd.DataFrame] = {}
    for pseudo in sources:
        train_countries = [c for c in sources if c != pseudo]
        raw, features = prepared[pseudo]
        query = features.loc[features.year.eq(2023)].copy()
        point = fit_ratio(prepared, train_countries, query)
        truth = base.truth_for(raw, query)
        residuals[pseudo] = query.assign(
            country=pseudo,
            abs_error=np.abs(truth - point),
            month=query.clock.dt.month.to_numpy(int),
        )
    return residuals


def radius_by_month(
    calibration: dict[str, pd.DataFrame],
    query: pd.DataFrame,
    exclude_country: str | None,
    scales: np.ndarray,
    bandwidth: float,
) -> tuple[dict[int, float], dict[int, float]]:
    radii, effective = {}, {}
    for month in sorted(query.clock.dt.month.unique()):
        month_cal = [
            frame.loc[frame.month.eq(month)]
            for country, frame in calibration.items()
            if country != exclude_country
        ]
        month_cal = [frame for frame in month_cal if not frame.empty]
        joined = pd.concat(month_cal, ignore_index=True)
        month_query = query.loc[query.clock.dt.month.eq(month)]
        query_mix = month_query[list(base.MODEL.GROUPS)].mean().to_numpy(float)
        radii[int(month)], effective[int(month)] = weighted_conformal_radius(
            joined, query_mix, scales, bandwidth
        )
    return radii, effective


def interval_metrics(y: np.ndarray, p: np.ndarray, radius: float) -> dict[str, float]:
    if not np.isfinite(radius):
        return {"coverage90": 1.0, "width90": math.inf, "winkler90": math.inf}
    lo, hi = p - radius, p + radius
    below, above = np.maximum(lo - y, 0), np.maximum(y - hi, 0)
    return {
        "coverage90": float(np.mean((y >= lo) & (y <= hi))),
        "width90": float(np.mean(hi - lo)),
        "winkler90": float(np.mean((hi - lo) + (2.0 / ALPHA) * (below + above))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    countries = list(base.COUNTRIES)
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(base.__file__).resolve(),
                  Path(base.cn5.__file__).resolve(), Path(base.DATA_MODULE.__file__).resolve(),
                  Path(base.DATA_MODULE.six_group.__file__).resolve(),
                  Path(base.DATA_MODULE.six_group.base.__file__).resolve(),
                  Path(base.MODEL.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    old_calibration_path = ROOT / "results/cn5-factor-expert-transfer-r1/source_calibration.json"
    old_calibration = {
        row["target"].upper(): float(row["radii90"]["ratio_recomposition"])
        for row in json.loads(old_calibration_path.read_text(encoding="utf-8"))
    }
    input_paths = [PROTOCOL, old_calibration_path]
    for code in countries:
        folder, manifest_path, _ = base.DATA_MODULE.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in base.DATA_MODULE.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    (out / "startup_hashes.json").write_text(json.dumps({
        "input_sha256": input_hashes, "code_sha256": code_hashes,
    }, indent=2) + "\n")

    prepared, audits = {}, {}
    for country in countries:
        raw, features, audits[country] = base.DATA_MODULE.prepare_country(country)
        prepared[country] = raw, features
    (out / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")

    sealed_rows, seals, tuning_rows = [], [], []
    prediction_frames: dict[str, pd.DataFrame] = {}
    for target in countries:
        sources = [c for c in countries if c != target]
        train22 = country_features(prepared, sources, 2022)
        scales = np.maximum(train22[list(base.MODEL.GROUPS)].std(ddof=0).to_numpy(float), 0.02)
        crossfit = calibration_for_outer(prepared, sources, target)

        # Nested source-only bandwidth selection: validate every pseudo-country
        # on 2023, calibrating from other pseudo-countries in the same month.
        score_rows = []
        for pseudo in sources:
            raw, features = prepared[pseudo]
            query = features.loc[features.year.eq(2023)].copy()
            query["month"] = query.clock.dt.month.to_numpy(int)
            point = fit_ratio(prepared, [c for c in sources if c != pseudo], query)
            truth = base.truth_for(raw, query)
            for bandwidth in BANDWIDTHS:
                radii, eff = radius_by_month(crossfit, query, pseudo, scales, bandwidth)
                for month in sorted(query.month.unique()):
                    mask = query.month.eq(month).to_numpy()
                    metrics = interval_metrics(truth[mask], point[mask], radii[int(month)])
                    score_rows.append({
                        "pseudo_target": pseudo, "month": int(month),
                        "bandwidth": "inf" if np.isinf(bandwidth) else bandwidth,
                        **metrics, "effective_n": eff[int(month)],
                    })
        score_frame = pd.DataFrame(score_rows)
        score_frame.to_csv(out / f"{target}_inner_bandwidth_scores.csv", index=False)
        by_pseudo = score_frame.groupby(["bandwidth", "pseudo_target"], as_index=False).agg(
            coverage90=("coverage90", "mean"),
            winkler90=("winkler90", "mean"),
            width90=("width90", "mean"),
            median_effective_n=("effective_n", "median"),
        )
        summaries = by_pseudo.groupby("bandwidth", as_index=False).agg(
            coverage90=("coverage90", "mean"),
            winkler90=("winkler90", "mean"),
            width90=("width90", "mean"),
            worst_country_coverage=("coverage90", "min"),
            median_effective_n=("median_effective_n", "median"),
        )
        pooled = summaries.loc[summaries.bandwidth.eq("inf")].iloc[0]
        min_coverage = max(0.88, float(pooled.coverage90) - 0.02)
        eligible = summaries.loc[
            summaries.coverage90.ge(min_coverage)
            & summaries.worst_country_coverage.ge(0.75)
        ]
        if eligible.empty:
            selected = "inf"
        else:
            selected = str(eligible.sort_values("winkler90", kind="mergesort").iloc[0].bandwidth)
        tuning_rows.append({
            "outer_target": target, "selected_bandwidth": selected,
            "pooled_inner_coverage": float(pooled.coverage90),
            "pooled_inner_winkler": float(pooled.winkler90),
            "selected_inner_coverage": float(summaries.loc[summaries.bandwidth.eq(selected), "coverage90"].iloc[0]),
            "selected_inner_winkler": float(summaries.loc[summaries.bandwidth.eq(selected), "winkler90"].iloc[0]),
            "selected_inner_width": float(summaries.loc[summaries.bandwidth.eq(selected), "width90"].iloc[0]),
            "selected_inner_worst_country_coverage": float(summaries.loc[summaries.bandwidth.eq(selected), "worst_country_coverage"].iloc[0]),
            "selected_inner_median_effective_n": float(summaries.loc[summaries.bandwidth.eq(selected), "median_effective_n"].iloc[0]),
        })

        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        center = fit_ratio(prepared, sources, query)
        pooled_radii, pooled_eff = radius_by_month(crossfit, query, None, scales, math.inf)
        bw = math.inf if selected == "inf" else float(selected)
        weighted_radii, weighted_eff = radius_by_month(crossfit, query, None, scales, bw)
        saved = pd.DataFrame({
            "time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
            "ratio_recomposition": center,
            "month": query.clock.dt.month.to_numpy(int),
            "selected_bandwidth": selected,
            "pooled_radius": [pooled_radii[m] for m in query.clock.dt.month.to_numpy(int)],
            "weighted_radius": [weighted_radii[m] for m in query.clock.dt.month.to_numpy(int)],
            "pooled_effective_n": [pooled_eff[m] for m in query.clock.dt.month.to_numpy(int)],
            "weighted_effective_n": [weighted_eff[m] for m in query.clock.dt.month.to_numpy(int)],
            "old_h2_radius": old_calibration[target.upper()],
        })
        saved["pooled_lo"] = saved.ratio_recomposition - saved.pooled_radius
        saved["pooled_hi"] = saved.ratio_recomposition + saved.pooled_radius
        saved["weighted_lo"] = saved.ratio_recomposition - saved.weighted_radius
        saved["weighted_hi"] = saved.ratio_recomposition + saved.weighted_radius
        saved["old_h2_lo"] = saved.ratio_recomposition - saved.old_h2_radius
        saved["old_h2_hi"] = saved.ratio_recomposition + saved.old_h2_radius
        saved["old_h2_effective_n"] = np.nan
        path = out / f"{target}_heldout_predictions.csv.gz"
        saved.to_csv(path, index=False, compression="gzip")
        prediction_frames[target] = saved
        seals.append({"target": target, "path": path.name, "sha256": sha(path), "rows": len(saved)})
        print(f"sealed CN5 weighted-conformal target {target}: {len(saved)} hours; h={selected}", flush=True)

    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    pd.DataFrame(tuning_rows).to_csv(out / "bandwidth_selection.csv", index=False)
    # Verify outer predictions are frozen before reading their 2024 labels.
    for row in seals:
        if sha(out / row["path"]) != row["sha256"]:
            raise RuntimeError(f"prediction seal mismatch before scoring: {row['target']}")

    # Score only after all outer predictions are sealed.
    metric_rows = []
    for target in countries:
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        truth = base.truth_for(raw, query)
        saved = pd.read_csv(out / f"{target}_heldout_predictions.csv.gz")
        if len(saved) != len(truth):
            raise RuntimeError(f"target alignment mismatch for {target}")
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            for method in ("pooled", "weighted", "old_h2"):
                lo, hi = saved[f"{method}_lo"].to_numpy(float)[mask], saved[f"{method}_hi"].to_numpy(float)[mask]
                y, p = truth[mask], saved.ratio_recomposition.to_numpy(float)[mask]
                width = hi - lo
                score = width + (2.0 / ALPHA) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
                metric_rows.append({
                    "target_country": target, "month": int(month), "method": method,
                    "hours": int(mask.sum()), "coverage90": float(np.mean((y >= lo) & (y <= hi))),
                    "width90": float(np.mean(width)), "winkler90": float(np.mean(score)),
                    "radius": float(np.mean(width) / 2), "selected_bandwidth": saved.selected_bandwidth.iloc[0],
                    "median_effective_n": float(saved[f"{method}_effective_n"].to_numpy(float)[mask].mean()),
                })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "monthly_metrics.csv", index=False)
    country = metrics.groupby(["target_country", "method"], as_index=False).agg(
        coverage90=("coverage90", "mean"), width90=("width90", "mean"),
        winkler90=("winkler90", "mean"), median_effective_n=("median_effective_n", "median"),
    )
    country.to_csv(out / "country_metrics.csv", index=False)
    country.groupby("method", as_index=False).agg(
        coverage90=("coverage90", "mean"), width90=("width90", "mean"),
        winkler90=("winkler90", "mean"), worst_country_coverage=("coverage90", "min"),
    ).to_csv(out / "macro_metrics.csv", index=False)

    for path, expected in input_hashes.items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f"input changed during run: {path}")
    for path, expected in code_hashes.items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f"code changed during run: {path}")
    if any(sha(out / row["path"]) != row["sha256"] for row in seals):
        raise RuntimeError("prediction seal mismatch after target scoring")
    print(f"completed CN5 composition-weighted conformal: {len(metrics)} country-month rows", flush=True)


if __name__ == "__main__":
    main()
