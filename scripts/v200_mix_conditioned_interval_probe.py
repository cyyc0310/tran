#!/usr/bin/env python3
"""V200 source-LOO interval probe using monthly fuel-share donor ordering."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v199_china_monthly_mix_shape_probe as v199

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/china-projected-cif-v200-mix-interval-r1"
K = 15
FLOOR = 1e-4
N_BOOT = 5000
SEED = 200


def qhigher(values, q=0.9):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))


def select_order(donors, target_idx, month_idx, target_mix, target_mean, mode):
    if mode == "mix":
        key = lambda i: (v199.bray_curtis(target_mix, v199.FEATURES[(donors[i]["province_idx"], month_idx)]), i)
    elif mode == "level":
        key = lambda i: (abs(donors[i]["monthly_mean"] - target_mean), i)
    else:
        raise ValueError(mode)
    return sorted(range(len(donors)), key=key)[:K]


def profile_error_calibration(donors, target_idx, month_idx, target_mix, mode):
    raw = []
    scaled = []
    for i, pseudo in enumerate(donors):
        pseudo_province = pseudo["province_idx"]
        candidates = [d for d in donors if d["province_idx"] != pseudo_province]
        if len(candidates) < K:
            raise ValueError(f"only {len(candidates)} profiles after strict province LOO")
        pseudo_mix = v199.FEATURES[(pseudo_province, month_idx)]
        order = select_order(candidates, target_idx, month_idx, pseudo_mix, pseudo["monthly_mean"], mode)
        neighbors = np.stack([candidates[j]["shape"] for j in order])
        center = neighbors.mean(axis=0)
        scale = np.maximum(neighbors.std(axis=0, ddof=1), FLOOR)
        residual = pseudo["shape"] - center
        raw.append(np.abs(residual).ravel())
        scaled.append((np.abs(residual) / scale).ravel())
    return qhigher(np.concatenate(raw)), qhigher(np.concatenate(scaled))


def interval_metrics(lower, upper, truth):
    width = upper - lower
    outside = np.maximum(lower - truth, 0) + np.maximum(truth - upper, 0)
    return {
        "coverage90": float(np.mean((truth >= lower) & (truth <= upper))),
        "width90": float(np.mean(width)),
        "winkler90": float(np.mean(width + 20 * outside)),
    }


def digest_json(path):
    return v199.sha256(path)


def make_intervals(donors, scenario, target_idx, month_idx, target_mean, target_mix):
    shapes = np.stack([d["shape"] for d in donors])
    mix_order = select_order(donors, target_idx, month_idx, target_mix, target_mean, "mix")
    level_order = select_order(donors, target_idx, month_idx, target_mix, target_mean, "level")
    mix_neighbors = shapes[mix_order]
    level_neighbors = shapes[level_order]
    q_mix, q_mix_scaled = profile_error_calibration(donors, target_idx, month_idx, target_mix, "mix")
    q_level, q_level_scaled = profile_error_calibration(donors, target_idx, month_idx, target_mix, "level")
    q_all = []
    for pseudo in donors:
        candidates = [d for d in donors if d["province_idx"] != pseudo["province_idx"]]
        residual = pseudo["shape"] - np.mean([d["shape"] for d in candidates], axis=0)
        q_all.append(np.abs(residual).ravel())
    q_all = qhigher(np.concatenate(q_all))

    target_shape_center = target_mean
    mix_center = target_shape_center + mix_neighbors.mean(axis=0)
    level_center = target_shape_center + level_neighbors.mean(axis=0)
    mix_scale = np.maximum(mix_neighbors.std(axis=0, ddof=1), FLOOR)
    level_scale = np.maximum(level_neighbors.std(axis=0, ddof=1), FLOOR)
    all_center = target_shape_center + shapes.mean(axis=0)
    all_law = target_shape_center + shapes
    mix_law = target_shape_center + mix_neighbors

    def symmetric(center, half):
        return center - half, center + half

    intervals = {
        "all30_law": (np.quantile(all_law, .05, axis=0), np.quantile(all_law, .95, axis=0), all_center),
        "mix15_law": (np.quantile(mix_law, .05, axis=0), np.quantile(mix_law, .95, axis=0), mix_center),
        "mix15_pooled": (*symmetric(mix_center, q_mix), mix_center),
        "mix15_scaled": (*symmetric(mix_center, q_mix_scaled * mix_scale), mix_center),
        "level15_pooled": (*symmetric(level_center, q_level), level_center),
        "level15_scaled": (*symmetric(level_center, q_level_scaled * level_scale), level_center),
        "all30_pooled": (*symmetric(all_center, q_all), all_center),
    }
    cal = {"q_mix_raw": q_mix, "q_mix_scaled": q_mix_scaled,
           "q_level_raw": q_level, "q_level_scaled": q_level_scaled,
           "q_all_raw": q_all}
    return intervals, cal, mix_order, level_order


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    provinces, data = v199.read_2025_shapes()
    v199.FEATURES, _ = v199.read_ember_shares(provinces)
    manifest_path = OUT / "prediction_manifest.jsonl"
    manifest_digest = hashlib.sha256()
    queries = []
    with manifest_path.open("w") as f:
        for track in ("same_scenario", "leave_scenario_out"):
            for scenario in v199.SCENARIOS:
                values, bounds = data[scenario]
                for target_idx in range(1, 32):
                    for month_idx, (lo, hi) in enumerate(bounds):
                        target_mean = float(values[lo:hi, target_idx].mean())
                        donors = v199.donor_archive(data, track, scenario, target_idx, month_idx, bounds)
                        mix = v199.FEATURES[(target_idx, month_idx)]
                        intervals, cal, mix_order, level_order = make_intervals(
                            donors, scenario, target_idx, month_idx, target_mean, mix
                        )
                        hashes = {m: {"lower": v199.digest(a), "upper": v199.digest(b), "center": v199.digest(c)}
                                  for m, (a, b, c) in intervals.items()}
                        record = {
                            "track": track, "scenario": scenario, "province": provinces[target_idx],
                            "month": month_idx + 1, "target_month_mean_oracle": target_mean,
                            "donor_archive": [{"scenario": d["scenario"], "province": provinces[d["province_idx"]]}
                                              for d in donors],
                            "mix_neighbors": [provinces[donors[i]["province_idx"]] for i in mix_order],
                            "level_neighbors": [provinces[donors[i]["province_idx"]] for i in level_order],
                            "calibration": cal, "interval_hashes": hashes,
                        }
                        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                        f.write(line)
                        manifest_digest.update(line.encode())
                        queries.append((track, scenario, target_idx, month_idx, lo, hi, intervals, hashes))
    seal = {
        "status": "SEALED_BEFORE_SCORING_TARGET_HOURS",
        "protocol_sha256": digest_json(ROOT / "docs/V200_MIX_CONDITIONED_INTERVAL_PROBE_PROTOCOL.md"),
        "runner_sha256": digest_json(Path(__file__)),
        "v199_runner_sha256": digest_json(ROOT / "scripts/v199_china_monthly_mix_shape_probe.py"),
        "ember_sha256": digest_json(v199.EMBER),
        "scenario_workbook_sha256": {s: digest_json(v199.V194 / f"CEF data for Scenario {s}.xlsx") for s in v199.SCENARIOS},
        "query_count": len(queries), "manifest_sha256": manifest_digest.hexdigest(),
        "k": K, "scale_floor": FLOOR,
        "scope": "retrospective development on previously opened China model-world labels; oracle target monthly means",
    }
    (OUT / "prediction_seal.json").write_text(json.dumps(seal, indent=2, ensure_ascii=False) + "\n")

    scores = []
    for track, scenario, target_idx, month_idx, lo, hi, intervals, hashes in queries:
        values, _bounds = data[scenario]
        truth = values[lo:hi, target_idx]
        target_mean = float(truth.mean())
        for method, (lower, upper, center) in intervals.items():
            observed_hash = {"lower": v199.digest(lower), "upper": v199.digest(upper), "center": v199.digest(center)}
            if hashes[method] != observed_hash:
                raise AssertionError("prediction seal mismatch")
            row = {"track": track, "scenario": scenario, "province": provinces[target_idx],
                   "month": month_idx + 1, "method": method, **interval_metrics(lower, upper, truth)}
            row["shape_mae"] = float(np.mean(np.abs((center - target_mean) - (truth - target_mean))))
            scores.append(row)
    frame = pd.DataFrame(scores)
    frame.to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.10g")
    metrics = ["coverage90", "width90", "winkler90", "shape_mae"]
    frame.groupby(["track", "method"])[metrics].mean().reset_index().to_csv(OUT / "summary_macro.csv", index=False)
    frame.groupby(["track", "scenario", "method"])[metrics].mean().reset_index().to_csv(OUT / "summary_scenario.csv", index=False)
    frame.groupby(["track", "province", "method"])[metrics].mean().reset_index().to_csv(OUT / "summary_province.csv", index=False)

    compare = []
    rng = np.random.default_rng(SEED)
    for track in ("same_scenario", "leave_scenario_out"):
        sub = frame[frame.track == track]
        wide = sub.pivot(index=["scenario", "province", "month"], columns="method", values=metrics)
        provinces_scored = sorted(wide.index.get_level_values("province").unique())
        for baseline in ("level15_scaled", "all30_pooled", "all30_law", "mix15_pooled"):
            for metric in ("coverage90", "width90", "winkler90"):
                delta = wide[(metric, "mix15_scaled")] - wide[(metric, baseline)]
                province_delta = np.array([delta.xs(p, level="province").mean() for p in provinces_scored])
                draws = rng.integers(0, len(provinces_scored), size=(N_BOOT, len(provinces_scored)))
                boot = province_delta[draws].mean(axis=1)
                compare.append({"track": track, "comparison": f"mix15_scaled_minus_{baseline}", "metric": metric,
                               "paired_mean_delta": float(delta.mean()), "province_macro_delta": float(province_delta.mean()),
                               "cluster_bootstrap_low": float(np.quantile(boot, .025)),
                               "cluster_bootstrap_high": float(np.quantile(boot, .975)),
                               "n_queries": len(delta), "n_provinces": len(provinces_scored),
                               "bootstrap_replicates": N_BOOT, "seed": SEED})
    pd.DataFrame(compare).to_csv(OUT / "paired_province_bootstrap.csv", index=False, float_format="%.10g")
    print(pd.read_csv(OUT / "summary_macro.csv").to_string(index=False), flush=True)
    print(pd.read_csv(OUT / "paired_province_bootstrap.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
