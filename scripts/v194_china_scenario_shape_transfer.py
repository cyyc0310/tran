#!/usr/bin/env python3
"""V194 model-world stress test: target monthly CIF oracle + donor hourly shapes.

The prediction API accepts target monthly means and donor profiles only. Target
hourly values are passed to the scoring API after prediction digests are sealed.
This is a simulator-world diagnostic, not observed-China validation.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/china-projected-cif-v194-shape-transfer-r1"
OUT = ROOT / "results/china-projected-cif-v194-shape-results-r3"
SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
ALL_YEARS = [2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
# Complete-case window frozen before any target-hour scores were computed.
YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
EXPECTED_PROVINCES = ["Mainland China", "Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia", "Liaoning", "Jilin", "Heilongjiang", "Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian", "Jiangxi", "Shandong", "Henan", "Hubei", "Hunan", "Guangdong", "Guangxi", "Hainan", "Chongqing", "Sichuan", "Guizhou", "Yunnan", "Tibet", "Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang"]
METHODS = ["constant_anchor", "all30_mean_shape", "all30_shape_law", "nearest5_mean_shape", "nearest5_shape_law", "permuted30_shape_law"]


def read_data():
    data = {}
    for si, scenario in enumerate(SCENARIOS):
        path = DATA / f"CEF data for Scenario {scenario}.xlsx"
        wb = load_workbook(path, read_only=True, data_only=True)
        if [str(y) for y in ALL_YEARS] != wb.sheetnames:
            raise ValueError(f"unexpected sheets in {path}: {wb.sheetnames}")
        for year in YEARS:
            ws = wb[str(year)]
            rows = ws.iter_rows(values_only=True)
            header = next(rows)
            if tuple(header) != ("Time", *EXPECTED_PROVINCES):
                raise ValueError(f"unexpected header {scenario}/{year}")
            arr = np.asarray(list(rows), dtype=np.float64)
            hours = 8784 if calendar.isleap(year) else 8760
            if arr.shape != (hours, len(header)) or not np.array_equal(arr[:, 0], np.arange(1, hours + 1)):
                raise ValueError(f"invalid rows/time {scenario}/{year}: {arr.shape}")
            vals = arr[:, 1:]
            if not np.isfinite(vals).all():
                raise ValueError(f"non-finite CIF {scenario}/{year}")
            # Month slices use Gregorian calendar boundaries from hour 1 at Jan 1 00:00.
            bounds, cursor = [], 0
            for month in range(1, 13):
                n = calendar.monthrange(year, month)[1] * 24
                bounds.append((cursor, cursor + n))
                cursor += n
            if cursor != hours:
                raise AssertionError((year, cursor, hours))
            data[(scenario, year)] = (vals, bounds)
        wb.close()
        print(f"loaded {scenario}", flush=True)
    return data


def donor_month_profile(data, scenario, year, province_idx, bounds, month_idx):
    vals, _ = data[(scenario, year)]
    lo, hi = bounds[month_idx]
    x = vals[lo:hi, province_idx]
    mu = float(x.mean())
    return mu, x - mu


def make_donors(data, track, scenario, year, target_idx, bounds, month_idx, target_mu):
    # Returns a fixed-budget list: (scenario, province, monthly_mean, centered shape).
    if track == "same_scenario":
        source_scenarios = [scenario]
        budget = [(scenario, p) for p in range(1, len(EXPECTED_PROVINCES)) if p != target_idx]
    else:
        source_scenarios = [s for s in SCENARIOS if s != scenario]
        budget = []
        counts = [8, 8, 7, 7]
        seed = 194_000_000 + SCENARIOS.index(scenario) * 1_000_000 + YEARS.index(year) * 10_000 + target_idx * 100 + month_idx
        rng = np.random.default_rng(seed)
        for donor_scenario, count in zip(source_scenarios, counts):
            choices = np.asarray([p for p in range(1, len(EXPECTED_PROVINCES)) if p != target_idx])
            chosen = rng.choice(choices, size=count, replace=False)
            budget.extend((donor_scenario, int(p)) for p in chosen)
    out = []
    for donor_scenario, p in budget:
        mu, shape = donor_month_profile(data, donor_scenario, year, p, bounds, month_idx)
        out.append((donor_scenario, p, mu, shape))
    if len(out) != 30:
        raise AssertionError(f"donor budget {len(out)}")
    return out


def predict(target_mu, donors, seed):
    """Prediction-only function. It never receives target labels/hourly truth."""
    shapes = np.stack([x[3] for x in donors])
    means = np.asarray([x[2] for x in donors])
    order = np.argsort(np.abs(means - target_mu), kind="stable")[:5]
    nearest = shapes[order]
    rng = np.random.default_rng(seed)
    permuted = np.stack([row[rng.permutation(row.size)] for row in shapes])
    return {
        "constant_anchor": np.full(shapes.shape[1], target_mu),
        "all30_mean_shape": target_mu + shapes.mean(axis=0),
        "all30_shape_law": target_mu + shapes,
        "nearest5_mean_shape": target_mu + nearest.mean(axis=0),
        "nearest5_shape_law": target_mu + nearest,
        "permuted30_shape_law": target_mu + permuted,
    }


def canonical_digest(arr):
    arr = np.asarray(arr, dtype="<f8", order="C")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def crps_empirical(particles, y):
    # O(k log k), exact empirical CRPS for equally weighted particles.
    x = np.sort(np.asarray(particles, dtype=np.float64), axis=0)
    first = np.mean(np.abs(x - y), axis=0)
    k = x.shape[0]
    weights = (2 * np.arange(1, k + 1) - k - 1).reshape(-1, 1)
    pair_term = np.sum(weights * x, axis=0) / (k * k)
    return float(np.mean(first - pair_term))


def score_point(pred, y):
    err = pred - y
    return {"hour_mae": float(np.mean(np.abs(err))), "hour_rmse": float(np.sqrt(np.mean(err * err)))}


def score_law(particles, y):
    lo, hi = np.quantile(particles, [0.05, 0.95], axis=0, method="linear")
    width = hi - lo
    outside = np.maximum(lo - y, 0) + np.maximum(y - hi, 0)
    winkler = width + (2 / 0.10) * outside
    return {"crps": crps_empirical(particles, y), "coverage90": float(np.mean((y >= lo) & (y <= hi))), "width90": float(np.mean(width)), "winkler90": float(np.mean(winkler))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="write to a fresh revision directory if default exists")
    args = ap.parse_args()
    out = OUT
    if out.exists():
        existing = [p for p in out.iterdir() if p.name != "run_freeze.json"]
        if not args.force and existing:
            raise SystemExit(f"refusing to overwrite {out}; use --force to create a timestamped revision")
        if args.force:
            out = OUT.with_name(OUT.name + "-" + time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    data = read_data()
    meta = {"protocol": "V194 frozen shape-transfer protocol + data-completeness amendment", "protocol_sha256": hashlib.sha256((ROOT / "docs/V194_CHINA_SCENARIO_SHAPE_TRANSFER_PROTOCOL.md").read_bytes()).hexdigest(), "amendment_sha256": hashlib.sha256((ROOT / "docs/V194_DATA_COMPLETENESS_AMENDMENT.md").read_bytes()).hexdigest(), "scope": "simulator-world only; target monthly means are retrospective oracle anchors", "scenario_order": SCENARIOS, "years_in_source": ALL_YEARS, "complete_case_years_scored": YEARS, "excluded_missing_years": [2055, 2060], "province_order": EXPECTED_PROVINCES, "methods": METHODS, "tracks": ["same_scenario", "leave_scenario_out"], "prediction_seal": "per-query SHA256 of all outputs, before scoring calls", "data_source": "Li et al. (2026) Figshare v3; CC BY 4.0; modeled projections"}
    (out / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    score_rows = []

    # Stage 1: compute only coarse target means + donor profiles, generate and hash predictions.
    # The target-hour matrix is held outside predict(); no target hour enters this stage's API.
    query_count = 0
    manifest_hash = hashlib.sha256()
    manifest_path = out / "prediction_manifest.jsonl"
    with manifest_path.open("w") as manifest_file:
        for track in meta["tracks"]:
            for scenario in SCENARIOS:
                for year in YEARS:
                    vals, bounds = data[(scenario, year)]
                    for target_idx in range(1, len(EXPECTED_PROVINCES)):
                        province = EXPECTED_PROVINCES[target_idx]
                        for month_idx, (lo, hi) in enumerate(bounds):
                            # Coarse target oracle is the only target-derived input.
                            target_mu = float(vals[lo:hi, target_idx].mean())
                            donors = make_donors(data, track, scenario, year, target_idx, bounds, month_idx, target_mu)
                            qseed = 194_900_000 + (SCENARIOS.index(scenario) * 1_000_000 + YEARS.index(year) * 10_000 + target_idx * 100 + month_idx)
                            preds = predict(target_mu, donors, qseed)
                            p_hashes = {m: canonical_digest(preds[m]) for m in METHODS}
                            donor_hash = canonical_digest(np.stack([d[3] for d in donors]))
                            manifest_row = {"track": track, "scenario": scenario, "year": year, "province": province, "month": month_idx + 1, "target_mean": target_mu, "donor_profiles": [{"scenario": d[0], "province": EXPECTED_PROVINCES[d[1]]} for d in donors], "donor_shape_sha256": donor_hash, "prediction_sha256": p_hashes}
                            line = json.dumps(manifest_row, sort_keys=True, separators=(",", ":")) + "\n"
                            manifest_file.write(line)
                            manifest_hash.update(line.encode())
                            query_count += 1
                    print(f"sealed {track} {scenario} {year} ({query_count} queries total)", flush=True)
    seal = {**meta, "query_count": query_count, "prediction_manifest_sha256": manifest_hash.hexdigest(), "manifest_file": "prediction_manifest.jsonl", "seal_status": "SEALED_BEFORE_SCORING", "target_hour_truth_accessed_by_predict_function": False}
    (out / "prediction_seal.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n")
    print("prediction manifest sealed; beginning independent scoring", flush=True)

    # Stage 2: score the sealed predictions against target hourly labels.
    manifest_reader = manifest_path.open()
    scored_queries = 0
    for track in meta["tracks"]:
        for scenario in SCENARIOS:
            for year in YEARS:
                vals, bounds = data[(scenario, year)]
                for target_idx in range(1, len(EXPECTED_PROVINCES)):
                    province = EXPECTED_PROVINCES[target_idx]
                    for month_idx, (lo, hi) in enumerate(bounds):
                        target_mu = float(vals[lo:hi, target_idx].mean())
                        donors = make_donors(data, track, scenario, year, target_idx, bounds, month_idx, target_mu)
                        qseed = 194_900_000 + (SCENARIOS.index(scenario) * 1_000_000 + YEARS.index(year) * 10_000 + target_idx * 100 + month_idx)
                        preds = predict(target_mu, donors, qseed)
                        sealed_row = json.loads(next(manifest_reader))
                        if sealed_row["track"] != track or sealed_row["scenario"] != scenario or sealed_row["year"] != year or sealed_row["province"] != province or sealed_row["month"] != month_idx + 1:
                            raise AssertionError("prediction manifest query order mismatch")
                        if sealed_row["donor_shape_sha256"] != canonical_digest(np.stack([d[3] for d in donors])):
                            raise AssertionError("donor shape differs from sealed prediction run")
                        if sealed_row["prediction_sha256"] != {m: canonical_digest(preds[m]) for m in METHODS}:
                            raise AssertionError("prediction differs from sealed forecast artifact")
                        target_y = vals[lo:hi, target_idx].copy()
                        # Functional isolation check: changing within-month label order without changing mean cannot change forecasts.
                        original_mu = float(target_y.mean())
                        permuted_y = target_y[np.random.default_rng(194_700_000 + target_idx * 100 + month_idx).permutation(target_y.size)]
                        if not np.isclose(permuted_y.mean(), original_mu, atol=1e-12, rtol=0):
                            raise AssertionError("month-label permutation changed mean")
                        permuted_mu = float(permuted_y.mean())
                        if not np.isclose(permuted_mu, target_mu, atol=1e-12, rtol=0):
                            raise AssertionError("permuted-label oracle mean changed beyond floating tolerance")
                        # Reuse the exact sealed oracle anchor: a permutation changes summation order at machine precision.
                        permuted_input_preds = predict(target_mu, donors, qseed)
                        if any(not np.array_equal(preds[m], permuted_input_preds[m]) for m in METHODS):
                            raise AssertionError("prediction changed under target within-month label permutation")
                        for method in METHODS:
                            pred = preds[method]
                            entry = {"track": track, "scenario": scenario, "year": year, "province": province, "month": month_idx + 1, "method": method, "level_mae": abs(float(pred.mean()) - target_mu), "shape_mae": None, "shape_rmse": None, "hour_mae": None, "hour_rmse": None, "crps": None, "coverage90": None, "width90": None, "winkler90": None}
                            center_truth = target_y - target_mu
                            if method.endswith("_law"):
                                particles = pred
                                entry.update(score_law(particles, target_y))
                                point = particles.mean(axis=0)
                                entry["shape_mae"] = float(np.mean(np.abs(point - target_mu - center_truth)))
                                entry["shape_rmse"] = float(np.sqrt(np.mean((point - target_mu - center_truth) ** 2)))
                                entry.update(score_point(point, target_y))
                            else:
                                entry["shape_mae"] = float(np.mean(np.abs(pred - target_mu - center_truth)))
                                entry["shape_rmse"] = float(np.sqrt(np.mean((pred - target_mu - center_truth) ** 2)))
                                entry.update(score_point(pred, target_y))
                            score_rows.append(entry)
                        scored_queries += 1
                print(f"scored {track} {scenario} {year}", flush=True)
    if manifest_reader.readline():
        raise AssertionError("unconsumed prediction manifest rows")
    manifest_reader.close()
    import pandas as pd
    df = pd.DataFrame(score_rows)
    df.to_csv(out / "monthly_scores.csv", index=False)
    # Equal-weight monthly query macro means; report both tracks, plus per scenario/method.
    metric_cols = ["level_mae", "shape_mae", "shape_rmse", "hour_mae", "hour_rmse", "crps", "coverage90", "width90", "winkler90"]
    summary = df.groupby(["track", "scenario", "method"], dropna=False)[metric_cols].mean().reset_index()
    summary.to_csv(out / "summary_scenario.csv", index=False)
    macro = df.groupby(["track", "method"], dropna=False)[metric_cols].mean().reset_index()
    macro.to_csv(out / "summary_macro.csv", index=False)
    counts = df.groupby(["track", "method"]).size().reset_index(name="scored_month_queries")
    counts.to_csv(out / "score_counts.csv", index=False)
    print("V194 complete", out, flush=True)
    print(macro.to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
