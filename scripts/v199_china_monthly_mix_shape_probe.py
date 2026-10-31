#!/usr/bin/env python3
"""Retrospective V199 probe: can public monthly fuel shares select CIF shapes?"""
from __future__ import annotations

import calendar
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
V194 = ROOT / "results/china-projected-cif-v194-shape-transfer-r1"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
OUT = ROOT / "results/china-projected-cif-v199-monthly-mix-shape-r2"
SCENARIOS = ("S1", "S2", "S3", "S4", "S5")
FEATURE_SOURCES = ("Coal", "Gas", "Other fossil", "Bioenergy", "Wind", "Solar", "Hydro", "Nuclear")
N_NEIGHBORS = 15
RANDOM_DRAWS = 100
BOOTSTRAPS = 5000
SEED = 199


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_2025_shapes():
    province_names = [
        "Mainland China", "Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia",
        "Liaoning", "Jilin", "Heilongjiang", "Shanghai", "Jiangsu", "Zhejiang",
        "Anhui", "Fujian", "Jiangxi", "Shandong", "Henan", "Hubei", "Hunan",
        "Guangdong", "Guangxi", "Hainan", "Chongqing", "Sichuan", "Guizhou",
        "Yunnan", "Tibet", "Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang",
    ]
    data = {}
    for scenario in SCENARIOS:
        path = V194 / f"CEF data for Scenario {scenario}.xlsx"
        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb["2025"]
        rows = ws.iter_rows(values_only=True)
        header = next(rows)
        if tuple(header) != ("Time", *province_names):
            raise ValueError(f"unexpected header in {path.name}")
        arr = np.asarray(list(rows), dtype=np.float64)
        if arr.shape != (8760, len(header)) or not np.array_equal(arr[:, 0], np.arange(1, 8761)):
            raise ValueError(f"unexpected 2025 matrix in {path.name}: {arr.shape}")
        values = arr[:, 1:]
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite hourly CIF in {path.name}")
        bounds, cursor = [], 0
        for month in range(1, 13):
            hi = cursor + calendar.monthrange(2025, month)[1] * 24
            bounds.append((cursor, hi))
            cursor = hi
        data[scenario] = (values, bounds)
        wb.close()
        print(f"loaded {scenario}", flush=True)
    return province_names, data


def read_ember_shares(province_names):
    frame = pd.read_csv(EMBER, low_memory=False)
    frame = frame[
        frame["Date"].astype(str).str.startswith("2025-")
        & frame["Electricity source"].isin(FEATURE_SOURCES)
        & frame["Is aggregated source"].astype(str).str.lower().eq("false")
    ].copy()
    frame["Generation (TWh)"] = pd.to_numeric(frame["Generation (TWh)"], errors="raise")
    aliases = {"Tibet": "Tibet (Xizang)"}
    rows = []
    for p_idx, province in enumerate(province_names[1:], start=1):
        ember_province = aliases.get(province, province)
        for month in range(1, 13):
            date = f"2025-{month:02d}-01"
            block = frame[(frame.Province == ember_province) & (frame.Date == date)]
            values = block.set_index("Electricity source")["Generation (TWh)"]
            if set(values.index) != set(FEATURE_SOURCES) or len(values) != len(FEATURE_SOURCES):
                raise ValueError(f"missing/duplicate Ember fuel categories: {ember_province} {date}")
            vector = values.reindex(FEATURE_SOURCES).to_numpy(dtype=np.float64)
            if not np.isfinite(vector).all() or np.any(vector < 0) or vector.sum() <= 0:
                raise ValueError(f"invalid generation vector {ember_province} {date}")
            rows.append((p_idx, month - 1, vector / vector.sum()))
    if len(rows) != 31 * 12:
        raise AssertionError(f"unexpected Ember feature count {len(rows)}")
    features = {(p, m): x for p, m, x in rows}
    return features, frame


def donor_archive(data, track, scenario, target_idx, month_idx, bounds):
    if track == "same_scenario":
        budget = [(scenario, p) for p in range(1, 32) if p != target_idx]
    else:
        source_scenarios = [s for s in SCENARIOS if s != scenario]
        counts = (8, 8, 7, 7)
        seed = 194_000_000 + SCENARIOS.index(scenario) * 1_000_000 + target_idx * 100 + month_idx
        rng = np.random.default_rng(seed)
        budget = []
        for donor_scenario, count in zip(source_scenarios, counts):
            choices = np.asarray([p for p in range(1, 32) if p != target_idx])
            budget.extend((donor_scenario, int(p)) for p in rng.choice(choices, size=count, replace=False))
    lo, hi = bounds[month_idx]
    out = []
    for donor_scenario, p in budget:
        values, _ = data[donor_scenario]
        hourly = values[lo:hi, p]
        out.append({"scenario": donor_scenario, "province_idx": p,
                    "monthly_mean": float(hourly.mean()), "shape": hourly - hourly.mean()})
    if len(out) != 30:
        raise AssertionError(len(out))
    return out


def bray_curtis(a, b):
    den = np.sum(a + b)
    return float(np.sum(np.abs(a - b)) / den) if den else 0.0


def digest(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8", order="C").tobytes()).hexdigest()


def score(prediction, truth):
    err = np.asarray(prediction) - np.asarray(truth)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err * err)))


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    provinces, data = read_2025_shapes()
    features, ember_rows = read_ember_shares(provinces)
    query_manifest = OUT / "prediction_manifest.jsonl"
    seal_hash = hashlib.sha256()
    query_meta = []

    # Prediction stage: target hourly values are not passed into the shape predictor.
    with query_manifest.open("w") as f:
        for track in ("same_scenario", "leave_scenario_out"):
            for scenario in SCENARIOS:
                values, bounds = data[scenario]
                for target_idx in range(1, 32):
                    for month_idx, (lo, hi) in enumerate(bounds):
                        target_mean = float(values[lo:hi, target_idx].mean())  # shared retrospective oracle anchor
                        donors = donor_archive(data, track, scenario, target_idx, month_idx, bounds)
                        target_mix = features[(target_idx, month_idx)]
                        mix_order = sorted(
                            range(len(donors)),
                            key=lambda i: (bray_curtis(target_mix, features[(donors[i]["province_idx"], month_idx)]), i),
                        )[:N_NEIGHBORS]
                        oracle_order = sorted(
                            range(len(donors)),
                            key=lambda i: (abs(donors[i]["monthly_mean"] - target_mean), i),
                        )[:N_NEIGHBORS]
                        mix_shape = np.mean([donors[i]["shape"] for i in mix_order], axis=0)
                        oracle_shape = np.mean([donors[i]["shape"] for i in oracle_order], axis=0)
                        all30_shape = np.mean([d["shape"] for d in donors], axis=0)
                        pred = {"constant_shape": np.zeros(hi - lo), "all30_mean": all30_shape,
                                "mix_bc_k15": mix_shape, "level_oracle_k15": oracle_shape}
                        row = {
                            "track": track, "scenario": scenario, "province": provinces[target_idx],
                            "month": month_idx + 1, "target_mean_oracle": target_mean,
                            "target_mix": target_mix.tolist(),
                            "mix_selected_donors": [
                                {"scenario": donors[i]["scenario"], "province": provinces[donors[i]["province_idx"]]}
                                for i in mix_order
                            ],
                            "level_selected_donors": [
                                {"scenario": donors[i]["scenario"], "province": provinces[donors[i]["province_idx"]]}
                                for i in oracle_order
                            ],
                            "prediction_hashes": {k: digest(v) for k, v in pred.items()},
                        }
                        line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        f.write(line)
                        seal_hash.update(line.encode())
                        query_meta.append((track, scenario, target_idx, month_idx, lo, hi, pred))
    metadata = {
        "status": "PREDICTIONS_SEALED_BEFORE_THIS_RUN_SCORED_TARGET_HOURS",
        "caveat": "2025 CEF target labels were previously opened/scored under V194; this is not a blind evaluation",
        "protocol_sha256": sha256(ROOT / "docs/V199_CHINA_MONTHLY_MIX_SHAPE_PROBE_PROTOCOL.md"),
        "runner_sha256": sha256(Path(__file__)),
        "ember_snapshot_sha256": sha256(EMBER),
        "input_workbook_sha256": {
            s: sha256(V194 / f"CEF data for Scenario {s}.xlsx") for s in SCENARIOS
        },
        "feature_sources": list(FEATURE_SOURCES), "k": N_NEIGHBORS,
        "random_draws": RANDOM_DRAWS, "query_count": len(query_meta),
        "manifest_sha256": seal_hash.hexdigest(),
        "scope": "retrospective model-world shape signal probe; target month mean is oracle",
    }
    (OUT / "prediction_seal.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    # Score after the manifest is sealed. Resample only fixed-profile archives for the random baseline.
    scores = []
    random_seed_sequence = np.random.SeedSequence(SEED).spawn(len(query_meta))
    for qi, (track, scenario, target_idx, month_idx, lo, hi, pred) in enumerate(query_meta):
        values, _ = data[scenario]
        target = values[lo:hi, target_idx]
        target_shape = target - float(target.mean())
        donors = donor_archive(data, track, scenario, target_idx, month_idx, data[scenario][1])
        if not all(np.isfinite(v).all() for v in pred.values()):
            raise AssertionError("non-finite sealed prediction")
        methods = dict(pred)
        random_rng = np.random.default_rng(random_seed_sequence[qi])
        random_errors = []
        for _draw in range(RANDOM_DRAWS):
            selected = random_rng.choice(len(donors), N_NEIGHBORS, replace=False)
            random_shape = np.mean([donors[i]["shape"] for i in selected], axis=0)
            random_errors.append(score(random_shape, target_shape))
        for method, prediction in methods.items():
            mae, rmse = score(prediction, target_shape)
            scores.append({"track": track, "scenario": scenario, "province": provinces[target_idx],
                           "month": month_idx + 1, "method": method, "shape_mae": mae, "shape_rmse": rmse})
        scores.append({"track": track, "scenario": scenario, "province": provinces[target_idx],
                       "month": month_idx + 1, "method": "random_k15_expected_100", 
                       "shape_mae": float(np.mean([x[0] for x in random_errors])),
                       "shape_rmse": float(np.mean([x[1] for x in random_errors]))})
    score_df = pd.DataFrame(scores)
    score_df.to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.10g")
    score_df.groupby(["track", "method"])[["shape_mae", "shape_rmse"]].mean().reset_index().to_csv(
        OUT / "summary_macro.csv", index=False, float_format="%.10g")
    score_df.groupby(["track", "scenario", "method"])[["shape_mae", "shape_rmse"]].mean().reset_index().to_csv(
        OUT / "summary_scenario.csv", index=False, float_format="%.10g")
    score_df.groupby(["track", "province", "method"])[["shape_mae", "shape_rmse"]].mean().reset_index().to_csv(
        OUT / "summary_province.csv", index=False, float_format="%.10g")

    # Descriptive interval for paired point-shape loss deltas, clustered by province.
    contrasts = []
    rng = np.random.default_rng(SEED + 1)
    provinces_scored = sorted(score_df.province.unique())
    for track in ("same_scenario", "leave_scenario_out"):
        sub = score_df[score_df.track == track]
        wide = sub.pivot(index=["scenario", "province", "month"], columns="method", values="shape_mae").reset_index()
        for baseline in ("all30_mean", "random_k15_expected_100", "level_oracle_k15"):
            wide["delta"] = wide["mix_bc_k15"] - wide[baseline]
            per_province = wide.groupby("province").delta.mean().reindex(provinces_scored)
            draws = rng.integers(0, len(provinces_scored), size=(BOOTSTRAPS, len(provinces_scored)))
            boot = per_province.to_numpy()[draws].mean(axis=1)
            contrasts.append({"track": track, "comparison": f"mix_bc_k15_minus_{baseline}",
                              "mean_query_delta": float(wide.delta.mean()),
                              "province_macro_delta": float(per_province.mean()),
                              "cluster_bootstrap_low": float(np.quantile(boot, .025)),
                              "cluster_bootstrap_high": float(np.quantile(boot, .975)),
                              "n_queries": len(wide), "n_provinces": len(per_province),
                              "bootstrap_replicates": BOOTSTRAPS, "seed": SEED + 1})
    pd.DataFrame(contrasts).to_csv(OUT / "paired_province_bootstrap.csv", index=False, float_format="%.10g")

    # The reported Ember intensity is an extra-emissions-data diagnostic, never a predictor here.
    ember_all = pd.read_csv(EMBER, low_memory=False)
    ember_all = ember_all[(ember_all.Date.astype(str).str.startswith("2025-"))
                          & ember_all["Electricity source"].eq("Total generation")].copy()
    ember_all["Date"] = pd.to_datetime(ember_all.Date)
    intensity = ember_all.set_index(["Province", ember_all.Date.dt.month])["Emissions intensity (gCO2e/kWh)"]
    agreement = []
    for scenario in SCENARIOS:
        values, bounds = data[scenario]
        for p_idx, province in enumerate(provinces[1:], start=1):
            ember_province = "Tibet (Xizang)" if province == "Tibet" else province
            for month_idx, (lo, hi) in enumerate(bounds):
                ember_ci = intensity.loc[(ember_province, month_idx + 1)]
                cef_mean = float(values[lo:hi, p_idx].mean())
                agreement.append({"scenario": scenario, "province": province, "month": month_idx + 1,
                                  "ember_reported_generation_intensity": float(ember_ci),
                                  "projected_cif_oracle_month_mean_t_per_mwh": cef_mean,
                                  "projected_cif_oracle_month_mean_g_per_kwh": cef_mean * 1000.0,
                                  "difference_ember_minus_projection_g_per_kwh": float(ember_ci - cef_mean * 1000.0)})
    agreement_df = pd.DataFrame(agreement)
    agreement_df.to_csv(OUT / "ember_vs_projection_level_diagnostic.csv", index=False, float_format="%.10g")
    summary = {
        "score_queries_per_track": int(score_df[score_df.method == "mix_bc_k15"].groupby("track").size().iloc[0]),
        "score_rows": int(len(score_df)), "province_count": int(len(provinces_scored)),
        "ember_intensity_vs_projected_month_mean": {
            "pearson": float(agreement_df[["ember_reported_generation_intensity", "projected_cif_oracle_month_mean_g_per_kwh"]].corr().iloc[0, 1]),
            "mae": float(agreement_df.difference_ember_minus_projection_g_per_kwh.abs().mean()),
            "mean_signed_difference": float(agreement_df.difference_ember_minus_projection_g_per_kwh.mean()),
            "rows": int(len(agreement_df)),
        },
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(pd.read_csv(OUT / "summary_macro.csv").to_string(index=False), flush=True)
    print(pd.read_csv(OUT / "paired_province_bootstrap.csv").to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
