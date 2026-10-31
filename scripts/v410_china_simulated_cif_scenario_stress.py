#!/usr/bin/env python3
"""Frozen S1-to-S2:S5 scenario stress for coarse-mix hourly CIF transfer.

This uses simulated projected provincial carbon factors as stress labels, not
measured hourly CIF. See docs/V410_CHINA_SIMULATED_CIF_SCENARIO_STRESS_PROTOCOL.md.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "v410-china-simulated-hourly-cif-stress-r1"
OUT_DIR = DATA_DIR
PROTOCOL = ROOT / "docs" / "V410_CHINA_SIMULATED_CIF_SCENARIO_STRESS_PROTOCOL.md"
NBS = ROOT / "results" / "v394-nbs-all-province-long-panel-r1" / "official_current_values.csv"
SCENARIOS = ("S1", "S2", "S3", "S4", "S5")
TARGET_SCENARIOS = ("S2", "S3", "S4", "S5")
METHODS = ("source_hour_profile", "local_thermal_anchor", "coarse_mix_ridge")
TARGET_MONTHS = tuple(range(5, 13))
THERMAL_EF = 700.0
RIDGE_ALPHA = 10.0
INTERVAL_LEVEL = 0.90
TAUS = (300.0, 500.0, 700.0)
RATES = (0.01, 0.05, 0.10)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_scenario(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["2025"]
    values = ws.iter_rows(values_only=True)
    columns = list(next(values))
    records = list(values)
    frame = pd.DataFrame(records, columns=columns)
    if len(frame) != 8760 or frame["Time"].tolist() != list(range(1, 8761)):
        raise ValueError(f"Unexpected 2025 hourly index in {path.name}")
    provinces = [c for c in columns if c not in {"Time", "Mainland China"}]
    # Figshare values are tCO2/MWh; numerically convert to gCO2/kWh.
    data = {p: pd.to_numeric(frame[p], errors="raise").to_numpy(float) * 1000.0 for p in provinces}
    wb.close()
    return provinces, data


def month_index() -> np.ndarray:
    days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours = days * 24
    return np.repeat(np.arange(1, 13), hours)


def load_lagged_nbs(provinces: list[str]) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, object]]:
    raw = pd.read_csv(NBS)
    raw = raw[(raw["year"] == 2025) & raw["technology"].isin(["total", "thermal", "hydro", "nuclear", "wind", "solar"])]
    values: dict[str, dict[tuple[int, str], float]] = {}
    for row in raw.itertuples(index=False):
        if pd.notna(row.value_100m_kwh):
            values.setdefault(row.province, {})[(int(row.month), row.technology)] = float(row.value_100m_kwh)

    result: dict[str, dict[int, np.ndarray]] = {}
    usable_months: dict[str, list[int]] = {}
    for province in provinces:
        nbs_province = {"Inner Mongolia": "InnerMongolia"}.get(province, province)
        result[province] = {}
        usable_months[province] = []
        for target_month in TARGET_MONTHS:
            feature_month = target_month - 2
            cats = ["thermal", "hydro", "nuclear", "wind", "solar"]
            vals = [values.get(nbs_province, {}).get((feature_month, c), np.nan) for c in cats]
            total = values.get(nbs_province, {}).get((feature_month, "total"), np.nan)
            if not np.isfinite(total) or total <= 0 or not np.all(np.isfinite(vals)):
                continue
            shares = np.asarray(vals, dtype=float) / total
            if np.any(shares < -1e-10) or shares.sum() > 1.02:
                raise ValueError(f"Invalid NBS generation shares: {province}, target month {target_month}")
            result[province][target_month] = shares
            usable_months[province].append(target_month)

    common = sorted(set.intersection(*(set(usable_months[p]) for p in provinces)))
    if common != list(TARGET_MONTHS):
        raise ValueError(f"Frozen target months unavailable for full cohort: {common}")
    return result, {"feature_categories": ["thermal", "hydro", "nuclear", "wind", "solar"],
                    "province_name_aliases": {"Inner Mongolia": "InnerMongolia"},
                    "lag_months": 2, "target_months": list(TARGET_MONTHS),
                    "common_months": common, "nbs_snapshot_sha256": sha256(NBS)}


def make_design(target_month: int, shares: np.ndarray, months: np.ndarray) -> np.ndarray:
    mask = months == target_month
    hour = np.arange(8760)[mask]
    day = hour // 24
    hod = hour % 24
    year_phase = 2 * np.pi * (day + 0.5) / 365.0
    day_phase = 2 * np.pi * (hod + 0.5) / 24.0
    week_phase = 2 * np.pi * (hour + 0.5) / 168.0
    time_features = np.column_stack((np.sin(day_phase), np.cos(day_phase),
                                     np.sin(year_phase), np.cos(year_phase),
                                     np.sin(week_phase), np.cos(week_phase)))
    return np.column_stack((np.repeat(shares[None, :], len(hour), axis=0), time_features))


def regional_sufficient_stats(x: np.ndarray, y: np.ndarray) -> tuple[int, np.ndarray, float, np.ndarray, np.ndarray]:
    n = x.shape[0]
    return n, x.sum(axis=0), float(y.sum()), x.T @ x, x.T @ y


def combine_stats(stats: list[tuple[int, np.ndarray, float, np.ndarray, np.ndarray]], excluded: set[int]):
    chosen = [s for i, s in enumerate(stats) if i not in excluded]
    n = sum(s[0] for s in chosen)
    sx = sum((s[1] for s in chosen), np.zeros_like(chosen[0][1]))
    sy = sum(s[2] for s in chosen)
    sxx = sum((s[3] for s in chosen), np.zeros_like(chosen[0][3]))
    sxy = sum((s[4] for s in chosen), np.zeros_like(chosen[0][4]))
    return n, sx, sy, sxx, sxy


def ridge_coeff(stats: list[tuple[int, np.ndarray, float, np.ndarray, np.ndarray]], excluded: set[int]) -> tuple[np.ndarray, float]:
    n, sx, sy, sxx, sxy = combine_stats(stats, excluded)
    mean_x, mean_y = sx / n, sy / n
    centered_xx = sxx - np.outer(sx, sx) / n
    centered_xy = sxy - sx * (sy / n)
    beta = np.linalg.solve(centered_xx + RIDGE_ALPHA * np.eye(centered_xx.shape[0]), centered_xy)
    intercept = mean_y - mean_x @ beta
    return beta, float(intercept)


def quantile_order_stat(scores: list[float], level: float) -> float:
    if not scores:
        raise ValueError("No calibration scores")
    rank = min(len(scores), max(1, math.ceil(level * (len(scores) + 1))))
    return float(np.sort(np.asarray(scores, dtype=float))[rank - 1])


def build_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    provinces, s1 = read_scenario(DATA_DIR / "CEF data for Scenario S1.xlsx")
    shares, nbs_info = load_lagged_nbs(provinces)
    months = month_index()
    scenario_data = {"S1": s1}
    for scenario in TARGET_SCENARIOS:
        got, this = read_scenario(DATA_DIR / f"CEF data for Scenario {scenario}.xlsx")
        if got != provinces:
            raise ValueError(f"Province order differs in {scenario}")
        scenario_data[scenario] = this

    province_ix = {p: i for i, p in enumerate(provinces)}
    x_by_p: dict[str, dict[int, np.ndarray]] = {}
    y_by_p: dict[str, dict[int, np.ndarray]] = {}
    stats_by_p: dict[str, tuple[int, np.ndarray, float, np.ndarray, np.ndarray]] = {}
    for p in provinces:
        x_by_p[p], y_by_p[p] = {}, {}
        xs, ys = [], []
        for m in TARGET_MONTHS:
            mask = months == m
            x = make_design(m, shares[p][m], months)
            y = s1[p][mask]
            if len(x) != len(y):
                raise ValueError("Feature/label length mismatch")
            x_by_p[p][m] = x
            y_by_p[p][m] = y
            xs.append(x)
            ys.append(y)
        stats_by_p[p] = regional_sufficient_stats(np.vstack(xs), np.concatenate(ys))
    regional_stats = [stats_by_p[p] for p in provinces]
    s1_matrix = np.column_stack([s1[p] for p in provinces])
    s1_sum = s1_matrix.sum(axis=1)

    raw_rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    calibration_summary: list[dict[str, object]] = []

    for target in provinces:
        pi = province_ix[target]
        sources = [p for p in provinces if p != target]
        source_ix = [province_ix[p] for p in sources]
        target_beta, target_intercept = ridge_coeff(regional_stats, {pi})

        cal_scores: dict[str, dict[int, list[float]]] = {m: {month: [] for month in TARGET_MONTHS} for m in METHODS}
        for q in sources:
            qi = province_ix[q]
            profile_q = (s1_sum - s1_matrix[:, pi] - s1_matrix[:, qi]) / (len(provinces) - 2)
            beta_q, intercept_q = ridge_coeff(regional_stats, {pi, qi})
            for month in TARGET_MONTHS:
                mask = months == month
                yq = y_by_p[q][month]
                profile_pred = profile_q[mask]
                anchor_pred = np.full(mask.sum(), THERMAL_EF * shares[q][month][0])
                ridge_pred = x_by_p[q][month] @ beta_q + intercept_q
                for method, pred in (("source_hour_profile", profile_pred),
                                     ("local_thermal_anchor", anchor_pred),
                                     ("coarse_mix_ridge", ridge_pred)):
                    cal_scores[method][month].append(float(np.max(np.abs(yq - pred))))

        q_by_method = {method: {month: quantile_order_stat(cal_scores[method][month], INTERVAL_LEVEL)
                                for month in TARGET_MONTHS} for method in METHODS}
        for method in METHODS:
            for month in TARGET_MONTHS:
                calibration_summary.append({"target_province": target, "method": method, "month": month,
                                            "n_calibration_regions": len(cal_scores[method][month]),
                                            "order_statistic_width": q_by_method[method][month],
                                            "rank_1based": min(len(sources), math.ceil(INTERVAL_LEVEL * (len(sources) + 1))),
                                            "region_max_abs_error_scores": json.dumps(cal_scores[method][month], separators=(",", ":"))})

        profile_target = (s1_sum - s1_matrix[:, pi]) / len(sources)
        for scenario in TARGET_SCENARIOS:
            labels = scenario_data[scenario][target]
            for month in TARGET_MONTHS:
                mask = months == month
                y = labels[mask]
                preds = {
                    "source_hour_profile": profile_target[mask],
                    "local_thermal_anchor": np.full(mask.sum(), THERMAL_EF * shares[target][month][0]),
                    "coarse_mix_ridge": x_by_p[target][month] @ target_beta + target_intercept,
                }
                n_hours = len(y)
                if n_hours != int(np.sum(mask)):
                    raise AssertionError("Bad target-month mask")
                for method in METHODS:
                    pred = preds[method]
                    q = q_by_method[method][month]
                    lower, upper = pred - q, pred + q
                    in_interval = (y >= lower - 1e-12) & (y <= upper + 1e-12)
                    row = {"scenario": scenario, "province": target, "month": month,
                           "method": method, "hours": n_hours,
                           "point_mae": float(np.mean(np.abs(y - pred))),
                           "point_rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
                           "simultaneous_month_coverage": bool(np.all(in_interval)),
                           "hour_fraction_covered": float(np.mean(in_interval)),
                           "interval_half_width": q, "interval_width": 2 * q,
                           "thermal_share_t_minus_2": float(shares[target][month][0]),
                           "actual_month_mean": float(np.mean(y)),
                           "pred_month_mean": float(np.mean(pred))}
                    raw_rows.append(row)
                    for tau in TAUS:
                        k_actual = int(np.count_nonzero(y > tau))
                        k_lower = int(np.count_nonzero(lower > tau))
                        k_upper = int(np.count_nonzero(upper > tau))
                        if k_lower > k_upper:
                            raise AssertionError("Invalid K interval")
                        for rate in RATES:
                            required = math.ceil(rate * n_hours)
                            if k_lower >= required:
                                decision = "positive"
                            elif k_upper < required:
                                decision = "negative"
                            else:
                                decision = "abstain"
                            tail_rows.append({"scenario": scenario, "province": target, "month": month,
                                              "method": method, "tau_gco2_kwh": tau,
                                              "event_rate": rate, "hours": n_hours,
                                              "k_actual": k_actual, "k_lower": k_lower, "k_upper": k_upper,
                                              "k_width": k_upper - k_lower,
                                              "k_contains_actual": k_lower <= k_actual <= k_upper,
                                              "simultaneous_month_coverage": bool(np.all(in_interval)),
                                              "decision": decision})

    metrics = pd.DataFrame(raw_rows)
    tails = pd.DataFrame(tail_rows)
    cal = pd.DataFrame(calibration_summary)
    summary_rows = []
    for (scenario, method), group in metrics.groupby(["scenario", "method"], sort=True):
        by_province = group.groupby("province", sort=True)
        summary_rows.append({"scenario": scenario, "method": method,
                             "region_equal_point_mae": float(by_province.point_mae.mean().mean()),
                             "region_equal_point_rmse": float(by_province.point_rmse.mean().mean()),
                             "region_equal_interval_width": float(by_province.interval_width.mean().mean()),
                             "region_equal_simultaneous_month_coverage": float(by_province.simultaneous_month_coverage.mean().mean()),
                             "pooled_simultaneous_month_coverage": float(group.simultaneous_month_coverage.mean()),
                             "worst_province_month_coverage": float(by_province.simultaneous_month_coverage.mean().min()),
                             "n_provinces": int(group.province.nunique()), "n_province_months": int(len(group))})
    summary = pd.DataFrame(summary_rows)
    tail_summary_rows = []
    for (scenario, method, tau, rate), group in tails.groupby(["scenario", "method", "tau_gco2_kwh", "event_rate"], sort=True):
        by_province = group.groupby("province", sort=True)
        counts = group.decision.value_counts(normalize=True)
        tail_summary_rows.append({"scenario": scenario, "method": method, "tau_gco2_kwh": tau,
                                  "event_rate": rate,
                                  "region_equal_k_width": float(by_province.k_width.mean().mean()),
                                  "region_equal_k_containment": float(by_province.k_contains_actual.mean().mean()),
                                  "positive_rate": float(counts.get("positive", 0.0)),
                                  "negative_rate": float(counts.get("negative", 0.0)),
                                  "abstention_rate": float(counts.get("abstain", 0.0)),
                                  "simultaneous_month_coverage": float(by_province.simultaneous_month_coverage.mean().mean()),
                                  "n_province_months": int(len(group))})
    tail_summary = pd.DataFrame(tail_summary_rows)
    return metrics, tails, summary, {"nbs": nbs_info, "calibration": cal, "tail_summary": tail_summary}


def main() -> None:
    metrics, tails, summary, extra = build_results()
    metrics.to_csv(OUT_DIR / "region_month_metrics.csv", index=False)
    tails.to_csv(OUT_DIR / "tail_count_intervals.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_by_scenario_method.csv", index=False)
    extra["calibration"].to_csv(OUT_DIR / "calibration_region_month_scores.csv", index=False)
    extra["tail_summary"].to_csv(OUT_DIR / "tail_frontier.csv", index=False)
    input_names = ["figshare_metadata.json", "Annotation of the dataset.pdf", "Input parameters of the dataset.xlsx"] + [f"CEF data for Scenario {s}.xlsx" for s in SCENARIOS]
    inputs = {name: sha256(DATA_DIR / name) for name in input_names}
    summary_json = {"status": "complete", "protocol_sha256": sha256(PROTOCOL),
                    "runner_sha256": sha256(Path(__file__)), "figshare_item_doi": "10.6084/m9.figshare.28953545",
                    "figshare_item_version": 3, "figshare_license": "CC BY 4.0",
                    "development_scenario": "S1", "stress_scenarios": list(TARGET_SCENARIOS),
                    "inputs_sha256": inputs, "nbs_feature_contract": extra["nbs"],
                    "target": "projected provincial hourly carbon emission factor from planning/operation simulation and carbon-flow model; tCO2/MWh converted to gCO2/kWh by x1000",
                    "eligible_months": list(TARGET_MONTHS), "provinces": sorted(metrics.province.unique().tolist()),
                    "n_provinces": int(metrics.province.nunique()), "n_scenarios_scored": 4,
                    "n_region_month_method_scenario_rows": int(len(metrics)),
                    "n_tail_rows": int(len(tails)),
                    "calibration": {"level": INTERVAL_LEVEL, "unit": "source-region-month maximum absolute error",
                                    "leave_one_source_region_out": True, "outer_target_region_excluded_from_calibration": True,
                                    "finite_sample_rank": "ceil(level*(n+1)), clipped to [1,n]"},
                    "limitations": ["All hourly targets are model-generated scenarios, not observations or measured hourly CIF.",
                                   "Scenarios share one planning/operation modeling framework; scenario shifts are not independent real-world systems.",
                                   "The historical NBS API snapshot has no as-published vintages; t-2 timing is emulated.",
                                   "NBS generation shares are production-side and above-scale; projected provincial CEF includes modeled network carbon flow.",
                                   "No scenario S2-S5 target labels enter fitting or calibration; S1 is development only."]}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nTail frontier:")
    print(extra["tail_summary"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
