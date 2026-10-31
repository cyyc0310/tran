"""V140: group-disjoint same-budget conditional K-risk baselines."""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import v139_physical_reserve_calibration as v139
import v90_coarse_capacity_identification as v90

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V140_MONTHLY_CONDITIONAL_RISK_PROTOCOL.md"
SOURCE = ROOT / "results/v139-physical-reserve-calibration-r1/source_scores.csv"
OUT = ROOT / "results/v140-monthly-conditional-risk-r1"
SEEDS = (41, 83)
ALPHAS = (.10, .20)
METHODS = ("physical", "anchor_absolute", "anchor_normalized", "direct_cqr_all",
           "direct_cqr_active", "occupancy_cqr")
PARAMS = {"n_estimators": 150, "learning_rate": .05, "max_depth": 2,
          "min_samples_leaf": 15, "random_state": 20261028, "loss": "quantile"}


def build_panel(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = v90.source_data
    regions = tuple(sorted(source.region.unique()))
    v90.REGIONS = regions
    data.ALL_REGIONS = regions
    data.START, data.STOP = pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")
    hourly, _, audits, hashes = data.prepare_regions()
    capacities, capacity_hashes = {}, {}
    for year in (2023, 2024):
        capacities[year], capacity_hashes[str(year)] = v90.capacities(year)
    groups = {}
    for region, frame in hourly.items():
        frame = frame.loc[frame.valid.astype(bool)].copy()
        frame["month"] = pd.to_datetime(frame.time, utc=True).dt.strftime("%Y-%m-01")
        groups[region] = {month: group for month, group in frame.groupby("month")}
    records = []
    for row in source[["region", "month", "hours"]].drop_duplicates().itertuples(index=False):
        group = groups[row.region][row.month]
        if len(group) != row.hours:
            raise ValueError(f"month hours differ: {row.region}/{row.month}")
        margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0))
                   for fuel in v90.FUELS}
        total = float(sum(margins.values(), F(0)))
        vintage = int(row.month[:4]) - 1
        caprow = capacities[vintage].loc[row.region] if row.region in capacities[vintage].index else pd.Series(dtype=float)
        caps = {name: sum(float(caprow.get(fuel, 0)) for fuel in members)
                for name, members in v90.GROUPS.items()}
        cap_total = sum(caps.values())
        month_number = int(row.month[5:7])
        record = {"region": row.region, "month": row.month, "hours": row.hours,
                  "log_total_mwh": math.log1p(total), "month_sin": math.sin(2 * math.pi * month_number / 12),
                  "month_cos": math.cos(2 * math.pi * month_number / 12)}
        for fuel in v90.FUELS:
            record[f"fuelshare_{fuel}"] = float(margins[fuel]) / total if total else 0.0
        for name, members in v90.GROUPS.items():
            energy = float(sum((margins[f] for f in members), F(0)))
            denominator = energy + row.hours * caps[name]
            record[f"capacity_share_{name}"] = caps[name] / cap_total if cap_total else 0.0
            record[f"bounded_utilization_{name}"] = energy / denominator if denominator else 0.0
        original = dict(v90.EF)
        try:
            for tau in (600, 900):
                bounds = []
                for multiplier in (.9, 1.1):
                    ef = {fuel: value * multiplier if fuel in ("coal", "gas", "oil", "other") else value
                          for fuel, value in original.items()}
                    v90.EF.clear(); v90.EF.update(ef)
                    values = v90.v66_set(margins, tau, row.hours)
                    bounds.append((min(values), max(values)))
                record[f"monthly_lower_{tau}"] = min(x[0] for x in bounds)
                record[f"monthly_upper_{tau}"] = max(x[1] for x in bounds)
        finally:
            v90.EF.clear(); v90.EF.update(original)
        records.append(record)
    panel = pd.DataFrame(records)
    return panel, {"hourly_input_hashes": hashes, "capacity_hashes": capacity_hashes, "source_audits": audits}


def projected_predictions(frame: pd.DataFrame, method: str, models=None) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = frame.physical_lower.to_numpy(float), frame.physical_upper.to_numpy(float)
    if method.startswith("anchor"):
        a = frame.anchor_lower_fraction.to_numpy(float) * frame.hours.to_numpy(float)
        b = frame.anchor_upper_fraction.to_numpy(float) * frame.hours.to_numpy(float)
    else:
        lower, upper = models
        raw_a, raw_b = lower.predict(frame[FEATURES]), upper.predict(frame[FEATURES])
        raw_a, raw_b = np.minimum(raw_a, raw_b), np.maximum(raw_a, raw_b)
        if method == "occupancy_cqr":
            a, b = lo + np.clip(raw_a, 0, 1) * (hi - lo), lo + np.clip(raw_b, 0, 1) * (hi - lo)
        else:
            hours = frame.hours.to_numpy(float)
            a, b = raw_a * hours, raw_b * hours
    return np.clip(a, lo, hi), np.clip(b, lo, hi)


def fitted_models(frame: pd.DataFrame, method: str):
    fit = frame.copy()
    if method in ("direct_cqr_active", "occupancy_cqr"):
        fit = fit.loc[fit.physical_upper.gt(fit.physical_lower)].copy()
    if len(fit) < 2:
        raise ValueError("insufficient fit data for conditional model")
    weight = 1.0 / fit.groupby("region").region.transform("size").to_numpy(float)
    models = []
    for quantile, target_column in ((.05, "factor_box_K_lower"), (.95, "factor_box_K_upper")):
        if method == "occupancy_cqr":
            y = (fit[target_column] - fit.physical_lower) / (fit.physical_upper - fit.physical_lower)
        else:
            y = fit[target_column] / fit.hours
        model = GradientBoostingRegressor(alpha=quantile, **PARAMS)
        model.fit(fit[FEATURES], y, sample_weight=weight)
        models.append(model)
    return models


def score(frame: pd.DataFrame, a: np.ndarray, b: np.ndarray, normalized: bool) -> np.ndarray:
    error = np.maximum.reduce([a - frame.factor_box_K_lower.to_numpy(float),
                               frame.factor_box_K_upper.to_numpy(float) - b, np.zeros(len(frame))])
    scale = ((frame.physical_upper - frame.physical_lower).to_numpy(float)
             if normalized else frame.hours.to_numpy(float))
    values = np.divide(error, scale, out=np.zeros(len(frame)), where=scale > 0)
    if np.any(values > 1 + 1e-10) or np.any(values < 0):
        raise ValueError("conformity scores outside [0,1]")
    return values


FEATURES: list[str] = []


def main() -> None:
    global FEATURES
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_hash = v139.sha(PROTOCOL)
    source = pd.read_csv(SOURCE)
    panel, data_audit = build_panel(source)
    source = source.merge(panel.drop(columns="hours"), on=["region", "month"], validate="many_to_one")
    regions = sorted(source.region.unique())
    common = ([f"fuelshare_{fuel}" for fuel in v90.FUELS] + ["log_total_mwh", "month_sin", "month_cos",
              "anchor_lower_fraction", "anchor_upper_fraction", "physical_lower_fraction",
              "physical_upper_fraction", "physical_width_fraction"])
    capacity_features = ([f"capacity_share_{g}" for g in v90.GROUPS]
                         + [f"bounded_utilization_{g}" for g in v90.GROUPS] + ["capacity_unavailable"])
    output, membership, radii, failures = [], [], [], []
    for held_out in regions:
        for seed in SEEDS:
            others = [r for r in regions if r != held_out]
            ordered = sorted(others, key=lambda r: hashlib.sha256(f"{seed}/{held_out}/{r}".encode()).hexdigest())
            cal_regions, fit_regions = ordered[:12], ordered[12:]
            if (set(cal_regions) & set(fit_regions) or held_out in others
                    or len(cal_regions) != 12 or len(fit_regions) != 12):
                raise ValueError("invalid disjoint region roles")
            for role, roster in (("fit", fit_regions), ("calibration", cal_regions)):
                for region in roster:
                    membership.append({"target_region": held_out, "seed": seed, "role": role, "region": region})
            for budget in ("monthly", "monthly_capacity"):
                FEATURES = common + (capacity_features if budget == "monthly_capacity" else [])
                for tau in (600, 900):
                    frame = source.loc[source.tau.eq(tau)].copy()
                    frame["physical_lower"] = frame[f"monthly_lower_{tau}"] if budget == "monthly" else frame.capacity_lower
                    frame["physical_upper"] = frame[f"monthly_upper_{tau}"] if budget == "monthly" else frame.capacity_upper
                    frame["physical_lower_fraction"] = frame.physical_lower / frame.hours
                    frame["physical_upper_fraction"] = frame.physical_upper / frame.hours
                    frame["physical_width_fraction"] = (frame.physical_upper - frame.physical_lower) / frame.hours
                    if (not np.isfinite(frame[FEATURES].to_numpy(float)).all()
                            or (frame.physical_lower > frame.factor_box_K_lower).any()
                            or (frame.physical_upper < frame.factor_box_K_upper).any()):
                        raise ValueError("invalid features or physical label containment")
                    fit = frame.loc[frame.region.isin(fit_regions)]
                    calibration = frame.loc[frame.region.isin(cal_regions)]
                    target = frame.loc[frame.region.eq(held_out)]
                    for method in METHODS:
                        normalized = method in ("anchor_normalized", "occupancy_cqr")
                        if method == "physical":
                            a, b = target.physical_lower.to_numpy(float), target.physical_upper.to_numpy(float)
                            region_scores = pd.Series(0.0, index=cal_regions)
                        else:
                            models = fitted_models(fit, method) if "cqr" in method else None
                            ca, cb = projected_predictions(calibration, method, models)
                            cs = pd.DataFrame({"region": calibration.region.to_numpy(),
                                               "score": score(calibration, ca, cb, normalized)})
                            region_scores = cs.groupby("region").score.max().sort_values()
                            a, b = projected_predictions(target, method, models)
                        for alpha in ALPHAS:
                            n = len(region_scores)
                            rank = math.ceil((n + 1) * (1 - alpha))
                            radius = float(region_scores.iloc[rank - 1]) if rank <= n else 1.0
                            radii.append({"target_region": held_out, "seed": seed, "budget": budget,
                                          "tau": tau, "method": method, "alpha": alpha,
                                          "n_calibration_regions": n, "rank": rank, "radius": radius})
                            scale = ((target.physical_upper - target.physical_lower).to_numpy(float)
                                     if normalized else target.hours.to_numpy(float))
                            lower = np.maximum(target.physical_lower.to_numpy(int), np.floor(a - radius * scale + 1e-9).astype(int))
                            upper = np.minimum(target.physical_upper.to_numpy(int), np.ceil(b + radius * scale - 1e-9).astype(int))
                            for j, row in enumerate(target.itertuples(index=False)):
                                lo, hi = int(lower[j]), int(upper[j])
                                if not row.physical_lower <= lo <= hi <= row.physical_upper:
                                    failures.append({"region": held_out, "month": row.month, "seed": seed,
                                                     "budget": budget, "tau": tau, "method": method,
                                                     "alpha": alpha, "check": "physical_subset"})
                                record = {"region": held_out, "month": row.month, "tau": tau, "seed": seed,
                                          "budget": budget, "method": method, "alpha": alpha, "hours": row.hours,
                                          "lower": lo, "upper": hi, "width": hi - lo,
                                          "covered": lo <= row.factor_box_K_lower and hi >= row.factor_box_K_upper,
                                          "empty_fallback": False, "capacity_unavailable": row.capacity_unavailable,
                                          "factor_box_K_lower": row.factor_box_K_lower,
                                          "factor_box_K_upper": row.factor_box_K_upper,
                                          "robust_interval_score90": (hi - lo + 20 * max(lo - row.factor_box_K_lower, 0)
                                                                       + 20 * max(row.factor_box_K_upper - hi, 0))}
                                for rate in v139.RATES:
                                    required = math.ceil(rate * row.hours - 1e-10)
                                    pos, neg = lo >= required, hi < required
                                    record[f"positive_{rate}"] = pos
                                    record[f"negative_{rate}"] = neg
                                    record[f"abstain_{rate}"] = not (pos or neg)
                                    record[f"error_{rate}"] = ((pos and row.factor_box_K_lower < required)
                                                               or (neg and row.factor_box_K_upper >= required))
                                output.append(record)
        print(f"outer region {held_out}", flush=True)
    rows = pd.DataFrame(output)
    expected = len(source) * len(SEEDS) * 2 * len(ALPHAS) * len(METHODS)
    keys = ["region", "month", "tau", "seed", "budget", "method", "alpha"]
    if len(rows) != expected or rows.duplicated(keys).any():
        raise ValueError("scoring denominator failure")
    for key, group in rows.groupby(keys[:-1]):
        narrow, wide = group.sort_values("alpha", ascending=False).iloc[0], group.sort_values("alpha").iloc[0]
        if wide.lower > narrow.lower or wide.upper < narrow.upper:
            failures.append({"key": str(key), "check": "alpha_nesting"})
    summaries = []
    for (budget, seed), group in rows.groupby(["budget", "seed"]):
        summary = v139.summarize(group)
        stress = group.groupby(["method", "alpha", "tau", "region"]).robust_interval_score90.mean()
        stress = stress.groupby(["method", "alpha", "tau"]).mean().reset_index()
        summary = summary.merge(stress, on=["method", "alpha", "tau"], validate="one_to_one")
        summary["budget"], summary["seed"] = budget, seed
        summaries.append(summary)
    frontier = pd.concat(summaries, ignore_index=True)
    if protocol_hash != v139.sha(PROTOCOL):
        raise RuntimeError("protocol changed during run")
    OUT.mkdir(parents=True)
    source.to_csv(OUT / "monthly_feature_panel.csv", index=False)
    rows.to_csv(OUT / "outer_fold_rows.csv", index=False)
    frontier.to_csv(OUT / "region_equal_frontier.csv", index=False)
    pd.DataFrame(membership).to_csv(OUT / "region_roles.csv", index=False)
    pd.DataFrame(radii).to_csv(OUT / "calibration_radii.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "assertion_failures.csv", index=False)
    payload = {"status": "OPENED_SOURCE_MONTHLY_CONDITIONAL_RISK_DEVELOPMENT", "region_months": len(panel),
               "source_rows": len(source), "scored_rows": len(rows), "assertion_failures": len(failures),
               "alphas": ALPHAS, "seeds": SEEDS, "model_params": PARAMS,
               "common_features": common, "optional_capacity_features": capacity_features,
               "source_sha256": v139.sha(SOURCE), "protocol_sha256": protocol_hash,
               "script_sha256": v139.sha(Path(__file__)), "data_audit": data_audit,
               "frontier": json.loads(frontier.to_json(orient="records")),
               "limitations": ["Source-pool development; no new test regions/time opened.",
                               "Monthly target-period totals are retrospective, not forecasts.",
                               "EIA proxy labels and assumed EF box, not Chinese hourly CIF labels.",
                               "CQR and physical normalization are established techniques, not novelty by naming."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(frontier.loc[frontier.alpha.eq(.1), ["budget", "seed", "tau", "method", "macro_coverage",
          "macro_width", "possible_positive_coverage", "robust_interval_score90"]].to_string(index=False))
    print(f"assertion failures: {len(failures)}")


if __name__ == "__main__":
    main()
