"""V139: paired region-LOO calibration in physical tail-risk coordinates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import v92_sharp_coarse_capacity_identification as sharp

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V139_PHYSICAL_RESERVE_CALIBRATION_PROTOCOL.md"
SOURCE = ROOT / "results/v120-support-gap-region-loo-cv-r2/source_outer_fold_features.csv"
ANCHOR = ROOT / "results/v111-factor-robust-monthly-anchor-r1/source_calibration_rows.csv"
REFERENCE = ROOT / "results/v120-support-gap-region-loo-cv-r2/outer_fold_fallback_rows.csv"
ENERGY_AUDIT = ROOT / "results/v100-no-peak-screen-capacity-r1/nonselective_cohort.csv"
VERIFIED = ROOT / "results/v103-factor-uncertainty-sharp-set-r1/factor_robust_rows.csv"
OUT = ROOT / "results/v139-physical-reserve-calibration-r1"
ALPHAS = (.05, .10, .20)
METHODS = ("physical", "raw_absolute", "projected_absolute", "physical_width", "directional_reserve")
RATES = (.01, .05, .10)


def guard_audit() -> list[dict]:
    records = []
    for name, fuel, amount, expected in (
            ("infeasible_all_event", "coal", 3, None),
            ("infeasible_no_event", "gas", 3, None),
            ("feasible_all_event", "coal", 2, (2,)),
            ("feasible_no_event", "gas", 2, (0,))):
        margins = {f: 0 for f in sharp.v90.FUELS}
        margins[fuel] = amount
        caps = {g: 0 for g in sharp.v90.GROUPS}
        caps["thermal"] = 1
        actual = sharp.sharp_group_set(margins, caps, 600, 2)
        records.append({"case": name, "expected": expected, "actual": actual, "passed": actual == expected})
    if not all(r["passed"] for r in records):
        raise ValueError("solver guard audit failed")
    return records


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> pd.DataFrame:
    source = pd.read_csv(SOURCE)
    anchor = pd.read_csv(ANCHOR)
    keys = ["region", "month", "tau"]
    source = source.merge(anchor[keys + ["anchor_lower_fraction", "anchor_upper_fraction"]],
                          on=keys, how="left", validate="one_to_one")
    energy = pd.read_csv(ENERGY_AUDIT)
    source = source.merge(energy[["region", "month", "hours", "monthly_group_energy_feasible"]],
                          on=["region", "month"], how="left", validate="many_to_one",
                          suffixes=("", "_energy_audit"))
    if source.isna().any().any():
        raise ValueError("missing merged inputs")
    if not source.hours.eq(source.hours_energy_audit).all():
        raise ValueError("source/audit hours differ")
    source["archived_capacity_lower"] = source.capacity_lower
    source["archived_capacity_upper"] = source.capacity_upper
    source["archived_physical_conflict"] = ((source.capacity_lower > source.factor_box_K_lower)
                                             | (source.capacity_upper < source.factor_box_K_upper))
    source["capacity_unavailable"] = ~source.monthly_group_energy_feasible.astype(bool)
    # Availability uses monthly margins/capacity only, never the target risk labels.
    source.loc[source.capacity_unavailable, "capacity_lower"] = 0
    source.loc[source.capacity_unavailable, "capacity_upper"] = source.loc[source.capacity_unavailable, "hours"]
    verified = pd.read_csv(VERIFIED).query("rate == .01")
    paired = source.loc[~source.capacity_unavailable].merge(
        verified[["region", "month", "tau", "robust_lower", "robust_upper"]],
        on=keys, how="left", validate="one_to_one")
    if (paired[["robust_lower", "robust_upper"]].isna().any().any()
            or not paired.capacity_lower.eq(paired.robust_lower).all()
            or not paired.capacity_upper.eq(paired.robust_upper).all()):
        raise ValueError("feasible source endpoints do not match V103 audited endpoints")
    h = source.hours.to_numpy(float)
    lo, hi = source.capacity_lower.to_numpy(float), source.capacity_upper.to_numpy(float)
    yl, yh = source.factor_box_K_lower.to_numpy(float), source.factor_box_K_upper.to_numpy(float)
    a = source.anchor_lower_fraction.to_numpy(float) * h
    b = source.anchor_upper_fraction.to_numpy(float) * h
    ap, bp = np.clip(a, lo, hi), np.clip(b, lo, hi)
    if np.any((lo > yl) | (yl > yh) | (yh > hi) | (ap > bp)):
        raise ValueError("invalid physical/label/anchor order")
    source["base_lower"], source["base_upper"] = a, b
    source["projected_lower"], source["projected_upper"] = ap, bp
    lower_error, upper_error = np.maximum(ap - yl, 0), np.maximum(yh - bp, 0)
    source["score_raw_absolute"] = np.maximum.reduce([a - yl, yh - b, np.zeros(len(h))]) / h
    source["score_projected_absolute"] = np.maximum(lower_error, upper_error) / h
    source["score_physical_width"] = np.divide(np.maximum(lower_error, upper_error), hi - lo,
                                               out=np.zeros(len(h)), where=(hi > lo))
    lower_ratio = np.divide(lower_error, ap - lo, out=np.zeros(len(h)), where=(ap > lo))
    upper_ratio = np.divide(upper_error, hi - bp, out=np.zeros(len(h)), where=(hi > bp))
    if np.any((lower_error > 0) & (ap == lo)) or np.any((upper_error > 0) & (hi == bp)):
        raise ValueError("positive error with zero directional reserve")
    source["score_directional_reserve"] = np.maximum(lower_ratio, upper_ratio)
    for method in ("physical_width", "directional_reserve"):
        if not source[f"score_{method}"].between(-1e-12, 1 + 1e-12).all():
            raise ValueError("physical normalized score outside [0,1]")
    return source


def interval(row: pd.Series, method: str, radius: float) -> tuple[int, int, bool]:
    lo, hi = float(row.capacity_lower), float(row.capacity_upper)
    if method == "physical" or not math.isfinite(radius):
        return int(lo), int(hi), False
    a, b = float(row.projected_lower), float(row.projected_upper)
    if method == "raw_absolute":
        lower, upper = row.base_lower - radius * row.hours, row.base_upper + radius * row.hours
    elif method == "projected_absolute":
        lower, upper = a - radius * row.hours, b + radius * row.hours
    elif method == "physical_width":
        lower, upper = a - radius * (hi - lo), b + radius * (hi - lo)
    elif method == "directional_reserve":
        lower, upper = a - radius * (a - lo), b + radius * (hi - b)
    else:
        raise ValueError(method)
    lower, upper = max(lo, lower), min(hi, upper)
    if lower > upper + 1e-9:
        return int(lo), int(hi), True
    return max(int(lo), math.floor(lower + 1e-9)), min(int(hi), math.ceil(upper - 1e-9)), False


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (method, alpha, tau), group in rows.groupby(["method", "alpha", "tau"]):
        region = group.groupby("region").agg(coverage=("covered", "mean"), width=("width", "mean"),
                                               all_months_covered=("covered", "all"),
                                               empty_fallback=("empty_fallback", "mean"))
        record = {"method": method, "alpha": alpha, "tau": tau, "rows": len(group),
                  "regions": group.region.nunique(), "macro_coverage": region.coverage.mean(),
                  "pooled_coverage": group.covered.mean(), "macro_width": region.width.mean(),
                  "all_month_region_coverage": region.all_months_covered.mean(),
                  "empty_fallback_rate": region.empty_fallback.mean()}
        for name, subset in (("possible_positive", group.loc[group.factor_box_K_upper.gt(0)]),
                             ("robust_positive", group.loc[group.factor_box_K_lower.gt(0)]),
                             ("capacity_feasible", group.loc[~group.capacity_unavailable]),
                             ("capacity_unavailable", group.loc[group.capacity_unavailable])):
            record[f"{name}_rows"] = len(subset)
            record[f"{name}_coverage"] = (subset.groupby("region").covered.mean().mean()
                                                  if len(subset) else None)
        for rate in RATES:
            decisions = group.groupby("region")[[f"positive_{rate}", f"negative_{rate}",
                                                    f"abstain_{rate}", f"error_{rate}"]].mean().mean()
            for key, value in decisions.items():
                record[key] = value
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_hash = sha(PROTOCOL)
    boundary_audit = guard_audit()
    source = prepare()
    outputs, calibration_rows, failures = [], [], []
    for (region, tau), target in source.groupby(["region", "tau"]):
        train = source.loc[source.region.ne(region) & source.tau.eq(tau)]
        train_regions = sorted(train.region.unique())
        if region in train_regions or len(train_regions) != source.region.nunique() - 1:
            raise ValueError("outer-fold region leakage or missing calibration regions")
        for alpha in ALPHAS:
            n = len(train_regions)
            rank = math.ceil((n + 1) * (1 - alpha))
            for method in METHODS:
                if method == "physical":
                    radius = 1.0
                else:
                    scores = train.groupby("region")[f"score_{method}"].max().sort_values()
                    radius = float(scores.iloc[rank - 1]) if rank <= n else float("inf")
                    for calibration_region, score in scores.items():
                        calibration_rows.append({"target_region": region, "tau": tau, "alpha": alpha,
                                                 "method": method, "calibration_region": calibration_region,
                                                 "score": score, "n_regions": n, "rank": rank,
                                                 "radius": radius})
                for _, row in target.iterrows():
                    lower, upper, fallback = interval(row, method, radius)
                    if not row.capacity_lower <= lower <= upper <= row.capacity_upper:
                        failures.append({"region": region, "tau": tau, "month": row.month,
                                         "method": method, "check": "physical_subset"})
                    covered = lower <= row.factor_box_K_lower and upper >= row.factor_box_K_upper
                    record = {"region": region, "month": row.month, "tau": tau, "alpha": alpha,
                              "method": method, "hours": row.hours, "lower": lower, "upper": upper,
                              "width": upper - lower, "covered": bool(covered), "radius": radius,
                              "empty_fallback": fallback, "capacity_unavailable": bool(row.capacity_unavailable),
                              "capacity_lower": row.capacity_lower,
                              "capacity_upper": row.capacity_upper,
                              "factor_box_K_lower": row.factor_box_K_lower,
                              "factor_box_K_upper": row.factor_box_K_upper}
                    for rate in RATES:
                        required = math.ceil(rate * row.hours - 1e-10)
                        pos, neg = lower >= required, upper < required
                        record[f"positive_{rate}"] = pos
                        record[f"negative_{rate}"] = neg
                        record[f"abstain_{rate}"] = not (pos or neg)
                        record[f"error_{rate}"] = ((pos and row.factor_box_K_lower < required)
                                                   or (neg and row.factor_box_K_upper >= required))
                    outputs.append(record)
    rows = pd.DataFrame(outputs)
    expected = len(source) * len(ALPHAS) * len(METHODS)
    if len(rows) != expected or rows.duplicated(["region", "month", "tau", "alpha", "method"]).any():
        raise ValueError("denominator or duplicate failure")
    # A raw-anchor empty-intersection fallback can violate nesting; audit it separately.
    for keys, group in rows.groupby(["region", "month", "tau", "method"]):
        ordered = group.sort_values("alpha", ascending=False)
        previous = None
        for row in ordered.itertuples(index=False):
            if previous and (row.lower > previous.lower or row.upper < previous.upper):
                failures.append({"region": keys[0], "month": keys[1], "tau": keys[2],
                                 "method": keys[3], "check": "alpha_nesting",
                                 "expected_raw_fallback_caveat": keys[3] == "raw_absolute"})
            previous = row
    summary = summarize(rows)
    width_rows = rows.loc[rows.method.eq("physical_width")].set_index(["region", "month", "tau", "alpha"])
    directional = rows.loc[rows.method.eq("directional_reserve")].set_index(width_rows.index.names)
    equivalent = bool(width_rows[["lower", "upper"]].eq(directional[["lower", "upper"]]).all().all())
    reference = pd.read_csv(REFERENCE)
    references = []
    for (tail, tau), group in reference.groupby(["support_tail", "tau"]):
        per_region = group.groupby("region")[["fallback_covered", "fallback_width"]].mean()
        references.append({"support_tail": tail, "tau": tau, "macro_coverage": per_region.fallback_covered.mean(),
                           "macro_width": per_region.fallback_width.mean(), "rows": len(group)})
    if sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    OUT.mkdir(parents=True)
    source.to_csv(OUT / "source_scores.csv", index=False)
    source.loc[source.capacity_unavailable | source.archived_physical_conflict,
               ["region", "month", "tau", "hours", "capacity_unavailable", "archived_physical_conflict",
                "archived_capacity_lower", "archived_capacity_upper", "capacity_lower", "capacity_upper",
                "factor_box_K_lower", "factor_box_K_upper"]].to_csv(OUT / "archived_input_contract_audit.csv", index=False)
    rows.to_csv(OUT / "outer_fold_rows.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT / "calibration_membership.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "assertion_failures.csv", index=False)
    (OUT / "solver_boundary_audit.json").write_text(json.dumps(boundary_audit, indent=2) + "\n")
    summary.to_csv(OUT / "region_equal_frontier.csv", index=False)
    archived = pd.DataFrame(references)
    archived["valid_performance_comparator"] = False
    archived.to_csv(OUT / "invalidated_archived_v120_reference.csv", index=False)
    payload = {"status": "PHYSICAL_RESERVE_SOURCE_REGION_LOO_DEVELOPMENT", "source_rows": len(source),
               "region_months": len(source[["region", "month"]].drop_duplicates()),
               "regions": int(source.region.nunique()), "scored_rows": len(rows), "assertion_failures": len(failures),
               "capacity_unavailable_region_months": int(source.loc[source.capacity_unavailable, ["region", "month"]].drop_duplicates().shape[0]),
               "archived_physical_conflict_rows": int(source.archived_physical_conflict.sum()),
               "feasible_endpoint_pairs_matched_v103": int(len(source.loc[~source.capacity_unavailable])),
               "solver_boundary_cases_passed": len(boundary_audit),
               "solver_sha256": sha(Path(sharp.__file__)),
               "normalized_method_outputs_identical": equivalent,
               "frontier": json.loads(summary.to_json(orient="records")),
               "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (SOURCE, ANCHOR, REFERENCE, ENERGY_AUDIT, VERIFIED)},
               "protocol_sha256": protocol_hash, "script_sha256": sha(Path(__file__)),
               "limitations": ["Opened EIA source pool, not blind confirmation or Chinese validation.",
                               "76/589 months lack a feasible capacity contract and use [0,T] fallback; U.S. capacity is not a confirmed Chinese input.",
                               "Region exchangeability is an assumption; empirical LOO coverage is not a guarantee.",
                               "Normalized scores are existing methodology; no novelty claim from scaling alone."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(summary[["method", "alpha", "tau", "macro_coverage", "macro_width", "all_month_region_coverage",
                   "possible_positive_coverage", "robust_positive_coverage"]].to_string(index=False))
    print(f"assertion failures: {len(failures)}; normalized outputs identical: {equivalent}")


if __name__ == "__main__":
    main()
