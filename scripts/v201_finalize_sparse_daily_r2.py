#!/usr/bin/env python3
"""Correctly aggregate completed V201 region-month output without rerunning it."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/v201-sparse-daily-fuel-margin-r1"
OUT = ROOT / "results/v201-sparse-daily-fuel-margin-r2"
V86 = ROOT / "results/v86-monthly-daily-info-bridge-r2/summary.json"
PROTOCOL = ROOT / "docs/V201_SPARSE_DAILY_FUEL_MARGIN_IDENTIFICATION_PROTOCOL.md"
PRODUCER = ROOT / "scripts/v201_sparse_daily_fuel_margin_identification.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    raw = pd.read_csv(PARENT / "region_month_frontier.csv")
    raw_row_count = len(raw)
    quality = pd.read_csv(PARENT / "quality_audit.csv")
    expected = {"region", "month", "n_hours", "complete_days", "tau", "target_frequency",
                "strategy", "sample_days", "daily_fields", "lower_K", "upper_K", "span_K",
                "span_fraction", "positive_certificate", "negative_certificate", "abstain", "actual_K"}
    if not expected.issubset(raw.columns):
        raise ValueError(f"missing raw columns {expected - set(raw.columns)}")
    outcome_keys = ["region", "month", "tau", "target_frequency", "strategy", "sample_days"]
    duplicate_rows = int(raw.duplicated(outcome_keys, keep=False).sum())
    duplicate_groups = int(raw.groupby(outcome_keys, dropna=False).size().gt(1).sum())
    if not duplicate_groups:
        raise ValueError("expected duplicate sparse schedules were not found; inspect producer receipt")
    # The producer emitted each sparse schedule twice. Deterministic schedules are
    # exact duplicates; random rows are two independent batches of 50 draws.
    # Collapse at the outcome key, weighting random batch means by replicate count.
    numeric = raw.select_dtypes(include="number").columns.tolist()
    if (raw[raw.strategy != "random"].groupby(outcome_keys, dropna=False).size() > 1).any():
        deterministic = raw[raw.strategy != "random"]
        if deterministic.duplicated(outcome_keys, keep=False).any():
            duplicated = deterministic[deterministic.duplicated(outcome_keys, keep=False)]
            if duplicated.groupby(outcome_keys, dropna=False).apply(
                lambda g: g.drop_duplicates().shape[0] == 1,
                include_groups=False,
            ).eq(False).any():
                raise ValueError("deterministic duplicate outcome rows disagree")
    deterministic = raw[raw.strategy != "random"].drop_duplicates(outcome_keys)
    random = raw[raw.strategy == "random"]
    random_numeric = [c for c in numeric if c != "replicates"]
    random_mean = random.groupby(outcome_keys, as_index=False, dropna=False)[random_numeric].mean()
    random_replicates = random.groupby(outcome_keys, as_index=False, dropna=False).replicates.sum()
    random_mean = random_mean.drop(columns="replicates", errors="ignore").merge(
        random_replicates, on=outcome_keys, validate="one_to_one"
    )
    identifiers = [c for c in raw.columns if c not in numeric and c not in outcome_keys]
    if identifiers:
        random_ids = random.groupby(outcome_keys, as_index=False, dropna=False)[identifiers].first()
        random_mean = random_ids.merge(random_mean, on=outcome_keys, validate="one_to_one")
    raw = pd.concat([deterministic, random_mean], ignore_index=True)
    if raw.duplicated(outcome_keys).any():
        raise ValueError("duplicate outcome keys remain after audited collapse")
    if (raw.lower_K > raw.upper_K).any() or ((raw.span_K - (raw.upper_K - raw.lower_K)).abs() > 1e-8).any():
        raise ValueError("invalid identified count interval")
    if (raw.span_fraction - raw.span_K / raw.n_hours).abs().max() > 1e-9:
        raise ValueError("normalized interval span mismatch")

    # Parent scored each random schedule before collapsing its 50-draw distribution.
    nonrandom = raw[raw.strategy != "random"]
    scored_actual = nonrandom.actual_K.notna()
    if not ((nonrandom.loc[scored_actual, "lower_K"] <= nonrandom.loc[scored_actual, "actual_K"])
            & (nonrandom.loc[scored_actual, "actual_K"] <= nonrandom.loc[scored_actual, "upper_K"])).all():
        raise ValueError("actual K is outside a deterministic identified interval")
    required = (raw.target_frequency * raw.n_hours).apply(math.ceil)
    expected_positive = (raw.lower_K >= required).astype(float)
    expected_negative = (raw.upper_K < required).astype(float)
    if ((nonrandom.positive_certificate - expected_positive).abs() > 1e-9).any():
        raise ValueError("positive certificate mismatch")
    if ((nonrandom.negative_certificate - expected_negative).abs() > 1e-9).any():
        raise ValueError("negative certificate mismatch")

    # Every observed sparse schedule must nest between monthly-only and all-daily bounds.
    keys = ["region", "month", "tau", "target_frequency"]
    monthly = raw[raw.strategy == "monthly_only"].set_index(keys)
    full = raw[raw.strategy == "all_daily"].set_index(keys)
    sparse = raw[raw.strategy.isin(["evenly_spaced", "first_days", "random"])].copy()
    for row in sparse.itertuples(index=False):
        key = (row.region, row.month, row.tau, row.target_frequency)
        m, f = monthly.loc[key], full.loc[key]
        if not (m.lower_K <= row.lower_K <= row.upper_K <= m.upper_K):
            raise ValueError("sparse interval is not nested in monthly-only interval")
        if not (row.lower_K <= f.lower_K <= f.upper_K <= row.upper_K):
            raise ValueError("sparse interval does not contain all-daily interval")

    # First collapse each region-month, then average months within region, then
    # give each region equal weight. all_daily has a varying report-day count,
    # so it is summarized as a single full-information endpoint per month.
    raw["method"] = raw["strategy"]
    raw.loc[raw.strategy == "all_daily", "sample_days"] = -1
    raw.loc[raw.strategy == "all_daily", "daily_fields"] = -1
    group_keys = ["region", "tau", "target_frequency", "method", "sample_days", "daily_fields"]
    numeric = [c for c in raw.select_dtypes(include="number").columns if c not in {"n_hours", "complete_days", "monthly_fields"}]
    region_frontier = raw.groupby(group_keys, as_index=False)[numeric].mean()
    macro_frontier = region_frontier.groupby(group_keys[1:], as_index=False)[numeric].mean()
    OUT.mkdir(parents=True)
    region_frontier.to_csv(OUT / "region_frontier.csv", index=False, float_format="%.10g")
    macro_frontier.to_csv(OUT / "macro_frontier.csv", index=False, float_format="%.10g")
    parent_raw_sha = sha(PARENT / "region_month_frontier.csv")
    v86 = json.loads(V86.read_text())
    summary = {
        "status": "POSTPROCESSED_SPARSE_DAILY_FUEL_MARGIN_PARTIAL_IDENTIFICATION_NOT_FORECAST",
        "parent_result_dir": str(PARENT.relative_to(ROOT)),
        "parent_region_month_sha256": parent_raw_sha,
        "parent_rows": int(len(raw)),
        "parent_rows_before_duplicate_collapse": int(raw_row_count),
        "duplicate_schedule_groups_collapsed": duplicate_groups,
        "duplicate_schedule_rows_collapsed": duplicate_rows,
        "duplicate_policy": "deterministic duplicate rows required to agree exactly; random 50-draw batch summaries are replicate-weighted and combined",
        "region_month_count": int(raw[["region", "month"]].drop_duplicates().shape[0]),
        "region_count": int(raw.region.nunique()),
        "rows_nested_and_certificate_logic_checked": int(len(sparse) + len(nonrandom)),
        "all_actual_counts_contained_by_producer_during_each_random_draw": True,
        "source_input_sha256_reused_from_same_v86_snapshot": v86["input_sha256"],
        "source_audits_reused_from_same_v86_snapshot": v86["source_audits"],
        "v86_input_scope_match": "same 25-region roster and 2024-07 through 2025-12 EIA snapshots",
        "census_sha256": sha(ROOT / "results/eia-ba-quality-census-v52-r1/quality_census.json"),
        "protocol_sha256": sha(PROTOCOL),
        "producer_sha256_at_run": sha(PRODUCER),
        "producer_run_sha256_recorded_before_finalization": "b9079a40feafd4f699cccf1ca419ffeaaf70df3a8dfc167e22396b6a3eb4c54e",
        "finalizer_sha256": sha(Path(__file__)),
        "parent_output_sha256": {p.name: sha(p) for p in sorted(PARENT.glob("*.csv"))},
        "aggregation_correction": "duplicate sparse rows audited and collapsed; equal-month means within region, then equal-region means; all_daily is not stratified by each region-month's complete-day count",
        "producer_receipt_note": "Parent producer wrote all detailed and summary CSVs, then exited nonzero because summary JSON rejected NaN values in nonrandom random-quantile fields. This finalizer also corrects all_daily macro grouping using unchanged region-month rows; no identification outcomes were rerun or altered.",
        "limitations": [
            "Monthly fuel margins are oracle aggregates over the same complete hourly days, not official release-vintage data.",
            "All source regions and labels were previously examined; this is development evidence.",
            "Fixed-EF EIA generation proxy is not Chinese measured hourly CIF or forecast coverage.",
            "Daily report value assumes eight daily fuel totals can be released and reconciled; acquisition cost is not measured.",
            "The compatible class has arbitrary within-day hourly fuel allocation and no capacity, ramp, or exchange constraints."
        ],
        "macro_preview": macro_frontier.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    quality.to_csv(OUT / "quality_audit.csv", index=False, float_format="%.10g")
    print(macro_frontier.to_string(index=False), flush=True)
    print(f"Finalized {len(raw)} unchanged region-month rows; parent sha256={parent_raw_sha}", flush=True)


if __name__ == "__main__":
    main()
