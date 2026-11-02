"""Test whether pre-origin coarse-month support flags identify held-out risk."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from run_episodic_coldstart_transfer import COUNTRIES, DATA, make_panel
from run_energy_charts_loco import load_country


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results/five-group-capacity-increment-r2/monthly_metrics.csv"
PARENT_PROTOCOL = ROOT / "docs/FIVE_GROUP_CAPACITY_INCREMENT_PROTOCOL_20261101.md"
PARENT_RUNNER = ROOT / "scripts/run_five_group_capacity_increment.py"
LAG_COLUMNS = tuple(f"{group}_lag{lag}" for lag in range(2, 8)
                    for group in ("thermal", "hydro", "nuclear", "wind", "solar"))
X_COLUMNS = (*LAG_COLUMNS, "thermal_mean", "thermal_std", "thermal_change", "total_cv", "observations")
METHODS = ("M_five_group_recomposition", "M_factor")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_months(panel: pd.DataFrame, year: int, expected: int | None = None) -> pd.DataFrame:
    block = panel.loc[panel.year.eq(year)].copy()
    if block.empty:
        raise ValueError(f"missing {year} feature rows")
    block["month"] = block.clock.dt.month.astype(int)
    monthly = block.groupby(["year", "month"], as_index=False)[list(X_COLUMNS)].first()
    if expected is not None and len(monthly) != expected:
        raise ValueError(f"expected {expected} eligible monthly vectors for {year}, found {len(monthly)}")
    if monthly.empty:
        raise ValueError(f"no eligible monthly vectors for {year}")
    return monthly


def scores(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    scaler = StandardScaler().fit(source)
    source_z, target_z = scaler.transform(source), scaler.transform(target)
    if not np.isfinite(source_z).all() or not np.isfinite(target_z).all():
        raise ValueError("nonfinite public input features")
    # Fixed, label-free score definitions; covariance shrinkage is fit only on source months.
    lw = LedoitWolf().fit(source_z)
    mahal = np.sqrt(np.maximum(lw.mahalanobis(target_z), 0.))
    distances = np.sqrt(((target_z[:, None, :] - source_z[None, :, :]) ** 2).mean(axis=2))
    nearest = distances.min(axis=1)
    lo, hi = source_z.min(axis=0), source_z.max(axis=0)
    excursion = np.maximum(lo - target_z, 0.) + np.maximum(target_z - hi, 0.)
    return {
        "nearest_source_month_zrms": nearest,
        "source_shrunk_mahalanobis": mahal,
        "mean_range_excursion_z": excursion.mean(axis=1),
        "outside_coordinate_count": (excursion > 1e-10).sum(axis=1).astype(float),
    }


def centered_spearman(frame: pd.DataFrame, x: str, y: str) -> float:
    xv = frame[x] - frame.groupby("target_country")[x].transform("mean")
    yv = frame[y] - frame.groupby("target_country")[y].transform("mean")
    return float(spearmanr(xv, yv).statistic)


def auc_by_monthly_event(frame: pd.DataFrame, score_col: str, event_col: str) -> float | None:
    y = frame[event_col].astype(int).to_numpy()
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, frame[score_col].to_numpy(float)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((DATA / "acquisition_manifest.json").read_text())
    panels: dict[str, pd.DataFrame] = {}
    for name in COUNTRIES:
        raw, _ = load_country(name, manifest)
        panel, _ = make_panel(raw)
        panel["country"] = name.upper()
        panels[name.upper()] = panel

    outcome = pd.read_csv(METRICS_PATH)
    outcome = outcome.loc[outcome.method.isin(METHODS)].copy()
    rows: list[dict] = []
    for target in sorted(panels):
        source_names = [name for name in sorted(panels) if name != target]
        source_vectors = np.vstack([unique_months(panels[name], 2022)[list(X_COLUMNS)].to_numpy(float)
                                    for name in source_names])
        target_months = unique_months(panels[target], 2024, expected=12)
        target_vectors = target_months[list(X_COLUMNS)].to_numpy(float)
        fold_scores = scores(source_vectors, target_vectors)
        for i, month in enumerate(target_months.month):
            rows.append({"target_country": target, "month": int(month),
                         "n_source_country_month_vectors": int(len(source_vectors)),
                         **{name: float(values[i]) for name, values in fold_scores.items()}})
    support = pd.DataFrame(rows)
    merged = support.merge(outcome, on=["target_country", "month"], how="inner", validate="one_to_many")
    if len(merged) != len(METHODS) * 5 * 12:
        raise ValueError(f"expected 120 held-out month/method rows, found {len(merged)}")
    merged["undercoverage_85"] = merged.coverage90.lt(.85)
    merged["undercoverage_90"] = merged.coverage90.lt(.90)

    score_columns = ["nearest_source_month_zrms", "source_shrunk_mahalanobis",
                     "mean_range_excursion_z", "outside_coordinate_count"]
    summary_rows = []
    outcome_columns = ["mae", "interval_score90", "coverage90"]
    for method in METHODS:
        block = merged.loc[merged.method.eq(method)].copy()
        for score_col in score_columns:
            for target_metric in outcome_columns:
                y_col = target_metric
                rho = float(spearmanr(block[score_col], block[y_col]).statistic)
                centered = centered_spearman(block, score_col, y_col)
                summary_rows.append({"method": method, "support_score": score_col,
                                     "outcome": target_metric, "spearman_all_country_months": rho,
                                     "within_country_demeaned_spearman": centered,
                                     "n_country_months": int(len(block))})
            summary_rows.append({"method": method, "support_score": score_col,
                                 "outcome": "undercoverage_85_auc",
                                 "spearman_all_country_months": auc_by_monthly_event(block, score_col, "undercoverage_85"),
                                 "within_country_demeaned_spearman": None,
                                 "n_country_months": int(len(block))})
            summary_rows.append({"method": method, "support_score": score_col,
                                 "outcome": "undercoverage_90_auc",
                                 "spearman_all_country_months": auc_by_monthly_event(block, score_col, "undercoverage_90"),
                                 "within_country_demeaned_spearman": None,
                                 "n_country_months": int(len(block))})
    args.output.mkdir(parents=True, exist_ok=False)
    merged.to_csv(args.output / "country_month_support_and_outcomes.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output / "support_risk_associations.csv", index=False)
    country_summary = merged.groupby(["method", "target_country"], as_index=False).agg(
        mean_nearest_distance=("nearest_source_month_zrms", "mean"),
        mean_mae=("mae", "mean"), mean_coverage90=("coverage90", "mean"),
        months_below_85=("undercoverage_85", "sum"), months_below_90=("undercoverage_90", "sum"))
    country_summary.to_csv(args.output / "country_level_summary.csv", index=False)
    inputs = [METRICS_PATH, DATA / "acquisition_manifest.json",
              *(DATA / f"{name}_2022_2024.json" for name in COUNTRIES),
              DATA / "audit.json"]
    code_files = [PARENT_PROTOCOL, PARENT_RUNNER,
                  ROOT / "scripts/run_episodic_coldstart_transfer.py",
                  ROOT / "scripts/run_energy_charts_loco.py",
                  ROOT / "src/transcif/research/multicountry.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    code = Path(__file__).resolve()
    summary = {
        "analysis": "source-only public-month feature support as risk indicator in country-held-out development",
        "source_months": "2022; one monthly row per source country/month; held-out country excluded",
        "target_months": "2024; one feature vector per month; features use only t-2 through t-7 coarse monthly information",
        "feature_columns": list(X_COLUMNS),
        "support_scores": score_columns,
        "outer_targets": sorted(panels), "methods_scored": list(METHODS),
        "support_score_formula_fit_to_target_labels": False,
        "target_labels_used_only_for_posthoc_association_scoring": True,
        "outcomes_read_only_for_posthoc_evaluation": True,
        "support_score_formulas_are_exploratory_after_prior_development_review": True,
        "known_limitations": [
            "five countries and one target year; all domains were previously inspected",
            "2022 source feature history is short; record n_source_country_month_vectors per fold",
            "monthly coverage/MAE outcomes aggregate hourly errors and do not provide independent hourly samples",
            "associations are exploratory and do not validate China accuracy or nominal coverage",
        ],
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "parent_analysis_code_and_protocol_sha256": {str(path): sha256(path) for path in code_files},
        "analysis_code_sha256": sha256(code),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).query("outcome in ['mae', 'coverage90', 'undercoverage_85_auc']")
          .to_string(index=False))


if __name__ == "__main__":
    main()
