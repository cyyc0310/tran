"""Apply the frozen support scores to the already-viewed Czech and Serbian LOCO folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_energy_charts_loco as base  # noqa: E402
import run_energy_charts_czech_expansion as czech  # noqa: E402
import run_energy_charts_serbia_expansion as serbia  # noqa: E402
from run_episodic_coldstart_transfer import make_panel  # noqa: E402
from analyze_loco_public_input_support_risk import X_COLUMNS, scores, unique_months  # noqa: E402


OLD_DATA = ROOT / "data/energy-charts-v2-panel-r1"
NEW_DATA = ROOT / "data/energy-charts-v2-panel-r2"
CZ_RESULTS = ROOT / "docs/evidence/energy-charts-czech-expansion-r4/monthly_metrics.csv"
RS_RESULTS = ROOT / "docs/evidence/energy-charts-serbia-expansion-r2/monthly_metrics.csv"
CZ_EVIDENCE = ROOT / "docs/evidence/energy-charts-czech-expansion-r4"
RS_EVIDENCE = ROOT / "docs/evidence/energy-charts-serbia-expansion-r2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_common_and_new() -> dict[str, pd.DataFrame]:
    old_manifest = json.loads((OLD_DATA / "acquisition_manifest.json").read_text())
    panels: dict[str, pd.DataFrame] = {}
    for code in base.COUNTRIES:
        base.DATA = OLD_DATA
        raw, _ = base.load_country(code, old_manifest)
        panel, _ = make_panel(raw)
        panels[code.upper()] = panel

    cz_manifest = json.loads((NEW_DATA / "cz_acquisition_manifest.json").read_text())
    base.DATA = NEW_DATA
    raw_cz, _ = base.load_country("cz", {"countries": [{"country": "cz", "sha256": cz_manifest["sha256"]}]})
    panel_cz, _ = make_panel(raw_cz)
    panels["CZ"] = panel_cz

    rs_manifest = json.loads((NEW_DATA / "acquisition_manifest.json").read_text())
    raw_rs, _ = serbia.load_serbia(rs_manifest["sha256"])
    panel_rs, _ = make_panel(raw_rs)
    panels["RS"] = panel_rs
    return panels


def run_one(target: str, roster: list[str], panels: dict[str, pd.DataFrame],
            outcomes: pd.DataFrame, experiment: str) -> list[dict]:
    sources = [country for country in roster if country != target]
    source = np.vstack([unique_months(panels[country], 2022)[list(X_COLUMNS)].to_numpy(float)
                        for country in sources])
    target_months = unique_months(panels[target], 2024, expected=12)
    score_matrix = scores(source, target_months[list(X_COLUMNS)].to_numpy(float))
    target_outcome = outcomes.loc[outcomes.target_country.eq(target)].copy()
    if experiment == "czech":
        target_outcome = target_outcome.loc[target_outcome.method.isin(["ratio_recomposition", "factor_history"])]
    else:
        target_outcome = target_outcome.loc[target_outcome.method.isin(["ratio_recomposition", "factor_history"])]
    if len(target_outcome) != 24:
        raise ValueError(f"{target}: expected 12 months × 2 frozen methods, found {len(target_outcome)}")
    if not set(target_outcome.month.astype(int)) == set(range(1, 13)):
        raise ValueError(f"{target}: outcome file does not contain all 2024 months")
    rows = []
    for index, month in enumerate(target_months.month.astype(int)):
        for _, result in target_outcome.loc[target_outcome.month.eq(month)].iterrows():
            rows.append({
                "experiment": experiment, "target_country": target, "method": result.method,
                "month": int(month), "n_source_country_month_vectors": len(source),
                **{name: float(value[index]) for name, value in score_matrix.items()},
                "mae": float(result.mae), "coverage90": float(result.coverage90),
                "interval_score90": float(result.interval_score90),
                "undercoverage_85": bool(result.coverage90 < .85),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels = load_common_and_new()
    old = sorted(code.upper() for code in base.COUNTRIES)
    cz_roster = [*old, "CZ"]
    rs_roster = [*old, "RS"]
    cz_outcomes = pd.read_csv(CZ_RESULTS)
    rs_outcomes = pd.read_csv(RS_RESULTS)
    rows = run_one("CZ", cz_roster, panels, cz_outcomes, "czech-six-country-loco")
    rows += run_one("RS", rs_roster, panels, rs_outcomes, "serbia-six-country-loco")
    frame = pd.DataFrame(rows)
    frame["undercoverage_90"] = frame.coverage90.lt(.90)
    summaries = []
    score_columns = ["nearest_source_month_zrms", "source_shrunk_mahalanobis",
                     "mean_range_excursion_z", "outside_coordinate_count"]
    for (experiment, country, method), group in frame.groupby(["experiment", "target_country", "method"]):
        for score_col in score_columns:
            summaries.append({
                "experiment": experiment, "target_country": country, "method": method,
                "support_score": score_col, "mean_support_score": float(group[score_col].mean()),
                "spearman_month_score_mae": float(spearmanr(group[score_col], group.mae).statistic),
                "spearman_month_score_coverage": float(spearmanr(group[score_col], group.coverage90).statistic),
                "months_below_85": int(group.undercoverage_85.sum()),
                "months_below_90": int(group.undercoverage_90.sum()),
                "mean_mae": float(group.mae.mean()), "mean_coverage90": float(group.coverage90.mean()),
                "mean_interval_score90": float(group.interval_score90.mean()),
            })
    args.output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output / "monthly_external_development.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "domain_summary.csv", index=False)
    input_paths = [CZ_RESULTS, RS_RESULTS,
                   OLD_DATA / "acquisition_manifest.json", OLD_DATA / "audit.json",
                   *(OLD_DATA / f"{code}_2022_2024.json" for code in base.COUNTRIES),
                   NEW_DATA / "cz_acquisition_manifest.json", NEW_DATA / "cz_2022_2024.json",
                   NEW_DATA / "acquisition_manifest.json", NEW_DATA / "rs_2022_2024.json",
                   ROOT / "scripts/analyze_loco_public_input_support_risk.py"]
    code_paths = [Path(__file__).resolve(), ROOT / "scripts/analyze_loco_public_input_support_risk.py",
                  ROOT / "scripts/run_energy_charts_loco.py",
                  ROOT / "scripts/run_episodic_coldstart_transfer.py",
                  ROOT / "scripts/run_energy_charts_czech_expansion.py",
                  ROOT / "scripts/run_energy_charts_serbia_expansion.py",
                  ROOT / "src/transcif/research/multicountry.py",
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py",
                  ROOT / "docs/ENERGY_CHARTS_CZECH_EXPANSION_PROTOCOL_20261101.md",
                  ROOT / "docs/ENERGY_CHARTS_SERBIA_EXPANSION_PROTOCOL_20261101.md"]
    summary = {
        "analysis": "frozen support score applied to CZ/RS six-country held-out development folds",
        "roster_cz": cz_roster, "roster_rs": rs_roster,
        "input_information": "five-group t-2..t-7 coarse monthly generation plus history; support formulas unchanged from prior five-country analysis",
        "label_use": "pre-existing held-out metrics used only to evaluate the unchanged support scores",
        "independent_confirmation": False,
        "known_limits": [
            "CZ and RS target metrics were already viewed in prior expansion experiments",
            "RS generation mapping assumes unreported categories are zero and has missing-hour exclusions",
            "only two additional target domains; no population-level risk calibration",
            "support score construction uses 2022 source monthly feature vectors, not exact official issued-vintage records",
        ],
        "input_sha256": {str(path): sha256(path) for path in input_paths},
        "code_sha256": {str(path): sha256(path) for path in code_paths},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).query("support_score == 'mean_range_excursion_z'")
          [["experiment", "target_country", "method", "mean_support_score", "spearman_month_score_mae",
            "spearman_month_score_coverage", "months_below_85", "mean_mae", "mean_coverage90"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
