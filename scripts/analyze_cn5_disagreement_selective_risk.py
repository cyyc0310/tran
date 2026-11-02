#!/usr/bin/env python3
"""Post-hoc source-LOCO test of whether head disagreement can rank risky days."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402

PREDICTIONS = ROOT / "results/cn5-disagreement-risk-pseudotarget-r2/double_heldout_pseudotarget_predictions.csv.gz"
OUTER_AUDIT = ROOT / "results/cn5-disagreement-risk-pseudotarget-r2/fold_audit.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    input_hashes = {str(path.resolve()): sha(path) for path in (PREDICTIONS, OUTER_AUDIT)}
    code_hashes = {str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                   str(Path(nine.__file__).resolve()): sha(Path(nine.__file__)),
                   str(Path(nine.six_group.__file__).resolve()): sha(Path(nine.six_group.__file__)),
                   str(Path(nine.six_group.base.__file__).resolve()): sha(Path(nine.six_group.base.__file__))}
    source = pd.read_csv(PREDICTIONS)

    daily_rows = []
    for code in nine.COUNTRIES:
        raw, features, _ = nine.prepare_country(code)
        h2 = features.loc[features.year.eq(2023) & features.clock.dt.month.ge(7)].copy()
        clock = pd.DataFrame({
            "time": raw.iloc[h2.raw_row.to_numpy()].time.astype(str).to_numpy(),
            "local_day": h2.clock.dt.strftime("%Y-%m-%d").to_numpy(),
        })
        rows = source.loc[source.country.eq(code.upper())].merge(
            clock, on="time", how="inner", validate="one_to_one")
        if len(rows) != len(h2):
            raise ValueError(f"hour support mismatch for {code}: {len(rows)} != {len(h2)}")
        rows["abs_error"] = np.abs(rows.truth-rows.ratio)
        daily = rows.groupby("local_day", as_index=False).agg(
            daily_mae=("abs_error", "mean"),
            factor_ratio_disagreement=("factor_ratio_disagreement", "mean"),
            three_model_spread=("three_model_spread", "mean"),
            hours=("abs_error", "size"),
        )
        for signal in ("factor_ratio_disagreement", "three_model_spread"):
            rho = float(daily.daily_mae.corr(daily[signal], method="spearman"))
            for retain in (.9, .75, .5):
                n_keep = max(1, int(np.ceil(retain*len(daily))))
                kept = daily.sort_values([signal, "local_day"], kind="mergesort").head(n_keep)
                oracle = daily.nsmallest(n_keep, "daily_mae")
                daily_rows.append({
                    "country": code.upper(), "signal": signal, "rank_spearman": rho,
                    "target_day_retention": retain, "actual_day_retention": len(kept)/len(daily),
                    "hour_retention": float(kept.hours.sum()/daily.hours.sum()),
                    "all_day_mae": float(daily.daily_mae.mean()),
                    "retained_day_mae": float(kept.daily_mae.mean()),
                    "retained_to_all_mae_ratio": float(kept.daily_mae.mean()/daily.daily_mae.mean()),
                    "oracle_retained_mae": float(oracle.daily_mae.mean()),
                    "oracle_to_all_mae_ratio": float(oracle.daily_mae.mean()/daily.daily_mae.mean()),
                    "days_total": int(len(daily)), "days_retained": int(len(kept)),
                })
    detailed = pd.DataFrame(daily_rows)
    detailed.to_csv(out/"country_risk_coverage.csv", index=False)
    macro = detailed.groupby(["signal", "target_day_retention"], as_index=False).agg(
        macro_rank_spearman=("rank_spearman", "mean"),
        actual_day_retention=("actual_day_retention", "mean"),
        hour_retention=("hour_retention", "mean"),
        retained_to_all_mae_ratio=("retained_to_all_mae_ratio", "mean"),
        worst_country_retained_ratio=("retained_to_all_mae_ratio", "max"),
        best_country_retained_ratio=("retained_to_all_mae_ratio", "min"),
        oracle_retained_to_all_mae_ratio=("oracle_to_all_mae_ratio", "mean"),
    )
    macro.to_csv(out/"macro_risk_coverage.csv", index=False)
    summary = {
        "purpose": "posthoc selective-risk ranking diagnostic on double-held-out source pseudo-targets",
        "targets": [c.upper() for c in nine.COUNTRIES],
        "label": "fixed-factor generation proxy",
        "risk_signals": ["factor_ratio_disagreement", "three_model_spread"],
        "retention_unit": "complete local days; smallest predicted disagreement days retained",
        "threshold": "within-country retrospective risk quantile; ranking upper-bound, not deployable threshold",
        "macro": macro.to_dict("records"),
    }
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    (out/"startup_hashes.json").write_text(json.dumps({"inputs": input_hashes,
        "code": code_hashes}, indent=2)+"\n", encoding="utf-8")
    if any(sha(Path(path)) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("input changed during analysis")
    if any(sha(Path(path)) != digest for path, digest in code_hashes.items()):
        raise RuntimeError("code changed during analysis")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
