"""Run frozen V321 LP unchanged under wider/narrower residual EF supports."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts/v321_unobserved_generation_remainder_bounds.py"
OUT = ROOT / "results/v322-residual-factor-sensitivity-r1"
PROTOCOL = ROOT / "docs/V322_RESIDUAL_FACTOR_SUPPORT_SENSITIVITY_PROTOCOL.md"
BASE_V321 = ROOT / "results/v321-unobserved-generation-remainder-r1"
SCENARIOS = [0.5, 1.0, 1.5]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("v321_runner", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen V321 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(summary: dict[str, object], diameter_path: Path, upper: float) -> dict[str, object]:
    data = pd.read_csv(diameter_path)
    widths = data.pivot(
        index=["province", "period"],
        columns=["supply_information", "monthly_receipts_observed"],
        values="worst_case_diameter_kgco2e_kwh_proxy",
    )
    rng = np.random.default_rng(32200 + int(upper * 10))
    result: dict[str, object] = {
        "unobserved_ef_support": [0.0, upper],
        "summary_by_information_set": summary["summary_by_information_set"],
    }
    for name, high, low in [
        ("category_gain_without_receipts", ("five_categories", False), ("total_only", False)),
        ("category_gain_with_receipts", ("five_categories", True), ("total_only", True)),
    ]:
        shrink = 1 - widths[high] / widths[low]
        by_province = shrink.groupby(level="province").mean().to_numpy()
        boot = np.array([
            rng.choice(by_province, size=len(by_province), replace=True).mean()
            for _ in range(20000)
        ])
        result[name] = {
            "mean_paired_shrink": float(shrink.mean()),
            "median_paired_shrink": float(shrink.median()),
            "positive_cells": int((shrink > 1e-9).sum()),
            "positive_provinces": int((shrink.groupby(level="province").mean() > 1e-9).sum()),
            "province_cluster_bootstrap_95pct": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_runner()
    all_scenarios = []
    width_tables = {}
    for upper in SCENARIOS:
        if upper == 1.0:
            out_dir = BASE_V321
            summary_path = out_dir / "summary.json"
            diameter_path = out_dir / "province_period_diameters.csv"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            out_dir = OUT / f"ef-upper-{upper:.1f}"
            module.UNOBSERVED_EF_MAX = upper
            module.OUT = out_dir
            module.main()
            summary_path = out_dir / "summary.json"
            diameter_path = out_dir / "province_period_diameters.csv"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["unobserved_ef_support_note"] = (
                f"V322 sensitivity scenario [0, {upper:.1f}]. The endpoint is an analyst-set scenario, not a validated engineering limit."
            )
            summary["v322_protocol_sha256"] = sha256(PROTOCOL)
            summary["v322_runner_sha256"] = sha256(Path(__file__))
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        all_scenarios.append(metrics(summary, diameter_path, upper))
        frame = pd.read_csv(diameter_path)
        width_tables[upper] = frame.pivot(
            index=["province", "period"],
            columns=["supply_information", "monthly_receipts_observed"],
            values="worst_case_diameter_kgco2e_kwh_proxy",
        )

    # Monotonic audit: enlarging a feasible factor support cannot narrow any
    # interval under otherwise identical constraints.
    monotone = True
    violations = []
    for lower, upper in zip(SCENARIOS, SCENARIOS[1:]):
        for arm in width_tables[lower].columns:
            delta = width_tables[upper][arm] - width_tables[lower][arm]
            if (delta < -1e-8).any():
                monotone = False
                violations.append({
                    "lower_support": lower,
                    "upper_support": upper,
                    "arm": arm,
                    "negative_cell_count": int((delta < -1e-8).sum()),
                    "minimum_width_change": float(delta.min()),
                })
    summary = {
        "experiment": "V322 unobserved generation factor-support sensitivity",
        "protocol_sha256": sha256(PROTOCOL),
        "sensitivity_script_sha256": sha256(Path(__file__)),
        "base_v321_script_sha256": sha256(BASE_SCRIPT),
        "scenarios": all_scenarios,
        "mean_diameter_monotonicity_with_wider_support": {
            "pass": monotone,
            "violations": violations,
        },
        "interpretation": "Scenario sensitivity only; none of the factor endpoints is asserted to be a validated engineering bound.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
