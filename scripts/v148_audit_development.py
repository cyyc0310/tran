"""Post-run endpoint, reproduction and gradient audits for V146--V148."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v139_physical_reserve_calibration import sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v148-development-audit-r2"
V146 = ROOT / "results/v146-critical-composition-calibration-r1"
V147 = ROOT / "results/v147-conditional-curve-scale-r1"
V148 = ROOT / "results/v148-quantile-stability-r2"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    endpoint_tables, checked_hashes = [], []
    for name, directory in (("v146", V146), ("v147", V147), ("v148", V148)):
        summary = json.loads((directory / "summary.json").read_text())
        if summary["assertion_failures"]:
            raise ValueError(f"upstream audit failed: {name}")
        for relative, expected in summary["input_hashes"].items():
            if sha(ROOT / relative) != expected:
                raise ValueError(f"input changed: {relative}")
            checked_hashes.append({"experiment": name, "kind": "input", "path": relative, "sha256": expected})
        for relative, expected in summary["output_hashes"].items():
            if sha(directory / relative) != expected:
                raise ValueError(f"output changed: {name}/{relative}")
            checked_hashes.append({"experiment": name, "kind": "output", "path": str((directory / relative).relative_to(ROOT)), "sha256": expected})
        rows = pd.read_csv(directory / "target_rows.csv")
        if name == "v148":
            rows["method"] = rows.base + "/" + rows.variant + "/" + rows.order
        rows["lower_covered"] = rows.lower.le(rows.factor_box_K_lower)
        rows["upper_covered"] = rows.upper.ge(rows.factor_box_K_upper)
        if not (rows.lower_covered & rows.upper_covered).eq(rows.covered).all():
            raise ValueError("joint versus individual endpoint indicator mismatch")
        table = rows.groupby(["region", "seed", "tau", "method", "alpha"]).agg(
            months=("month", "size"), lower_coverage=("lower_covered", "mean"),
            upper_coverage=("upper_covered", "mean"), coverage=("covered", "mean"), width=("width", "mean"),
        ).reset_index().assign(experiment=name)
        expected = table.region.map({"PACE": 18, "LGEE": 24, "SC": 24})
        if expected.isna().any() or not table.months.eq(expected).all():
            raise ValueError("per-region endpoint summary denominator mismatch")
        endpoint_tables.append(table)
    earlier = pd.read_csv(V146 / "target_rows.csv")
    scaled = pd.read_csv(V147 / "target_rows.csv")
    scaled["family"] = np.where(scaled.method.str.startswith("horizontal"), "horizontal_composition_joint", "vertical_hull_joint")
    keys = ["region", "month", "tau", "seed", "alpha"]
    matched = scaled.merge(earlier, left_on=keys + ["family"], right_on=keys + ["method"],
                           suffixes=("_scaled", "_unscaled"), validate="many_to_one")
    if len(matched) != len(scaled):
        raise ValueError("V147 matching silently dropped a target prediction")
    scaled_differences = int((matched.lower_scaled.ne(matched.lower_unscaled)
                              | matched.upper_scaled.ne(matched.upper_unscaled)).sum())
    r1 = pd.read_csv(ROOT / "results/v148-quantile-stability-r1/target_rows.csv")
    r2 = pd.read_csv(V148 / "target_rows.csv")
    all_keys = keys + ["base", "variant", "order"]
    pair = r1.merge(r2, on=all_keys, suffixes=("_r1", "_r2"), validate="one_to_one")
    if len(pair) != len(r1) or len(pair) != len(r2):
        raise ValueError("V148 preflight/final denominator mismatch")
    preflight_differences = int((pair.lower_r1.ne(pair.lower_r2) | pair.upper_r1.ne(pair.upper_r2)).sum())
    if scaled_differences or preflight_differences:
        raise ValueError("expected exact target reproduction failed")
    gradients = np.load(V148 / "initial_source_gradients.npz", allow_pickle=False)
    manifest = pd.read_csv(V148 / "gradient_manifest.csv")
    checks = []
    for row in manifest.itertuples(index=False):
        grad = gradients["gradient"][row.offset:row.stop]
        ids = gradients["profile_id"][row.offset:row.stop]
        if (hashlib.sha256(grad.tobytes()).hexdigest() != row.gradient_sha256
                or hashlib.sha256(ids.tobytes()).hexdigest() != row.profile_sha256):
            raise ValueError("stored source gradient/profile digest mismatch")
        checks.append({"target_region": row.target_region, "seed": row.seed, "tau": row.tau,
            "base": row.base, "variant": row.variant, "order": row.order, "side": row.side,
            "gradient_values": len(grad), "gradient_unique_values": int(np.unique(grad).size),
            "minimum": float(grad.min()), "maximum": float(grad.max())})
    if manifest.iloc[0].offset != 0 or manifest.iloc[-1].stop != len(gradients["gradient"]):
        raise ValueError("gradient manifest does not span array")
    if not np.array_equal(manifest.offset.to_numpy()[1:], manifest.stop.to_numpy()[:-1]):
        raise ValueError("gradient manifest has gaps or overlaps")
    diag = pd.read_csv(V148 / "training_diagnostics.csv")
    check_frame = pd.DataFrame(checks)
    joined = check_frame.merge(diag, on=["target_region", "seed", "tau", "base", "variant", "order", "side"],
                               suffixes=("_stored", "_reported"), validate="one_to_one")
    if len(joined) != len(manifest) or len(joined) != len(diag):
        raise ValueError("gradient diagnostic matching silently dropped a block")
    if not joined.gradient_unique_values_stored.eq(joined.gradient_unique_values_reported).all():
        raise ValueError("gradient diagnostics differ from stored values")
    OUT.mkdir(parents=True)
    pd.concat(endpoint_tables, ignore_index=True).to_csv(OUT / "individual_endpoint_coverage.csv", index=False)
    pd.DataFrame(checked_hashes).to_csv(OUT / "verified_hashes.csv", index=False)
    joined.to_csv(OUT / "gradient_contract_audit.csv", index=False)
    report = {"assertion_failures": 0, "verified_hashes": len(checked_hashes),
        "v147_matched_target_rows": len(matched), "v147_changed_bounds": scaled_differences,
        "v148_preflight_matched_rows": len(pair), "v148_preflight_changed_bounds": preflight_differences,
        "source_gradient_blocks": len(manifest), "source_gradient_values": len(gradients["gradient"]),
        "script_sha256": sha(Path(__file__)), "output_hashes": {p.name: sha(p) for p in sorted(OUT.iterdir())},
        "scope": "V148 r2 preserves r1 predictions while repairing helper API contracts and saving full source gradients."}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
