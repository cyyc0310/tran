"""Post-hoc support-gate audit for the V51 identification-width risk feature."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v51-identification-width-risk-support-gate"
HALVES = ("h1", "h2")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty result directory: {OUT}")
    frames, inputs = [], {}
    for half in HALVES:
        source = ROOT / f"results/v51-identification-width-risk-r3-{half}/outer_source_month_intervals.csv"
        inputs[str(source)] = sha256(source)
        data = pd.read_csv(source)
        ordinary = data[data.method == "conditional_identification_width"].set_index(["heldout_region", "month"])
        conservative = data[data.method == "conditional_identification_width_max_source"].set_index(["heldout_region", "month"])
        if not ordinary.index.equals(conservative.index):
            raise ValueError("Ordinary and max-source month supports differ")
        outside = ordinary.outside_train_bbox.astype(bool)
        output = ordinary[["coverage", "width", "interval_score", "lower_miss", "upper_miss", "point_mae"]].copy()
        for column in output.columns:
            output[column] = output[column].where(~outside, conservative[column])
        output["heldout_region"] = output.index.get_level_values("heldout_region")
        output["month"] = output.index.get_level_values("month")
        output["half"] = half.upper()
        output["used_max_source"] = outside.to_numpy()
        output = output.reset_index(drop=True)
        frames.append(output)
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(OUT / "gated_source_month.csv", index=False)
    per_source = rows.groupby(["half", "heldout_region"], as_index=False).agg(
        months=("month", "size"), coverage=("coverage", "mean"), width=("width", "mean"),
        interval_score=("interval_score", "mean"), lower_miss=("lower_miss", "mean"),
        upper_miss=("upper_miss", "mean"), outside_fraction=("used_max_source", "mean"),
    )
    per_source.to_csv(OUT / "source_summary.csv", index=False)
    macro = per_source.groupby("half", as_index=False).agg(
        sources=("heldout_region", "nunique"), coverage=("coverage", "mean"),
        width=("width", "mean"), interval_score=("interval_score", "mean"),
        lower_miss=("lower_miss", "mean"), upper_miss=("upper_miss", "mean"),
        outside_fraction=("outside_fraction", "mean"),
    )
    macro.to_csv(OUT / "macro.csv", index=False)
    output_names = ("gated_source_month.csv", "source_summary.csv", "macro.csv")
    summary = {
        "status": "POSTHOC_SUPPORT_GATE_DIAGNOSTIC",
        "rule": "Use the normalized pooled-q90 radius when target monthly risk features are inside the calibration source axis-aligned bounding box; use max-source normalized-q90 radius outside.",
        "selection": "Post-hoc diagnostic, not preregistered or an algorithm claim; gate is deliberately a simple support check.",
        "inputs_sha256": inputs,
        "outputs_sha256": {name: sha256(OUT / name) for name in output_names},
        "macro": macro.to_dict("records"),
        "interpretation": "Does not guarantee coverage. The same feature-support violations occur in covered sources and the H1 MISO tail failure remains.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(macro.to_string(index=False))
    print(per_source.sort_values(["half", "coverage"]).to_string(index=False))


if __name__ == "__main__":
    main()
