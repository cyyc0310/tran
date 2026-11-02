"""Apply the frozen foreign-only public-month model to each Chinese province."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from transcif.research.public_inference import infer, nbs_months, verify_input_seal

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    origin = pd.Timestamp(args.origin)
    if origin.tzinfo is None or not 1 <= args.hours <= 168:
        raise ValueError("origin must include a timezone and hours must be in 1..168")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    source_paths = [args.model, args.panel, args.manifest, args.seal]
    input_hashes = {str(path.resolve()): sha256(path) for path in source_paths}
    panel_bytes = args.panel.read_bytes()
    manifest_bytes = args.manifest.read_bytes()
    seal = json.loads(args.seal.read_text(encoding="utf-8"))
    available_at = verify_input_seal(seal, panel_bytes, manifest_bytes)
    panel = pd.read_csv(args.panel)
    model = joblib.load(args.model)

    predictions: list[pd.DataFrame] = []
    features: list[pd.DataFrame] = []
    accepted_months: list[pd.DataFrame] = []
    province_rows: list[dict] = []
    for province in sorted(panel.province.dropna().unique()):
        try:
            months, input_audit = nbs_months(panel, province, origin, available_at)
            prediction, public_features, metadata = infer(model, months, origin, args.hours)
        except ValueError as exc:
            province_rows.append({
                "province": province,
                "status": "rejected",
                "reason": str(exc),
            })
            continue

        prediction.insert(0, "province", province)
        public_features.insert(0, "province", province)
        predictions.append(prediction)
        features.append(public_features)
        accepted_months.append(months.assign(province=province).reset_index())
        province_rows.append({
            "province": province,
            "status": "forecasted",
            "latest_allowed_month": input_audit["latest_allowed_month"],
            "eligible_months": input_audit["eligible_months"],
            "excluded_months": len(input_audit["excluded_months"]),
            "prediction_rows": len(prediction),
            "outside_source_coordinate_support": metadata[
                "outside_source_coordinate_support"
            ],
            "input_audit": input_audit,
        })

    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
        pd.concat(features, ignore_index=True).to_csv(output / "public_features.csv", index=False)
        pd.concat(accepted_months, ignore_index=True).to_csv(
            output / "accepted_months.csv", index=False
        )

    summary = {
        "experiment": "all-province Chinese public-month label-free inference",
        "origin": origin.isoformat(),
        "horizon_hours": args.hours,
        "forecast_target": "generation-side fixed-factor proxy; not consumption CIF",
        "target_hourly_labels_read": False,
        "target_model_fitting": False,
        "source_model_metadata": model["metadata"],
        "input_snapshot_available_at": available_at.isoformat(),
        "province_count_in_panel": int(panel.province.nunique()),
        "forecasted_provinces": sum(row["status"] == "forecasted" for row in province_rows),
        "rejected_provinces": sum(row["status"] == "rejected" for row in province_rows),
        "provinces": province_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if any(sha256(path) != input_hashes[str(path.resolve())] for path in source_paths):
        raise RuntimeError("an input changed during inference")
    provenance = {
        "sha256": {
            **input_hashes,
            str(Path(__file__).resolve()): sha256(Path(__file__)),
            str((ROOT / "src/transcif/research/public_inference.py").resolve()): sha256(
                ROOT / "src/transcif/research/public_inference.py"
            ),
        },
        "output_sha256": {
            path.name: sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "province_count_in_panel", "forecasted_provinces", "rejected_provinces"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
