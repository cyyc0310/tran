#!/usr/bin/env python3
"""Seal score outputs and the already-frozen response label for AEMO fuelshare r1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/aemo-fuelshare-response-transfer-r1"
LABEL_DIR = ROOT / "results/aemo-cn5-common-ef-cross-domain-r5"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    prediction_seal = json.loads((OUT / "prediction_seal.json").read_text())
    label_seal = json.loads((LABEL_DIR / "target_label_seal.json").read_text())
    prediction = OUT / "candidate_predictions.csv.gz"
    labels = LABEL_DIR / "target_fixed_ef_labels.csv"
    if sha(prediction) != prediction_seal["candidate_prediction_sha256"]:
        raise ValueError("candidate predictions do not match prediction seal")
    if sha(labels) != label_seal["labels_sha256"]:
        raise ValueError("common fixed-EF labels do not match their pre-existing seal")
    payload = {
        "status": "SCORED_AFTER_PREDICTION_SEAL",
        "prediction_sha256": sha(prediction),
        "target_label_sha256": sha(labels),
        "preexisting_target_label_seal_sha256": sha(LABEL_DIR / "target_label_seal.json"),
        "baseline_prediction_sha256": prediction_seal["baseline_prediction_sha256"],
        "score_files_sha256": {
            name: sha(OUT / name) for name in ("region_metrics.csv", "macro_metrics.csv", "status.json")
        },
        "protocol_sha256": prediction_seal["protocol_sha256"],
        "prediction_seal_sha256": sha(OUT / "prediction_seal.json"),
        "scored_hours_all_source_configs": 64989,
        "scored_hours_per_source_config": 21663,
    }
    (OUT / "score_seal.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
