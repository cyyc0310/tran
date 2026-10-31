"""Source-held-out interval comparison for coarse-mix-conditioned risk."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/disjoint-calibration-v50"
MIX = ROOT / "results/disjoint-calibration-v50-full-mix-probe"
OUT = ROOT / "results/disjoint-calibration-v50-coarse-mix-conditional-interval"
sys.path.insert(0, str(ROOT))
from scripts import disjoint_calibration_v50 as helper  # noqa: E402

SOURCES, EXTERNAL = helper.LEGACY, helper.EXTERNAL
BASES = helper.BASES
METHODS = ("GlobalABS", "MCStd", "Component", "HETScale", "HETLocationScale", "CQRplus")
N_METHOD = {"MCStd": 1, "HETScale": 3}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def inverse(values: np.ndarray, level: float) -> float:
    ordered = np.sort(values.reshape(-1), kind="stable")
    return float(ordered[max(0, math.ceil(level * len(ordered)) - 1)])


def equal_source_quantile(scores: dict[str, np.ndarray], level: float = .9) -> float:
    sources = tuple(scores)
    counts = {source: scores[source].shape[0] for source in sources}
    denominator = math.lcm(*(count * 24 for count in counts.values()))
    total = len(sources) * denominator
    values, weights = [], []
    for source in sources:
        values.append(scores[source].reshape(-1))
        weight = denominator // (counts[source] * 24)
        weights.append(np.full(scores[source].size, weight, dtype=object))
    value, weight = np.concatenate(values), np.concatenate(weights)
    order = np.argsort(value, kind="stable")
    cumulative = np.cumsum(weight[order], dtype=object)
    threshold = math.ceil(level * total - 1e-12)
    return float(value[order[int(np.searchsorted(cumulative, threshold, side="left"))]])


def metrics(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> dict[str, float]:
    hit = (y >= lo) & (y <= hi)
    lower, upper = y < lo, y > hi
    width = hi - lo
    score = width + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
    return {
        "coverage90": float(hit.mean()), "width90": float(width.mean()),
        "interval_score90": float(score.mean()), "lower_miss": float(lower.mean()),
        "upper_miss": float(upper.mean()),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve existing interval experiment: {OUT}")
    freeze = read_json(RUN / "freeze.json")
    pred_seal = read_json(RUN / "prediction_seal.json")
    cal_seal = read_json(RUN / "calibration_seal.json")
    if (read_json(RUN / "completed.json").get("status") != "COMPLETE"
            or cal_seal.get("risk_refits") != 0
            or pred_seal.get("freeze_sha256") != sha(RUN / "freeze.json")
            or cal_seal.get("prediction_seal_sha256") != sha(RUN / "prediction_seal.json")):
        raise ValueError("Completed V50 seals required")

    mix_summary = read_json(MIX / "summary.json")
    mix_table_path = MIX / "source_feature.csv"
    if (mix_summary.get("status") != "COMPLETE_EXPLORATORY_INFORMATION_VALUE_PROBE"
            or sha(mix_table_path) != mix_summary.get("feature_csv_sha256")):
        raise ValueError("Hash-matched full-mix feature table required")
    feature_table = pd.read_csv(mix_table_path).set_index("source")
    features = feature_table["mean_legal_monthly_EF_proxy"].to_dict()
    if set(features) != set(EXTERNAL):
        raise ValueError("Exact ten source feature records required")

    prediction_entries = {(e["outer"], e["region"]): e for e in pred_seal["entries"]
                          if e.get("role") == "calibration" and e.get("region") in EXTERNAL}
    label_entries = {e["region"]: e for e in cal_seal["labels"]}
    if (set(prediction_entries) != {(outer, source) for outer in SOURCES for source in EXTERNAL}
            or set(label_entries) != set(EXTERNAL)):
        raise ValueError("Exact fixed external grid required")

    labels, score_cache, het_score_cache, pred_cache = {}, {}, {}, {}
    opened_predictions = []
    for source in EXTERNAL:
        label_path = RUN / "external_labels" / f"{source}.npz"
        if label_entries[source]["path"] != str(label_path) or sha(label_path) != label_entries[source]["sha256"]:
            raise ValueError("External label seal mismatch")
        with np.load(label_path, allow_pickle=False) as z:
            positions, truth, origins = z["positions"].astype(int), z["truth"].copy(), z["origins"].copy()
        keep = helper.eligible(pd.DatetimeIndex(pd.to_datetime(origins, utc=True)), 90)
        if int(keep.sum()) != freeze["lag_counts"]["90"][source]:
            raise ValueError("Frozen 90-day source support mismatch")
        labels[source] = (positions[keep], truth[keep])
        score_cache[source], het_score_cache[source], pred_cache[source] = {}, {}, {}
        for outer in SOURCES:
            entry = prediction_entries[(outer, source)]
            path = RUN / "predictions" / f"{outer}__{source}.npz"
            if entry["path"] != str(path) or sha(path) != entry["sha256"]:
                raise ValueError("External prediction seal mismatch")
            with np.load(path, allow_pickle=False) as z:
                if (z["outer"].item() != outer or z["region"].item() != source
                        or z["arms"].tolist() != list(BASES) or z["methods"].tolist() != list(METHODS)):
                    raise ValueError("Prediction archive identity mismatch")
                pos, truth = labels[source]
                lo0, hi0, scale = (np.take(z[name][0, 1], pos, axis=0) for name in ("lo0", "hi0", "scales"))
                score = np.maximum(np.maximum((lo0 - truth) / scale, (truth - hi0) / scale), 0.0)
                if not np.isfinite(score).all():
                    raise ValueError("Nonfinite hourly calibration score")
                score_cache[source][outer] = score
                het_bounds = {name: np.take(z[name][0, N_METHOD["HETScale"]], pos, axis=0)
                              for name in ("lo0", "hi0", "scales")}
                het_score_cache[source][outer] = np.maximum(np.maximum(
                    (het_bounds["lo0"] - truth) / het_bounds["scales"],
                    (truth - het_bounds["hi0"]) / het_bounds["scales"]), 0.0)
                pred_cache[source][outer] = {
                    "lo0": lo0.copy(), "hi0": hi0.copy(), "scale": scale.copy(),
                    "truth": truth.copy(), "raw": np.take(z["raw"][0], pos, axis=0),
                }
                # Keep HETScale bounds as a separate, fixed strong-risk baseline.
                pred_cache[source][outer]["het"] = het_bounds
            opened_predictions.append(str(path))

    # The label arrays above are already filtered to eligible positions; helper preserves index identity.
    rows = []
    for outer in SOURCES:
        for held in EXTERNAL:
            training_sources = tuple(source for source in EXTERNAL if source != held)
            training_scores = {source: score_cache[source][outer] for source in training_sources}
            global_q = equal_source_quantile(training_scores, .9)
            train_x = np.array([features[source] for source in training_sources], float)
            train_y = np.array([inverse(training_scores[source], .9) for source in training_sources], float)
            design = np.column_stack([np.ones(len(train_x)), train_x])
            beta = np.linalg.lstsq(design, train_y, rcond=None)[0]
            q_local = max(0.0, float(np.array([1.0, features[held]]) @ beta))
            stored = pred_cache[held][outer]
            y = stored["truth"]
            q2_rows = (
                ("global_MCStd", global_q, stored["lo0"], stored["hi0"], stored["scale"]),
                ("mix_conditional_MCStd", q_local, stored["lo0"], stored["hi0"], stored["scale"]),
            )
            for name, q, lo0, hi0, scale in q2_rows:
                lo, hi = np.clip(lo0 - q * scale, 0, 1000), np.clip(hi0 + q * scale, 0, 1000)
                rows.append({"outer": outer, "held_source": held, "method": name,
                             "q_prediction": q, **metrics(lo, hi, y)})

            # Standard heteroscedastic scale model, with its own global equal-region calibration.
            het_scores = {source: het_score_cache[source][outer] for source in training_sources}
            het_q = equal_source_quantile(het_scores, .9)
            held_het = stored["het"]
            rows.append({"outer": outer, "held_source": held, "method": "global_HETScale",
                         "q_prediction": het_q,
                         **metrics(np.clip(held_het["lo0"] - het_q * held_het["scales"], 0, 1000),
                                   np.clip(held_het["hi0"] + het_q * held_het["scales"], 0, 1000), y)})

            raw = stored["raw"]
            center = raw.mean(axis=1, keepdims=True)
            expanded = np.sort(np.clip(center + 2.0 * (raw - center), 0, 1000), axis=1)
            rows.append({"outer": outer, "held_source": held, "method": "raw_anchor_factor2",
                         "q_prediction": 2.0,
                         **metrics(expanded[:, 6], expanded[:, 121], y)})

    if len(rows) != len(SOURCES) * len(EXTERNAL) * 4:
        raise ValueError("Expected fixed 280 held-source interval rows")
    frame = pd.DataFrame(rows)
    OUT.mkdir()
    rows_path = OUT / "held_source_metrics.csv"
    frame.to_csv(rows_path, index=False)
    source_macro = frame.groupby(["held_source", "method"], sort=False)[
        ["coverage90", "width90", "interval_score90", "lower_miss", "upper_miss"]].mean().reset_index()
    macro_path = OUT / "source_macro_metrics.csv"
    source_macro.to_csv(macro_path, index=False)
    overall = source_macro.groupby("method", sort=False)[
        ["coverage90", "width90", "interval_score90", "lower_miss", "upper_miss"]].mean().reset_index()
    summary = {
        "status": "COMPLETE_DEVELOPMENT_SOURCE_HELDOUT_INTERVAL_COMPARISON",
        "protocol": "docs/V50_COARSE_MIX_CONDITIONAL_INTERVAL_PROTOCOL.md",
        "protocol_sha256": sha(ROOT / "docs/V50_COARSE_MIX_CONDITIONAL_INTERVAL_PROTOCOL.md"),
        "script_sha256": sha(Path(__file__)),
        "prediction_packages_opened": len(opened_predictions), "external_label_packages_opened": len(labels),
        "target_prediction_packages_opened": 0, "target_or_unseen_scores_opened": 0,
        "point_or_risk_refits": 0, "conditional_diagnostic_fits": len(SOURCES) * len(EXTERNAL),
        "source_units": len(EXTERNAL), "held_source_rows": len(rows),
        "same_budget_monthly_feature_sha256": sha(MIX / "source_feature.csv"),
        "source_macro_metrics": overall.to_dict(orient="records"),
        "held_source_metrics_sha256": sha(rows_path), "source_macro_metrics_sha256": sha(macro_path),
        "interpretation_limit": "Ten previously used external regions only. No distribution-free region-conditional guarantee or unseen-domain evidence.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "source_macro_metrics": summary["source_macro_metrics"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
