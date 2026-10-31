"""Source-held-out test of coarse monthly mix as a level-bias correction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/disjoint-calibration-v50"
MIX = ROOT / "results/disjoint-calibration-v50-full-mix-probe"
OUT = ROOT / "results/disjoint-calibration-v50-monthly-level-correction"
sys.path.insert(0, str(ROOT))
from scripts import disjoint_calibration_v50 as helper  # noqa: E402

SOURCES, EXTERNAL = helper.LEGACY, helper.EXTERNAL
BASES = helper.BASES
METHODS = ("GlobalABS", "MCStd", "Component", "HETScale", "HETLocationScale", "CQRplus")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def crps(samples: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Exact empirical CRPS for case-by-sample arrays."""
    x = np.asarray(samples, float)
    y = np.asarray(observation, float)
    if x.ndim != 2 or y.shape != (x.shape[0],) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite case-by-sample CRPS inputs required")
    ordered = np.sort(x, axis=1, kind="stable")
    n = ordered.shape[1]
    coeff = 2 * np.arange(1, n + 1) - n - 1
    half_pairwise = (ordered * coeff[None, :]).sum(axis=1) / (n * n)
    return np.mean(np.abs(x - y[:, None]), axis=1) - half_pairwise


def interval_metrics(raw: np.ndarray, truth: np.ndarray, offset: float) -> dict[str, float]:
    shifted = np.clip(raw + offset, 0.0, 1000.0)
    center = shifted.mean(axis=1, keepdims=True)
    expanded = np.sort(np.clip(center + 2.0 * (shifted - center), 0.0, 1000.0), axis=1)
    lo, hi = expanded[:, 6], expanded[:, 121]
    hit = (truth >= lo) & (truth <= hi)
    lower, upper = truth < lo, truth > hi
    width = hi - lo
    score = width + 20 * np.maximum(lo - truth, 0) + 20 * np.maximum(truth - hi, 0)
    return {
        "hourly_crps": float(crps(shifted.transpose(0, 2, 1).reshape(-1, shifted.shape[1]),
                                  truth.reshape(-1)).mean()),
        "daily_mean_crps": float(crps(shifted.mean(axis=2), truth.mean(axis=1)).mean()),
        "coverage90": float(hit.mean()), "width90": float(width.mean()),
        "interval_score90": float(score.mean()), "lower_miss": float(lower.mean()),
        "upper_miss": float(upper.mean()),
        "fraction_clipped_after_shift": float(np.mean((raw + offset < 0) | (raw + offset > 1000))),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve existing level-correction probe: {OUT}")
    freeze = read_json(RUN / "freeze.json")
    prediction_seal = read_json(RUN / "prediction_seal.json")
    calibration_seal = read_json(RUN / "calibration_seal.json")
    if (read_json(RUN / "completed.json").get("status") != "COMPLETE"
            or prediction_seal.get("freeze_sha256") != sha(RUN / "freeze.json")
            or calibration_seal.get("prediction_seal_sha256") != sha(RUN / "prediction_seal.json")
            or calibration_seal.get("risk_refits") != 0):
        raise ValueError("Completed, hash-linked V50 source run required")
    mix_summary = read_json(MIX / "summary.json")
    feature_path = MIX / "source_feature.csv"
    if (mix_summary.get("status") != "COMPLETE_EXPLORATORY_INFORMATION_VALUE_PROBE"
            or sha(feature_path) != mix_summary.get("feature_csv_sha256")):
        raise ValueError("Hash-bound full-mix feature table required")
    feature_frame = pd.read_csv(feature_path).set_index("source")
    features = feature_frame["mean_legal_monthly_EF_proxy"].to_dict()
    if set(features) != set(EXTERNAL):
        raise ValueError("Exact ten full-mix source features required")

    entries = {(entry["outer"], entry["region"]): entry for entry in prediction_seal["entries"]
               if entry.get("role") == "calibration" and entry.get("region") in EXTERNAL}
    label_entries = {entry["region"]: entry for entry in calibration_seal["labels"]}
    if set(entries) != {(outer, source) for outer in SOURCES for source in EXTERNAL} or set(label_entries) != set(EXTERNAL):
        raise ValueError("Exact 70 external prediction and ten label archives required")

    labels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    archives: dict[tuple[str, str], np.ndarray] = {}
    opened = []
    for source in EXTERNAL:
        label_path = RUN / "external_labels" / f"{source}.npz"
        if label_entries[source]["path"] != str(label_path) or sha(label_path) != label_entries[source]["sha256"]:
            raise ValueError("External label archive seal mismatch")
        with np.load(label_path, allow_pickle=False) as z:
            origins, positions, truth = z["origins"].astype(np.int64), z["positions"].astype(int), z["truth"].astype(float)
        if not np.array_equal(origins, helper.CAL_DAYS.asi8[positions]) or truth.shape != (len(positions), 24):
            raise ValueError("External label support identity mismatch")
        keep = helper.eligible(pd.DatetimeIndex(pd.to_datetime(origins, utc=True)), 90)
        if int(keep.sum()) != freeze["lag_counts"]["90"][source]:
            raise ValueError("Frozen 90-day support count mismatch")
        labels[source] = (positions[keep], truth[keep])
        for outer in SOURCES:
            entry = entries[(outer, source)]
            path = RUN / "predictions" / f"{outer}__{source}.npz"
            if entry["path"] != str(path) or sha(path) != entry["sha256"]:
                raise ValueError("External prediction package seal mismatch")
            with np.load(path, allow_pickle=False) as z:
                if (z["outer"].item() != outer or z["region"].item() != source
                        or z["arms"].tolist() != list(BASES) or z["methods"].tolist() != list(METHODS)
                        or not np.array_equal(z["origins"], helper.CAL_DAYS.asi8)):
                    raise ValueError("External prediction archive identity mismatch")
                raw = np.take(z["raw"][0], labels[source][0], axis=0).astype(float, copy=True)
            archives[(outer, source)] = raw
            opened.append(str(path))

    bias_lookup = {}
    for outer in SOURCES:
        for source in EXTERNAL:
            raw = archives[(outer, source)]
            truth = labels[source][1]
            center = raw.mean(axis=1)
            bias_lookup[(outer, source)] = float(np.mean(truth - center))

    bias_rows, metric_rows = [], []
    for outer in SOURCES:
        for held in EXTERNAL:
            train_sources = tuple(source for source in EXTERNAL if source != held)
            train_bias = np.array([bias_lookup[(outer, source)] for source in train_sources], float)
            global_offset = float(train_bias.mean())
            design = np.column_stack([np.ones(len(train_sources)),
                                      np.array([features[source] for source in train_sources], float)])
            beta = np.linalg.lstsq(design, train_bias, rcond=None)[0]
            mix_offset = float(np.array([1.0, features[held]]) @ beta)
            observed = bias_lookup[(outer, held)]
            bias_rows.append({
                "outer": outer, "held_source": held, "monthly_mix_EF_proxy": features[held],
                "observed_level_bias": observed, "global_bias_offset": global_offset,
                "mix_bias_offset": mix_offset, "no_correction_abs_bias": abs(observed),
                "global_abs_residual_bias": abs(observed - global_offset),
                "mix_abs_residual_bias": abs(observed - mix_offset),
            })
            raw, truth = archives[(outer, held)], labels[held][1]
            for method, offset in (("no_correction_factor2_anchor", 0.0),
                                   ("global_source_bias_shift", global_offset),
                                   ("monthly_mix_bias_shift", mix_offset)):
                metric_rows.append({"outer": outer, "held_source": held, "method": method,
                                    "predicted_level_offset": offset,
                                    **interval_metrics(raw, truth, offset)})

    OUT.mkdir()
    bias_path, fold_path = OUT / "held_source_bias_predictions.csv", OUT / "held_source_fold_metrics.csv"
    pd.DataFrame(bias_rows).to_csv(bias_path, index=False)
    fold = pd.DataFrame(metric_rows)
    fold.to_csv(fold_path, index=False)
    source_macro = fold.groupby(["held_source", "method"], sort=False)[
        ["hourly_crps", "daily_mean_crps", "coverage90", "width90", "interval_score90",
         "lower_miss", "upper_miss", "fraction_clipped_after_shift"]].mean().reset_index()
    source_path = OUT / "source_macro_metrics.csv"
    source_macro.to_csv(source_path, index=False)
    overall = source_macro.groupby("method", sort=False)[
        ["hourly_crps", "daily_mean_crps", "coverage90", "width90", "interval_score90",
         "lower_miss", "upper_miss", "fraction_clipped_after_shift"]].mean().reset_index()
    bias_macro = pd.DataFrame(bias_rows).groupby("held_source", sort=False)[
        ["observed_level_bias", "no_correction_abs_bias", "global_abs_residual_bias", "mix_abs_residual_bias"]].mean().reset_index()
    bias_summary = {
        "no_correction_MAE": float(bias_macro.no_correction_abs_bias.mean()),
        "global_bias_shift_MAE": float(bias_macro.global_abs_residual_bias.mean()),
        "monthly_mix_shift_MAE": float(bias_macro.mix_abs_residual_bias.mean()),
        "mix_wins_vs_global_sources": int((bias_macro.mix_abs_residual_bias < bias_macro.global_abs_residual_bias).sum()),
        "source_count": len(bias_macro),
    }
    summary = {
        "status": "COMPLETE_SOURCE_HELDOUT_LEVEL_INFORMATION_PROBE",
        "protocol": "docs/V50_MONTHLY_MIX_LEVEL_CORRECTION_PROTOCOL.md",
        "protocol_sha256": sha(ROOT / "docs/V50_MONTHLY_MIX_LEVEL_CORRECTION_PROTOCOL.md"),
        "script_sha256": sha(Path(__file__)),
        "prediction_packages_opened": len(opened), "external_label_packages_opened": len(labels),
        "target_prediction_packages_opened": 0, "target_or_unseen_scores_opened": 0,
        "point_or_risk_refits": 0, "source_level_regressions": len(SOURCES) * len(EXTERNAL),
        "source_units": len(EXTERNAL), "held_source_fold_rows": len(metric_rows),
        "monthly_mix_feature_sha256": sha(feature_path),
        "bias_prediction_summary": bias_summary,
        "source_macro_metrics": overall.to_dict(orient="records"),
        "input_hashes": {"freeze": sha(RUN / "freeze.json"),
                         "prediction_seal": sha(RUN / "prediction_seal.json"),
                         "calibration_seal": sha(RUN / "calibration_seal.json"),
                         "mix_feature_table": sha(feature_path)},
        "bias_csv_sha256": sha(bias_path), "fold_metrics_csv_sha256": sha(fold_path),
        "source_macro_metrics_csv_sha256": sha(source_path),
        "interpretation_limit": "Ten previously used sources only; source held-out descriptive evidence, not a China transfer or conditional coverage guarantee.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "bias_prediction_summary": summary["bias_prediction_summary"],
        "source_macro_metrics": summary["source_macro_metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
