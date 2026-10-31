"""Read-only source influence diagnostics for sealed V50 external calibration scores."""
from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/disjoint-calibration-v50"
OUT = ROOT / "results/disjoint-calibration-v50-source-diagnostic"
sys.path.insert(0, str(ROOT))
from scripts import disjoint_calibration_v50 as helper

SOURCES, EXTERNAL = helper.LEGACY, helper.EXTERNAL
BASES, METHODS, LAGS, RULES = helper.BASES, ("GlobalABS", "MCStd", "Component", "HETScale", "HETLocationScale", "CQRplus"), helper.LAGS, helper.RULES


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def weighted_inverse(values: dict[str, np.ndarray], days_per_source: dict[str, int], sources: tuple[str, ...], level: Fraction) -> float:
    """Exact source-equal, day-hour-equal empirical inverse CDF; no interpolation."""
    if level > 1:
        return math.inf
    denominator = math.lcm(*(days_per_source[s] * 24 for s in sources))
    per_hour_mass = {s: denominator // (days_per_source[s] * 24) for s in sources}
    total_mass = len(sources) * denominator
    flat_parts, mass_parts = [], []
    for source in sources:
        part = values[source].reshape(-1)
        flat_parts.append(part)
        mass_parts.append(np.full(part.size, per_hour_mass[source], dtype=object))
    score = np.concatenate(flat_parts)
    mass = np.concatenate(mass_parts)
    order = np.argsort(score, kind="stable")
    cumulative = np.cumsum(mass[order], dtype=object)
    target = (level.numerator * total_mass + level.denominator - 1) // level.denominator
    at = int(np.searchsorted(cumulative, target, side="left"))
    return float(score[order[at]])


def local_inverse(values: np.ndarray, level: float) -> float:
    ordered = np.sort(values.reshape(-1), kind="stable")
    return float(ordered[max(0, math.ceil(level * len(ordered)) - 1)])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve existing diagnostic output: {OUT}")
    freeze = read_json(RUN / "freeze.json")
    prediction_seal = read_json(RUN / "prediction_seal.json")
    calibration_seal = read_json(RUN / "calibration_seal.json")
    completed = read_json(RUN / "completed.json")
    if completed.get("status") != "COMPLETE" or completed.get("target_values_opened") is not False:
        raise ValueError("Completed V50 run with sealed target values required")
    if (prediction_seal.get("complete_grid") is not True or len(prediction_seal.get("entries", [])) != 77
            or calibration_seal.get("complete_grid") is not True or calibration_seal.get("risk_refits") != 0
            or calibration_seal.get("source2025_response_opened") is not False
            or prediction_seal.get("freeze_sha256") != sha(RUN / "freeze.json")
            or calibration_seal.get("freeze_sha256") != sha(RUN / "freeze.json")
            or calibration_seal.get("prediction_seal_sha256") != sha(RUN / "prediction_seal.json")
            or completed.get("prediction_seal_sha256") != sha(RUN / "prediction_seal.json")
            or completed.get("calibration_seal_sha256") != sha(RUN / "calibration_seal.json")):
        raise ValueError("Completed prediction/calibration seal chain required")

    prediction_lookup = {}
    for entry in prediction_seal["entries"]:
        if entry.get("role") == "calibration" and entry.get("region") in EXTERNAL:
            key = (entry["outer"], entry["region"])
            if key in prediction_lookup:
                raise ValueError("Duplicate external calibration archive")
            prediction_lookup[key] = entry
    if set(prediction_lookup) != {(outer, source) for outer in SOURCES for source in EXTERNAL}:
        raise ValueError("Exact seventy outer-by-external packages required")

    label_lookup = {entry["region"]: entry for entry in calibration_seal["labels"]}
    if set(label_lookup) != set(EXTERNAL):
        raise ValueError("Exact ten external label packages required")
    official = read_json(RUN / "calibration.json")["records"]
    if sha(RUN / "calibration.json") != calibration_seal["quantile_json_sha256"]:
        raise ValueError("Official calibration JSON hash mismatch")
    if sha(RUN / "calibration.csv") != calibration_seal["quantile_csv_sha256"]:
        raise ValueError("Official calibration CSV hash mismatch")
    if len(official) != 1260:
        raise ValueError("Expected fixed 1260 calibration thresholds")
    official_by_key = {(r["outer"], r["base"], r["method"], int(r["lag"]), r["rule"]): r for r in official}
    if len(official_by_key) != 1260:
        raise ValueError("Official calibration keys are not unique")

    # Load only fixed external label archives and their explicitly sealed prediction packages.
    labels = {}
    for source in EXTERNAL:
        entry = label_lookup[source]
        path = RUN / "external_labels" / f"{source}.npz"
        if entry["path"] != str(path) or sha(path) != entry["sha256"]:
            raise ValueError("External label path/hash does not match calibration seal")
        with np.load(path, allow_pickle=False) as z:
            positions = z["positions"].astype(int, copy=True)
            truth = z["truth"].astype(float, copy=True)
            keys = z["keys"].tolist()
            origins = z["origins"].astype(np.int64, copy=True)
        if truth.shape != (len(positions), 24) or len(keys) != len(positions) or len(origins) != len(positions):
            raise ValueError("External label archive shape mismatch")
        if not np.isfinite(truth).all() or np.any((positions < 0) | (positions >= len(helper.CAL_DAYS))):
            raise ValueError("External truth/positions invalid")
        if not np.array_equal(origins, helper.CAL_DAYS.asi8[positions]):
            raise ValueError("External label origin alignment mismatch")
        if keys != [f"{source}/{day:%Y-%m-%d}" for day in pd.DatetimeIndex(pd.to_datetime(origins, utc=True))]:
            raise ValueError("External label key alignment mismatch")
        labels[source] = (positions, truth, origins)

    # Scores are indexed outer/base/method/source, retaining day/hour only for the predeclared eligible days.
    scores: dict[tuple[str, str, str, str], np.ndarray] = {}
    for outer in SOURCES:
        for source in EXTERNAL:
            entry = prediction_lookup[(outer, source)]
            path = RUN / "predictions" / f"{outer}__{source}.npz"
            if entry["path"] != str(path) or sha(path) != entry["sha256"]:
                raise ValueError("External prediction path/hash does not match prediction seal")
            positions, truth, origins = labels[source]
            with np.load(path, allow_pickle=False) as z:
                if (z["outer"].item() != outer or z["region"].item() != source
                        or z["arms"].tolist() != list(BASES) or z["methods"].tolist() != list(METHODS)
                        or not np.array_equal(z["origins"], helper.CAL_DAYS.asi8)
                        or z["keys"].tolist() != [f"{source}/{day:%Y-%m-%d}" for day in helper.CAL_DAYS]):
                    raise ValueError("Prediction archive is outside fixed external calibration grid")
                lo0 = np.take(z["lo0"], positions, axis=2)
                hi0 = np.take(z["hi0"], positions, axis=2)
                scale = np.take(z["scales"], positions, axis=2)
            raw_scores = np.maximum(np.maximum((lo0 - truth[None, None, :, :]) / scale,
                                               (truth[None, None, :, :] - hi0) / scale), 0.0)
            if not np.isfinite(raw_scores).all():
                raise ValueError("Nonfinite recalculated calibration score")
            for lag in LAGS:
                keep = helper.eligible(pd.DatetimeIndex(pd.to_datetime(origins, utc=True)), lag)
                if not keep.any():
                    raise ValueError("Empty fixed-lag source support")
                for bi, base in enumerate(BASES):
                    for mi, method in enumerate(METHODS):
                        scores[(outer, base, method, source, lag)] = raw_scores[bi, mi, keep]

    rows = []
    max_official_delta = 0.0
    calibration_csv = list(csv.DictReader((RUN / "calibration.csv").open(newline="")))
    if len(calibration_csv) != 1260:
        raise ValueError("Official calibration CSV row count mismatch")
    csv_by_key = {(r["outer"], r["base"], r["method"], int(r["lag"]), r["rule"]): r for r in calibration_csv}
    if len(csv_by_key) != 1260:
        raise ValueError("Official calibration CSV keys are not unique")

    for outer in SOURCES:
        for base in BASES:
            for method in METHODS:
                for lag in LAGS:
                    source_scores = {source: scores[(outer, base, method, source, lag)] for source in EXTERNAL}
                    counts = {source: source_scores[source].shape[0] for source in EXTERNAL}
                    expected_counts = freeze["lag_counts"][str(lag)]
                    if counts != expected_counts:
                        raise ValueError("Lag support count differs from frozen source-day counts")
                    for rule in RULES:
                        key = (outer, base, method, lag, rule)
                        official_record = official_by_key[key]
                        csv_record = csv_by_key[key]
                        k = len(EXTERNAL)
                        level = Fraction(9, 10) if rule == "equal_region" else Fraction(9 * (k + 1), 10 * k)
                        q_all = weighted_inverse(source_scores, counts, EXTERNAL, level)
                        q_official = official_record["q"]
                        if q_official is None or not math.isfinite(float(q_official)):
                            raise ValueError("Fixed V50 official threshold unexpectedly infinite")
                        delta = abs(q_all - float(q_official))
                        max_official_delta = max(max_official_delta, delta)
                        if delta > 1e-10 or abs(float(csv_record["q"]) - float(q_official)) > 1e-12:
                            raise ValueError("Independent threshold reconstruction failed")
                        if official_record["source_counts"] != expected_counts:
                            raise ValueError("Official per-source counts mismatch")
                        for source in EXTERNAL:
                            local = source_scores[source]
                            q_loco = weighted_inverse(source_scores, counts,
                                tuple(s for s in EXTERNAL if s != source),
                                Fraction(9, 10) if rule == "equal_region" else Fraction(10, 10))
                            rows.append({
                                "outer": outer, "base": base, "budget": helper.BUDGETS[BASES.index(base)],
                                "method": method, "lag": lag, "rule": rule, "source": source,
                                "n_days": counts[source], "q_official": float(q_official),
                                "q_rebuilt": q_all, "q_loco": q_loco,
                                "delta_q_loco_minus_full": q_loco - q_all,
                                "local_score_p50": local_inverse(local, .50),
                                "local_score_p90": local_inverse(local, .90),
                                "local_score_p99": local_inverse(local, .99),
                                "local_exceedance_fraction_at_q": float(np.mean(local > q_official)),
                                "local_coverage_fraction_at_q": float(np.mean(local <= q_official)),
                                "official_q_rebuild_abs_delta": delta,
                            })

    if len(rows) != 12600 or max_official_delta > 1e-10:
        raise ValueError("Complete source diagnostic grid and exact official threshold reconstruction required")
    OUT.mkdir()
    csv_path = OUT / "source_effects.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    values = pd.DataFrame(rows)
    summary = {
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "protocol": "docs/V50_CALIBRATION_SOURCE_DIAGNOSTIC_PROTOCOL.md",
        "protocol_sha256": sha(ROOT / "docs/V50_CALIBRATION_SOURCE_DIAGNOSTIC_PROTOCOL.md"),
        "producer_script_sha256": sha(Path(__file__)),
        "prediction_packages_opened": 70,
        "external_label_packages_opened": 10,
        "target_prediction_packages_opened": 0,
        "target_truth_or_scoring_files_opened": 0,
        "refits": 0,
        "configuration_count": 1260,
        "source_configuration_rows": len(rows),
        "official_q_reconstruction_max_abs_delta": max_official_delta,
        "loco_delta_summary": values.groupby(["lag", "rule"])["delta_q_loco_minus_full"].agg(
            ["min", "median", "mean", "max" ]).reset_index().to_dict(orient="records"),
        "source_local_tail_summary": values.groupby("source")[["local_score_p50", "local_score_p90",
            "local_score_p99", "local_exceedance_fraction_at_q"]].mean().reset_index().to_dict(orient="records"),
        "source_loco_influence_summary": values.groupby("source")["delta_q_loco_minus_full"].agg(
            ["min", "median", "mean", "max"]).reset_index().to_dict(orient="records"),
        "input_hashes": {
            "freeze.json": sha(RUN / "freeze.json"),
            "prediction_seal.json": sha(RUN / "prediction_seal.json"),
            "calibration_seal.json": sha(RUN / "calibration_seal.json"),
            "calibration.json": sha(RUN / "calibration.json"),
            "calibration.csv": sha(RUN / "calibration.csv"),
        },
        "source_effects_csv_sha256": sha(csv_path),
        "interpretation_limit": "Descriptive calibration-source influence only; not independent target validation, a coverage guarantee, or a basis for source/lag/threshold selection.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "prediction_packages_opened",
        "external_label_packages_opened", "target_prediction_packages_opened",
        "target_truth_or_scoring_files_opened", "refits", "source_configuration_rows",
        "official_q_reconstruction_max_abs_delta")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
