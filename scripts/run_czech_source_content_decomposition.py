"""Decompose Czech source-content and calibration-pool effects with matched PX controls."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_czech_expansion as czech  # noqa: E402
from transcif.research.multicountry import FACTORS, FINE, predict, weights  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

base = czech.base
OLD_COUNTRIES = tuple(base.COUNTRIES)
TARGETS = OLD_COUNTRIES
CZ = "cz"
OLD_DATA = czech.OLD_DATA
NEW_DATA = czech.NEW_DATA
OLD_MANIFEST = czech.OLD_MANIFEST
OLD_AUDIT = czech.OLD_AUDIT
CZ_RAW = czech.CZ_RAW
CZ_MANIFEST = czech.CZ_MANIFEST
PROTOCOL = ROOT / "docs/CZECH_SOURCE_CONTENT_DECOMPOSITION_PROTOCOL_20261101.md"
BASELINE = ROOT / "docs/evidence/energy-charts-loco-capacity-r3/results"
SEED = 20261101


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_block(prepared: dict, code: str, mask: pd.Series | np.ndarray):
    raw, features, _ = prepared[code]
    part = features.loc[mask].copy()
    fuel = czech.fuel_for(raw, part)
    coarse = base.coarse_values(raw.iloc[part.raw_row.to_numpy()])
    return part.reset_index(drop=True), fuel, coarse


def combine(blocks):
    frame = pd.concat([b[0] for b in blocks], ignore_index=True)
    fuel = np.concatenate([b[1] for b in blocks], axis=0)
    coarse = np.concatenate([b[2] for b in blocks], axis=0)
    if not (len(frame) == len(fuel) == len(coarse)):
        raise ValueError("unaligned source feature/fuel/coarse blocks")
    return frame, fuel, coarse


def subset_block(block, idx: np.ndarray, country: str):
    frame = block[0].iloc[idx].copy().reset_index(drop=True)
    frame["country"], frame["region"] = country, country
    return frame, block[1][idx], block[2][idx]


def pseudo_indices(pool_n: int, size: int, seed: int) -> np.ndarray:
    if size > pool_n:
        raise ValueError(f"matched pseudo source size {size} exceeds old source pool {pool_n}")
    return np.random.default_rng(seed).choice(pool_n, size=size, replace=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="root seed for matched pseudo-source train/calibration draws")
    args = parser.parse_args()
    seed = args.seed
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    old_manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    cz_manifest = json.loads(CZ_MANIFEST.read_text(encoding="utf-8"))
    manifests = {c: old_manifest for c in OLD_COUNTRIES}
    manifests[CZ] = {"countries": [{"country": CZ, "sha256": cz_manifest["sha256"]}]}
    prepared, audits = {}, {}
    for country in (*OLD_COUNTRIES, CZ):
        prepared[country] = czech.prepare_country(country, manifests[country])
        audits[country] = prepared[country][2]
    (out / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")

    input_paths = [OLD_MANIFEST, OLD_AUDIT, CZ_RAW, CZ_MANIFEST]
    input_paths += [OLD_DATA / f"{c}_2022_2024.json" for c in OLD_COUNTRIES]
    input_hashes = {str(p.resolve()): sha(p) for p in input_paths}
    baseline_paths = [BASELINE / "country_metrics.csv"] + [
        BASELINE / f"{c.upper()}_heldout_predictions.csv.gz" for c in TARGETS
    ]
    baseline_hashes = {str(p.resolve()): sha(p) for p in baseline_paths}
    code_paths = [
        Path(__file__).resolve(), Path(czech.__file__).resolve(), Path(base.__file__).resolve(),
        ROOT / "src/transcif/research/multicountry.py",
        ROOT / "src/transcif/research/cross_country.py",
        ROOT / "src/transcif/research/monthly_adaptation.py", PROTOCOL,
    ]
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    (out / "startup_hashes.json").write_text(
        json.dumps({"input_sha256": input_hashes, "baseline_sha256": baseline_hashes,
                    "code_sha256": code_hashes}, indent=2) + "\n"
    )

    assignments = []
    prediction_rows = []
    prediction_objects = {}
    seals = []

    for fold_index, target in enumerate(TARGETS):
        sources = [c for c in OLD_COUNTRIES if c != target]
        old_train_blocks = [make_block(prepared, c, prepared[c][1].year.eq(2022)) for c in sources]
        old_cal_blocks = [make_block(prepared, c, prepared[c][1].year.eq(2023)
                                     & prepared[c][1].clock.dt.month.le(6)) for c in sources]
        old_train = combine(old_train_blocks)
        old_cal = combine(old_cal_blocks)
        cz_train = make_block(prepared, CZ, prepared[CZ][1].year.eq(2022))
        cz_cal = make_block(prepared, CZ, prepared[CZ][1].year.eq(2023)
                            & prepared[CZ][1].clock.dt.month.le(6))
        rng_seed_train = seed + fold_index
        rng_seed_cal = seed + 100 + fold_index
        idx_train = pseudo_indices(len(old_train[0]), len(cz_train[0]), rng_seed_train)
        idx_cal = pseudo_indices(len(old_cal[0]), len(cz_cal[0]), rng_seed_cal)
        px_train = subset_block(old_train, idx_train, "PX")
        px_cal = subset_block(old_cal, idx_cal, "PX")
        for period, idx, pool, seed in [
            ("train_2022", idx_train, old_train[0], rng_seed_train),
            ("cal_2023h1", idx_cal, old_cal[0], rng_seed_cal),
        ]:
            chosen = pool.iloc[idx]
            for position, (_, row) in enumerate(chosen.iterrows()):
                assignments.append({"target": target.upper(), "period": period, "seed": seed,
                                    "px_position": position, "sampled_country": row.country,
                                    "sampled_region": row.region, "sampled_row": int(row.raw_row),
                                    "clock": row.clock.isoformat()})

        train_blocks = {"old4": old_train, "cz": combine([old_train, cz_train]),
                        "px": combine([old_train, px_train])}
        cal_blocks = {"old4": old_cal, "cz": combine([old_cal, cz_cal]),
                      "px": combine([old_cal, px_cal])}
        models = {}
        for train_arm, block in train_blocks.items():
            models[train_arm] = base.fit(block[0], block[1], block[2])
        raw_target, feat_target, _ = prepared[target]
        query = feat_target.loc[feat_target.year.eq(2024)].copy()
        if query.empty:
            raise ValueError(f"empty target query for {target}")

        points = {arm: predict(model, query) for arm, model in models.items()}
        target_dir = out / target.upper()
        target_dir.mkdir()
        for train_arm, prediction in points.items():
            if any(not np.isfinite(values).all() for values in prediction.values()):
                raise ValueError(f"nonfinite {train_arm} prediction for {target}")
            path = target_dir / f"{train_arm}_prediction.csv.gz"
            pd.DataFrame({"time": raw_target.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                          **prediction}).to_csv(path, index=False, compression="gzip")
            seals.append({"target": target.upper(), "train_arm": train_arm,
                          "file": str(path.relative_to(out)), "sha256": sha(path), "hours": len(query)})
            prediction_objects[(target, train_arm)] = (query, path, prediction)

        # Build the three calibration radii for each point model. Calibration labels are source-only.
        for train_arm, model in models.items():
            for cal_arm, cal in cal_blocks.items():
                cal_y = cal[1] @ FACTORS / cal[1].sum(axis=1)
                cal_point = predict(model, cal[0])
                cal_w = weights(cal[0])
                radii = {method: weighted_quantile(np.abs(cal_y - values), cal_w, 0.9)
                         for method, values in cal_point.items()}
                prediction_rows.append({"target": target, "train_arm": train_arm,
                                        "cal_arm": cal_arm, "query": query,
                                        "radii": radii})

    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    pd.DataFrame(assignments).to_csv(out / "pseudo_source_assignments.csv.gz", index=False,
                                     compression="gzip")
    if any(sha(Path(path)) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("frozen input changed during execution")
    if any(sha(Path(path)) != digest for path, digest in baseline_hashes.items()):
        raise RuntimeError("frozen comparison baseline changed during execution")
    if any(sha(Path(path)) != digest for path, digest in code_hashes.items()):
        raise RuntimeError("frozen code or protocol changed during execution")

    baseline_check = []
    for target in TARGETS:
        query, _, prediction = prediction_objects[(target, "old4")]
        reference = pd.read_csv(BASELINE / f"{target.upper()}_heldout_predictions.csv.gz")
        for method, values in prediction.items():
            observed = reference[method].to_numpy(float)
            max_abs = float(np.max(np.abs(values - observed)))
            within_tolerance = max_abs <= 1e-8
            baseline_check.append({"target": target.upper(), "method": method,
                                   "rows": len(values), "within_tolerance": within_tolerance,
                                   "max_abs_difference": max_abs,
                                   "historical_comparison_eligible": within_tolerance})
            if method != "direct_history" and not within_tolerance:
                raise RuntimeError(f"historical baseline mismatch for {target}/{method}: {max_abs}")
    (out / "baseline_reproduction.json").write_text(json.dumps(baseline_check, indent=2) + "\n")

    scores = []
    for item in prediction_rows:
        target, train_arm, cal_arm = item["target"], item["train_arm"], item["cal_arm"]
        query, path, prediction = prediction_objects[(target, train_arm)]
        expected = next(x["sha256"] for x in seals if x["target"] == target.upper()
                        and x["train_arm"] == train_arm)
        if sha(path) != expected:
            raise RuntimeError(f"sealed prediction changed before scoring: {path}")
        # Target hourly labels are loaded only after every target forecast is sealed.
        truth_fuel = czech.fuel_for(prepared[target][0], query)
        truth = truth_fuel @ FACTORS / truth_fuel.sum(axis=1)
        for method, radius in item["radii"].items():
            point = prediction[method]
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                d = pd.DataFrame({"day": query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy(),
                                  "truth": truth[mask], "prediction": point[mask]})
                centered = d[["truth", "prediction"]] - d.groupby("day")[["truth", "prediction"]].transform("mean")
                err = truth[mask] - point[mask]
                scores.append({"target": target.upper(), "train_arm": train_arm, "cal_arm": cal_arm,
                               "method": method, "month": int(month), "hours": int(mask.sum()),
                               "mae": float(np.abs(err).mean()),
                               "shape_mae": float(np.abs(centered.truth - centered.prediction).mean()),
                               "coverage90": float((np.abs(err) <= radius).mean()),
                               "width90": float(2 * radius),
                               "interval_score90": float(np.mean(2 * radius + 20 * np.maximum(np.abs(err) - radius, 0)))})
    monthly = pd.DataFrame(scores)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    country = monthly.groupby(["target", "train_arm", "cal_arm", "method"])[metrics].mean().reset_index()
    country.to_csv(out / "country_metrics.csv", index=False)
    macro = country.groupby(["train_arm", "cal_arm", "method"])[metrics].mean().reset_index()
    macro.to_csv(out / "macro_metrics.csv", index=False)

    status = {"status": "DEVELOPMENT_SOURCE_CONTENT_AND_CALIBRATION_DECOMPOSITION",
              "targets": [c.upper() for c in TARGETS], "source_training_year": 2022,
              "source_calibration_period": "2023-01 through 2023-06", "target_year": 2024,
              "source_country_attribution": "Energy-Charts.info, CC BY 4.0",
              "target_hourly_labels_used_before_seal": False,
              "monthly_input": "current revised hourly data with simulated t-2 delay",
              "labels": "fixed-factor generation proxy, not measured or institutional CIF",
              "pseudo_source": "matched-count, no-replacement sample from old4 source pool; not an independent country",
              "pseudo_source_root_seed": seed,
              "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                          "scikit_learn": sklearn.__version__},
              "historical_baseline_exception": "BE/direct_history is not compared to the archived prediction because the archive has no estimator-library version metadata; all within-run contrasts remain paired"}
    (out / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    output_files = [p for p in out.rglob("*") if p.is_file() and p.name != "provenance.json"]
    provenance = {
        "input_sha256": input_hashes, "code_sha256": code_hashes,
        "baseline_sha256": {str(p.resolve()): sha(p) for p in baseline_paths},
        "output_sha256": {str(p.relative_to(out)): sha(p) for p in output_files},
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Macro metrics by training and calibration arm:")
    print(macro.loc[macro.method.isin(["ratio_recomposition", "macro_recomposition",
                                       "direct_latest", "factor_history"])]
          .to_string(index=False))
    eligible = [x for x in baseline_check if x["historical_comparison_eligible"]]
    exceptions = [x for x in baseline_check if not x["historical_comparison_eligible"]]
    print(f"Historical baseline checks within 1e-8: {len(eligible)}/{len(baseline_check)}")
    print("Documented historical comparison exceptions:", exceptions)


if __name__ == "__main__":
    main()
