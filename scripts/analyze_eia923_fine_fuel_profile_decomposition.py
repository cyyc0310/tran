"""Post-hoc decomposition of fine-fuel monthly levels and source hour profiles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_live_eia923_ba_vintage as vintage
import run_actual_issue_cross_country as source_model
import run_eia923_fine_fuel_composition_increment as fine
from transcif.research.cross_country import EF, FUELS, labels, metrics

OUT = ROOT / "results/eia923-fine-fuel-profile-decomposition-r1"
EIA923 = ROOT / "docs/evidence/eia923-live-monthly-2026-08-r1/f923_2026.zip"
PROTOCOL = ROOT / "docs/EIA923_FINE_FUEL_PROFILE_DECOMPOSITION_PROTOCOL_20261101.md"
PRIOR = ROOT / "results/eia923-fine-fuel-composition-increment-r1"
PRIOR_FILES = [PRIOR / name for name in (
    "ba_month_method_metrics.csv", "hourly_predictions_and_labels.csv.gz", "provenance.json")]
SOURCE_INPUTS = list(fine.SOURCE_INPUTS)
CODE_INPUTS = [Path(__file__).resolve(), Path(vintage.__file__).resolve(),
               Path(source_model.__file__).resolve(), Path(fine.__file__).resolve(),
               PROTOCOL, ROOT / "src/transcif/data/eia930.py",
               ROOT / "src/transcif/research/cross_country.py"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    required = [EIA923, PROTOCOL, *PRIOR_FILES, *SOURCE_INPUTS, *CODE_INPUTS]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("a frozen input, protocol, prior result, or code asset is missing")

    prior = pd.read_csv(PRIOR / "ba_month_method_metrics.csv")
    hourly = pd.read_csv(PRIOR / "hourly_predictions_and_labels.csv.gz")
    profile_arm = "MplusF_EIA923_internal_mix"
    prior_actual = prior.loc[prior.arm.isin(["M5_EIA923_report", profile_arm])].copy()
    hourly = hourly.loc[hourly.arm.eq(profile_arm)].copy()
    if len(hourly) != 14592 or prior_actual.groupby("arm").size().to_dict() != {
            "M5_EIA923_report": 20, profile_arm: 20}:
        raise ValueError("prior result no longer matches the frozen 20-cell queue")

    monthly, _ = vintage.parse_monthly_generation(EIA923, 2026, range(1, 9),
                                                   tuple(vintage.DEFAULT_BAS))
    shares = monthly.set_index(["ba", "month"])
    raw, _, _, _ = source_model.fit_source()
    source = raw["GB"]
    source_monthly = fine.fine_monthly_source_inputs(source)
    calibration = source_monthly.loc[
        pd.to_datetime(source_monthly.clock).dt.year.eq(2023)].copy()
    fuel_columns = list(FUELS)
    good = np.isfinite(calibration[fuel_columns].to_numpy(float)).all(axis=1)
    calibration = calibration.loc[good]
    calibration_y = labels(source.loc[calibration.index])
    static_source = calibration[fuel_columns].to_numpy(float) @ EF
    residual = calibration_y - static_source
    bias = float(residual.mean())
    sigma = max(float(np.sqrt(np.mean((residual - bias) ** 2))), 1.)

    static_rows: list[dict] = []
    for (ba, month), group in hourly.groupby(["ba", "month"], sort=True):
        row = shares.loc[(ba, month)]
        fuel_mwh = np.asarray([float(row[f"eia923_{fuel}_mwh"]) for fuel in FUELS])
        total = float(fuel_mwh.sum())
        if total <= 0 or np.any(fuel_mwh < 0) or not np.isfinite(fuel_mwh).all():
            raise ValueError(f"invalid EIA-923 composition for {ba}/{month}")
        point = np.full(len(group), float((fuel_mwh / total) @ EF))
        frame = pd.DataFrame({"clock": pd.to_datetime(group.local_time)})
        score = metrics(frame, group.y_fixed_ef_proxy.to_numpy(float), point, bias, sigma)
        static_rows.append({"ba": ba, "month": month,
                            "arm": "MplusF_static_EF_EIA923_report",
                            "method": "monthly_fuel_share_times_fixed_EF",
                            **score, "source_calibration_bias": bias,
                            "source_calibration_sigma": sigma})
    static = pd.DataFrame(static_rows)
    paired = pd.concat([prior_actual, static], ignore_index=True, sort=False)
    macro = (paired.groupby(["arm", "method"], as_index=False)
             .agg(ba_months=("ba", "size"), hours=("hours", "sum"),
                  macro_mae=("mae", "mean"), macro_shape_mae=("shape_mae", "mean"),
                  macro_crps=("crps_gaussian", "mean"), macro_coverage90=("coverage90", "mean"),
                  mean_width90=("width90", "mean"),
                  macro_interval_score90=("interval_score90", "mean")))

    OUT.mkdir(parents=True)
    paired.to_csv(OUT / "ba_month_method_metrics.csv", index=False)
    macro.to_csv(OUT / "macro_metrics.csv", index=False)
    provenance = {
        "status": "POSTHOC_DEVELOPMENT_DECOMPOSITION_NOT_INDEPENDENT_CONFIRMATION",
        "paired_ba_months": 20, "hours": 14592,
        "queue_source": "reused and hash-linked prior fine-fuel information-value experiment",
        "arms": ["M5_EIA923_report", profile_arm, "MplusF_static_EF_EIA923_report"],
        "static_arm_source_calibration": {"year": 2023, "hours": int(len(residual)),
                                          "monthly_feature_lag": 2,
                                          "bias": bias, "sigma": sigma},
        "target": "current EIA-930 fixed-EF generation-intensity proxy",
        "input_hashes": {str(p.relative_to(ROOT)): sha(p)
                         for p in [EIA923, *PRIOR_FILES, *SOURCE_INPUTS]},
        "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in CODE_INPUTS},
        "limitations": ["same target BA-months were already inspected in the prior experiment",
                        "EIA-923 release postdates evaluated months",
                        "static arm is an attribution baseline, not a candidate model",
                        "EIA-923 is above the China five-group information budget",
                        "hourly label is a fixed-EF proxy, not consumption-side CIF"],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
