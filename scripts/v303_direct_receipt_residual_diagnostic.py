"""V303: post-hoc test of direct receipt volume vs V300 annual-factor residuals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V303_DIRECT_RECEIPT_RESIDUAL_DIAGNOSTIC_PROTOCOL.md"
RECEIPTS = ROOT / "data/china/cec_province_crossprov_receipts_2023.csv"
PREDICTIONS = ROOT / "results/v300-official-provincial-ef-mix-transfer-r1/province_predictions_2023.csv"
OUT = ROOT / "results/v303-direct-receipt-residual-diagnostic-r1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - actual
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "bias": float(np.mean(err)),
            "p90_absolute_error": float(np.quantile(np.abs(err), 0.9))}


def loo_correction(frame: pd.DataFrame) -> np.ndarray:
    x = frame.receipt_share_pct.to_numpy(float)
    residual = (frame.coarse5_mix_pred - frame.actual_2023).to_numpy(float)
    correction = np.empty_like(residual)
    for i in range(len(frame)):
        train = np.arange(len(frame)) != i
        center = float(x[train].mean())
        scale = float(x[train].std(ddof=0))
        ztrain = (x[train] - center) / scale if scale > 0 else np.zeros(train.sum())
        ztest = (x[i] - center) / scale if scale > 0 else 0.0
        design = np.column_stack([np.ones(train.sum()), ztrain])
        beta, *_ = np.linalg.lstsq(design, residual[train], rcond=None)
        correction[i] = float(np.array([1.0, ztest]) @ beta)
    return correction


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    receipts = pd.read_csv(RECEIPTS)
    predictions = pd.read_csv(PREDICTIONS)
    frame = predictions.merge(receipts, on="province", validate="one_to_one")
    if len(frame) != 30 or frame.province.nunique() != 30:
        raise ValueError("expected exact 30-province MEE/V300 cohort")
    frame["receipt_share_pct"] = 100 * frame.receipt_10k_kwh / (
        frame.consumption_100m_kwh * 10000
    )
    frame["coarse5_residual"] = frame.coarse5_mix_pred - frame.actual_2023
    frame["abs_coarse5_residual"] = frame.coarse5_residual.abs()
    signed = spearmanr(frame.receipt_share_pct, frame.coarse5_residual)
    absolute = spearmanr(frame.receipt_share_pct, frame.abs_coarse5_residual)
    correction = loo_correction(frame)
    frame["receipt_share_loo_error_correction"] = correction
    frame["receipt_share_loo_corrected_pred"] = frame.coarse5_mix_pred + correction
    actual = frame.actual_2023.to_numpy(float)
    model_rows = [
        {"method": "V300_frozen_coarse5", **score(actual, frame.coarse5_mix_pred.to_numpy(float))},
        {"method": "V300_plus_LOO_receipt_share_residual_correction",
         **score(actual, frame.receipt_share_loo_corrected_pred.to_numpy(float))},
    ]
    OUT.mkdir(parents=True)
    frame[["province", "province_zh", "actual_2023", "coarse5_mix_pred",
           "coarse5_residual", "abs_coarse5_residual", "consumption_100m_kwh",
           "receipt_10k_kwh", "receipt_share_pct", "receipt_share_loo_error_correction",
           "receipt_share_loo_corrected_pred"]].to_csv(OUT / "province_receipt_residuals.csv", index=False)
    pd.DataFrame(model_rows).to_csv(OUT / "diagnostic_scores.csv", index=False)
    association = {
        "feature": "receipt_share_pct",
        "spearman_signed_rho": float(signed.statistic),
        "spearman_signed_p": float(signed.pvalue),
        "spearman_abs_rho": float(absolute.statistic),
        "spearman_abs_p": float(absolute.pvalue),
    }
    summary = {
        "status": "POSTHOC_SAME_YEAR_RECEIPT_VOLUME_DIAGNOSTIC_NOT_PROSPECTIVE_VALIDATION",
        "n_provinces": len(frame),
        "models": model_rows,
        "association": association,
        "protocol_sha256": sha(PROTOCOL),
        "receipt_table_sha256": sha(RECEIPTS),
        "v300_prediction_sha256": sha(PREDICTIONS),
        "script_sha256": sha(Path(__file__)),
        "limitations": [
            "Receipt volume is same-year 2023 information and post-hoc, not eligible for the lagged-mix forecast.",
            "Aggregate receipt volume does not identify sending province, generation source, or incoming carbon intensity.",
            "The source table may follow a market/transmission scope that is not identical to the MEE/NBS consumption-factor accounting boundary.",
            "Only 30 provinces are evaluated; the one-feature correction is a diagnostic cross-fit, not a validated predictor.",
            "The target is an annual consumption-side factor, not an hourly generation-side CIF.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(pd.DataFrame(model_rows).to_string(index=False))
    print(json.dumps(association, ensure_ascii=False, indent=2))
    print(frame.sort_values("abs_coarse5_residual", ascending=False)[
        ["province", "coarse5_residual", "receipt_share_pct"]
    ].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
