"""V302: post-hoc diagnostic of annual power-flow proxies vs V301 residuals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V302_OFFICIAL_FLOW_RESIDUAL_DIAGNOSTIC_PROTOCOL.md"
POWER = ROOT / "data/china/cec_province_power_balance_2022.csv"
PREDICTIONS = ROOT / "results/v301-mix-transfer-2022-temporal-replication-r1/province_predictions_2022.csv"
OUT = ROOT / "results/v302-official-flow-residual-diagnostic-r1"
FEATURES = ("balance_gap_pct", "sendout_share_pct")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - actual
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "p90_absolute_error": float(np.quantile(np.abs(err), 0.9)),
    }


def loo_residual_correction(frame: pd.DataFrame, feature: str) -> np.ndarray:
    """Cross-fit a one-feature linear residual correction across 30 provinces."""
    x = frame[feature].to_numpy(float)
    residual = (frame.coarse5_mix_pred - frame.actual_2022).to_numpy(float)
    correction = np.empty_like(residual)
    for i in range(len(frame)):
        train = np.arange(len(frame)) != i
        mean = float(x[train].mean())
        scale = float(x[train].std(ddof=0))
        z_train = (x[train] - mean) / scale if scale > 0 else np.zeros(train.sum())
        z_test = (x[i] - mean) / scale if scale > 0 else 0.0
        design = np.column_stack([np.ones(train.sum()), z_train])
        beta, *_ = np.linalg.lstsq(design, residual[train], rcond=None)
        correction[i] = float(np.array([1.0, z_test]) @ beta)
    return correction


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    powers = pd.read_csv(POWER)
    predictions = pd.read_csv(PREDICTIONS)
    frame = predictions.merge(powers, on="province", validate="one_to_one")
    if len(frame) != 30 or frame.province.nunique() != 30:
        raise ValueError("expected exact 30-province V301 factor cohort")
    frame["balance_gap_pct"] = 100 * (
        frame.consumption_100m_kwh - frame.generation_100m_kwh
    ) / frame.consumption_100m_kwh
    frame["sendout_share_pct"] = 100 * frame.cross_province_sendout_10k_kwh / (
        frame.consumption_100m_kwh * 10000
    )
    frame["coarse5_residual"] = frame.coarse5_mix_pred - frame.actual_2022
    frame["abs_coarse5_residual"] = frame.coarse5_residual.abs()

    actual = frame.actual_2022.to_numpy(float)
    baseline = frame.coarse5_mix_pred.to_numpy(float)
    model_rows = [{"method": "V301_frozen_coarse5", **score(actual, baseline)}]
    association_rows = []
    for feature in FEATURES:
        signed = spearmanr(frame[feature], frame.coarse5_residual)
        absolute = spearmanr(frame[feature], frame.abs_coarse5_residual)
        association_rows.append({
            "feature": feature,
            "spearman_signed_rho": float(signed.statistic),
            "spearman_signed_p": float(signed.pvalue),
            "spearman_abs_rho": float(absolute.statistic),
            "spearman_abs_p": float(absolute.pvalue),
        })
        correction = loo_residual_correction(frame, feature)
        frame[f"{feature}_loo_error_correction"] = correction
        frame[f"{feature}_loo_corrected_pred"] = baseline + correction
        model_rows.append({
            "method": f"V301_plus_LOO_residual_correction_{feature}",
            **score(actual, baseline + correction),
        })

    OUT.mkdir(parents=True)
    keep = [
        "province", "province_zh", "actual_2022", "coarse5_mix_pred",
        "coarse5_residual", "abs_coarse5_residual", "consumption_100m_kwh",
        "generation_100m_kwh", "cross_province_sendout_10k_kwh",
        "balance_gap_pct", "sendout_share_pct",
        "balance_gap_pct_loo_error_correction", "balance_gap_pct_loo_corrected_pred",
        "sendout_share_pct_loo_error_correction", "sendout_share_pct_loo_corrected_pred",
    ]
    frame[keep].to_csv(OUT / "province_flow_residuals.csv", index=False)
    pd.DataFrame(model_rows).to_csv(OUT / "diagnostic_scores.csv", index=False)
    pd.DataFrame(association_rows).to_csv(OUT / "proxy_associations.csv", index=False)
    summary = {
        "status": "POSTHOC_SAME_YEAR_FLOW_PROXY_DIAGNOSTIC_NOT_PROSPECTIVE_VALIDATION",
        "n_provinces": len(frame),
        "target": "2022 MEE/NBS provincial consumption CO2 factor",
        "baseline_source": "frozen V301 coarse5 out-of-province prediction",
        "models": model_rows,
        "associations": association_rows,
        "protocol_sha256": sha(PROTOCOL),
        "power_table_sha256": sha(POWER),
        "v301_prediction_sha256": sha(PREDICTIONS),
        "script_sha256": sha(Path(__file__)),
        "limitations": [
            "All power-flow proxies are contemporaneous with the target year and post-hoc; they are not eligible forecast features in this audit.",
            "Consumption minus generation is only a balance proxy, not observed net imports; statistical scope, plant use, and grid losses matter.",
            "Province cross-province sendout is one-sided and contains no incoming volume or origin-specific carbon intensity.",
            "The one-feature residual corrections use only 30 provinces and are diagnostic cross-fits, not validated forecast models.",
            "The MEE/NBS targets are consumption-side annual factors, not hourly generation-side CIF labels.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(pd.DataFrame(model_rows).to_string(index=False))
    print(pd.DataFrame(association_rows).to_string(index=False))
    print("largest residuals")
    print(frame.sort_values("abs_coarse5_residual", ascending=False)[
        ["province", "coarse5_residual", "balance_gap_pct", "sendout_share_pct"]
    ].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
