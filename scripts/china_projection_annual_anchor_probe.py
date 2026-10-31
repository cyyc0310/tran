"""As-of annual-factor anchor probe on an independent China hourly CEF projection.

This is a simulator stress test, not validation against observed hourly CIF.
The 2025 hourly outcomes come from the published CISPO-China projection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


ROOT = Path.cwd()
SOURCE = ROOT / "results/external-china-cif-projection-source"
MEE_2022_KG_PER_KWH = {
    "Beijing": 0.5580,
    "Tianjin": 0.7041,
    "Hebei": 0.7252,
    "Shanxi": 0.7096,
    "Inner Mongolia": 0.6849,
    "Liaoning": 0.5626,
    "Jilin": 0.4932,
    "Heilongjiang": 0.5368,
    "Shanghai": 0.5849,
    "Jiangsu": 0.5978,
    "Zhejiang": 0.5153,
    "Anhui": 0.6782,
    "Fujian": 0.4092,
    "Jiangxi": 0.5752,
    "Shandong": 0.6410,
    "Henan": 0.6058,
    "Hubei": 0.4364,
    "Hunan": 0.4900,
    "Guangdong": 0.4403,
    "Guangxi": 0.4044,
    "Hainan": 0.4184,
    "Chongqing": 0.5227,
    "Sichuan": 0.1404,
    "Guizhou": 0.4989,
    "Yunnan": 0.1073,
    "Shaanxi": 0.6558,
    "Gansu": 0.4772,
    "Qinghai": 0.1567,
    "Ningxia": 0.6423,
    "Xinjiang": 0.6231,
}
MEE_2021_KG_PER_KWH = {
    "Beijing": 0.5688,
    "Tianjin": 0.7355,
    "Hebei": 0.7901,
    "Shanxi": 0.7222,
    "Inner Mongolia": 0.7025,
    "Liaoning": 0.5876,
    "Jilin": 0.5629,
    "Heilongjiang": 0.6342,
    "Shanghai": 0.5834,
    "Jiangsu": 0.6451,
    "Zhejiang": 0.5422,
    "Anhui": 0.7075,
    "Fujian": 0.4711,
    "Jiangxi": 0.5835,
    "Shandong": 0.6838,
    "Henan": 0.6369,
    "Hubei": 0.3672,
    "Hunan": 0.5138,
    "Guangdong": 0.4715,
    "Guangxi": 0.5154,
    "Hainan": 0.4524,
    "Chongqing": 0.4743,
    "Sichuan": 0.1255,
    "Guizhou": 0.5182,
    "Yunnan": 0.1235,
    "Shaanxi": 0.6336,
    "Gansu": 0.4955,
    "Qinghai": 0.1326,
    "Ningxia": 0.6546,
    "Xinjiang": 0.6577,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = []
    workbooks = {}
    for scenario in ("S1", "S2", "S3", "S4", "S5"):
        workbook = SOURCE / f"CEF data for Scenario {scenario}.xlsx"
        workbooks[scenario] = workbook
        wb = load_workbook(workbook, read_only=True, data_only=True)
        ws = wb["2025"]
        values = list(ws.iter_rows(values_only=True))
        frame = pd.DataFrame(values[1:], columns=list(values[0]))
        frame = frame.loc[frame["Time"].notna()].copy()
        if len(frame) != 8760:
            raise ValueError(f"Expected 8760 {scenario} 2025 hourly rows, found {len(frame)}")

        provinces = list(MEE_2022_KG_PER_KWH)
        target_means = {
            province: float(frame[province].to_numpy(dtype=float).mean() * 1000.0)
            for province in provinces
        }
        level_features = np.array(
            [
                [
                    1.0,
                    MEE_2022_KG_PER_KWH[province] * 1000.0,
                    (MEE_2022_KG_PER_KWH[province] - MEE_2021_KG_PER_KWH[province]) * 1000.0,
                ]
                for province in provinces
            ],
            dtype=float,
        )
        loo_level_predictions = {}
        loo_level_shape_predictions = {}
        loo_local_conditioned_shapes = {}
        all_hourly_anomalies = {
            name: frame[name].to_numpy(dtype=float) * 1000.0 - target_means[name]
            for name in provinces
        }
        for target_index, province in enumerate(provinces):
            training_mask = np.arange(len(provinces)) != target_index
            coef, *_ = np.linalg.lstsq(
                level_features[training_mask],
                np.array([target_means[name] for name in provinces])[training_mask],
                rcond=None,
            )
            loo_level_predictions[province] = float(level_features[target_index] @ coef)
            donors = [name for name in provinces if name != province]
            donor_anomalies = np.stack(
                [frame[name].to_numpy(dtype=float) * 1000.0 - target_means[name] for name in donors],
                axis=0,
            )
            loo_level_shape_predictions[province] = (
                loo_level_predictions[province] + donor_anomalies.mean(axis=0)
            )
            shape_features = level_features[:, 1:]
            shape_coef, *_ = np.linalg.lstsq(
                shape_features[training_mask],
                np.stack([all_hourly_anomalies[name] for name in provinces if name != province]),
                rcond=None,
            )
            loo_local_conditioned_shapes[province] = (
                loo_level_predictions[province] + shape_features[target_index] @ shape_coef
            )

        for province, factor_kg_kwh in MEE_2022_KG_PER_KWH.items():
            truth = frame[province].to_numpy(dtype=float) * 1000.0
            local_anchor = factor_kg_kwh * 1000.0
            national_anchor = 536.6  # official 2022 national average, gCO2/kWh
            local_trend = max(
                0.0,
                (factor_kg_kwh + 3.0 * (factor_kg_kwh - MEE_2021_KG_PER_KWH[province])) * 1000.0,
            )
            for method, prediction in (
                ("local_MEE_2022_flat", local_anchor),
                ("national_MEE_2022_flat", national_anchor),
                ("local_MEE_2021_2022_linear_extrapolation_to_2025", local_trend),
                ("source_LOO_level_transfer_linear", loo_level_predictions[province]),
                ("source_LOO_level_plus_cross_region_hourly_shape", loo_level_shape_predictions[province]),
                ("source_LOO_level_plus_local_factor_conditioned_shape", loo_local_conditioned_shapes[province]),
            ):
                prediction_vector = (
                    np.full_like(truth, float(prediction))
                    if np.isscalar(prediction)
                    else np.asarray(prediction, dtype=float)
                )
                residual = truth - prediction_vector
                rows.append(
                    {
                        "scenario": scenario,
                        "province": province,
                        "method": method,
                        "n_hours": len(truth),
                        "target_annual_mean": float(np.mean(truth)),
                        "target_hourly_sd": float(np.std(truth)),
                        "prediction": float(np.mean(prediction_vector)),
                        "mae": float(np.mean(np.abs(residual))),
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                        "bias": float(np.mean(residual)),
                        "target_hourly_p90": float(np.quantile(truth, .9)),
                    }
                )

    province_scores = pd.DataFrame(rows)
    output = ROOT / "results/china-projection-annual-anchor"
    output.mkdir(parents=True, exist_ok=True)
    province_scores.to_csv(output / "province_scores.csv", index=False)
    macro = (
        province_scores.groupby(["scenario", "method"])
        .agg(
            provinces=("province", "nunique"),
            source_equal_mae=("mae", "mean"),
            source_equal_rmse=("rmse", "mean"),
            source_equal_abs_bias=("bias", lambda x: float(np.mean(np.abs(x)))),
            mean_hourly_target_sd=("target_hourly_sd", "mean"),
        )
        .reset_index()
    )
    macro.to_csv(output / "macro.csv", index=False)
    checksums = {
        **{f"projection_{scenario}": sha256(workbook) for scenario, workbook in workbooks.items()},
        "annotation_pdf": sha256(SOURCE / "Annotation of the dataset.pdf"),
        "figshare_metadata": sha256(SOURCE / "figshare_metadata.json"),
    }
    summary = {
        "status": "EXTERNAL_CHINA_MODEL_PROJECTION_STRESS_TEST_NOT_OBSERVED_VALIDATION",
        "target": "2025 hourly provincial consumption-based CEF projections across five CISPO scenarios, gCO2/kWh",
        "asof_inputs": "official MEE/NBS 2021 provincial factor (published 2024-04-12), 2022 provincial factor (published 2024-12-26), and 2022 national average; all precede the 2025 target start",
        "baselines": "flat local 2022 factor, flat national 2022 factor, untuned local 2021-2022 linear trend extrapolated three years, and leave-one-province-out linear mapping from official 2021/2022 factors to the held-out scenario's annual mean; hourly-shape rows are explicitly same-year cross-province oracle diagnostics and are not deployable forecasts",
        "scope_limit": "This evaluates as-of annual anchoring against model-generated 2025 scenarios. It does not validate observed hourly CIF, cross-country transfer, uncertainty coverage, or real China forecasts; the hourly-shape variants use held-out-year outcomes from other provinces.",
        "n_provinces": len(MEE_2022_KG_PER_KWH),
        "n_scenarios": len(workbooks),
        "n_hours_per_province": len(frame),
        "inputs_sha256": checksums,
        "macro": macro.to_dict("records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
