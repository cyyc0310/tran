"""Post-hoc sensitivity of monthly fuel-margin CIF bounds to modelled load shapes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v65-load-shape-information-value-r3"
WORKBOOK = ROOT / "results/china-hourly-load-dataset-audit-v64-r1/Data output.xlsx"
OLD_LOAD = ROOT / "results/china-hourly-load-dataset-audit-v64-r1/Appendix 1_Hourly electric power load final.csv"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
PROTOCOL = ROOT / "docs/V65_LOAD_SHAPE_INFORMATION_VALUE_PROTOCOL.md"
FUELS = ("Coal", "Gas", "Other fossil", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy")
EF = np.array([1000.0, 400.0, 400.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def margin_only_bounds(margins: np.ndarray, slots: int) -> tuple[float, float, float]:
    active = margins > max(1e-12, margins.sum() * 1e-12)
    ef, mass = EF[active], margins[active]
    overall = float(ef @ mass / mass.sum())
    low, high = float(ef.min()), float(ef.max())
    if np.isclose(low, high):
        return low, high, overall
    lower = ((slots - 1) * low + overall) / slots
    below_high = ef < high
    upper = ((slots - 1) * high + float(ef[below_high] @ mass[below_high] / mass[below_high].sum())) / slots
    return lower, upper, overall


def fixed_slot_lp(margins: np.ndarray, slot_totals: np.ndarray) -> tuple[float, float]:
    n_slots, n_fuels = len(slot_totals), len(FUELS)
    rows, cols, values = [], [], []
    for slot in range(n_slots):
        for fuel in range(n_fuels):
            var = slot * n_fuels + fuel
            rows.extend((slot, n_slots + fuel))
            cols.extend((var, var))
            values.extend((1.0, 1.0))
    a_eq = coo_matrix((values, (rows, cols)), shape=(n_slots + n_fuels, n_slots * n_fuels)).tocsr()
    b_eq = np.concatenate((slot_totals, margins))
    objective = np.tile(EF, n_slots) / np.repeat(slot_totals, n_fuels) / n_slots
    lo = linprog(objective, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    hi = linprog(-objective, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not lo.success or not hi.success:
        raise RuntimeError(f"infeasible proxy shape LP: {lo.message}; {hi.message}")
    return float(lo.fun), float(-hi.fun)


def load_monthly_shapes() -> dict[tuple[str, str], dict[str, np.ndarray]]:
    shapes: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    xls = pd.ExcelFile(WORKBOOK)
    for sheet in xls.sheet_names:
        if not sheet.endswith("load curve"):
            continue
        year = int(sheet[:4])
        if year < 2016:
            continue
        frame = pd.read_excel(WORKBOOK, sheet_name=sheet)
        tcol = frame.columns[0]
        frame = frame.dropna(subset=[tcol]).copy()
        frame[tcol] = pd.to_datetime(frame[tcol], errors="coerce")
        frame = frame.dropna(subset=[tcol])
        frame["month"] = frame[tcol].dt.strftime("%Y-%m")
        frame["hod"] = frame[tcol].dt.hour
        for col in frame.columns[1:-2]:
            province = str(col).split(" (")[0]
            values = pd.to_numeric(frame[col], errors="coerce")
            for month, indices in frame.groupby("month").groups.items():
                by_hour = values.loc[indices].groupby(frame.loc[indices, "hod"]).sum().reindex(range(24), fill_value=0).to_numpy(float)
                if np.isfinite(by_hour).all() and by_hour.sum() > 0:
                    shapes[(province, month)] = {"weather_reconstructed": by_hour}
    return shapes


OLD_CODE_TO_PROVINCE = {
    "BJ": "Beijing", "TJ": "Tianjin", "HB": "Hebei", "SX": "Shanxi", "NM": "Inner Mongolia",
    "LN": "Liaoning", "JL": "Jilin", "HL": "Heilongjiang", "SH": "Shanghai", "JS": "Jiangsu",
    "ZJ": "Zhejiang", "AH": "Anhui", "FJ": "Fujian", "JX": "Jiangxi", "SD": "Shandong",
    "HA": "Henan", "HB.1": "Hubei", "HN": "Hunan", "GD": "Guangdong", "GX": "Guangxi",
    "HI": "Hainan", "CQ": "Chongqing", "SC": "Sichuan", "GZ": "Guizhou", "YN": "Yunnan",
    "XZ": "Tibet (Xizang)", "SN": "Shaanxi", "GS": "Gansu", "QH": "Qinghai", "NX": "Ningxia", "XJ": "Xinjiang",
}


def add_old_2018_shapes(shapes: dict[tuple[str, str], dict[str, np.ndarray]]) -> None:
    old = pd.read_csv(OLD_LOAD, sep=";")
    vals = old.iloc[:, 1:].to_numpy(float)
    if vals.shape != (8760, 31):
        raise ValueError(f"Unexpected 2018 Zenodo array dimensions: {vals.shape}")
    for j, code in enumerate(old.columns[1:]):
        province = OLD_CODE_TO_PROVINCE.get(code)
        if not province:
            raise ValueError(f"Unmapped 2018 province code: {code}")
        for month in range(1, 13):
            start = pd.Timestamp(2018, month, 1)
            stop = start + pd.offsets.MonthBegin(1)
            indices = (pd.date_range("2018-01-01", periods=8760, freq="h").month == month)
            by_hour = vals[indices, j].reshape(-1, 24).sum(axis=0)
            shapes.setdefault((province, start.strftime("%Y-%m")), {})["zenodo_template"] = by_hour


def main() -> None:
    protocol_hash = sha256(PROTOCOL)
    ember = pd.read_csv(EMBER)
    ember = ember.loc[~ember["Is aggregated source"].astype(str).str.lower().eq("true")].copy()
    load_shapes = load_monthly_shapes()
    add_old_2018_shapes(load_shapes)
    rows = []
    for (province, month), group in ember.groupby(["Province", "Date"], sort=True):
        if province == "China (total)":
            continue
        month_key = month[:7]
        fuel_rows = group.set_index("Electricity source")
        if not set(FUELS).issubset(fuel_rows.index) or (province, month_key) not in load_shapes:
            continue
        margins = fuel_rows.loc[list(FUELS), "Generation (TWh)"].to_numpy(float)
        if not np.isfinite(margins).all() or (margins < 0).any() or margins.sum() <= 0:
            continue
        lo0, hi0, exact = margin_only_bounds(margins, 24)
        flat_totals = np.full(24, margins.sum() / 24)
        flat_lo, flat_hi = fixed_slot_lp(margins, flat_totals)
        for source, raw_shape in load_shapes[(province, month_key)].items():
            shape = np.asarray(raw_shape, float)
            if shape.shape != (24,) or not np.isfinite(shape).all() or (shape < 0).any() or shape.sum() <= 0:
                continue
            slot_totals = shape / shape.sum() * margins.sum()
            lo, hi = fixed_slot_lp(margins, slot_totals)
            rows.append({"province": province, "month": month_key, "shape_source": source,
                         "monthly_energy_weighted_cif_exact_g_kwh": exact,
                         "margin_lower": lo0, "margin_upper": hi0, "margin_width": hi0 - lo0,
                         "flat_slot_lower": flat_lo, "flat_slot_upper": flat_hi, "flat_slot_width": flat_hi - flat_lo,
                         "proxy_shape_lower": lo, "proxy_shape_upper": hi, "proxy_shape_width": hi - lo,
                         "width_reduction_fraction": 1 - (hi - lo) / (hi0 - lo0) if hi0 > lo0 else 0.0,
                         "proxy_width_minus_flat_g_kwh": (hi - lo) - (flat_hi - flat_lo),
                         "generation_TWh": margins.sum(), "slot_count": int(np.count_nonzero(slot_totals > 0))})
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No matched complete province-months")
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "province_month_shape_bounds.csv", index=False)
    summary = (detail.groupby("shape_source").agg(rows=("month", "count"), provinces=("province", "nunique"),
               mean_margin_width=("margin_width", "mean"), mean_proxy_shape_width=("proxy_shape_width", "mean"),
               mean_flat_slot_width=("flat_slot_width", "mean"),
               median_width_reduction=("width_reduction_fraction", "median"),
               mean_width_reduction=("width_reduction_fraction", "mean"),
               median_proxy_width_minus_flat=("proxy_width_minus_flat_g_kwh", "median"),
               mean_proxy_width_minus_flat=("proxy_width_minus_flat_g_kwh", "mean"),
               positive_proxy_width_minus_flat_fraction=("proxy_width_minus_flat_g_kwh", lambda x: (x > 1e-9).mean()),
               p10_width_reduction=("width_reduction_fraction", lambda x: x.quantile(.1)),
               positive_reduction_fraction=("width_reduction_fraction", lambda x: (x > 1e-9).mean()))
               .reset_index())
    summary.to_csv(OUT / "shape_source_summary.csv", index=False)
    province = (detail.groupby(["shape_source", "province"]).agg(months=("month", "count"),
               mean_margin_width=("margin_width", "mean"), mean_proxy_shape_width=("proxy_shape_width", "mean"),
               mean_flat_slot_width=("flat_slot_width", "mean"),
               median_width_reduction=("width_reduction_fraction", "median"),
               median_proxy_width_minus_flat=("proxy_width_minus_flat_g_kwh", "median"),
               positive_reduction_fraction=("width_reduction_fraction", lambda x: (x > 1e-9).mean()))
               .reset_index())
    province.to_csv(OUT / "province_summary.csv", index=False)
    paired_2018 = detail.loc[detail.month.str.startswith("2018-")].pivot(
        index=["province", "month"], columns="shape_source", values="proxy_shape_width").dropna()
    paired_delta = paired_2018["weather_reconstructed"] - paired_2018["zenodo_template"]
    pd.DataFrame({"weather_minus_zenodo_width_g_kwh": paired_delta}).reset_index().to_csv(
        OUT / "paired_2018_source_comparison.csv", index=False)
    payload = {"status": "POSTHOC_PROXY_SHAPE_IDENTIFIABILITY_SENSITIVITY_NOT_FORECAST_OR_VALIDATION",
               "region_month_shape_rows": len(detail), "period": [detail.month.min(), detail.month.max()],
               "provinces": sorted(detail.province.unique().tolist()), "sources": summary.to_dict("records"),
               "paired_2018_model_vs_zenodo": {"province_months": int(len(paired_2018)),
                   "mean_weather_proxy_width": float(paired_2018.weather_reconstructed.mean()),
                   "mean_zenodo_proxy_width": float(paired_2018.zenodo_template.mean()),
                   "mean_weather_minus_zenodo_width_g_kwh": float(paired_delta.mean()),
                   "median_weather_minus_zenodo_width_g_kwh": float(paired_delta.median()),
                   "weather_shape_narrower_fraction": float((paired_delta < 0).mean()),
                   "paired_width_correlation": float(paired_2018.corr().iloc[0, 1])},
               "ef_gco2e_kwh": dict(zip(FUELS, EF.tolist())), "protocol_sha256": protocol_hash,
               "script_sha256": sha256(Path(__file__)), "input_sha256": {str(p): sha256(p) for p in [WORKBOOK, OLD_LOAD, EMBER]},
               "limitations": ["The load curves are model-derived demand, not hourly generation or CIF measurements.",
                              "Fuel margins are the target-month retrospective Ember snapshot; this is not a forecast experiment.",
                              "The calculation assumes within-month generation follows the load-derived hour-of-day profile.",
                              "A narrower feasible set under this assumption does not establish empirical coverage or predictive validity."]}
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("Frozen protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(summary.to_string(index=False))
    print(f"rows={len(detail)} provinces={detail.province.nunique()} outputs={OUT}")


if __name__ == "__main__":
    main()
