"""V235: does a public reconstructed China load profile tighten monthly-fuel CIF sets?"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V235_CHINA_HOURLY_LOAD_PROXY_INFORMATION_PROTOCOL.md"
LOAD = Path("/tmp/transcif_v235_china_hourly_load_v3.xlsx")
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
OUT = ROOT / "results/v235-china-hourly-load-proxy-information-r1"
YEARS = tuple(range(2018, 2025))
SOURCES = ("Coal", "Gas", "Other fossil", "Bioenergy", "Wind", "Solar", "Hydro", "Nuclear")
EF = np.array([1000., 400., 800., 400., 0., 0., 0., 0.])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hourly_bounds(hourly_total: float, margins: np.ndarray) -> tuple[float, float]:
    """Sharp CIF bounds for one hour under fuel-month caps and known total."""
    month_total = float(margins.sum())
    lower = np.maximum(0., margins - (month_total - hourly_total))
    upper = np.minimum(margins, hourly_total)
    remaining = hourly_total - float(lower.sum())

    def allocate(order: np.ndarray) -> float:
        x = lower.copy()
        need = remaining
        for idx in order:
            amount = min(float(upper[idx] - lower[idx]), need)
            x[idx] += amount
            need -= amount
            if need <= 1e-10:
                break
        if need > 1e-8:
            raise RuntimeError(f"infeasible allocation residue {need}")
        return float(x @ EF / hourly_total)

    return allocate(np.argsort(EF)), allocate(np.argsort(-EF))


def read_load_profiles() -> dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]]:
    wb = load_workbook(LOAD, read_only=True, data_only=True)
    out: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for year in YEARS:
        ws = wb[f"{year} load curve"]
        iterator = ws.iter_rows(values_only=True)
        header = next(iterator)
        province_columns = {}
        for col, raw_name in enumerate(header[1:], start=1):
            if not isinstance(raw_name, str) or "(" not in raw_name:
                continue
            english = raw_name.split("(", 1)[0].strip()
            province_columns[english] = col
        rows = []
        for row in iterator:
            if not isinstance(row[0], datetime):
                continue
            rows.append(row)
        frame = pd.DataFrame(rows, columns=header)
        frame = frame.rename(columns={header[col]: province for province, col in province_columns.items()})
        time_col = frame.columns[0]
        frame["time"] = pd.to_datetime(frame[time_col])
        expected = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        if not frame.time.equals(pd.Series(expected, name="time")):
            frame = frame.set_index("time").reindex(expected).rename_axis("time").reset_index()
        if len(frame) != len(expected) or frame.time.isna().any() or frame.time.duplicated().any():
            raise RuntimeError(f"invalid hourly timestamps for {year}")
        for province in province_columns:
            values = pd.to_numeric(frame[province], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise RuntimeError(f"invalid hourly load: {province} {year}")
            times = pd.DatetimeIndex(frame.time)
            for month in range(1, 13):
                mask = times.month == month
                daily = values[mask].reshape(-1, 24).sum(axis=1)
                daily_flat = np.repeat(daily / 24., 24)
                out[(province, year, month)] = (values[mask], daily_flat)
    wb.close()
    return out


def read_ember() -> dict[tuple[str, int, int], np.ndarray]:
    frame = pd.read_csv(EMBER, low_memory=False)
    frame = frame[
        frame["Electricity source"].isin(SOURCES)
        & frame["Is aggregated source"].astype(str).str.lower().eq("false")
        & frame["Date"].astype(str).str[:4].astype(int).isin(YEARS)
    ].copy()
    frame["Generation (TWh)"] = pd.to_numeric(frame["Generation (TWh)"], errors="raise")
    frame["year"] = frame["Date"].astype(str).str[:4].astype(int)
    frame["month"] = frame["Date"].astype(str).str[5:7].astype(int)
    out = {}
    for (province, year, month), block in frame.groupby(["Province", "year", "month"], sort=True):
        if len(block) != len(SOURCES) or set(block["Electricity source"]) != set(SOURCES):
            continue
        margins = block.set_index("Electricity source")["Generation (TWh)"].reindex(SOURCES).to_numpy(float)
        if np.isfinite(margins).all() and np.all(margins >= 0.) and margins.sum() > 0.:
            out[(province, int(year), int(month))] = margins
    return out


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    if not LOAD.exists() or not EMBER.exists():
        raise FileNotFoundError("V235 input missing")
    protocol_hash = sha256(PROTOCOL)
    load_profiles = read_load_profiles()
    margins_by_key = read_ember()
    province_alias = {"Tibet (Xizang)": "Tibet"}
    rows = []
    for (province, year, month), margins in sorted(margins_by_key.items()):
        load_province = province_alias.get(province, province)
        key = (load_province, year, month)
        if key not in load_profiles:
            continue
        hourly, daily_flat = load_profiles[key]
        if len(hourly) != len(daily_flat) or not np.isfinite(hourly).all():
            raise RuntimeError(f"bad load block {key}")
        monthly_total = float(margins.sum())
        hourly_full = monthly_total * hourly / float(hourly.sum())
        hourly_daily = monthly_total * daily_flat / float(daily_flat.sum())
        supports = np.flatnonzero(margins > 1e-12)
        full_width = float(EF[supports].max() - EF[supports].min())
        for regime, totals in (("daily_flat", hourly_daily), ("hourly_profile", hourly_full)):
            bounds = np.asarray([hourly_bounds(float(g), margins) for g in totals])
            width = bounds[:, 1] - bounds[:, 0]
            rows.append({
                "province": province, "year": year, "month": month, "regime": regime,
                "hours": len(totals), "active_sources": len(supports),
                "full_support_width": full_width, "mean_width": float(width.mean()),
                "median_width": float(np.median(width)), "p90_width": float(np.quantile(width, .9)),
                "mean_reduction_from_monthly_hull": float(1. - width.mean() / full_width) if full_width else 0.,
                "hours_at_full_support_width": float(np.mean(width >= full_width - 1e-8)) if full_width else 1.,
                "max_hourly_total_share_of_month": float(totals.max() / monthly_total),
            })
    detail = pd.DataFrame(rows)
    unique = detail[["province", "year", "month"]].drop_duplicates()
    if unique.province.nunique() < 20 or unique.year.nunique() < 5:
        raise RuntimeError(f"frozen coverage gate failed: provinces={unique.province.nunique()}, years={unique.year.nunique()}")
    summary = detail.groupby("regime", as_index=False).agg(
        mean_width_equal_region_month=("mean_width", "mean"),
        median_width_equal_region_month=("mean_width", "median"),
        p90_width_equal_region_month=("mean_width", lambda s: float(np.quantile(s, .9))),
        mean_reduction=("mean_reduction_from_monthly_hull", "mean"),
        share_hours_full_support=("hours_at_full_support_width", "mean"),
        max_hourly_total_month_share=("max_hourly_total_share_of_month", "max"),
        province_months=("province", "count"), provinces=("province", "nunique"),
        years=("year", "nunique"),
    )
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "province_month_bounds.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    meta = {
        "status": "information_set_geometry_not_CIF_forecast_validation",
        "protocol_sha256": protocol_hash,
        "load_input_sha256": sha256(LOAD), "ember_input_sha256": sha256(EMBER),
        "load_input_doi": "10.6084/m9.figshare.29832701.v3",
        "ef_gco2_per_kwh": EF.tolist(),
        "province_months": int(unique.shape[0]), "provinces": int(unique.province.nunique()),
        "years": sorted(int(x) for x in unique.year.unique()),
        "interpretation": "hourly demand profile is normalized to monthly Ember generation total as an explicit proxy; not observed generation or a CIF label",
        "target_labels_used": False,
        "bound_scope": "pointwise sharp CIF bounds under fixed hourly totals plus monthly eight-source margins",
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(summary.to_string(index=False))
    print(f"province_months={len(unique)} output={OUT}")


if __name__ == "__main__":
    main()
