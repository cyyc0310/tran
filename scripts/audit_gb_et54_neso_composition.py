"""Audit alignment of the UK official ET 5.4 monthly table with NESO GB generation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COARSE = ("thermal", "hydro", "nuclear", "wind", "solar")
ET54_COLUMNS = {
    "thermal": ("Coal\n[note 7]", "Oil", "Gas", "Bioenergy\n[note 10]",
                "Other fuels \n[note 10]"),
    "hydro": (" Hydro",),
    "nuclear": (" Nuclear",),
    "wind": ("Total wind \n[note 8]\n[note 9]",),
    "solar": ("Solar",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shares(values: pd.DataFrame) -> pd.DataFrame:
    values = values.loc[:, COARSE].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().all(axis=1) & np.isfinite(values).all(axis=1)
    valid &= values.ge(0).all(axis=1) & values.sum(axis=1).gt(0)
    return values.loc[valid].div(values.loc[valid].sum(axis=1), axis=0)


def load_et54(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Month", header=None, engine="openpyxl")
    headers = raw.iloc[6].tolist()
    data = raw.iloc[7:].copy()
    data.columns = headers
    data = data.loc[data["Month"].notna()].copy()
    data["month"] = pd.to_datetime(data["Month"], format="%B %Y", errors="coerce").dt.to_period("M")
    data = data.loc[data.month.notna() & data.month.dt.year.eq(2024)]
    output = pd.DataFrame({
        "month": data.month,
        **{group: sum(pd.to_numeric(data[col], errors="coerce") for col in cols)
           for group, cols in ET54_COLUMNS.items()},
    })
    output = output.set_index("month").sort_index()
    return shares(output)


def load_neso(path: Path) -> pd.DataFrame:
    columns = ["DATETIME", "GAS", "COAL", "NUCLEAR", "WIND", "WIND_EMB",
               "HYDRO", "BIOMASS", "OTHER", "SOLAR"]
    raw = pd.read_csv(path, usecols=columns)
    raw["stamp"] = pd.to_datetime(raw.DATETIME, errors="raise")
    raw = raw.loc[raw.stamp.dt.year.eq(2024)].copy()
    groups = pd.DataFrame({
        "month": raw.stamp.dt.to_period("M"),
        "thermal": raw.GAS + raw.COAL + raw.BIOMASS + raw.OTHER,
        "hydro": raw.HYDRO,
        "nuclear": raw.NUCLEAR,
        "wind": raw.WIND + raw.WIND_EMB,
        "solar": raw.SOLAR,
    })
    monthly = groups.groupby("month", sort=True)[list(COARSE)].sum(min_count=1)
    return shares(monthly)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--et54", type=Path, required=True, help="Official ET 5.4 xlsx")
    parser.add_argument("--neso", type=Path, default=ROOT / "data/neso-gb/df_fuel_ckan.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    official = load_et54(args.et54)
    neso = load_neso(args.neso)
    common = official.index.intersection(neso.index)
    if len(common) != 12:
        raise ValueError(f"expected 12 common 2024 months, got {len(common)}")
    rows = []
    for month in common:
        left, right = official.loc[month], neso.loc[month]
        row = {"month": str(month), "total_variation_distance": float((left - right).abs().sum() / 2),
               "l1_share_distance": float((left - right).abs().sum())}
        for group in COARSE:
            row[f"et54_{group}_share"] = float(left[group])
            row[f"neso_{group}_share"] = float(right[group])
            row[f"{group}_absolute_share_error"] = float(abs(left[group] - right[group]))
        rows.append(row)
    monthly = pd.DataFrame(rows)
    monthly.to_csv(args.output / "monthly_composition_comparison.csv", index=False)
    summary = {
        "status": "CURRENT_REVISED_OFFICIAL_MONTHLY_INPUT_ALIGNMENT_AUDIT",
        "months": len(monthly),
        "mean_l1_share_distance": float(monthly.l1_share_distance.mean()),
        "median_l1_share_distance": float(monthly.l1_share_distance.median()),
        "mean_total_variation_distance": float(monthly.total_variation_distance.mean()),
        "mean_absolute_share_error_by_group": {
            group: float(monthly[f"{group}_absolute_share_error"].mean()) for group in COARSE
        },
        "inputs": {"et54_path": str(args.et54), "et54_sha256": sha256(args.et54),
                   "neso_path": str(args.neso), "neso_sha256": sha256(args.neso),
                   "script_sha256": sha256(Path(__file__))},
        "interpretation": "Current revised ET 5.4 MPP statistics vs current NESO GB hourly generation; not an as-issued historical backtest or CIF score.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
