#!/usr/bin/env python
"""Download fuel-price tracks for TransCIF-FD (roadmap E-class, #4).

Sources (both keyless):
    * World Bank Pink Sheet (monthly): Australian thermal coal (NEWC,
      global coal proxy) + natural gas for Europe / Japan / US — covers
      the AU / UK / US jurisdictions at regime level, which is where
      coal-vs-gas dispatch switching lives.
    * FRED fredgraph.csv (daily): Henry Hub spot (US gas refinement).

Writes ``data_2023/prices/prices_2023.csv`` with one row per month of
2023: coal_newc, gas_eu, gas_jp, gas_us (pink sheet) and gas_us_daily
(FRED monthly mean).

Usage:
    .venv/bin/python scripts/data/download_prices.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent.parent / "data_2023"
OUT = DATA / "prices" / "prices_2023.csv"
PINK = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP&cosd=2023-01-01&coed=2023-12-31"

# Pink-sheet column labels (Monthly Prices sheet, commodities are COLUMNS).
PINK_COLS = {
    "coal_newc": "Coal, Australian",
    "gas_eu": "Natural gas, Europe",
    "gas_jp": "Liquefied natural gas, Japan",
    "gas_us": "Natural gas, US",
}


def month_col(date_cell):
    """Pink-sheet monthly header cells parse as '2023M01' or Timestamps."""
    s = str(date_cell)
    if "M" in s and len(s) == 7:
        return int(s[5:])
    try:
        return pd.Timestamp(s).month
    except (ValueError, TypeError):
        return None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request  # noqa: PLC0415

    pink_path = OUT.parent / "CMO-Monthly.xlsx"
    if not pink_path.exists():
        print("[prices] downloading World Bank pink sheet ...")
        urllib.request.urlretrieve(PINK, pink_path)

    xls = pd.ExcelFile(pink_path)
    sheet = [s for s in xls.sheet_names if "monthly" in s.lower()][0]
    raw = pd.read_excel(pink_path, sheet_name=sheet, header=None)
    # Layout: row 4 holds commodity names (columns), row 5 units, column 0
    # holds month labels like '2023M01' from row 6 on.
    names = raw.iloc[4].astype(str).str.strip()
    col_of = {}
    for col, label in PINK_COLS.items():
        match = [j for j, v in names.items() if v == label]
        if not match:
            raise RuntimeError(f"pink sheet: column '{label}' not found")
        col_of[col] = match[0]
    months = {}
    for i in range(5, len(raw)):
        m = month_col(raw.iloc[i, 0])
        if m is not None and 1 <= m <= 12:
            months[m] = i

    out = {"month": list(range(1, 13))}
    for col, _ in PINK_COLS.items():
        j = col_of[col]
        out[col] = [float(raw.iloc[months[m], j]) if m in months else np.nan
                    for m in range(1, 13)]

    df = pd.DataFrame(out)

    # FRED daily Henry Hub -> monthly mean (US refinement).
    try:
        fred = pd.read_csv(FRED, parse_dates=["observation_date"])
        fred = fred[(fred["observation_date"].dt.year == 2023)]
        fred["month"] = fred["observation_date"].dt.month
        df["gas_us_daily"] = df["month"].map(
            fred.groupby("month")["DHHNGSP"].mean())
        print("[prices] FRED daily US gas merged")
    except Exception as e:  # noqa: BLE001
        print(f"[prices] FRED unavailable ({e}); monthly-only track")
        df["gas_us_daily"] = np.nan

    df.to_csv(OUT, index=False)
    print(f"[prices] wrote {OUT}")
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
