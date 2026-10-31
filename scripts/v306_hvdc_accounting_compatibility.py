"""Check whether public 2023 point-to-point UHVDC totals fit V304 margins.

This is a transparent arithmetic audit, not a flow reconstruction. The line
totals are annual system totals from the NEA reliability report. Corridor
classification uses published endpoints. Because those are gross physical
line totals and V304 margins come from a separate regional accounting table,
the script reports incompatibility as a warning against substitution.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARGINS = ROOT / "data/china/mee_regional_factor_trade_margins_2023.csv"

# NEA 2023 reliability report, unit: 100 million kWh (亿千瓦时).
# Fufeng, Jinsu, and Jiansu have Sichuan (Southwest) origins and East-region
# destinations. Binjin is separated because its supply origins span Sichuan
# and Yunnan, which fall into different factor regions. Yahu terminates in
# Jiangxi and is kept separate from the East aggregate.
SW_TO_EAST = {
    "Fufeng": 262.77,
    "Jinsu": 304.20,
    "Jiansu": 234.05,
}
MIXED_ORIGIN = {"Binjin": 238.17}
SW_TO_JIANGXI = {"Yahu": 183.20}


def read_margin_100m_kwh() -> float:
    with MARGINS.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        for row in rows:
            if row["region"] == "Southwest":
                # Source file is in 10,000 kWh; convert to 100 million kWh.
                return int(row["outgoing_10k_kwh"]) / 10_000
    raise RuntimeError("Southwest margin not found")


def main() -> None:
    margin = read_margin_100m_kwh()
    known_east = sum(SW_TO_EAST.values())
    with_mixed = known_east + sum(MIXED_ORIGIN.values())
    with_yahu = with_mixed + sum(SW_TO_JIANGXI.values())
    print(f"Southwest outgoing margin: {margin:.4f} 亿kWh")
    print(f"Three known Sichuan-to-East UHVDC lines: {known_east:.2f} 亿kWh")
    print(f"Including mixed-origin Binjin without allocation: {with_mixed:.2f} 亿kWh")
    print(f"Including Yahu (destination Jiangxi): {with_yahu:.2f} 亿kWh")
    print(f"Yahu destination assignment: review against the paper's region map")


if __name__ == "__main__":
    main()
