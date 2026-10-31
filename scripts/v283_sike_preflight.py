"""V283 preflight only: inspect native EIA-930 activity, never CO2 label files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V283_SIKE_LABEL_PREFLIGHT_PROTOCOL.md"
FILES = tuple(sorted((ROOT / "data/native-eia-2025").glob("EIA930_BALANCE_2025_*.csv")))
BA = "SIKE"
FUEL_COLUMNS = (
    "Net Generation (MW) from Coal", "Net Generation (MW) from Natural Gas",
    "Net Generation (MW) from All Petroleum Products", "Net Generation (MW) from Nuclear",
    "Net Generation (MW) from Hydropower Excluding Pumped Storage",
    "Net Generation (MW) from Pumped Storage",
    "Net Generation (MW) from Solar without Integrated Battery Storage",
    "Net Generation (MW) from Solar with Integrated Battery Storage",
    "Net Generation (MW) from Wind without Integrated Battery Storage",
    "Net Generation (MW) from Wind with Integrated Battery Storage",
    "Net Generation (MW) from Geothermal", "Net Generation (MW) from Other Fuel Sources",
    "Net Generation (MW) from Unknown Fuel Sources",
)
USE = ("Balancing Authority", "Data Date", "Hour Number", "Net Generation (MW)", *FUEL_COLUMNS)
OUT = ROOT / "results/v283-sike-label-preflight"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    frames = []
    for path in FILES:
        for part in pd.read_csv(path, usecols=lambda col: col in USE, chunksize=500_000):
            target = part[part["Balancing Authority"].eq(BA)]
            if len(target):
                frames.append(target)
    if not frames:
        raise RuntimeError("SIKE absent from native EIA-930")
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["Data Date"], errors="coerce")
    for column in ("Net Generation (MW)", *FUEL_COLUMNS):
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.).clip(lower=0.)
    month_rows = []
    for month, group in data.groupby(data.date.dt.month):
        positive = group["Net Generation (MW)"] > 0
        month_rows.append({"month": int(month), "rows": int(len(group)),
            "positive_generation_hours": int(positive.sum()),
            "positive_generation_mwh": float(group["Net Generation (MW)"].sum()),
            "active_fuels": [c.removeprefix("Net Generation (MW) from ")
                for c in FUEL_COLUMNS if group[c].sum() > 0]})
    result = {"ba": BA, "row_count": int(len(data)),
        "months_present": len(month_rows), "positive_generation_mwh": float(data["Net Generation (MW)"].sum()),
        "fuel_totals_mwh": {c.removeprefix("Net Generation (MW) from "): float(data[c].sum())
                            for c in FUEL_COLUMNS if data[c].sum() > 0},
        "months": month_rows, "preflight_gate_months": 10,
        "status": "failed; official EIA hourly CO2 workbook was not opened",
        "protocol_sha256": sha(PROTOCOL),
        "input_files": {str(p.relative_to(ROOT)): sha(p) for p in FILES},
        "script_sha256": sha(Path(__file__))}
    OUT.mkdir(parents=True)
    (OUT / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
