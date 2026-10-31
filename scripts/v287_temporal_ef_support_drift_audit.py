"""Post-hoc audit: target-year EIA fuel-factor drift vs prior-year supports."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import v279_asymmetric_ef_sensor_frontier as sensor
import v280_eia_official_label_pace_lgee as v280

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("PACE", "LGEE")
YEARS = (2020, 2021, 2022, 2023)
FACTORS = {"coal": ("CO2 Factor: COL", "NG: COL"),
           "gas": ("CO2 Factor: NG", "NG: NG"),
           "oil": ("CO2 Factor: OIL", "NG: OIL")}
FACTOR_INDEX = {"coal": 0, "gas": 1, "oil": 2}
CONVERSION = 453.59237
OUT = ROOT / "results/v287-temporal-ef-support-drift-audit"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def parse_day(v: object) -> date | None:
    if isinstance(v, datetime): return v.date()
    if isinstance(v, date): return v
    return None


def scan(ba: str) -> dict[int, dict[str, np.ndarray]]:
    path = v280.DATA / f"{ba}.xlsx"
    wb=load_workbook(path,read_only=True,data_only=True); ws=wb["Published Hourly Data"]
    rows=ws.iter_rows(values_only=True); headers=next(rows); ix={str(v):i for i,v in enumerate(headers) if v is not None}
    out={y:{f:[] for f in FACTORS} for y in range(2019,2024)}
    for row in rows:
        day=parse_day(row[ix["Local date"]])
        if day is None or day.year not in out: continue
        for fuel,(fc,gc) in FACTORS.items():
            try: gen=float(row[ix[gc]] or 0.); factor=float(row[ix[fc]])*CONVERSION
            except (TypeError,ValueError): continue
            if gen>0 and np.isfinite(factor): out[day.year][fuel].append(factor)
    wb.close()
    return {y:{f:np.asarray(v,float) for f,v in vals.items()} for y,vals in out.items()}


def main() -> None:
    if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
    rows=[]; hashes={}
    for ba in TARGETS:
        data=scan(ba); hashes[ba]=sha(v280.DATA/f"{ba}.xlsx")
        for year in YEARS:
            for fuel,idx in FACTOR_INDEX.items():
                prior=data[year-1][fuel]; target=data[year][fuel]
                if len(prior):
                    low=float(prior.min()); high=float(prior.max())
                else:
                    low,high=sensor.ALL_EF[idx],sensor.ALL_EF[idx]
                rows.append({"ba":ba,"target_year":year,"prior_year":year-1,"fuel":fuel,
                    "prior_active_hours":len(prior),"target_active_hours":len(target),
                    "prior_low_g_kwh":low,"prior_high_g_kwh":high,
                    "target_min_g_kwh":float(target.min()) if len(target) else np.nan,
                    "target_median_g_kwh":float(np.median(target)) if len(target) else np.nan,
                    "target_max_g_kwh":float(target.max()) if len(target) else np.nan,
                    "target_below_prior_fraction":float(np.mean(target<low)) if len(target) else np.nan,
                    "target_inside_prior_fraction":float(np.mean((target>=low)&(target<=high))) if len(target) else np.nan,
                    "target_above_prior_fraction":float(np.mean(target>high)) if len(target) else np.nan})
    frame=pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    frame.to_csv(OUT/"target_vs_prior_factor_support.csv",index=False)
    pooled=frame[frame.fuel.eq("coal")].copy()
    print(pooled.to_string(index=False))
    metadata={"status":"post-hoc label-source factor drift diagnostic after V286 temporal holdout","target_workbooks":hashes,
        "script_sha256":sha(Path(__file__)),"factor_conversion":"lbs/kWh to g/kWh x 453.59237",
        "outputs":{"target_vs_prior_factor_support.csv":sha(OUT/"target_vs_prior_factor_support.csv")},
        "caveat":"target-year factors are diagnostic only; they were not used in V286 prediction intervals"}
    (OUT/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")

if __name__=="__main__": main()
