"""Score the sealed TVA forecast using V6's frozen cross-snapshot boundary rule."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts import disjoint_calibration_v50 as helper
from scripts import confirm_v50_tva_asymmetric as forecast
from transcif.data.eia930 import normalize_balance
from transcif.research import eia_proxy

OUT=forecast.OUT
DAYS=forecast.DAYS
TARGET=forecast.TARGET
EF=forecast.EF
AUDIT_PATH=forecast.V6_AUDIT
AMENDMENT=ROOT/"docs/V50_TVA_CONFIRMATION_BOUNDARY_AMENDMENT.md"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def write_new(path,obj):
    with Path(path).open("x") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def target_labels(audit,pins):
    needed=(7,8)
    entries={int(e["source_snapshot"]):e for e in audit["native_sources"]}
    if not set(needed)<=set(entries): raise ValueError("V6 snapshots 7 and 8 required for target boundary")
    parts=[]
    for sid in needed:
        entry=entries[sid];path=Path(entry["path"])
        if sha(path)!=pins["raw_native"][str(path)]:raise ValueError("Native boundary snapshot hash mismatch")
        raw=pd.read_csv(path)
        raw=raw.loc[raw["Balancing Authority"].eq(TARGET)]
        frame=normalize_balance(raw);frame["source_snapshot"]=sid
        frame=frame.loc[(frame.valid_at>=DAYS[0])&(frame.valid_at<DAYS[-1]+pd.Timedelta(days=1))]
        if len(frame):parts.append(frame)
    if len(parts)!=2:raise ValueError("Both V6 boundary snapshots must contribute")
    frame=pd.concat(parts,ignore_index=True)
    if frame.duplicated(["region","valid_at"]).any():raise ValueError("Duplicate target boundary hour")
    frame.index=pd.DatetimeIndex(frame.valid_at)
    hourly,quality=eia_proxy._proxy_rows(frame,TARGET,audit["native_schemas"])
    hourly=hourly.loc[(hourly.time>=DAYS[0])&(hourly.time<DAYS[-1]+pd.Timedelta(days=1))]
    valid=hourly.loc[hourly.valid].copy()
    valid["day"]=pd.to_datetime(valid.time,utc=True).dt.floor("D")
    labels=[]
    for day,part in valid.groupby("day",sort=True):
        if len(part)!=24:raise ValueError("Incomplete daily label under frozen V6 rule")
        g=part[["generation_"+f for f in helper.FUELS]].to_numpy(float)
        labels.append((day,(g/g.sum(axis=1,keepdims=True))@EF))
    if len(labels)!=181 or [x[0] for x in labels]!=list(DAYS):
        raise ValueError("Exact 181-day V6 target support required")
    return np.stack([x[1] for x in labels]),quality

def main():
    seal_path=OUT/"prediction_seal.json"
    seal=read(seal_path)
    pins=seal["input_output_hashes"]
    if seal.get("status")!="TVA_PREDICTION_SEALED_LABELS_UNOPENED":
        raise ValueError("Prediction-only seal required")
    if sha(forecast.PROTOCOL)!=pins["protocol"] or sha(OUT/"tva_prediction_seal.npz")!=pins["prediction"]:
        raise ValueError("Frozen protocol/prediction seal mismatch")
    if sha(Path(forecast.__file__))!=pins["script"]:
        raise ValueError("Forecast-generation code changed after sealing")
    amendment=AMENDMENT.read_text()
    scorer_hash=sha(Path(__file__))
    if f"Scorer SHA256: {scorer_hash}" not in amendment:
        raise ValueError("Boundary amendment must pin this exact scorer before labels are read")
    if (OUT/"score_seal.json").exists():raise FileExistsError("TVA scoring is immutable")
    audit=read(AUDIT_PATH)
    truth,qa=target_labels(audit,pins)
    records=[]
    with np.load(OUT/"tva_prediction_seal.npz",allow_pickle=False) as z:
        keys=[f"{TARGET}/{d:%Y-%m-%d}" for d in DAYS]
        if z["keys"].tolist()!=keys or not np.array_equal(z["origins"],DAYS.asi8):
            raise ValueError("Prediction support mismatch")
        for method in seal["methods"]:
            lo,hi=z["lo_"+method],z["hi_"+method]
            if lo.shape!=truth.shape or hi.shape!=truth.shape or np.any(lo>hi):
                raise ValueError("Sealed intervals malformed")
            for i,day in enumerate(DAYS):
                w=hi[i]-lo[i]
                score=w+20*np.maximum(lo[i]-truth[i],0)+20*np.maximum(truth[i]-hi[i],0)
                hit=(truth[i]>=lo[i])&(truth[i]<=hi[i])
                records.append({"method":method,"day":f"{day:%Y-%m-%d}",
                    "coverage90":float(hit.mean()),"width90":float(w.mean()),
                    "interval_score90":float(score.mean()),
                    "lower_miss":float((truth[i]<lo[i]).mean()),
                    "upper_miss":float((truth[i]>hi[i]).mean())})
    table=pd.DataFrame(records)
    daily=OUT/"daily_metrics.csv";table.to_csv(daily,index=False)
    table["month"]=pd.to_datetime(table.day).dt.month
    monthly=table.groupby(["method","month"],as_index=False)[
        ["coverage90","width90","interval_score90","lower_miss","upper_miss"]].mean()
    monthly_path=OUT/"monthly_metrics.csv";monthly.to_csv(monthly_path,index=False)
    aggregate=table.groupby("method",sort=False)[
        ["coverage90","width90","interval_score90","lower_miss","upper_miss"]].mean().reset_index()
    aggregate_path=OUT/"aggregate_metrics.csv";aggregate.to_csv(aggregate_path,index=False)
    write_new(OUT/"score_seal.json",{"status":"TVA_CONFIRMATION_SCORED",
        "target":TARGET,"days":181,"labels_opened_after_prediction_seal":True,
        "target_refit_or_recalibration":False,"proxy_quality":qa,
        "boundary_snapshots":[7,8],"target_native_snapshot_hashes":{
            str(e["path"]):e["sha256"] for e in audit["native_sources"] if int(e["source_snapshot"]) in (7,8)},
        "prediction_seal_sha256":sha(seal_path),"prediction_sha256":sha(OUT/"tva_prediction_seal.npz"),
        "boundary_amendment_sha256":sha(AMENDMENT),"scorer_sha256":scorer_hash,
        "daily_metrics_sha256":sha(daily),"monthly_metrics_sha256":sha(monthly_path),
        "aggregate_metrics_sha256":sha(aggregate_path),
        "aggregate_metrics":aggregate.to_dict(orient="records"),
        "interpretation_limit":"One preselected region and one semester; not a multi-region or China transfer guarantee."})
    print(json.dumps({"status":"TVA_CONFIRMATION_SCORED",
                      "metrics":aggregate.to_dict(orient="records")},ensure_ascii=False))

if __name__=="__main__":main()
