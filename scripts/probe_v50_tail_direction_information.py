"""LOO diagnostic of monthly fuel features versus held-source tail imbalance."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import disjoint_calibration_v50 as helper

RUN=ROOT/"results/disjoint-calibration-v50"
DEV=ROOT/"results/disjoint-calibration-v50-crossfit-asymmetric-risk"
OUT=ROOT/"results/disjoint-calibration-v50-tail-direction-information"
FUELS=helper.FUELS
EF=helper.EF

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def main():
    if OUT.exists():raise FileExistsError("Preserve previous tail-direction diagnostic")
    freeze=json.loads((RUN/"freeze.json").read_text())
    feature_by={}
    input_hashes={str(RUN/"freeze.json"):sha(RUN/"freeze.json"),
                  str(DEV/"source_macro_metrics.csv"):sha(DEV/"source_macro_metrics.csv")}
    ef_rows={}
    for path in map(Path,freeze["external_monthly"]):
        input_hashes[str(path)]=sha(path)
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                if r["region"] in helper.EXTERNAL:ef_rows.setdefault(r["region"],[]).append(r)
    labels=RUN/"external_labels"
    for s in helper.EXTERNAL:
        lp=labels/f"{s}.npz";input_hashes[str(lp)]=sha(lp)
        with np.load(lp,allow_pickle=False) as z:
            pos=z["positions"].astype(int);orig=z["origins"].astype("int64")
        keep=helper.eligible(pd.DatetimeIndex(pd.to_datetime(orig,utc=True)),90)
        days=pd.DatetimeIndex(pd.to_datetime(orig[pos[keep]],utc=True))
        parsed=[]
        for r in ef_rows.get(s,[]):
            start,end,release=[pd.Timestamp(r[k]) for k in ("period_start","period_end","available_at")]
            share=np.array([float(r["fuelshare_"+f]) for f in FUELS])
            parsed.append((start,end,release,share))
        parsed.sort(key=lambda r:r[0])
        values=[]
        for d in days:
            avail=[r for r in parsed if r[0]<d and r[1]<=d and r[2]<=d]
            if len(avail)<3:raise ValueError("Three as-of monthly records required")
            share=np.mean([x[3] for x in avail[-3:]],axis=0)
            values.append([float(share@EF),float(share[[0,1,2,7]].sum()),float(share[[5,6]].sum())])
        feature_by[s]=np.mean(values,axis=0)
    source=pd.read_csv(DEV/"source_macro_metrics.csv")
    selected=source.loc[(source.arm=="B0-SizeET-AlignedKeep")&(source.risk_method=="MCStd")
        &(source.location_method=="ef_only")&(source.tail_calibration=="symmetric_90")]
    if set(selected.held_source)!=set(helper.EXTERNAL):raise ValueError("Exact ten source outcomes required")
    response={r.held_source:float(r.lower_miss-r.upper_miss) for r in selected.itertuples()}
    names=("mean_3month_EF","mean_fossil_share","mean_wind_solar_share")
    feats={s:dict(zip(names,feature_by[s])) for s in helper.EXTERNAL}
    rows=[]
    y=np.array([response[s] for s in helper.EXTERNAL])
    for feature in names:
        x=np.array([feats[s][feature] for s in helper.EXTERNAL])
        pred_mean=[];pred_linear=[]
        for i,held in enumerate(helper.EXTERNAL):
            mask=np.arange(len(y))!=i
            mean=float(y[mask].mean())
            design=np.column_stack([np.ones(mask.sum()),x[mask]])
            beta=np.linalg.lstsq(design,y[mask],rcond=None)[0]
            estimate=float(np.array([1.,x[i]])@beta)
            pred_mean.append(mean);pred_linear.append(estimate)
            rows.append({"feature":feature,"held_source":held,"feature_value":float(x[i]),
                "observed_tail_imbalance":float(y[i]),"loo_null_mean":mean,
                "loo_linear_prediction":estimate,"absolute_error_null":abs(y[i]-mean),
                "absolute_error_linear":abs(y[i]-estimate),
                "direction_zero_correct_null":int(np.sign(mean)==np.sign(y[i])),
                "direction_zero_correct_linear":int(np.sign(estimate)==np.sign(y[i]))})
    table=pd.DataFrame(rows)
    summaries=[]
    for name,group in table.groupby("feature",sort=False):
        summaries.append({"feature":name,
            "loo_mae_null":float(group.absolute_error_null.mean()),
            "loo_mae_linear":float(group.absolute_error_linear.mean()),
            "loo_rmse_null":float(np.sqrt(np.mean(group.absolute_error_null**2))),
            "loo_rmse_linear":float(np.sqrt(np.mean(group.absolute_error_linear**2))),
            "direction_accuracy_null":float(group.direction_zero_correct_null.mean()),
            "direction_accuracy_linear":float(group.direction_zero_correct_linear.mean()),
            "sources":len(group)})
    OUT.mkdir()
    table_path=OUT/"source_heldout_predictions.csv";table.to_csv(table_path,index=False)
    summary_path=OUT/"summary.json"
    summary={"status":"COMPLETE_MONTHLY_TAIL_DIRECTION_LOO_DIAGNOSTIC",
        "protocol":"docs/V50_TAIL_DIRECTION_INFORMATION_PROTOCOL.md",
        "protocol_sha256":sha(ROOT/"docs/V50_TAIL_DIRECTION_INFORMATION_PROTOCOL.md"),
        "script_sha256":sha(Path(__file__)),"independent_units":10,
        "hourly_label_values_opened":False,"target_or_unseen_values_opened":False,
        "features":names,"feature_values":feats,"tail_imbalance":response,
        "summaries":summaries,"input_hashes":input_hashes,
        "source_predictions_sha256":sha(table_path),
        "interpretation_limit":"Ten previously used regions; descriptive LOO diagnostic only."}
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":summary["status"],"summaries":summaries},ensure_ascii=False))

if __name__=="__main__":main()
