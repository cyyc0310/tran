"""Two-phase, frozen TVA confirmation. Run predict, verify the seal, then score."""
from __future__ import annotations
import csv, hashlib, json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import disjoint_calibration_v50 as helper
from scripts import disjoint_point_v50 as point
from scripts import probe_v50_crossfit_asymmetric_risk as dev
from scripts import risk_baselines_v50 as risk
from scripts import run_disjoint_calibration_v50 as runner
from transcif.data.eia930 import normalize_balance
from transcif.research import eia_proxy

OUT = ROOT / "results/disjoint-calibration-v50-tva-confirmation"
V50 = ROOT / "results/disjoint-calibration-v50"
V6_AUDIT = ROOT / "results/external-v6-data/audit.json"
PROTOCOL = ROOT / "docs/V50_TVA_ASYMMETRIC_CONFIRMATION_PROTOCOL.md"
TARGET = "TVA"
DAYS = pd.date_range("2026-01-01", "2026-06-30", tz="UTC").as_unit("ns")
SOURCES, EXTERNAL = helper.LEGACY, helper.EXTERNAL
EF = helper.EF
B0 = helper.BASES.index("B0-SizeET-AlignedKeep")
RISK_INDEX = {"MCStd": 1, "HETScale": 3, "CQRplus": 5}
FUEL_COLS = tuple("fuelshare_" + f for f in helper.FUELS)
MONTH_COLS = ("region", "period_start", "period_end", "available_at", "observed_hours",
              "expected_hours", "coverage", "total_mwh", *FUEL_COLS)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def write_new(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")

def pin_inputs():
    runner.check()
    audit = read(V6_AUDIT)
    c = audit["candidate_selection"]["candidates"][TARGET]
    if not c["qualifies"] or c["periods"]["2026"]["valid_full_utc_days"] != 181:
        raise ValueError("TVA must remain the metadata-qualified 181-day candidate")
    raw = {}
    for entry in audit["native_sources"]:
        p = Path(entry["path"])
        if sha(p) != entry["sha256"]: raise ValueError("V6 native hash mismatch")
        raw[str(p)] = entry["sha256"]
    return audit, {"v50_freeze": sha(V50 / "freeze.json"),
        "v50_prediction_seal": sha(V50 / "prediction_seal.json"),
        "v50_calibration_seal": sha(V50 / "calibration_seal.json"),
        "v6_audit": sha(V6_AUDIT), "raw_native": raw}

def target_monthly(audit):
    schemas = audit["native_schemas"]
    paths = [Path(x["path"]) for x in audit["native_sources"]
             if "2026" not in Path(x["path"]).name]
    parts, hashes = [], {}
    for i, path in enumerate(paths):
        hashes[str(path)] = sha(path)
        raw = pd.read_csv(path)
        raw = raw.loc[raw["Balancing Authority"].eq(TARGET)]
        frame = normalize_balance(raw)
        frame["source_snapshot"] = i
        frame = frame.loc[frame.valid_at < pd.Timestamp("2026-01-01", tz="UTC")]
        if len(frame): parts.append(frame)
    if len(parts) != 8: raise ValueError("Expected eight native snapshots through 2025")
    rows = pd.concat(parts, ignore_index=True)
    rows.index = pd.DatetimeIndex(rows.valid_at)
    if rows.index.has_duplicates or rows.region.ne(TARGET).any(): raise ValueError("Invalid TVA history")
    monthly, eligibility = eia_proxy._monthly_at_release(rows.sort_index(), TARGET, schemas)
    if not monthly or any(pd.Timestamp(r["period_end"]) > pd.Timestamp("2026-01-01", tz="UTC") for r in monthly):
        raise ValueError("Target monthly context must end before 2026H1")
    return monthly, {"native_historical_hashes": hashes, "monthly_eligibility": eligibility}

def features(rows, days):
    parsed = []
    for r in rows:
        start, end, release = [pd.Timestamp(r[k]).tz_convert("UTC") for k in ("period_start","period_end","available_at")]
        share = np.array([float(r[k]) for k in FUEL_COLS])
        if not np.isfinite(share).all() or np.any(share < 0) or not np.isclose(share.sum(),1.,atol=1e-8,rtol=0):
            raise ValueError("Invalid monthly fuel shares")
        parsed.append((start,end,release,float(share@EF),share))
    ef, group = [], []
    for day in days:
        legal = [x for x in parsed if x[0] < day and x[1] <= day and x[2] <= day]
        if len(legal) < 3: raise ValueError("Three released months required per query")
        chosen = legal[-3:]
        share = np.mean([x[4] for x in chosen],axis=0)
        ef.append(np.mean([x[3] for x in chosen]))
        group.append([share[[0,1,2,7]].sum(),share[[5,6]].sum(),share[3]])
    return np.asarray(ef),np.asarray(group)

def patch_roles():
    old_date, old_meta = point._dates, risk._metadata
    def date_role(region, values):
        if region != TARGET: return old_date(region, values)
        dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(values),unit="ns",utc=True)).as_unit("ns")
        if not dates.equals(DAYS): raise ValueError("Only exact TVA 2026H1 support allowed")
        return "external2026_confirmation", dates, 2024
    def meta(regions, keys, outer, *, training):
        if training: return old_meta(regions,keys,outer,training=training)
        r=np.asarray(regions,dtype=str)
        if set(r)=={TARGET}:
            expected=[f"{TARGET}/{d:%Y-%m-%d}" for d in DAYS]
            if outer not in risk.SOURCES or list(keys)!=expected or len(r)!=len(expected):
                raise ValueError("Only TVA 2026H1 frozen risk inference allowed")
            return r
        return old_meta(regions,keys,outer,training=training)
    point._dates, risk._metadata = date_role, meta
    return old_date,old_meta

def unpatch_roles(old): point._dates,risk._metadata=old

def source_data(frozen, tva_monthly):
    ef, groups = {}, {}
    for source in EXTERNAL:
        rows=[]
        for p in map(Path,frozen["external_monthly"]):
            with p.open(newline="") as f: rows += [r for r in csv.DictReader(f) if r["region"]==source]
        e,g=features(rows,helper.CAL_DAYS); ef[source],groups[source]=e,g
    ef[TARGET],groups[TARGET]=features(tva_monthly,DAYS)
    seal=read(V50/"calibration_seal.json")
    labels={x["region"]:x for x in seal["labels"]}
    predseal=read(V50/"prediction_seal.json")
    pe={(x["outer"],x["region"]):x for x in predseal["entries"]
        if x.get("role")=="calibration" and x.get("region") in EXTERNAL}
    if set(labels)!=set(EXTERNAL) or len(pe)!=len(SOURCES)*len(EXTERNAL): raise ValueError("V50 external grid incomplete")
    lc,rc,bc={},{},{}
    for s in EXTERNAL:
        lp=V50/"external_labels"/f"{s}.npz"
        if sha(lp)!=labels[s]["sha256"]: raise ValueError("V50 label hash mismatch")
        with np.load(lp,allow_pickle=False) as z: origins=z["origins"].astype("int64"); pos=z["positions"].astype(int); truth=z["truth"].astype(float)
        keep=helper.eligible(pd.DatetimeIndex(pd.to_datetime(origins,utc=True)),90)
        if not np.array_equal(origins,helper.CAL_DAYS.asi8[pos]) or truth.shape!=(len(pos),24) or int(keep.sum())!=frozen["lag_counts"]["90"][s]:
            raise ValueError("V50 label support mismatch")
        pos,truth=pos[keep],truth[keep]
        lc[s]=(pos,truth,ef[s][pos],groups[s][pos]); rc[s]={}; bc[s]={}
        for o in SOURCES:
            p=V50/"predictions"/f"{o}__{s}.npz"
            if sha(p)!=pe[(o,s)]["sha256"]: raise ValueError("V50 source prediction hash mismatch")
            with np.load(p,allow_pickle=False) as z:
                rc[s][o]=z["raw"][B0,pos].copy(); bc[s][o]={}
                for name,idx in RISK_INDEX.items():
                    bc[s][o][name]={k:z[arr][B0,idx,pos].copy() for k,arr in
                        (("lo","lo0"),("hi","hi0"),("scale","scales"))}
    return lc,rc,bc

def conformal_q(outer, method, bias, label_cache, bounds):
    scores={"sym":{},"lo":{},"hi":{}}
    for cal in EXTERNAL:
        donors=tuple(s for s in EXTERNAL if s!=cal)
        offset,_=dev.fit_location(donors,cal,bias,label_cache)
        y=label_cache[cal][1]; b=bounds[cal][outer][method]
        lo=np.maximum((b["lo"]+offset[:,None]-y)/b["scale"],0.)
        hi=np.maximum((y-b["hi"]-offset[:,None])/b["scale"],0.)
        scores["lo"][cal],scores["hi"][cal]=lo,hi;scores["sym"][cal]=np.maximum(lo,hi)
    matrix=np.concatenate([scores["sym"][s] for s in EXTERNAL])
    regions=[s for s in EXTERNAL for _ in range(len(scores["sym"][s]))]
    qsym=float(risk.calibrate(matrix,regions,"equal_region")["q"])
    return qsym,dev.equal_source_quantile(scores["lo"],.95),dev.equal_source_quantile(scores["hi"],.95)

def predict_phase():
    if OUT.exists(): raise FileExistsError("Preserve previous confirmation attempts")
    OUT.mkdir(parents=True); (OUT/"predictions").mkdir()
    write_new(OUT/"protocol_pin.json",{"path":str(PROTOCOL),"sha256":sha(PROTOCOL)})
    audit,pins=pin_inputs(); monthly,prep=target_monthly(audit)
    mp=OUT/"TVA_monthly_context.csv";pd.DataFrame(monthly,columns=MONTH_COLS).to_csv(mp,index=False)
    frozen=runner.check();lc,rc,bc=source_data(frozen,monthly)
    ef,groups=features(monthly,DAYS)
    old=patch_roles()
    try:
        legacy,rows=runner.monthly_rows(frozen);rows.extend(monthly)
        query=point.query(rows,TARGET,DAYS,None,None)
        if query["role"]!="external2026_confirmation":raise ValueError("Wrong TVA role")
        raw_by, bounds_by={},{}
        for outer in SOURCES:
            with (V50/"models"/f"{outer}.pkl").open("rb") as f: fitted=pickle.load(f)
            result=point.predict(fitted["point"],query);raw=result["raw"][B0]
            packet=helper.feature_packet(result["raw"],query["features"])
            keys=[f"{TARGET}/{d:%Y-%m-%d}" for d in DAYS]
            rp=fitted["risk"][B0].predict(packet["X"][B0],packet["point"][B0],
                packet["std_hour"][B0],packet["std_level"][B0],packet["std_shape"][B0],
                np.repeat(TARGET,len(DAYS)),keys)
            raw_by[outer]=raw;bounds_by[outer]={}
            for name,idx in RISK_INDEX.items():
                bounds_by[outer][name]={k:rp[k][idx].copy() for k in ("lo0","hi0","scales")}
        intervals=[];qrows=[]
        for outer in SOURCES:
            bias={s:(lc[s][1]-rc[s][outer].mean(axis=1)).mean(axis=1) for s in EXTERNAL}
            tva_stub={TARGET:(np.arange(len(DAYS)),np.zeros((len(DAYS),24)),ef,groups)}
            offset,_=dev.fit_location(EXTERNAL,TARGET,bias,{**lc,**tva_stub})
            for method in RISK_INDEX:
                qs,ql,qh=conformal_q(outer,method,bias,lc,bc);b=bounds_by[outer][method]
                low,high=b["lo0"]+offset[:,None],b["hi0"]+offset[:,None]
                if method=="MCStd":
                    intervals.append((outer,"candidate_EF_MCStd_asymmetric",
                        np.clip(low-ql*b["scales"],0,1000),np.clip(high+qh*b["scales"],0,1000)))
                    sym=risk.final_bounds(low,high,b["scales"],qs)
                    intervals.append((outer,"EF_MCStd_symmetric",sym["lo"],sym["hi"]))
                else:
                    sym=risk.final_bounds(low,high,b["scales"],qs)
                    intervals.append((outer,f"EF_{method}_symmetric",sym["lo"],sym["hi"]))
                qrows.append({"outer":outer,"risk_method":method,"q_symmetric":qs,"q_lower":ql,"q_upper":qh})
        output={}
        for name in sorted({r[1] for r in intervals}):
            view=[r for r in intervals if r[1]==name]
            if len(view)!=len(SOURCES):raise ValueError("Seven outer views required per method")
            output[name]=(np.median(np.stack([r[2] for r in view]),axis=0),
                          np.median(np.stack([r[3] for r in view]),axis=0))
        anchors=[]
        for outer in SOURCES:
            a=raw_by[outer];center=a.mean(axis=1,keepdims=True)
            expanded=np.sort(np.clip(center+2*(a-center),0,1000),axis=1)
            anchors.append((expanded[:,6],expanded[:,121]))
        output["simple_B0_anchor_factor2"]=(np.median(np.stack([a[0] for a in anchors]),axis=0),
                                               np.median(np.stack([a[1] for a in anchors]),axis=0))
    finally: unpatch_roles(old)
    arrays={"origins":DAYS.asi8,"keys":np.array([f"{TARGET}/{d:%Y-%m-%d}" for d in DAYS]),
            "monthly_ef":ef,"monthly_groups":groups}
    for outer,a in raw_by.items():arrays["raw_"+outer]=a
    for name,(lo,hi) in output.items():arrays["lo_"+name]=lo;arrays["hi_"+name]=hi
    pp=OUT/"tva_prediction_seal.npz"
    with pp.open("xb") as f:np.savez_compressed(f,**arrays)
    qp=OUT/"calibration_quantiles.csv";pd.DataFrame(qrows).to_csv(qp,index=False)
    pins.update({"protocol":sha(PROTOCOL),"script":sha(Path(__file__)),
        "monthly_context":sha(mp),"prediction":sha(pp),"quantiles":sha(qp),
        "native_historical":prep["native_historical_hashes"]})
    write_new(OUT/"prediction_seal.json",{"status":"TVA_PREDICTION_SEALED_LABELS_UNOPENED",
        "target":TARGET,"days":181,"outer_functions":list(SOURCES),
        "methods":list(output),"target_hourly_label_values_opened":False,
        "target_labels_used_for_fit_calibration_or_selection":False,
        "location_fits":len(SOURCES),"input_output_hashes":pins,"prep_audit":prep})
    print("SEALED",OUT/"prediction_seal.json")

def score_phase():
    sealp=OUT/"prediction_seal.json";seal=read(sealp);pins=seal["input_output_hashes"]
    if seal["status"]!="TVA_PREDICTION_SEALED_LABELS_UNOPENED" or sha(PROTOCOL)!=pins["protocol"] or sha(Path(__file__))!=pins["script"]:
        raise ValueError("Frozen prediction seal required")
    pred=OUT/"tva_prediction_seal.npz"
    if sha(pred)!=pins["prediction"]:raise ValueError("TVA prediction hash mismatch")
    if (OUT/"score_seal.json").exists():raise FileExistsError("Score already sealed")
    audit=read(V6_AUDIT);path=next(Path(e["path"]) for e in audit["native_sources"] if "2026_Jan_Jun" in Path(e["path"]).name)
    if sha(path)!=pins["raw_native"][str(path)]:raise ValueError("Target native hash mismatch")
    raw=pd.read_csv(path);raw=raw.loc[raw["Balancing Authority"].eq(TARGET)]
    frame=normalize_balance(raw);frame["source_snapshot"]=8
    frame=frame.loc[(frame.valid_at>=DAYS[0])&(frame.valid_at<DAYS[-1]+pd.Timedelta(days=1))]
    frame.index=pd.DatetimeIndex(frame.valid_at)
    hourly,qa=eia_proxy._proxy_rows(frame,TARGET,audit["native_schemas"])
    hourly=hourly.loc[(hourly.time>=DAYS[0])&(hourly.time<DAYS[-1]+pd.Timedelta(days=1))]
    valid=hourly.loc[hourly.valid].copy();valid["day"]=pd.to_datetime(valid.time,utc=True).dt.floor("D")
    truth=[]
    for day,part in valid.groupby("day",sort=True):
        if len(part)!=24:raise ValueError("Frozen full-day target support mismatch")
        g=part[["generation_"+f for f in helper.FUELS]].to_numpy(float)
        truth.append((day,(g/g.sum(axis=1,keepdims=True))@EF))
    if len(truth)!=181 or [x[0] for x in truth]!=list(DAYS):raise ValueError("Need exact 181 target days")
    y=np.stack([x[1] for x in truth]);metrics=[]
    with np.load(pred,allow_pickle=False) as z:
        if z["keys"].tolist()!=[f"{TARGET}/{d:%Y-%m-%d}" for d in DAYS]:raise ValueError("Scored keys mismatch")
        for name in seal["methods"]:
            lo,hi=z["lo_"+name],z["hi_"+name]
            for i,day in enumerate(DAYS):
                w=hi[i]-lo[i];sc=w+20*np.maximum(lo[i]-y[i],0)+20*np.maximum(y[i]-hi[i],0)
                hit=(y[i]>=lo[i])&(y[i]<=hi[i])
                metrics.append({"method":name,"day":f"{day:%Y-%m-%d}",
                    "coverage90":hit.mean(),"width90":w.mean(),"interval_score90":sc.mean(),
                    "lower_miss":(y[i]<lo[i]).mean(),"upper_miss":(y[i]>hi[i]).mean()})
    table=pd.DataFrame(metrics);daily=OUT/"daily_metrics.csv";table.to_csv(daily,index=False)
    table["month"]=pd.to_datetime(table.day).dt.month
    monthly=table.groupby(["method","month"],as_index=False)[["coverage90","width90","interval_score90","lower_miss","upper_miss"]].mean()
    mp=OUT/"monthly_metrics.csv";monthly.to_csv(mp,index=False)
    aggregate=table.groupby("method",sort=False)[["coverage90","width90","interval_score90","lower_miss","upper_miss"]].mean().reset_index()
    ap=OUT/"aggregate_metrics.csv";aggregate.to_csv(ap,index=False)
    write_new(OUT/"score_seal.json",{"status":"TVA_CONFIRMATION_SCORED","target":TARGET,"days":181,
        "labels_opened_after_prediction_seal":True,"target_refit_or_recalibration":False,"proxy_quality":qa,
        "target_native_sha256":sha(path),"prediction_seal_sha256":sha(sealp),"prediction_sha256":sha(pred),
        "daily_metrics_sha256":sha(daily),"monthly_metrics_sha256":sha(mp),"aggregate_metrics_sha256":sha(ap),
        "aggregate_metrics":aggregate.to_dict(orient="records"),
        "interpretation_limit":"One preselected region and one semester; not a multi-region or China transfer guarantee."})
    print(json.dumps({"status":"TVA_CONFIRMATION_SCORED","metrics":aggregate.to_dict(orient="records")},ensure_ascii=False))

if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in {"predict","score"}:raise SystemExit("Usage: confirm_v50_tva_asymmetric.py predict|score")
    (predict_phase if sys.argv[1]=="predict" else score_phase)()
