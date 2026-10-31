"""V230: LP-sharp hourly feasible CIF ranges under EF uncertainty."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import v224_eia_sparse_daily_maxrisk as dayrisk
import v225_partial_snapshot_sharp_risk_bounds as partial
import v226_frozen_four_field_confirmation as confirm
import v228_factor_aware_minimum_channels as factor_span

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"docs/V230_SHARP_FACTOR_UNCERTAINTY_LP_PROTOCOL.md"
OUT=ROOT/"results/v230-sharp-factor-uncertainty-lp-r1"
EF_TOLS=(0.,.10)
N_LOCAL_DAYS=12
REPEATS=100
BOOT=20_000
LEGACY_CHANNELS=("thermal","coal","gas")


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def lp_hour_bounds(fuel:np.ndarray,margins:np.ndarray,channels:list[str],ef_tol:float)->tuple[float,float]:
    total=float(fuel.sum())
    monthly_total=float(margins.sum())
    lower=np.maximum(0.,margins-(monthly_total-total))
    upper=np.minimum(margins,total)
    eq_rows=[np.ones(len(fuel))]
    eq_vals=[total]
    for name in channels:
        ids=factor_span.GROUPS[name]
        row=np.zeros(len(fuel)); row[ids]=1.
        eq_rows.append(row)
        eq_vals.append(float(fuel[ids].sum()))
    Aeq=np.asarray(eq_rows,float)
    beq=np.asarray(eq_vals,float)
    bounds=list(zip(lower,upper))
    ef_low=partial.EF*(1.-ef_tol)
    ef_high=partial.EF*(1.+ef_tol)
    lo=linprog(ef_low,A_eq=Aeq,b_eq=beq,bounds=bounds,method="highs")
    hi=linprog(-ef_high,A_eq=Aeq,b_eq=beq,bounds=bounds,method="highs")
    if not lo.success or not hi.success:
        raise RuntimeError(f"LP failed: low={lo.message}; high={hi.message}")
    return float(lo.fun/total),float(-hi.fun/total)


def day_score_map(d:dict,lo:np.ndarray,hi:np.ndarray)->dict[int,float]:
    dayids=dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
    center=float(d["monthly_y"])
    scores=np.maximum(np.abs(lo-center),np.abs(hi-center))
    return {int(day):float(np.max(scores[dayids==day])) for day in np.unique(dayids)}


def source_day_scores(d:dict)->np.ndarray:
    values=np.abs(np.asarray(d["cif"],float)-float(d["monthly_y"]))
    ids=dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
    return np.asarray([np.max(values[ids==day]) for day in np.unique(ids)],float)


def evaluate(y:np.ndarray,lo_bound:np.ndarray,hi_bound:np.ndarray,center:float,q:float,day_ids:np.ndarray)->tuple[float,...]:
    if not np.isfinite(q):
        return 1.,np.inf,np.inf,1.,1.
    pi_lo,pi_hi=center-q,center+q
    inside=(y>=pi_lo)&(y<=pi_hi)
    miss=np.maximum(pi_lo-y,0.)+np.maximum(y-pi_hi,0.)
    width=pi_hi-pi_lo
    winkler=width+(2/dayrisk.ALPHA)*miss
    robust_inside=(lo_bound>=pi_lo)&(hi_bound<=pi_hi)
    unique=np.unique(day_ids)
    daycov=float(np.mean([inside[day_ids==d].all() for d in unique]))
    robust_daycov=float(np.mean([robust_inside[day_ids==d].all() for d in unique]))
    return float(inside.mean()),float(width),float(winkler.mean()),daycov,robust_daycov


def main()->None:
    if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash=sha256(PROTOCOL)
    native,schemas,input_hashes,helpers=confirm.load_native()
    series=confirm.build_series(native,schemas,helpers)
    dev=set(confirm.DEV_REGIONS)
    eligible=[]; donor_map={}
    for key,d in sorted(series.items()):
        region,year,month=key
        if year!=2026 or month not in (7,8,9) or region in dev: continue
        if len(np.unique(dayrisk.day_ids(pd.DatetimeIndex(d["time"]))))<24: continue
        donors=[r for r in sorted({k[0] for k in series}) if r!=region and (r,2025,month) in series
                and len(np.unique(dayrisk.day_ids(pd.DatetimeIndex(series[(r,2025,month)]["time"]))))>=24]
        if len(donors)>=10:
            eligible.append(key); donor_map[key]=donors
    if len(eligible)<30 or len({k[0] for k in eligible})<10: raise RuntimeError("cohort gate failed")

    prepared={}; selection=[]; zero_error=0.
    for key in eligible:
        d=series[key]
        channels,_coef,_resid=factor_span.factor_span_fields(d)
        fuel=np.asarray(d["fuel_profile"],float)*np.asarray(d["fuel_totals"],float)[None,:]
        margins=np.asarray(d["fuel_totals"],float)
        lo8=[];hi8=[]; lospan=[];hispan=[];loleg=[];hileg=[]
        # Exact observed full-eight fields imply a point allocation; only EFs vary.
        for row in fuel:
            g=float(row.sum())
            lo8.append(float(row@(partial.EF*.9)/g));hi8.append(float(row@(partial.EF*1.1)/g))
        # LP factor-span and legacy group rows. Bounds are computed once per hour.
        for row in fuel:
            a,b=lp_hour_bounds(row,margins,channels,.10);lospan.append(a);hispan.append(b)
            a,b=lp_hour_bounds(row,margins,list(LEGACY_CHANNELS),.10);loleg.append(a);hileg.append(b)
        lo8=np.asarray(lo8);hi8=np.asarray(hi8);lospan=np.asarray(lospan);hispan=np.asarray(hispan)
        loleg=np.asarray(loleg);hileg=np.asarray(hileg)
        score_maps={
            ("factor_span",0.):day_score_map(d,np.asarray(d["cif"]),np.asarray(d["cif"])),
            ("factor_span",.10):day_score_map(d,lospan,hispan),
            ("legacy_four",0.):dict(zip(
                (lambda s: s.index.astype(int))(partial.build_day_scores(d,"total_thermal_coal_gas")[0]),
                partial.build_day_scores(d,"total_thermal_coal_gas")[0].to_numpy(float))),
            ("legacy_four",.10):day_score_map(d,loleg,hileg),
            ("full8",0.):day_score_map(d,np.asarray(d["cif"]),np.asarray(d["cif"])),
            ("full8",.10):day_score_map(d,lo8,hi8),
        }
        # EF=0.10 LP ranges must include the observed nominal-factor proxy.
        nominal=np.asarray(d["cif"],float)
        if np.any(lospan-1e-7>nominal) or np.any(hispan+1e-7<nominal):
            raise RuntimeError(f"EF stress envelope fails nominal witness at {key}")
        # Exact nominal identity for the factor-span method is from V228's row-span certificate.
        z=factor_span.reconstruct_cif(d,channels,_coef)
        zero_error=max(zero_error,float(np.max(np.abs(z-nominal))))
        prepared[key]={"data":d,"channels":channels,"bounds":{
            ("factor_span",0.):(nominal,nominal),("factor_span",.10):(lospan,hispan),
            ("legacy_four",0.):(nominal,nominal),("legacy_four",.10):(loleg,hileg),
            ("full8",0.):(nominal,nominal),("full8",.10):(lo8,hi8)},
            "scores":score_maps,"source":np.concatenate([
                source_day_scores(series[(r,2025,key[2])]) for r in donor_map[key]])}
        selection.append({"region":key[0],"month":key[2],"channels":channels,
                          "n_numeric_fields_including_total":len(channels)+1})
    if zero_error>1e-8: raise RuntimeError(f"zero uncertainty reconstruction failed: {zero_error}")

    rng=np.random.default_rng(230)
    records=[]; methods=("factor_span","legacy_four","full8")
    for key in eligible:
        p=prepared[key];d=p["data"]
        days=np.array(sorted(p["scores"][("factor_span",0.)]),dtype=np.int64)
        all_ids=dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
        y=np.asarray(d["cif"],float);center=float(d["monthly_y"])
        for rep in range(REPEATS):
            chosen=rng.choice(days,size=N_LOCAL_DAYS,replace=False)
            test=np.setdiff1d(days,chosen);mask=np.isin(all_ids,test)
            test_ids=all_ids[mask];yt=y[mask]
            for method in methods:
                for ef_tol in EF_TOLS:
                    q=dayrisk.radius(np.array([p["scores"][(method,ef_tol)][int(dy)] for dy in chosen]))
                    lo,hi=p["bounds"][(method,ef_tol)]
                    vals=evaluate(yt,lo[mask],hi[mask],center,q,test_ids)
                    records.append({"region":key[0],"month":key[2],"repeat":rep,"method":method,
                        "ef_tolerance":ef_tol,"hour_coverage":vals[0],"width":vals[1],
                        "winkler90":vals[2],"simultaneous_day_coverage":vals[3],
                        "robust_hour_set_containment":float(np.mean((lo[mask]>=center-q)&(hi[mask]<=center+q))) if np.isfinite(q) else 1.,
                        "robust_day_set_containment":vals[4]})
    reps=pd.DataFrame(records)
    rm=reps.groupby(["region","month","method","ef_tolerance"],as_index=False).agg(
        hour_coverage=("hour_coverage","mean"),width=("width","mean"),winkler90=("winkler90","mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage","mean"),
        robust_hour_set_containment=("robust_hour_set_containment","mean"),
        robust_day_set_containment=("robust_day_set_containment","mean"),n_repeats=("repeat","count"))
    reg=rm.groupby(["region","method","ef_tolerance"],as_index=False).agg(
        hour_coverage=("hour_coverage","mean"),width=("width","mean"),winkler90=("winkler90","mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage","mean"),
        robust_hour_set_containment=("robust_hour_set_containment","mean"),
        robust_day_set_containment=("robust_day_set_containment","mean"),n_months=("month","count"))
    summary=reg.groupby(["method","ef_tolerance"],as_index=False).agg(
        macro_hour_coverage=("hour_coverage","mean"),macro_width=("width","mean"),
        macro_winkler90=("winkler90","mean"),
        macro_simultaneous_day_coverage=("simultaneous_day_coverage","mean"),
        macro_robust_day_containment=("robust_day_set_containment","mean"),
        worst_ba_robust_day_containment=("robust_day_set_containment","min"),
        n_regions=("region","nunique"))
    rngb=np.random.default_rng(231); comps=[]
    wide=reg.pivot(index="region",columns=["method","ef_tolerance"])
    for method in methods:
        for baseline in ("full8","legacy_four"):
            for metric in ("width","winkler90","robust_day_set_containment"):
                for etol in EF_TOLS:
                    if method==baseline: continue
                    dlt=(wide[metric][(method,etol)]-wide[metric][(baseline,etol)]).to_numpy(float)
                    sim=np.mean(rngb.choice(dlt,size=(BOOT,len(dlt)),replace=True),axis=1)
                    comps.append({"method":method,"baseline":baseline,"ef_tolerance":etol,"metric":metric,
                        "delta":float(dlt.mean()),"ci95_low":float(np.quantile(sim,.025)),
                        "ci95_high":float(np.quantile(sim,.975)),"n_regions":len(dlt)})
    OUT.mkdir(parents=True)
    reps.to_csv(OUT/"replicate_metrics.csv",index=False);rm.to_csv(OUT/"region_month_metrics.csv",index=False)
    reg.to_csv(OUT/"region_metrics.csv",index=False);summary.to_csv(OUT/"summary.csv",index=False)
    pd.DataFrame(comps).to_csv(OUT/"paired_ba_bootstrap.csv",index=False)
    pd.DataFrame(selection).to_json(OUT/"channel_selection.json",orient="records",indent=2)
    metadata={"protocol_sha256":protocol_hash,"input_hashes":input_hashes,
        "target_ba_months":int(rm[["region","month"]].drop_duplicates().shape[0]),
        "target_region_count":int(reg.region.nunique()),"ef_tolerances":EF_TOLS,
        "local_days":N_LOCAL_DAYS,"repeats":REPEATS,"zero_uncertainty_reconstruction_error":zero_error,
        "lp":"scipy HiGHS, per-hour LP with monthly-margin and exact group equations",
        "label_caveat":"nominal EIA generation-derived fixed-factor proxy; not measured emissions",
        "status":"sharp per-hour stress reanalysis on already opened V226 cohort; not blind"}
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    if sha256(PROTOCOL)!=protocol_hash: raise RuntimeError("protocol changed during run")
    print(summary.to_string(index=False))
    print(f"BA_months={metadata['target_ba_months']} zero_error={zero_error:.3g} output={OUT}")


if __name__=="__main__": main()
