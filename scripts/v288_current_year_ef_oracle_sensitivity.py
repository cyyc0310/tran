"""Post-hoc V288: oracle target-year EF support sensitivity for V286 failures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v224_eia_sparse_daily_maxrisk as dayrisk
import v279_asymmetric_ef_sensor_frontier as sensor
import v280_eia_official_label_pace_lgee as v280
import v286_prior_year_ef_temporal_holdout as v286

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V288_CURRENT_YEAR_EF_ORACLE_SENSITIVITY.md"
DRIFT = ROOT / "results/v287-temporal-ef-support-drift-audit/target_vs_prior_factor_support.csv"
OUT = ROOT / "results/v288-current-year-ef-oracle-sensitivity-r1"
TARGETS = v286.TARGETS
DONORS = v286.DONORS
METHODS = ("cems_support", "prior_year_local_ef", "target_year_oracle_ef", "source_loo")
REPEATS = 100
CAL_DAYS = 9


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def annual_bounds(table: pd.DataFrame, ba: str, year: int, base_low: np.ndarray,
                  base_high: np.ndarray, prefix: str) -> tuple[np.ndarray,np.ndarray]:
    low,high=base_low.copy(),base_high.copy()
    fuelcols={"coal":("target_min_g_kwh","target_max_g_kwh"),
              "gas":("target_min_g_kwh","target_max_g_kwh"),
              "oil":("target_min_g_kwh","target_max_g_kwh")}
    idxmap={"coal":0,"gas":1,"oil":2}
    for fuel,idx in idxmap.items():
        row=table[(table.ba==ba)&(table.target_year==year)&(table.fuel==fuel)]
        if row.empty or int(row.iloc[0].target_active_hours)==0:
            continue
        low[idx]=float(row.iloc[0][prefix[0]])
        high[idx]=float(row.iloc[0][prefix[1]])
    return low,high


def main() -> None:
    if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash=sha(PROTOCOL)
    support=pd.read_csv(v286.SUPPORT)
    drift=pd.read_csv(DRIFT)
    targets={}
    factors={}
    target_meta={}
    for ba in TARGETS:
        path=v286.DATA/f"{ba}.xlsx"
        targets[ba],factors[ba],target_meta[ba]=v286.read_ba(ba,path,target=True)
    donors={}; donor_meta={}
    for ba in DONORS:
        path=v286.DONOR_DIR/f"{ba}.xlsx"
        donors[ba],_,donor_meta[ba]=v286.read_ba(ba,path,target=False)
    prepared={}
    for ba in TARGETS:
        for (year,month),d in targets[ba].items():
            quarter=(month-1)//3+1
            c_low,c_high=sensor.factor_support(quarter,support)
            p_low,p_high,_=v286.local_prior_bounds(ba,year,month,factors[ba],support)
            o_low,o_high=annual_bounds(drift,ba,year,c_low,c_high,
                ("target_min_g_kwh","target_max_g_kwh"))
            ranges={}
            for method,loef,hief in (("cems_support",c_low,c_high),
                ("prior_year_local_ef",p_low,p_high),
                ("target_year_oracle_ef",o_low,o_high)):
                lo,hi=sensor.robust_hour_ranges(d,"support_minimum",loef,hief)
                ranges[method]=(lo,hi,sensor.daily_scores(d,lo,hi))
            source=[]
            key=(year,month)
            for donor in DONORS:
                sd=donors[donor].get(key)
                if sd is not None: source.extend(sensor.nominal_source_scores(sd).tolist())
            if len(source)<100: raise RuntimeError(f"insufficient source days {ba}/{key}")
            prepared[(ba,year,month)]={"d":d,"ranges":ranges,"source":np.asarray(source,float)}
    rng=np.random.default_rng(286); records=[]
    for key,item in sorted(prepared.items()):
        ba,year,month=key; d=item['d']
        did=dayrisk.day_ids(pd.DatetimeIndex(d['time']))
        days=np.asarray(sorted(item['ranges']['cems_support'][2]),dtype=np.int64)
        for rep in range(REPEATS):
            sampled=rng.choice(days,size=CAL_DAYS,replace=False)
            test_days=np.setdiff1d(days,sampled); mask=np.isin(did,test_days)
            for method in METHODS:
                if method=='source_loo':
                    q=dayrisk.radius(item['source']); vals=sensor.score_source_baseline(d,q,mask); setcov=np.nan
                else:
                    lo,hi,scores=item['ranges'][method]
                    q=dayrisk.radius(np.asarray([scores[int(day)] for day in sampled]))
                    vals=sensor.score_test(d,lo,hi,q,mask); setcov=vals[4]
                records.append({'ba':ba,'year':year,'month':month,'repeat':rep,'method':method,
                    'hour_coverage':vals[0],'width':vals[1],'winkler90':vals[2],
                    'simultaneous_day_coverage':vals[3],
                    'ex_post_robust_day_set_containment':setcov})
    reps=pd.DataFrame(records)
    month=reps.groupby(['ba','year','month','method'],as_index=False).agg(
        hour_coverage=('hour_coverage','mean'),width=('width','mean'),winkler90=('winkler90','mean'),
        simultaneous_day_coverage=('simultaneous_day_coverage','mean'),
        ex_post_robust_day_set_containment=('ex_post_robust_day_set_containment','mean'))
    by_year=month.groupby(['ba','year','method'],as_index=False).agg(
        hour_coverage=('hour_coverage','mean'),width=('width','mean'),winkler90=('winkler90','mean'),
        simultaneous_day_coverage=('simultaneous_day_coverage','mean'),
        ex_post_robust_day_set_containment=('ex_post_robust_day_set_containment','mean'),n_months=('month','count'))
    macro=by_year.groupby('method',as_index=False).agg(
        macro_hour_coverage=('hour_coverage','mean'),macro_width=('width','mean'),
        macro_winkler90=('winkler90','mean'),
        macro_simultaneous_day_coverage=('simultaneous_day_coverage','mean'),
        macro_ex_post_robust_day_set_containment=('ex_post_robust_day_set_containment','mean'),
        n_ba_years=('year','count'))
    OUT.mkdir(parents=True)
    reps.to_csv(OUT/'replicate_metrics.csv',index=False)
    month.to_csv(OUT/'ba_month_metrics.csv',index=False)
    by_year.to_csv(OUT/'ba_year_metrics.csv',index=False)
    macro.to_csv(OUT/'macro_metrics.csv',index=False)
    metadata={'protocol_sha256':protocol_hash,'script_sha256':sha(Path(__file__)),
        'drift_audit_sha256':sha(DRIFT),'target_files':{b:v280.sha256(v286.DATA/f"{b}.xlsx") for b in TARGETS},
        'donor_files':{b:v280.sha256(v286.DONOR_DIR/f"{b}.xlsx") for b in DONORS},
        'support_sha256':sha(v286.SUPPORT),'seed':286,'calibration_days':CAL_DAYS,
        'repeats':REPEATS,'methods':list(METHODS),
        'caveat':'same-year EF support uses target-year factor columns; oracle diagnostic only, not deployable'}
    (OUT/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    if sha(PROTOCOL)!=protocol_hash: raise RuntimeError('protocol changed during run')
    print(macro.to_string(index=False)); print(by_year.to_string(index=False))

if __name__=='__main__': main()
