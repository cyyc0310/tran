"""V289 exploratory rolling EF-support update using only previous months."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import v224_eia_sparse_daily_maxrisk as dayrisk
import v279_asymmetric_ef_sensor_frontier as sensor
import v280_eia_official_label_pace_lgee as v280
import v286_prior_year_ef_temporal_holdout as v286

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"docs/V289_LAGGED_CURRENT_EF_UPDATE_PROTOCOL.md"
OUT=ROOT/"results/v289-lagged-current-ef-update-r1"
TARGETS=v286.TARGETS; DONORS=v286.DONORS; YEARS=v286.TARGET_YEARS
FACTORS=v286.FACTOR_COLUMNS; CONVERSION=v286.CONVERSION
METHODS=("cems_support","prior_year_local_ef","lagged_current_year_ef","source_loo")
REPEATS=100; CAL_DAYS=9


def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def date_value(v):
 if isinstance(v,datetime):return v.date()
 if isinstance(v,date):return v
 return None

def read_monthly_factor_history(ba:str)->dict[tuple[int,int],dict[str,np.ndarray]]:
 path=v286.DATA/f"{ba}.xlsx"; wb=load_workbook(path,read_only=True,data_only=True)
 ws=wb['Published Hourly Data']; it=ws.iter_rows(values_only=True); heads=next(it)
 ix={str(v):i for i,v in enumerate(heads) if v is not None}
 result=defaultdict(lambda:{fuel:[] for fuel in FACTORS})
 for row in it:
  day=date_value(row[ix['Local date']])
  if day is None or day.year not in YEARS:continue
  for fuel,(fc,gc) in FACTORS.items():
   try: gen=float(row[ix[gc]] or 0.); ef=float(row[ix[fc]])*CONVERSION
   except (TypeError,ValueError):continue
   if gen>0 and np.isfinite(ef):result[(day.year,day.month)][fuel].append(ef)
 wb.close()
 return {k:{f:np.asarray(v,float) for f,v in vals.items()} for k,vals in result.items()}

def update_bounds(ba,year,month,cems_low,cems_high,prior_factors,month_factors):
 p_low,p_high,_=v286.local_prior_bounds(ba,year,month,prior_factors,v286.pd.read_csv(v286.SUPPORT))
 low,high=p_low.copy(),p_high.copy()
 idx={"coal":0,"gas":1,"oil":2}
 history={fuel:[] for fuel in FACTORS}
 for m in range(1,month):
  for fuel,arr in month_factors.get((year,m),{}).items():
   if len(arr):history[fuel].extend(arr.tolist())
 for fuel,j in idx.items():
  if history[fuel]:
   x=np.asarray(history[fuel],float)
   low[j]=min(low[j],float(x.min()));high[j]=max(high[j],float(x.max()))
  else:
   low[j]=min(low[j],cems_low[j]);high[j]=max(high[j],cems_high[j])
 return low,high

def main():
 if OUT.exists():raise SystemExit(f"refusing to overwrite {OUT}")
 protocol_hash=sha(PROTOCOL); support=pd.read_csv(v286.SUPPORT)
 targets={}; prior={}; target_meta={}; month_factors={}
 for ba in TARGETS:
  path=v286.DATA/f"{ba}.xlsx"
  targets[ba],prior[ba],target_meta[ba]=v286.read_ba(ba,path,target=True)
  month_factors[ba]=read_monthly_factor_history(ba)
 donors={}; donor_meta={}
 for ba in DONORS:
  path=v286.DONOR_DIR/f"{ba}.xlsx"
  donors[ba],_,donor_meta[ba]=v286.read_ba(ba,path,target=False)
 prepared={}
 for ba in TARGETS:
  for (year,month),d in targets[ba].items():
   quarter=(month-1)//3+1;c_low,c_high=sensor.factor_support(quarter,support)
   p_low,p_high,_=v286.local_prior_bounds(ba,year,month,prior[ba],support)
   l_low,l_high=update_bounds(ba,year,month,c_low,c_high,prior[ba],month_factors[ba])
   ranges={}
   for method,loef,hief in (("cems_support",c_low,c_high),("prior_year_local_ef",p_low,p_high),
                            ("lagged_current_year_ef",l_low,l_high)):
    lo,hi=sensor.robust_hour_ranges(d,"support_minimum",loef,hief)
    ranges[method]=(lo,hi,sensor.daily_scores(d,lo,hi))
   source=[]
   for donor in DONORS:
    sd=donors[donor].get((year,month))
    if sd is not None:source.extend(sensor.nominal_source_scores(sd).tolist())
   if len(source)<100:raise RuntimeError(f"insufficient donor scores: {ba}/{year}-{month}")
   prepared[(ba,year,month)]={"d":d,"ranges":ranges,"source":np.asarray(source,float)}
 expected={(ba,y,m) for ba in TARGETS for y in YEARS for m in range(1,13)}
 if set(prepared)!=expected:raise RuntimeError(f"target roster incomplete: {sorted(expected-set(prepared))}")
 rng=np.random.default_rng(286);records=[]
 for key,item in sorted(prepared.items()):
  ba,year,month=key;d=item['d'];did=dayrisk.day_ids(pd.DatetimeIndex(d['time']))
  days=np.asarray(sorted(item['ranges']['cems_support'][2]),dtype=np.int64)
  for rep in range(REPEATS):
   sampled=rng.choice(days,size=CAL_DAYS,replace=False);test=np.setdiff1d(days,sampled);mask=np.isin(did,test)
   for method in METHODS:
    if method=='source_loo':
     q=dayrisk.radius(item['source']);v=sensor.score_source_baseline(d,q,mask);setcov=np.nan
    else:
     lo,hi,scores=item['ranges'][method];q=dayrisk.radius(np.asarray([scores[int(x)] for x in sampled]))
     v=sensor.score_test(d,lo,hi,q,mask);setcov=v[4]
    records.append({'ba':ba,'year':year,'month':month,'repeat':rep,'method':method,
     'hour_coverage':v[0],'width':v[1],'winkler90':v[2],
     'simultaneous_day_coverage':v[3],'ex_post_robust_day_set_containment':setcov})
 reps=pd.DataFrame(records)
 month=reps.groupby(['ba','year','month','method'],as_index=False).agg(
  hour_coverage=('hour_coverage','mean'),width=('width','mean'),winkler90=('winkler90','mean'),
  simultaneous_day_coverage=('simultaneous_day_coverage','mean'),
  ex_post_robust_day_set_containment=('ex_post_robust_day_set_containment','mean'))
 yr=month.groupby(['ba','year','method'],as_index=False).agg(
  hour_coverage=('hour_coverage','mean'),width=('width','mean'),winkler90=('winkler90','mean'),
  simultaneous_day_coverage=('simultaneous_day_coverage','mean'),
  ex_post_robust_day_set_containment=('ex_post_robust_day_set_containment','mean'),n_months=('month','count'))
 macro=yr.groupby('method',as_index=False).agg(
  macro_hour_coverage=('hour_coverage','mean'),macro_width=('width','mean'),macro_winkler90=('winkler90','mean'),
  macro_simultaneous_day_coverage=('simultaneous_day_coverage','mean'),
  macro_ex_post_robust_day_set_containment=('ex_post_robust_day_set_containment','mean'),n_ba_years=('year','count'))
 OUT.mkdir(parents=True)
 reps.to_csv(OUT/'replicate_metrics.csv',index=False);month.to_csv(OUT/'ba_month_metrics.csv',index=False)
 yr.to_csv(OUT/'ba_year_metrics.csv',index=False);macro.to_csv(OUT/'macro_metrics.csv',index=False)
 metadata={'protocol_sha256':protocol_hash,'script_sha256':sha(Path(__file__)),
  'target_files':target_meta,'donor_files':donor_meta,'support_sha256':sha(v286.SUPPORT),
  'seed':286,'repeats':REPEATS,'calibration_days':CAL_DAYS,'methods':list(METHODS),
  'status':'post-hoc lagged EF support development on previously opened target years'}
 (OUT/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
 if sha(PROTOCOL)!=protocol_hash:raise RuntimeError('protocol changed')
 print(macro.to_string(index=False));print(yr.to_string(index=False))

if __name__=='__main__':main()
