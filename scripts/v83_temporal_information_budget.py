"""V83: temporal aggregation budget for finite-block tail identification."""
from __future__ import annotations
import hashlib,json,sys
from fractions import Fraction as F
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import v54_unseen_local_ef_confirmation as source_data  # noqa:E402
sys.path.insert(0,str(ROOT))
from scripts import tail_count_v20 as model  # noqa:E402
PROTOCOL=ROOT/'docs/V83_TEMPORAL_INFORMATION_BUDGET_PROTOCOL.md'
OUT=ROOT/'results/v83-temporal-information-budget-r1'
TARGETS=('EPE','TIDC','AVRN','DEAA');TAUS=(100,300);BLOCKS=(24,12,6,3,1);GLOBAL_M=F(1000)
PERIODS=(('2024H2','2024-07-01','2025-01-01'),('2025H1','2025-01-01','2025-07-01'),('2025H2','2025-07-01','2026-01-01'))
EF=(1000,400,800,0,0,0,0,400)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def counts(hours,tau,M):
 n=len(hours);gs=[sum(x,F(0)) for x in hours]
 hs=[sum((v*ef for v,ef in zip(x,EF)),F(0)) for x in hours]
 H=sum(hs,F(0));G=sum(gs,F(0));Q=sum((g*g for g in gs),F(0))
 actual=sum(h>F(tau)*g for h,g in zip(hs,gs))
 if M<=tau:
  if actual:raise ValueError('event violates support upper bound')
  return {'actual':0,'coarse':(0,),'moment':(0,),'full':(0,) if n<=12 else None}
 coarse=model.coarse_counts(n,G,H,F(tau),M)
 moment=model.moment_counts(n,G,Q,H,F(tau),M)
 full=model.full_g_counts(gs,H,F(tau),M) if n<=6 else None
 return {'actual':actual,'coarse':coarse,'moment':moment,'full':full}

def combine(block_sets,key):
 return model.combine_blocks([b[key] for b in block_sets])

def main():
 protocol_hash=sha(PROTOCOL);source_data.ALL_REGIONS=tuple(source_data.ALL_REGIONS)+TARGETS
 hourly,monthly,audits,input_hashes=source_data.prepare_regions();monthly=monthly[monthly.region.isin(TARGETS)].copy()
 rows=[];quality=[]
 for label,start,stop in PERIODS:
  s,e=pd.Timestamp(start,tz='UTC'),pd.Timestamp(stop,tz='UTC');expected=len(pd.date_range(s,e-pd.Timedelta(days=1),freq='D'))
  for region in TARGETS:
   frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
   x=frame.loc[times.ge(s)&times.lt(e)&frame.valid.astype(bool)].copy();x['day']=pd.to_datetime(x.time,utc=True).dt.floor('D')
   cols=[f'generation_{f}' for f in source_data.FUELS];valid=[]
   for day,g in x.groupby('day',sort=True):
    g=g.assign(_time=pd.to_datetime(g.time,utc=True)).sort_values('_time')
    if len(g)==24 and list(g._time)==list(pd.date_range(day,periods=24,freq='h')):valid.append((day,g))
   quality.append({'region':region,'period':label,'complete_days':len(valid),'expected_days':expected,'coverage':len(valid)/expected})
   for day,g in valid:
    month_start=pd.Timestamp(day.strftime('%Y-%m-01'),tz='UTC');mr=monthly[(monthly.region==region)&(monthly.period_start==month_start)]
    if len(mr)>1:raise ValueError('duplicate month row')
    if len(mr)==0:M=GLOBAL_M
    else:
     active=[j for j,f in enumerate(source_data.FUELS) if float(mr.iloc[0][f'fuelshare_{f}'])>0]
     M=F(max(EF[j] for j in active)) if active else GLOBAL_M
    hours=[]
    for _,r in g.iterrows():
     z=[F(str(float(r[c]))) for c in cols]
     if any(v<0 for v in z) or sum(z)<=0:raise ValueError('invalid generation')
     hours.append(z)
    for tau in TAUS:
     for block_len in BLOCKS:
      bs=[counts(hours[i:i+block_len],tau,M) for i in range(0,24,block_len)]
      coarse=combine(bs,'coarse');moment=combine(bs,'moment')
      if not set(moment).issubset(coarse):raise ValueError('moment/coarse nesting failed')
      actual=sum(b['actual'] for b in bs)
      if actual not in coarse or actual not in moment:raise ValueError('observed daily K outside outer set')
      full=None
      if block_len<=6:
       full=combine(bs,'full')
       if actual not in full or not set(full).issubset(moment):raise ValueError('full-G/moment nesting failed')
      rows.append({'region':region,'period':label,'day':day.isoformat(),'tau':tau,'M':int(M),
       'block_hours':block_len,'blocks_per_day':24//block_len,'H_G_fields_per_day':2*(24//block_len),
       'Q_fields_per_day':24//block_len,'actual_K':actual,'coarse_Kge8':min(coarse)>=8,
       'moment_Kge8':min(moment)>=8,'fullG_Kge8':(min(full)>=8 if full is not None else None),
       'coarse_lower':min(coarse),'coarse_upper':max(coarse),'moment_lower':min(moment),'moment_upper':max(moment),
       'fullG_lower':(min(full) if full is not None else None),'fullG_upper':(max(full) if full is not None else None)})
 d=pd.DataFrame(rows);q=pd.DataFrame(quality)
 if d.empty:raise ValueError('no rows')
 OUT.mkdir(parents=True,exist_ok=False);d.to_csv(OUT/'day_identification.csv',index=False);q.to_csv(OUT/'quality_audit.csv',index=False)
 keys=['region','period','tau','M','block_hours','blocks_per_day','H_G_fields_per_day','Q_fields_per_day']
 summary=d.groupby(keys,dropna=False).agg(days=('day','size'),coarse_rate=('coarse_Kge8','mean'),moment_rate=('moment_Kge8','mean'),fullG_rate=('fullG_Kge8','mean')).reset_index()
 spans=d.assign(coarse_span=d.coarse_upper-d.coarse_lower,moment_span=d.moment_upper-d.moment_lower,
  fullG_span=d.fullG_upper-d.fullG_lower).groupby(keys,dropna=False)[['coarse_span','moment_span','fullG_span']].mean().reset_index()
 summary=summary.merge(spans,on=keys,validate='one_to_one');summary.to_csv(OUT/'summary.csv',index=False)
 payload={'status':'DEVELOPMENT_INFORMATION_BUDGET_IDENTIFICATION_NOT_FORECAST','regions':list(TARGETS),
  'periods':[x[0] for x in PERIODS],'thresholds':list(TAUS),'block_hours':list(BLOCKS),
  'rows':len(d),'quality':quality,'summary':summary.astype(object).where(pd.notna(summary),None).to_dict(orient='records'),
  'source_audits':audits,'input_sha256':input_hashes,'protocol_sha256':protocol_hash,'script_sha256':sha(Path(__file__)),
  'limitations':['All four target regions were opened in prior experiments; this is development, not held-out validation.',
   'Fixed-EF EIA generation proxy, contemporaneous set identification, no probability forecast or empirical coverage.',
   'G2 is a necessary outer set, not generally sharp; Q acquisition cost is not represented by scalar count alone.',
   'No additional H/G beyond the stated per-block aggregates; China public monthly/yearly tables do not provide these.']}
 if sha(PROTOCOL)!=protocol_hash:raise ValueError('protocol changed')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False))
 print(q.to_string(index=False));print(summary.to_string(index=False))

if __name__=='__main__':main()
