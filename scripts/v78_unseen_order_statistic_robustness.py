"""V78: robustness of frozen L1 tail certificates in the V77 regions."""
from __future__ import annotations
import hashlib,json,sys
from fractions import Fraction as F
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import v54_unseen_local_ef_confirmation as source_data  # noqa:E402
V77=ROOT/'results/v77-unseen-epe-tidc-risk-sketch-r1/day_certificates.csv'
PROTOCOL=ROOT/'docs/V78_UNSEEN_ORDER_STATISTIC_ROBUSTNESS_PROTOCOL.md'
OUT=ROOT/'results/v78-unseen-order-statistic-robustness-r1'
TARGETS=('EPE','TIDC')
PERIODS=(('2024H2','2024-07-01','2025-01-01'),('2025H1','2025-01-01','2025-07-01'),('2025H2','2025-07-01','2026-01-01'))
TAUS=(100,300);EPS=(F(0),F(1,1000),F(1,100),F(1,50),F(1,20),F(1,10));B=F(1000)

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def robust_ge8(hours,tau,eps):
 for start in range(0,24,6):
  block=hours[start:start+6]
  gs=[sum(x,F(0)) for x in block]
  hs=[sum((v*e for v,e in zip(x,(1000,400,800,0,0,0,0,400))),F(0)) for x in block]
  total=sum(gs,F(0));H=sum(hs,F(0));l1=max(gs)
  if H*(1-eps)<=F(tau)*total*(1+eps)+(B-F(tau))*l1*(1+eps):return False
 return True

def main():
 ph=sha(PROTOCOL)
 v77=pd.read_csv(V77)
 if set(v77.region.unique())!=set(TARGETS):raise ValueError('V77 target roster mismatch')
 source_data.ALL_REGIONS=tuple(source_data.ALL_REGIONS)+TARGETS
 hourly,_,audits,input_hashes=source_data.prepare_regions()
 rows=[]
 for period,start,stop in PERIODS:
  s,e=pd.Timestamp(start,tz='UTC'),pd.Timestamp(stop,tz='UTC')
  for region in TARGETS:
   frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
   x=frame.loc[times.ge(s)&times.lt(e)&frame.valid.astype(bool)].copy()
   x['utc_day']=pd.to_datetime(x.time,utc=True).dt.floor('D')
   cols=[f'generation_{fuel}' for fuel in source_data.FUELS]
   for day,g in x.groupby('utc_day',sort=True):
    g=g.assign(_time=pd.to_datetime(g.time,utc=True)).sort_values('_time')
    if len(g)!=24 or list(g._time)!=list(pd.date_range(day,periods=24,freq='h')):continue
    hours=[]
    for _,r in g.iterrows():
     z=[F(str(float(r[c]))) for c in cols]
     if any(a<0 for a in z) or sum(z)<=0:raise ValueError('bad hourly generation')
     hours.append(z)
    dkey=day.isoformat()
    for tau in TAUS:
     ref=v77[(v77.region==region)&(v77.period==period)&(v77.day==dkey)&(v77.tau==tau)]
     if len(ref)!=1:raise ValueError(f'missing V77 row {region}/{period}/{dkey}/{tau}')
     exact_full=bool(ref.iloc[0].T8_G24_Kge8);g2=bool(ref.iloc[0].T8_G2_Kge8)
     for eps in EPS:
      cert=robust_ge8(hours,tau,eps)
      if cert and not exact_full:raise ValueError('robust L1 certificate contradicts full-G set')
      rows.append({'region':region,'period':period,'day':dkey,'tau':tau,'epsilon':float(eps),
       'robust_L1_Kge8':cert,'exact_fullG_Kge8':exact_full,'exact_G2_Kge8':g2})
 d=pd.DataFrame(rows)
 if d.empty:raise ValueError('empty robustness panel')
 OUT.mkdir(parents=True,exist_ok=False)
 d.to_csv(OUT/'day_certificates.csv',index=False)
 group=d.groupby(['region','period','tau','epsilon'])
 summary=group.agg(days=('day','size'),robust_L1_days=('robust_L1_Kge8','sum'),fullG_days=('exact_fullG_Kge8','sum'),G2_days=('exact_G2_Kge8','sum')).reset_index()
 summary['L1_retention_of_fullG_pct']=summary.robust_L1_days/summary.fullG_days.where(summary.fullG_days>0)
 summary['L1_gain_vs_G2_days']=summary.robust_L1_days-summary.G2_days
 summary.to_csv(OUT/'summary.csv',index=False)
 # Exact-zero-error must match V77 row for row; each larger error box is nested.
 for keys,g in d.groupby(['region','period','day','tau']):
  g=g.sort_values('epsilon')
  if bool(g.iloc[0].robust_L1_Kge8)!=bool(g.iloc[0].exact_fullG_Kge8):raise ValueError('epsilon=0 mismatch with V77 full-G decision')
  if g.robust_L1_Kge8.astype(int).diff().fillna(0).gt(0).any():raise ValueError('robust certificates grew with epsilon')
 payload={'status':'UNSEEN_REGION_MEASUREMENT_SENSITIVITY_NOT_FORECAST','targets':list(TARGETS),
  'periods':[p[0] for p in PERIODS],'taus':list(TAUS),'epsilon':[float(x) for x in EPS],
  'rows':len(d),'summary':summary.to_dict(orient='records'),'source_audits':audits,
  'input_sha256':input_hashes,'v77_day_table_sha256':sha(V77),'protocol_sha256':ph,
  'script_sha256':sha(Path(__file__)),'limitations':['Error levels are hypothetical and not measured sensor precision.',
  'Same contemporaneous EIA proxy and fixed EF as V77; not forecast coverage.',
  'Requires 6-hour H/G and L1 measurements, absent from China public monthly/yearly tables.']}
 if sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed during run')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
 print(summary.to_string(index=False))

if __name__=='__main__':main()
