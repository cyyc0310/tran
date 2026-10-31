"""Frozen EPE/TIDC out-of-region check of V74 order-statistic certificates."""
from __future__ import annotations
import hashlib,json,sys
from fractions import Fraction as F
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import v54_unseen_local_ef_confirmation as source_data  # noqa:E402
sys.path.insert(0,str(ROOT))
from scripts import tail_count_v20 as model  # noqa:E402
PROTOCOL=ROOT/'docs/V77_UNSEEN_EPE_TIDC_RISK_SKETCH_CONFIRMATION_PROTOCOL.md'
OUT=ROOT/'results/v77-unseen-epe-tidc-risk-sketch-r1'
TARGETS=('EPE','TIDC');PERIODS=(('2024H2','2024-07-01','2025-01-01'),('2025H1','2025-01-01','2025-07-01'),('2025H2','2025-07-01','2026-01-01'))
TAUS=(100,300);B=F(1000);EF=(1000,400,800,0,0,0,0,400)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def make_counts(hours,tau):
 gs=[sum(x,F(0)) for x in hours];hs=[sum((v*e for v,e in zip(x,EF)),F(0)) for x in hours]
 stats={'minG_K0':True,'S2_Kle4':True,'L1_Kge8':True};sets={'T8':[],'T8_G2':[],'T8_G24':[]}
 for start in range(0,24,6):
  g=gs[start:start+6];hh=sum(hs[start:start+6],F(0));total=sum(g,F(0));o=sorted(g)
  stats['minG_K0'] &= hh<=F(tau)*o[0]
  stats['S2_Kle4'] &= hh<=F(tau)*sum(o[:2],F(0))
  stats['L1_Kge8'] &= hh>F(tau)*total+(B-F(tau))*o[-1]
  q=sum((x*x for x in g),F(0));actual=sum(v>F(tau)*z for v,z in zip(hs[start:start+6],gs[start:start+6]))
  c=(model.coarse_counts(6,total,hh,F(tau)),model.moment_counts(6,total,q,hh,F(tau)),model.full_g_counts(g,hh,F(tau)))
  if not set(c[2]).issubset(c[1]) or not set(c[1]).issubset(c[0]):raise ValueError('block set nesting failed')
  for key,poss in zip(('T8','T8_G2','T8_G24'),c):
   if actual not in poss:raise ValueError('observed block K outside compatible set')
   sets[key].append(poss)
 daily={k:model.combine_blocks(v) for k,v in sets.items()}
 if not set(daily['T8_G24']).issubset(daily['T8_G2']) or not set(daily['T8_G2']).issubset(daily['T8']):raise ValueError('daily set nesting failed')
 actual=sum(v>F(tau)*g for v,g in zip(hs,gs))
 if any(actual not in v for v in daily.values()):raise ValueError('observed daily K outside compatible set')
 return {'actual_K':actual,**stats,
  **{f'{x}_lower':min(v) for x,v in daily.items()},**{f'{x}_upper':max(v) for x,v in daily.items()},
  **{f'{x}_span':max(v)-min(v) for x,v in daily.items()},
  **{f'{x}_K0':max(v)==0 for x,v in daily.items()},
  **{f'{x}_Kle4':max(v)<=4 for x,v in daily.items()},
  **{f'{x}_Kge8':min(v)>=8 for x,v in daily.items()}}
def main():
 ph=sha(PROTOCOL);source_data.ALL_REGIONS=tuple(source_data.ALL_REGIONS)+TARGETS
 hourly,monthly,audits,input_hashes=source_data.prepare_regions();rows=[];valid_days={}
 for label,start,stop in PERIODS:
  s,e=pd.Timestamp(start,tz='UTC'),pd.Timestamp(stop,tz='UTC')
  expected=len(pd.date_range(s,e-pd.Timedelta(days=1),freq='D'))
  for region in TARGETS:
   frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
   keep=times.ge(s)&times.lt(e)&frame.valid.astype(bool)
   x=frame.loc[keep].copy();x['utc_day']=pd.to_datetime(x.time,utc=True).dt.floor('D')
   valid_days[(label,region)]=0
   cols=[f'generation_{f}' for f in source_data.FUELS]
   for day,g in x.groupby('utc_day',sort=True):
    g=g.assign(_time=pd.to_datetime(g.time,utc=True)).sort_values('_time')
    if len(g)!=24 or list(g._time)!=list(pd.date_range(day,periods=24,freq='h')):continue
    vals=[]
    for _,r in g.iterrows():
     hour=[F(str(float(r[c]))) for c in cols]
     if any(v<0 for v in hour) or sum(hour)<=0:raise ValueError('invalid target generation')
     vals.append(hour)
    valid_days[(label,region)]+=1
    for tau in TAUS:rows.append({'region':region,'period':label,'day':day.isoformat(),'tau':tau,**make_counts(vals,tau)})
   if valid_days[(label,region)]<.8*expected:raise ValueError(f'quality floor failed {region}/{label}: {valid_days[(label,region)]}/{expected}')
 d=pd.DataFrame(rows)
 if d.empty:raise ValueError('no target certificates')
 # No targeted certificate may disagree with the complete-G decision.
 pairs=[('minG_K0','T8_G24_K0'),('S2_Kle4','T8_G24_Kle4'),('L1_Kge8','T8_G24_Kge8')]
 for a,b in pairs:
  if (d[a]&~d[b]).any():raise ValueError(f'{a} issued a certificate excluded by complete G')
  if ((d[a])&(d.actual_K!=0)).any() and a=='minG_K0':raise ValueError('false K=0 certificate')
  if ((d[a])&(d.actual_K>4)).any() and a=='S2_Kle4':raise ValueError('false K<=4 certificate')
  if ((d[a])&(d.actual_K<8)).any() and a=='L1_Kge8':raise ValueError('false K>=8 certificate')
 OUT.mkdir(parents=True,exist_ok=False);d.to_csv(OUT/'day_certificates.csv',index=False)
 measures=['minG_K0','T8_G2_K0','T8_G24_K0','S2_Kle4','T8_G2_Kle4','T8_G24_Kle4','L1_Kge8','T8_G2_Kge8','T8_G24_Kge8']
 reg=d.groupby(['region','period','tau'])[measures].mean().reset_index();reg.to_csv(OUT/'region_summary.csv',index=False)
 summary=d.groupby(['period','tau'])[measures].mean().reset_index().merge(d.groupby(['period','tau']).size().rename('day_threshold_rows').reset_index(),on=['period','tau'])
 summary.to_csv(OUT/'summary.csv',index=False)
 payload={'status':'UNSEEN_REGION_RETROSPECTIVE_IDENTIFICATION_CONFIRMATION_NOT_FORECAST',
  'targets':list(TARGETS),'periods':[x[0] for x in PERIODS],'valid_days':{f'{k[0]}:{k[1]}':v for k,v in valid_days.items()},
  'thresholds':list(TAUS),'rows':len(d),'summary':summary.to_dict(orient='records'),
  'region_summary':reg.to_dict(orient='records'),'input_sha256':input_hashes,'release_audits':audits,
  'protocol_sha256':ph,'script_sha256':sha(Path(__file__)),'limitations':[
   'Two target regions and three periods; descriptive confirmation only, not population coverage.',
   'Same EIA fixed-EF generation proxy and contemporaneous hourly H/G base as V74.',
   'Not a forecast and not available under the China public monthly/annual-only input budget.']}
 if sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False));print(summary.to_string(index=False));print('days',valid_days)
if __name__=='__main__':main()
