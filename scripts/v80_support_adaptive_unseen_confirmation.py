"""V80: frozen support-adaptive tail identification on four unseen EIA areas."""
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
PROTOCOL=ROOT/'docs/V80_SUPPORT_ADAPTIVE_UNSEEN_CONFIRMATION_PROTOCOL.md'
OUT=ROOT/'results/v80-support-adaptive-unseen-confirmation-r1'
TARGETS=('CHPD','DOPD','GCPD','TPWR');TAUS=(100,300);GLOBAL_M=F(1000)
PERIODS=(('2024H2','2024-07-01','2025-01-01'),('2025H1','2025-01-01','2025-07-01'),('2025H2','2025-07-01','2026-01-01'))
EF=(1000,400,800,0,0,0,0,400)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def block(hours,tau,M):
 g=[sum(x,F(0)) for x in hours]
 h=[sum((v*e for v,e in zip(x,EF)),F(0)) for x in hours]
 H=sum(h,F(0));G=sum(g,F(0));Q=sum((x*x for x in g),F(0));actual=sum(a>F(tau)*b for a,b in zip(h,g))
 if M<=tau:
  if actual:raise ValueError('observed event above threshold despite support <= threshold')
  return {'actual':0,'g2':(0,),'full':(0,),'l1':False,'coarse':(0,)}
 return {'actual':actual,'g2':model.moment_counts(6,G,Q,H,F(tau),M),
  'full':model.full_g_counts(g,H,F(tau),M),'coarse':model.coarse_counts(6,G,H,F(tau),M),
  'l1':H>F(tau)*G+(M-F(tau))*max(g)}

def combine(blocks,key):return model.combine_blocks([b[key] for b in blocks])

def main():
 ph=sha(PROTOCOL);source_data.ALL_REGIONS=tuple(source_data.ALL_REGIONS)+TARGETS
 hourly,monthly,audits,input_hashes=source_data.prepare_regions()
 monthly=monthly.loc[monthly.region.isin(TARGETS)].copy();day_rows=[];quality=[]
 for period,start,stop in PERIODS:
  s,e=pd.Timestamp(start,tz='UTC'),pd.Timestamp(stop,tz='UTC');expected=len(pd.date_range(s,e-pd.Timedelta(days=1),freq='D'))
  for region in TARGETS:
   frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
   x=frame.loc[times.ge(s)&times.lt(e)&frame.valid.astype(bool)].copy();x['utc_day']=pd.to_datetime(x.time,utc=True).dt.floor('D')
   cols=[f'generation_{f}' for f in source_data.FUELS];valid=[]
   for day,g in x.groupby('utc_day',sort=True):
    g=g.assign(_time=pd.to_datetime(g.time,utc=True)).sort_values('_time')
    if len(g)==24 and list(g._time)==list(pd.date_range(day,periods=24,freq='h')):valid.append((day,g))
   passing=len(valid)>=.8*expected
   quality.append({'region':region,'period':period,'complete_days':len(valid),'expected_days':expected,'fraction':len(valid)/expected,'quality_pass':passing})
   if not passing:continue
   for day,g in valid:
    month_start=pd.Timestamp(day.strftime('%Y-%m-01'),tz='UTC')
    mr=monthly[(monthly.region==region)&(monthly.period_start==month_start)]
    if len(mr)>1:raise ValueError(f'duplicate month row {region}/{month_start}')
    if len(mr)==0:M=GLOBAL_M
    else:
     active=[j for j,fuel in enumerate(source_data.FUELS) if float(mr.iloc[0][f'fuelshare_{fuel}'])>0]
     M=F(max(EF[j] for j in active)) if active else GLOBAL_M
    hours=[]
    for _,r in g.iterrows():
     values=[F(str(float(r[c]))) for c in cols]
     if any(v<0 for v in values) or sum(values)<=0:raise ValueError('invalid hourly generation')
     hours.append(values)
    for tau in TAUS:
     global_blocks=[block(hours[i:i+6],tau,GLOBAL_M) for i in range(0,24,6)]
     local_blocks=[block(hours[i:i+6],tau,M) for i in range(0,24,6)]
     actual=sum(b['actual'] for b in local_blocks)
     full=combine(local_blocks,'full');g2=combine(local_blocks,'g2');global_full=combine(global_blocks,'full')
     cert=all(b['l1'] for b in local_blocks)
     if actual not in full or actual not in g2 or actual not in global_full:raise ValueError('observed K outside compatible set')
     if not set(full).issubset(g2):raise ValueError('local full-G/G2 nesting failed')
     if cert and min(full)<8:raise ValueError('local L1 produced false K>=8 certificate')
     day_rows.append({'region':region,'period':period,'day':day.isoformat(),'tau':tau,'local_support_M':int(M),
      'actual_K':actual,'globalM_fullG_Kge8':min(global_full)>=8,'localM_fullG_Kge8':min(full)>=8,
      'localM_G2_Kge8':min(g2)>=8,'localM_L1_Kge8':cert,
      'localM_fullG_lower':min(full),'localM_fullG_upper':max(full)})
 if not day_rows:raise ValueError('no quality-qualified windows')
 q=pd.DataFrame(quality);d=pd.DataFrame(day_rows);OUT.mkdir(parents=True,exist_ok=False)
 q.to_csv(OUT/'quality_audit.csv',index=False);d.to_csv(OUT/'day_certificates.csv',index=False)
 measures=['globalM_fullG_Kge8','localM_fullG_Kge8','localM_G2_Kge8','localM_L1_Kge8']
 summary=d.groupby(['region','period','tau','local_support_M'])[measures].mean().reset_index()
 summary['days']=d.groupby(['region','period','tau','local_support_M']).size().to_numpy()
 summary.to_csv(OUT/'summary.csv',index=False)
 payload={'status':'FROZEN_UNSEEN_REGION_RETROSPECTIVE_IDENTIFICATION_NOT_FORECAST','targets':list(TARGETS),
  'periods':[p[0] for p in PERIODS],'taus':list(TAUS),'quality':quality,'rows':len(d),
  'summary':summary.to_dict(orient='records'),'source_audits':audits,'input_sha256':input_hashes,
  'protocol_sha256':ph,'script_sha256':sha(Path(__file__)),'limitations':[
   'Four balancing authorities from a quality-only census; descriptive transfer confirmation only.',
   'Fixed-EF hourly generation proxy and contemporaneous identification, not probabilistic coverage.',
   'Local M derives from EIA-derived monthly shares; hourly H/G/L1 are additional telemetry absent from China monthly/yearly-only public information.']}
 if sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
 print(pd.DataFrame(quality).to_string(index=False));print(summary.to_string(index=False))

if __name__=='__main__':main()
