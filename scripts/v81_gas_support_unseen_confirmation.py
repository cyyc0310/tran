"""V81: confirm monthly active-support bounds on frozen unseen gas regions."""
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
PROTOCOL=ROOT/'docs/V81_GAS_SUPPORT_UNSEEN_CONFIRMATION_PROTOCOL.md'
OUT=ROOT/'results/v81-gas-support-unseen-confirmation-r1'
TARGETS=('AVRN','DEAA');TAUS=(100,300);GLOBAL_M=F(1000)
PERIODS=(('2024H2','2024-07-01','2025-01-01'),('2025H1','2025-01-01','2025-07-01'),('2025H2','2025-07-01','2026-01-01'))
EF=(1000,400,800,0,0,0,0,400)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def block(hours,tau,M):
 gs=[sum(x,F(0)) for x in hours];hs=[sum((v*e for v,e in zip(x,EF)),F(0)) for x in hours]
 H=sum(hs,F(0));G=sum(gs,F(0));Q=sum((x*x for x in gs),F(0));actual=sum(h>F(tau)*g for h,g in zip(hs,gs))
 if M<=tau:
  if actual:raise ValueError('event incompatible with known fuel support')
  return {'actual':0,'g2':(0,),'full':(0,),'l1':False}
 return {'actual':actual,'g2':model.moment_counts(6,G,Q,H,F(tau),M),
  'full':model.full_g_counts(gs,H,F(tau),M),
  'l1':H>F(tau)*G+(M-F(tau))*max(gs)}

def combine(blocks,key):return model.combine_blocks([b[key] for b in blocks])

def main():
 ph=sha(PROTOCOL);source_data.ALL_REGIONS=tuple(source_data.ALL_REGIONS)+TARGETS
 hourly,monthly,audits,input_hashes=source_data.prepare_regions();monthly=monthly[monthly.region.isin(TARGETS)].copy()
 quality=[];rows=[]
 for period,start,stop in PERIODS:
  s,e=pd.Timestamp(start,tz='UTC'),pd.Timestamp(stop,tz='UTC');expected=len(pd.date_range(s,e-pd.Timedelta(days=1),freq='D'))
  for region in TARGETS:
   frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
   x=frame.loc[times.ge(s)&times.lt(e)&frame.valid.astype(bool)].copy();x['day']=pd.to_datetime(x.time,utc=True).dt.floor('D')
   cols=[f'generation_{f}' for f in source_data.FUELS];valid=[]
   for day,g in x.groupby('day',sort=True):
    g=g.assign(_time=pd.to_datetime(g.time,utc=True)).sort_values('_time')
    if len(g)==24 and list(g._time)==list(pd.date_range(day,periods=24,freq='h')):valid.append((day,g))
   passing=len(valid)>=.8*expected
   quality.append({'region':region,'period':period,'complete_days':len(valid),'expected_days':expected,'fraction':len(valid)/expected,'quality_pass':passing})
   if not passing:continue
   for day,g in valid:
    mstart=pd.Timestamp(day.strftime('%Y-%m-01'),tz='UTC');mr=monthly[(monthly.period_start==mstart)&(monthly.region==region)]
    if len(mr)>1:raise ValueError('duplicate monthly fuel support')
    if len(mr)==0:M=GLOBAL_M
    else:
     active=[j for j,f in enumerate(source_data.FUELS) if float(mr.iloc[0][f'fuelshare_{f}'])>0]
     M=F(max(EF[j] for j in active)) if active else GLOBAL_M
    hours=[]
    for _,r in g.iterrows():
     a=[F(str(float(r[c]))) for c in cols]
     if any(z<0 for z in a) or sum(a)<=0:raise ValueError('invalid hourly fuel generation')
     hours.append(a)
    for tau in TAUS:
     gb=[block(hours[i:i+6],tau,GLOBAL_M) for i in range(0,24,6)]
     lb=[block(hours[i:i+6],tau,M) for i in range(0,24,6)]
     actual=sum(b['actual'] for b in lb);local=combine(lb,'full');g2=combine(lb,'g2');globalset=combine(gb,'full');cert=all(b['l1'] for b in lb)
     if any(actual not in z for z in (local,g2,globalset)):raise ValueError('actual K absent from a set')
     if not set(local).issubset(g2):raise ValueError('local full/G2 nesting failed')
     if cert and min(local)<8:raise ValueError('L1 false certificate')
     rows.append({'region':region,'period':period,'day':day.isoformat(),'tau':tau,'local_support_M':int(M),
      'actual_K':actual,'globalM_fullG_Kge8':min(globalset)>=8,'localM_fullG_Kge8':min(local)>=8,
      'localM_G2_Kge8':min(g2)>=8,'localM_L1_Kge8':cert})
 if not rows:raise ValueError('no quality-qualified score rows')
 q=pd.DataFrame(quality);d=pd.DataFrame(rows);OUT.mkdir(parents=True,exist_ok=False)
 q.to_csv(OUT/'quality_audit.csv',index=False);d.to_csv(OUT/'day_certificates.csv',index=False)
 metrics=['globalM_fullG_Kge8','localM_fullG_Kge8','localM_G2_Kge8','localM_L1_Kge8']
 summary=d.groupby(['region','period','tau','local_support_M'])[metrics].mean().reset_index()
 summary['days']=d.groupby(['region','period','tau','local_support_M']).size().to_numpy();summary.to_csv(OUT/'summary.csv',index=False)
 payload={'status':'UNSEEN_REGION_RETROSPECTIVE_IDENTIFICATION_CONFIRMATION_NOT_FORECAST','targets':list(TARGETS),
  'periods':[x[0] for x in PERIODS],'taus':list(TAUS),'quality':quality,'rows':len(d),
  'summary':summary.to_dict(orient='records'),'source_audits':audits,'input_sha256':input_hashes,
  'protocol_sha256':ph,'script_sha256':sha(Path(__file__)),'limitations':[
   'Two regions selected by quality and annual fuel-column metadata only; limited descriptive confirmation.',
   'Fixed EF EIA proxy and same-period identification, not forecasting or coverage.',
   'Requires local 6h H/G and G2/L1 telemetry beyond monthly fuel support; not China monthly-only deployable.']}
 if sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
 print(pd.DataFrame(quality).to_string(index=False));print(summary.to_string(index=False))

if __name__=='__main__':main()
