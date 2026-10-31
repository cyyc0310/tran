"""V79: development ablation using monthly active fuel support M."""
from __future__ import annotations
import hashlib,json,sys
from fractions import Fraction as F
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import v54_unseen_local_ef_confirmation as source_data  # noqa:E402
sys.path.insert(0,str(ROOT))
from scripts import tail_count_v20 as count_model  # noqa:E402
PROTOCOL=ROOT/'docs/V79_MONTHLY_SUPPORT_ENVELOPE_PROTOCOL.md'
OUT=ROOT/'results/v79-monthly-support-envelope-r1'
TARGETS=('EPE','TIDC');TAUS=(100,300);GLOBAL_M=F(1000)
PERIODS=(('2024H2','2024-07-01','2025-01-01'),('2025H1','2025-01-01','2025-07-01'),('2025H2','2025-07-01','2026-01-01'))
EF=(1000,400,800,0,0,0,0,400)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def block_sets(hours,tau,support):
 gs=[sum(x,F(0)) for x in hours]
 hs=[sum((v*e for v,e in zip(x,EF)),F(0)) for x in hours]
 H=sum(hs,F(0));G=sum(gs,F(0));Q=sum((g*g for g in gs),F(0))
 actual=sum(h>F(tau)*g for h,g in zip(hs,gs))
 return {'H':H,'G':G,'Q':Q,'actual':actual,
  'coarse':count_model.coarse_counts(6,G,H,F(tau),support),
  'g2':count_model.moment_counts(6,G,Q,H,F(tau),support),
  'full':count_model.full_g_counts(gs,H,F(tau),support),
  'l1':H>F(tau)*G+(support-F(tau))*max(gs)}

def main():
 ph=sha(PROTOCOL);source_data.ALL_REGIONS=tuple(source_data.ALL_REGIONS)+TARGETS
 hourly,monthly,audits,input_hashes=source_data.prepare_regions()
 monthly=monthly.loc[monthly.region.isin(TARGETS)].copy()
 rows=[]
 for period,start,stop in PERIODS:
  s,e=pd.Timestamp(start,tz='UTC'),pd.Timestamp(stop,tz='UTC')
  for region in TARGETS:
   frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
   keep=times.ge(s)&times.lt(e)&frame.valid.astype(bool)
   x=frame.loc[keep].copy();x['utc_day']=pd.to_datetime(x.time,utc=True).dt.floor('D')
   cols=[f'generation_{f}' for f in source_data.FUELS]
   for day,g in x.groupby('utc_day',sort=True):
    g=g.assign(_time=pd.to_datetime(g.time,utc=True)).sort_values('_time')
    if len(g)!=24 or list(g._time)!=list(pd.date_range(day,periods=24,freq='h')):continue
    month_start=pd.Timestamp(day.strftime('%Y-%m-01'),tz='UTC')
    mr=monthly[(monthly.region==region)&(monthly.period_start==month_start)]
    if len(mr)>1:raise ValueError(f'duplicate month row {region}/{month_start}')
    if len(mr)==0:
     M=GLOBAL_M
    else:
     active=[i for i,fuel in enumerate(source_data.FUELS) if float(mr.iloc[0][f'fuelshare_{fuel}'])>0]
     M=F(max(EF[i] for i in active))
    if M<=max(TAUS):raise ValueError('support must exceed all tested thresholds')
    hours=[]
    for _,r in g.iterrows():
     z=[F(str(float(r[c]))) for c in cols]
     if any(v<0 for v in z) or sum(z)<=0:raise ValueError('invalid hourly generation')
     hours.append(z)
    for tau in TAUS:
     out_global=[];out_local=[]
     for start_hour in range(0,24,6):
      part=hours[start_hour:start_hour+6]
      out_global.append(block_sets(part,tau,GLOBAL_M))
      out_local.append(block_sets(part,tau,M))
     def combine(pack,key):return count_model.combine_blocks([p[key] for p in pack])
     actual=sum(p['actual'] for p in out_local)
     full_local=combine(out_local,'full');g2_local=combine(out_local,'g2')
     global_l1=all(p['l1'] for p in out_global)
     local_l1=all(p['l1'] for p in out_local)
     if actual not in full_local or actual not in g2_local:raise ValueError('actual K not in local-support sets')
     if not set(full_local).issubset(g2_local):raise ValueError('local full/G2 nesting failed')
     if local_l1 and max(full_local)<8:raise ValueError('local L1 contradicts local full-G')
     rows.append({'region':region,'period':period,'day':day.isoformat(),'tau':tau,'monthly_support_M':int(M),
      'actual_K':actual,'globalM1000_L1_Kge8':global_l1,'localM_L1_Kge8':local_l1,
      'localM_G2_Kge8':min(g2_local)>=8,'localM_fullG_Kge8':min(full_local)>=8,
      'globalM1000_fullG_Kge8':min(combine(out_global,'full'))>=8,
      'localM_fullG_lower':min(full_local),'localM_fullG_upper':max(full_local)})
 d=pd.DataFrame(rows)
 if d.empty:raise ValueError('empty results')
 OUT.mkdir(parents=True,exist_ok=False);d.to_csv(OUT/'day_certificates.csv',index=False)
 metrics=['globalM1000_L1_Kge8','localM_L1_Kge8','localM_G2_Kge8','localM_fullG_Kge8','globalM1000_fullG_Kge8']
 summary=d.groupby(['region','period','tau','monthly_support_M'])[metrics].mean().reset_index()
 summary['days']=d.groupby(['region','period','tau','monthly_support_M']).size().to_numpy()
 summary.to_csv(OUT/'summary.csv',index=False)
 for c in ('globalM1000_L1_Kge8','localM_L1_Kge8'):
  if (d[c]&~d.localM_fullG_Kge8).any():raise ValueError(f'{c} has false certificates relative to local full G')
 payload={'status':'MONTHLY_ACTIVE_SUPPORT_DEVELOPMENT_ABLATION_NOT_UNSEEN_CONFIRMATION',
  'targets':list(TARGETS),'periods':[x[0] for x in PERIODS],'thresholds':list(TAUS),'rows':len(d),
  'summary':summary.to_dict(orient='records'),'monthly_support_values':sorted(d.monthly_support_M.unique().tolist()),
  'source_audits':audits,'input_sha256':input_hashes,'protocol_sha256':ph,'script_sha256':sha(Path(__file__)),
  'limitations':['EPE/TIDC are already opened regions; this is method development, not an independent confirmation.',
   'Monthly active support is derived from frozen EIA proxy monthly shares, not Chinese official data.',
   'Still requires contemporaneous 6h carbon energy, G and L1 summaries; not a monthly-only forecast.']}
 if sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
 print(summary.to_string(index=False))

if __name__=='__main__':main()
