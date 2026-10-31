"""V84: broaden the frozen temporal-information curve to the full V52 census."""
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
PROTOCOL=ROOT/'docs/V84_CENSUS_SCALE_TEMPORAL_BUDGET_PROTOCOL.md'
CENSUS=ROOT/'results/eia-ba-quality-census-v52-r1/quality_census.json'
OUT=ROOT/'results/v84-census-scale-temporal-budget-r1'
TARGETS=('AVA','BANC','BPAT','CHPD','DOPD','EPE','FMPP','FPL','GCPD','GRID','GVL','IID','JEA','LDWP','MISO','NWMT','NYIS','PSEI','SC','SCEG','SCL','SEC','TEC','TIDC','TPWR')
TAUS=(100,300);BLOCKS=(24,6,3,1);GLOBAL_M=F(1000)
PERIODS=(('2024H2','2024-07-01','2025-01-01'),('2025H1','2025-01-01','2025-07-01'),('2025H2','2025-07-01','2026-01-01'))
EF=(1000,400,800,0,0,0,0,400)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def block_counts(hours,tau,M):
 gs=[sum(x,F(0)) for x in hours]
 hs=[sum((v*ef for v,ef in zip(x,EF)),F(0)) for x in hours]
 actual=sum(h>F(tau)*g for h,g in zip(hs,gs));n=len(gs)
 if M<=tau:
  if actual:raise ValueError('observed event exceeds local fuel support')
  return {'actual':0,'coarse':(0,),'moment':(0,),'full':(0,) if n<=6 else None}
 H=sum(hs,F(0));G=sum(gs,F(0));Q=sum((g*g for g in gs),F(0))
 coarse=model.coarse_counts(n,G,H,F(tau),M)
 moment=model.moment_counts(n,G,Q,H,F(tau),M)
 full=model.full_g_counts(gs,H,F(tau),M) if n<=6 else None
 return {'actual':actual,'coarse':coarse,'moment':moment,'full':full}

def combine(parts,key):return model.combine_blocks([x[key] for x in parts])

def main():
 protocol_hash=sha(PROTOCOL);census=json.loads(CENSUS.read_text())
 if tuple(census['quality_qualified_regions_alphabetical'])!=TARGETS:raise ValueError('frozen census roster differs')
 source_data.ALL_REGIONS=tuple(dict.fromkeys(source_data.ALL_REGIONS+TARGETS))
 hourly,monthly,audits,input_hashes=source_data.prepare_regions();monthly=monthly[monthly.region.isin(TARGETS)].copy()
 qrows=[];rows=[]
 for label,start,stop in PERIODS:
  s,e=pd.Timestamp(start,tz='UTC'),pd.Timestamp(stop,tz='UTC');expected=len(pd.date_range(s,e-pd.Timedelta(days=1),freq='D'))
  for region in TARGETS:
   frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
   x=frame.loc[times.ge(s)&times.lt(e)&frame.valid.astype(bool)].copy();x['day']=pd.to_datetime(x.time,utc=True).dt.floor('D')
   cols=[f'generation_{f}' for f in source_data.FUELS];valid=[]
   for day,g in x.groupby('day',sort=True):
    g=g.assign(_time=pd.to_datetime(g.time,utc=True)).sort_values('_time')
    if len(g)==24 and list(g._time)==list(pd.date_range(day,periods=24,freq='h')):valid.append((day,g))
   qrows.append({'region':region,'period':label,'complete_days':len(valid),'expected_days':expected,'coverage':len(valid)/expected})
   for day,g in valid:
    ms=pd.Timestamp(day.strftime('%Y-%m-01'),tz='UTC');mr=monthly[(monthly.region==region)&(monthly.period_start==ms)]
    if len(mr)>1:raise ValueError('duplicate monthly row')
    if len(mr)==0:M=GLOBAL_M
    else:
     active=[j for j,f in enumerate(source_data.FUELS) if float(mr.iloc[0][f'fuelshare_{f}'])>0]
     M=F(max(EF[j] for j in active)) if active else F(0)
    hours=[]
    for _,r in g.iterrows():
     a=[F(str(float(r[c]))) for c in cols]
     if any(v<0 for v in a) or sum(a)<=0:raise ValueError('invalid hourly generation')
     hours.append(a)
    for tau in TAUS:
     for b in BLOCKS:
      parts=[block_counts(hours[i:i+b],tau,M) for i in range(0,24,b)]
      coarse=combine(parts,'coarse');moment=combine(parts,'moment');actual=sum(z['actual'] for z in parts)
      if not set(moment).issubset(coarse):raise ValueError('coarse/moment nesting failed')
      if actual not in coarse or actual not in moment:raise ValueError('actual K not contained')
      full=combine(parts,'full') if b<=6 else None
      if full is not None and (actual not in full or not set(full).issubset(moment)):
       raise ValueError('full-G/moment nesting failed')
      rows.append({'region':region,'period':label,'day':day.isoformat(),'tau':tau,'M':int(M),'block_hours':b,
       'blocks_per_day':24//b,'HG_fields':2*(24//b),'Q_fields':(0 if b==1 else 24//b),
       'actual_K':actual,'coarse_Kge8':min(coarse)>=8,'moment_Kge8':min(moment)>=8,
       'fullG_Kge8':(min(full)>=8 if full is not None else None),
       'coarse_span':max(coarse)-min(coarse),'moment_span':max(moment)-min(moment),
       'fullG_span':(max(full)-min(full) if full is not None else None)})
 d=pd.DataFrame(rows);q=pd.DataFrame(qrows)
 if d.empty:raise ValueError('empty census run')
 OUT.mkdir(parents=True,exist_ok=False);d.to_csv(OUT/'day_identification.csv',index=False);q.to_csv(OUT/'quality_audit.csv',index=False)
 keys=['region','period','tau','M','block_hours','blocks_per_day','HG_fields','Q_fields']
 summary=d.groupby(keys,dropna=False).agg(days=('day','size'),coarse_rate=('coarse_Kge8','mean'),moment_rate=('moment_Kge8','mean'),fullG_rate=('fullG_Kge8','mean'),mean_coarse_span=('coarse_span','mean'),mean_moment_span=('moment_span','mean'),mean_fullG_span=('fullG_span','mean')).reset_index()
 summary.to_csv(OUT/'summary.csv',index=False)
 # Equal region then equal time-window macro average; never count days as independent systems.
 rp=d.groupby(['region','period','tau','block_hours'])[['coarse_Kge8','moment_Kge8','fullG_Kge8']].mean().reset_index()
 reg=rp.groupby(['region','tau','block_hours'])[['coarse_Kge8','moment_Kge8','fullG_Kge8']].mean().reset_index()
 macro=reg.groupby(['tau','block_hours'])[['coarse_Kge8','moment_Kge8','fullG_Kge8']].mean().reset_index()
 macro.to_csv(OUT/'region_macro_summary.csv',index=False)
 # Quality-only sensitivity: remove all period rows under the inherited 80% floor.
 pass_pairs={(r,p) for r,p,c in q[['region','period','coverage']].itertuples(index=False,name=None) if c>=.8}
 dq=d[d.apply(lambda z:(z.region,z.period) in pass_pairs,axis=1)]
 rqp=dq.groupby(['region','period','tau','block_hours'])[['coarse_Kge8','moment_Kge8','fullG_Kge8']].mean().reset_index()
 rq=rqp.groupby(['region','tau','block_hours'])[['coarse_Kge8','moment_Kge8','fullG_Kge8']].mean().reset_index()
 qm=rq.groupby(['tau','block_hours'])[['coarse_Kge8','moment_Kge8','fullG_Kge8']].mean().reset_index();qm.to_csv(OUT/'quality_gated_macro_summary.csv',index=False)
 payload={'status':'CENSUS_SCALE_DEVELOPMENT_IDENTIFICATION_NOT_FORECAST','regions':list(TARGETS),
  'periods':[p[0] for p in PERIODS],'taus':list(TAUS),'block_hours':list(BLOCKS),'rows':len(d),
  'quality':qrows,'region_macro_summary':macro.astype(object).where(pd.notna(macro),None).to_dict(orient='records'),
  'quality_gated_macro_summary':qm.astype(object).where(pd.notna(qm),None).to_dict(orient='records'),'source_audits':audits,'input_sha256':input_hashes,
  'quality_census_sha256':sha(CENSUS),'protocol_sha256':protocol_hash,'script_sha256':sha(Path(__file__)),
  'limitations':['All 25 census-eligible regions have been opened in prior work; this is expansion, not unseen-region confirmation.',
   'Balancing areas are not independent power systems; region macro is descriptive only.',
   'Fixed-EF contemporaneous generation proxy, not actual CIF forecasting or empirical coverage.',
   'Block H/G/Q acquisition/validation cost is not represented by scalar count alone.']}
 if sha(PROTOCOL)!=protocol_hash:raise ValueError('protocol changed')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False))
 print('quality',q.to_string(index=False));print('macro',macro.to_string(index=False));print('quality-gated',qm.to_string(index=False))

if __name__=='__main__':main()
