from pathlib import Path
import sys, hashlib, pickle
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v215-eia2026-sparse-anchor-frontier-r1'
sys.path.insert(0,str(ROOT/'scripts'))
import v54_unseen_local_ef_confirmation as sd
REGIONS=('AVA','BANC','BPAT','CHPD','DOPD','EPE','FMPP','FPL','GCPD','GRID','GVL','IID','JEA','LDWP','MISO','NWMT','NYIS','PSEI','SC','SCEG','SCL','SEC','TEC','TIDC','TPWR')
FUELS=('coal','gas','oil','other','nuclear','hydro','wind','solar')
EF=np.array([1000.,400.,800.,400.,0.,0.,0.,0.])
CACHE=Path('/tmp/v215_eia_hourly_series.pkl')
if OUT.exists(): raise SystemExit(f'refusing to overwrite {OUT}')
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
current_hashes={str(Path(path).resolve()):sha256(Path(path).resolve()) for path in sd.default_native_paths()}
cache_loaded=False
if CACHE.exists():
 with CACHE.open('rb') as f: cached_series,cached_hashes=pickle.load(f)
 if cached_hashes==current_hashes:
  series,input_hashes=cached_series,cached_hashes
  cache_loaded=True
if not cache_loaded:
 sd.ALL_REGIONS=REGIONS
 hourly,_,release_audit,input_hashes=sd.prepare_regions()
 # Convert valid hourly mix vectors to derived hourly direct-emissions average CIF under the frozen V202 EF proxy.
 series={}
 for r in REGIONS:
  f=hourly[r].copy()
  f['time']=pd.to_datetime(f.time,utc=True)
  f=f.loc[f.valid & f.time.ge('2025-01-01T00:00:00Z') & f.time.lt('2026-07-01T00:00:00Z')].copy()
  cols=[f'generation_{x}' for x in FUELS]
  z=f[cols].to_numpy(float)
  good=np.isfinite(z).all(axis=1)&(z>=0).all(axis=1)&(z.sum(axis=1)>0)
  f=f.loc[good].copy();z=z[good]
  f['cif']=z@EF/z.sum(axis=1)
  for period,g in f.groupby(f.time.dt.strftime('%Y-%m')):
   year,month=map(int,period.split('-'))
   g=g.sort_values('time'); zz=g[cols].to_numpy(float)
   cif=g.cif.to_numpy(float); hours=g.time.dt.hour.to_numpy(int)
   total=zz.sum(axis=0)
   if len(cif)<24*28: continue
   means=np.array([np.mean(cif[hours==h]) if np.any(hours==h) else np.nan for h in range(24)])
   monthly_y=float(cif.mean())
   means=np.where(np.isfinite(means),means,monthly_y)
   series[(r,year,month)]={'time':g.time.to_numpy(),'cif':cif,'hours':hours,'mix':total/total.sum(),'monthly_y':monthly_y,'hod':means,'n':len(cif)}
 with CACHE.open('wb') as f: pickle.dump((series,input_hashes),f)
# Require every leaveout target month and source month needed by the design.
all_regions=REGIONS
missing25=[(r,m) for r in all_regions for m in range(1,13) if (r,2025,m) not in series]
missing26=[(r,m) for r in all_regions for m in range(1,7) if (r,2026,m) not in series]
print('MISSING_2025',missing25,flush=True)
print('MISSING_2026',missing26,flush=True)
REGIONS=tuple(r for r in all_regions if sum((r,2025,m) in series for m in range(1,13))>=9 and sum((r,2026,m) in series for m in range(1,7))>=3)
if len(REGIONS)<10: raise ValueError(f'too few usable regions: {len(REGIONS)}')
print('ELIGIBLE_REGIONS',REGIONS,flush=True)

def design(rows):
 return np.column_stack([np.ones(len(rows)),np.eye(12)[np.array([x[0] for x in rows])-1],np.array([x[1] for x in rows])])
def pred_level(train_regions, target_mix, month):
 rows=[]; y=[]
 for r in train_regions:
  for m in range(1,13):
   if (r,2025,m) not in series: continue
   d=series[(r,2025,m)]; proxy=float(d['mix']@EF)
   rows.append((m,proxy/1000));y.append(d['monthly_y'])
 A=design(rows); coef=np.linalg.lstsq(A,np.array(y),rcond=None)[0]
 return float(np.array([1.,*np.eye(12)[month-1],float(target_mix@EF)/1000])@coef)
def dist(a,b):return float(np.abs(a-b).sum()/np.sum(a+b))
def shape_for(target_region,target_year,month,donor_regions):
 tx=series[(target_region,target_year,month)]['mix']
 donor_regions=[q for q in donor_regions if (q,2025,month) in series]
 donors=sorted(donor_regions,key=lambda q:(dist(tx,series[(q,2025,month)]['mix']),q))[:10]
 if len(donors)<5: raise ValueError(f'only {len(donors)} shape donors for {target_region}/{target_year}/{month}')
 prof=np.mean(np.stack([series[(q,2025,month)]['hod']-series[(q,2025,month)]['monthly_y'] for q in donors]),axis=0)
 return prof,donors
# Nested leave-two-region-out source residuals: the held target's data are never training/calibration input.
source_residuals={}
for excluded in REGIONS:
 src=[r for r in REGIONS if r!=excluded]
 for q in src:
  train=[r for r in src if r!=q]
  for m in range(1,13):
   if (q,2025,m) not in series: continue
   d=series[(q,2025,m)]
   lv=pred_level(train,d['mix'],m)
   sh,_=shape_for(q,2025,m,train)
   pred=lv+sh[d['hours']]
   source_residuals[(excluded,q,m)]=d['cif']-pred
# Prospective leave-one-region-out 2026 H1 target predictions.
targets={}
for p in REGIONS:
 train=[r for r in REGIONS if r!=p]
 for m in range(1,7):
  if (p,2026,m) not in series: continue
  d=series[(p,2026,m)]
  lv=pred_level(train,d['mix'],m)
  sh,donors=shape_for(p,2026,m,train)
  pred=lv+sh[d['hours']]
  targets[(p,m)]={'d':d,'pred':pred,'resid':d['cif']-pred,'donors':donors,'level':lv,'shape':sh}
# Sample budget and design comparison, 20 repeated draws per query.
methods=['global_source','random24','single_day24','clock_stratified24','random48','two_adjacent_days48','clock_stratified48']
Ks={'random24':24,'single_day24':24,'clock_stratified24':24,'random48':48,'two_adjacent_days48':48,'clock_stratified48':48}
metrics={x:[] for x in methods};prov_metrics={x:{p:[] for p in REGIONS} for x in methods}; per_query={x:[] for x in methods}; rng=np.random.default_rng(215); draws=200
for p in REGIONS:
 for m in range(1,7):
  if (p,m) not in targets: continue
  d=targets[(p,m)]['d'];e=targets[(p,m)]['resid'];n=len(e)
  train_abs=np.concatenate([np.abs(source_residuals[(p,q,sm)]) for q in REGIONS if q!=p for sm in range(1,13) if (p,q,sm) in source_residuals])
  globalr=float(np.quantile(train_abs,.9,method='higher'))
  baseae=np.abs(e); vals=(float(np.mean(baseae<=globalr)),2*globalr,float(np.mean(2*globalr+20*np.maximum(baseae-globalr,0))),float(np.mean(baseae)))
  metrics['global_source'].append(vals);prov_metrics['global_source'][p].append(vals);per_query['global_source'].append({'region':p,'month':m,'n_hours':n,'draws':1,'coverage':vals[0],'width':vals[1],'winkler90':vals[2],'point_mae':vals[3]})
  outs={k:[] for k in methods if k!='global_source'}
  ts=pd.DatetimeIndex(d['time']); date_values=ts.floor('D')
  daily=[idx.to_numpy() for _,idx in pd.Series(np.arange(n),index=date_values).groupby(level=0) if len(idx)==24]
  consecutive=[(daily[j],daily[j+1]) for j in range(len(daily)-1) if date_values[daily[j][-1]]+pd.Timedelta(days=1)==date_values[daily[j+1][0]]]
  hour_rows=[np.where(d['hours']==h)[0] for h in range(24)]
  if not daily or not consecutive or min(map(len,hour_rows))<2: raise ValueError(f'insufficient real-time sampling support: {p}/{m}')
  for _ in range(draws):
   idx_single=daily[int(rng.integers(len(daily)))]
   idx_clock24=np.array([rng.choice(hour_rows[h]) for h in range(24)])
   idx_random24=rng.choice(n,24,False);idx_random48=rng.choice(n,48,False)
   d1,d2=consecutive[int(rng.integers(len(consecutive)))];idx_2day=np.r_[d1,d2]
   idx_clock48=np.array([rng.choice(hour_rows[h],2,False) for h in range(24)]).reshape(-1)
   for name,idx in [('random24',idx_random24),('single_day24',idx_single),('clock_stratified24',idx_clock24),('random48',idx_random48),('two_adjacent_days48',idx_2day),('clock_stratified48',idx_clock48)]:
    mask=np.ones(n,bool);mask[idx]=False;corr=float(np.mean(e[idx]));ae=np.abs(e[mask]-corr);rad=float(np.quantile(np.abs(e[idx]-corr),.9,method='higher'))
    vals=(float(np.mean(ae<=rad)),2*rad,float(np.mean(2*rad+20*np.maximum(ae-rad,0))),float(np.mean(ae)))
    outs[name].append(vals)
  for name in outs:
   val=np.mean(outs[name],axis=0);metrics[name].append(val);prov_metrics[name][p].append(val);per_query[name].append({'region':p,'month':m,'n_hours':n,'draws':draws,'coverage':val[0],'width':val[1],'winkler90':val[2],'point_mae':val[3]})
print('CACHE_USED',cache_loaded,flush=True)
print('INPUT_REGIONS',len(REGIONS),'queries',len(targets),'target_window','2026-01..2026-06','source_window','2025-01..2025-12','draws',draws,flush=True)
print('method coverage width winkler point_mae worst_region_coverage',flush=True)
region_values={}
for name in methods:
 region_values[name]=np.array([np.mean(prov_metrics[name][p],axis=0) for p in REGIONS if prov_metrics[name][p]])
 byregion=region_values[name]
 worst=min(float(v[0]) for v in byregion)
 print(name,round(byregion[:,0].mean(),4),round(byregion[:,1].mean(),2),round(byregion[:,2].mean(),2),round(byregion[:,3].mean(),2),round(worst,4),flush=True)
# Paired bootstrap with the region as the independent resampling unit.
rr=np.random.default_rng(2150);bootstrap_rows=[]
for candidate,base in [('clock_stratified48','global_source'),('clock_stratified48','random48'),('clock_stratified48','single_day24'),('clock_stratified48','clock_stratified24')]:
 diff=region_values[candidate]-region_values[base]
 print('PAIRED_REGION_BOOT',candidate,'minus',base,flush=True)
 for j,label in enumerate(['coverage','width','winkler','point_mae']):
  b=np.mean(diff[rr.integers(0,len(diff),(20000,len(diff))) ,j],axis=1)
  ci=np.quantile(b,[.025,.975]);delta=float(diff[:,j].mean())
  print(label,'delta',round(delta,3),'CI',[round(float(x),3) for x in ci],flush=True)
  bootstrap_rows.append({'candidate':candidate,'baseline':base,'metric':label,'delta':delta,'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'resamples':20000,'cluster_unit':'region'})
print('TARGET_MONTHS',[(m,int(np.mean([targets[(p,m)]['d']['n'] for p in REGIONS if (p,m) in targets]))) for m in range(1,7)],flush=True)
print('QUERY_COUNTS_BY_REGION',{p:sum((p,m) in targets for m in range(1,7)) for p in REGIONS},flush=True)
print('INPUT_SHA256',input_hashes,flush=True)
import json
if OUT.exists(): raise SystemExit(f'refusing to overwrite {OUT}')
OUT.mkdir(parents=True)
pd.DataFrame([{'method':name,**row} for name,rows in per_query.items() for row in rows]).to_csv(OUT/'query_metrics.csv',index=False,float_format='%.10g')
pd.DataFrame([{'method':name,'region':p,**dict(zip(['coverage','width','winkler90','point_mae'],np.mean(prov_metrics[name][p],axis=0)))} for name in methods for p in REGIONS if prov_metrics[name][p]]).to_csv(OUT/'region_metrics.csv',index=False,float_format='%.10g')
pd.DataFrame(bootstrap_rows).to_csv(OUT/'paired_region_bootstrap.csv',index=False,float_format='%.10g')
summary={'status':'EIA_TEMPORAL_HOLDOUT_OPERATIONAL_MIX_DERIVED_CIF_SPARSE_ANCHOR_DEVELOPMENT','regions':list(REGIONS),'n_regions':len(REGIONS),'n_target_queries':len(targets),'target_window':'2026-01..2026-06','source_window':'2025-01..2025-12','fuel_categories':list(FUELS),'fixed_ef_g_per_kwh':EF.tolist(),'anchor_draws_per_query':draws,'methods':methods,'missing_2025_region_months':[list(x) for x in missing25],'missing_2026_region_months':[list(x) for x in missing26],'input_sha256':input_hashes,'limits':['EIA hourly fuel-generation estimates, not independently observed CIF','CIF labels mechanically derived from the same hourly fuel-generation shares with fixed emission factors','same native EIA data family already used by prior project experiments; not an independent data-vintage replication','2026 H1 targets exclude all history from the held-out region but this is retrospective development, not a blind test','empirical anchor q90 is descriptive and has no finite-sample coverage guarantee','some region-months fail the minimum valid-hour threshold'],'aggregation':'equal region weighting, then equal available month weighting within region','bootstrap':'20000 paired resamples of 24 regions; region is the independent unit'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('WROTE',OUT,flush=True)
