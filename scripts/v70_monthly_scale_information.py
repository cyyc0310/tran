"""Test whether legal monthly generation scale adds cross-region tail-rate skill."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import v54_unseen_local_ef_confirmation as source_data  # noqa:E402
import v68_coarse_mix_tail_frequency_forecast as v68  # noqa:E402
from v69_composition_tree_ablation import equal_region_weights,bootstrap  # noqa:E402
PROTOCOL=ROOT/'docs/V70_MONTHLY_SCALE_INFORMATION_PROTOCOL.md'
OUT=ROOT/'results/v70-monthly-scale-information-r1';TAUS=(300.,600.);SEED=68027
ARMS={'ef_calendar':np.array([16,18,19]),'ef_scale_calendar':np.array([16,17,18,19]),
      'mix_calendar':np.array([*range(8),18,19]),'mix_scale_calendar':np.array([*range(8),17,18,19])}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 hourly,monthly,audits,input_hashes=source_data.prepare_regions();labels=v68.monthly_labels(hourly);rows=[]
 for _,lab in labels.iterrows():
  x=v68.month_features(monthly,str(lab.region),lab.period_start)
  if x is None:continue
  rows.append({'region':lab.region,'month':lab.month,'period_start':lab.period_start,'period_end':lab.period_end,
   'label_available_at':lab.label_available_at,'features':x,**{k:lab[k] for k in lab.index if k.startswith(('rate_tau_','any_tau_'))}})
 panel=pd.DataFrame(rows);pred=[]
 for month in v68.TARGET_MONTHS:
  origin=pd.Timestamp(f'{month}-01',tz='UTC')
  for region in sorted(hourly):
   q=panel.loc[panel.region.eq(region)&panel.month.eq(month)];qx=v68.month_features(monthly,region,origin)
   if q.empty or qx is None:continue
   tr=panel.loc[panel.region.ne(region)&panel.period_end.le(origin)&panel.label_available_at.le(origin)]
   X=np.stack(tr.features.to_list());w=equal_region_weights(tr.region)
   for tau in TAUS:
    col=f'rate_tau_{int(tau)}';y=tr[col].to_numpy(float);actual=float(q.iloc[0][col])
    for arm,idx in ARMS.items():
     model=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=12,max_features=.8,random_state=SEED,n_jobs=1)
     model.fit(X[:,idx],y,sample_weight=w);p=float(np.clip(model.predict(qx[idx].reshape(1,-1))[0],0,1))
     pred.append({'region':region,'month':month,'tau':int(tau),'arm':arm,'actual':actual,'prediction':p,
      'abs_error':abs(p-actual),'positive':int(q.iloc[0][f'any_tau_{int(tau)}'])})
 d=pd.DataFrame(pred);eligible=panel.loc[panel.month.isin(v68.TARGET_MONTHS),['region','month']].drop_duplicates()
 if len(d)!=len(eligible)*len(TAUS)*len(ARMS):raise ValueError(f'row mismatch {len(d)}')
 OUT.mkdir(parents=True,exist_ok=False);d.to_csv(OUT/'paired_predictions.csv',index=False)
 summ=d.groupby(['tau','arm']).agg(region_months=('month','count'),positive=('positive','sum'),mae=('abs_error','mean'),rmse=('abs_error',lambda s:float(np.sqrt(np.mean(s*s))))).reset_index();summ.to_csv(OUT/'summary.csv',index=False)
 comps=[]
 for tau in TAUS:
  z=d.loc[d.tau.eq(tau)].pivot(index=['region','month'],columns='arm',values='abs_error').reset_index()
  for left,right in [('ef_scale_calendar','ef_calendar'),('mix_scale_calendar','mix_calendar'),('mix_calendar','ef_calendar'),('mix_scale_calendar','ef_scale_calendar')]:
   x=z[['region','month',left,right]].copy();x['delta']=x[left]-x[right];lo,hi=bootstrap(x[['region','delta']])
   comps.append({'tau':tau,'candidate':left,'reference':right,'candidate_mae':float(x[left].mean()),'reference_mae':float(x[right].mean()),
    'mae_delta':float(x.delta.mean()),'relative_improvement':float((x[right].mean()-x[left].mean())/x[right].mean()),
    'improvement_fraction':float((x[left]<x[right]).mean()),'cluster_bootstrap_95_low':lo,'cluster_bootstrap_95_high':hi})
 cp=pd.DataFrame(comps);cp.to_csv(OUT/'paired_comparisons.csv',index=False)
 payload={'status':'DEVELOPMENT_NOT_INDEPENDENT_CONFIRMATION','regions':sorted(d.region.unique()),'target_months':list(v68.TARGET_MONTHS),
  'rows':len(d),'taus':list(TAUS),'arms':{k:v.tolist() for k,v in ARMS.items()},'estimator':{'n_estimators':100,'min_samples_leaf':12,'max_features':.8,'seed':SEED},
  'hourly_label_release_lag_assumption_days':30,'release_audits':audits,'input_sha256':input_hashes,'protocol_sha256':sha(PROTOCOL),
  'script_sha256':sha(Path(__file__)),'summary':summ.to_dict(orient='records'),'paired_comparisons':cp.to_dict(orient='records'),
  'limitations':['Hourly label first-release archive unavailable; 30-day lag is an assumption.','All regions previously opened.','Fixed-EF generation-side proxy, not consumption CIF.']}
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2));print(json.dumps({'summary':payload['summary'],'paired_comparisons':payload['paired_comparisons']},indent=2))
if __name__=='__main__':main()
