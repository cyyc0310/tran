"""One-region frozen confirmation of V70's tau=600 scale interaction."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import v54_unseen_local_ef_confirmation as source_data  # noqa:E402
import v68_coarse_mix_tail_frequency_forecast as v68  # noqa:E402
from v69_composition_tree_ablation import equal_region_weights  # noqa:E402
PROTOCOL=ROOT/'docs/V71_GVL_UNSEEN_REGION_CONFIRMATION_PROTOCOL.md'
OUT=ROOT/'results/v71-gvl-unseen-confirmation-r2';TARGET='GVL';TAU=600;SEED=68027
ARMS={'ef_calendar':np.array([16,18,19]),'ef_scale_calendar':np.array([16,17,18,19])}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 # Extend only the quality-screened, never-scored target. Existing 17 regions
 # remain the fixed source pool and their labels are excluded when target=GVL.
 source_data.ALL_REGIONS=tuple(source_data.ALL_REGIONS)+(TARGET,)
 hourly,monthly,audits,input_hashes=source_data.prepare_regions()
 labels=v68.monthly_labels(hourly);rows=[]
 for _,lab in labels.iterrows():
  x=v68.month_features(monthly,str(lab.region),lab.period_start)
  if x is None:continue
  rows.append({'region':lab.region,'month':lab.month,'period_start':lab.period_start,'period_end':lab.period_end,
   'label_available_at':lab.label_available_at,'valid_hours':lab.valid_hours,'features':x,**{k:lab[k] for k in lab.index if k.startswith(('rate_tau_','any_tau_'))}})
 panel=pd.DataFrame(rows);pred=[]
 for month in v68.TARGET_MONTHS:
  origin=pd.Timestamp(f'{month}-01',tz='UTC')
  target=panel.loc[panel.region.eq(TARGET)&panel.month.eq(month)]
  qx=v68.month_features(monthly,TARGET,origin)
  if target.empty or qx is None:raise ValueError(f'missing eligible target {TARGET}/{month}')
  train=panel.loc[panel.region.ne(TARGET)&panel.period_end.le(origin)&panel.label_available_at.le(origin)]
  if train.region.nunique()!=17:raise ValueError(f'unexpected source pool: {train.region.nunique()}')
  X=np.stack(train.features.to_list());w=equal_region_weights(train.region);col=f'rate_tau_{TAU}';y=train[col].to_numpy(float);source_names=sorted(train.region.unique())
  actual=float(target.iloc[0][col])
  for arm,idx in ARMS.items():
   model=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=12,max_features=.8,random_state=SEED,n_jobs=1)
   model.fit(X[:,idx],y,sample_weight=w);p=float(np.clip(model.predict(qx[idx].reshape(1,-1))[0],0,1))
   pred.append({'region':TARGET,'month':month,'tau':TAU,'arm':arm,'actual_event_rate':actual,'prediction':p,
    'abs_error':abs(p-actual),'valid_hours':int(target.iloc[0].valid_hours),'training_regions':int(train.region.nunique()),'source_region_names':source_names,
    'training_region_months':int(len(train)),'origin':origin.isoformat()})
 detail=pd.DataFrame(pred)
 if len(detail)!=6*len(ARMS):raise ValueError(f'expected 12 predictions, got {len(detail)}')
 OUT.mkdir(parents=True,exist_ok=False);detail.to_csv(OUT/'gvl_month_predictions.csv',index=False)
 summary=detail.groupby('arm').agg(months=('month','count'),mae=('abs_error','mean'),rmse=('abs_error',lambda s:float(np.sqrt(np.mean(s*s)))),mean_actual_rate=('actual_event_rate','mean')).reset_index()
 summary.to_csv(OUT/'summary.csv',index=False)
 payload={'status':'SINGLE_UNSEEN_REGION_FROZEN_CONFIRMATION_NOT_GENERALIZATION','target_region':TARGET,'target_months':list(v68.TARGET_MONTHS),
  'tau':TAU,'rows':len(detail),'source_regions':sorted(set(r for names in detail.source_region_names for r in names)),'arms':{k:v.tolist() for k,v in ARMS.items()},
  'estimator':{'n_estimators':100,'min_samples_leaf':12,'max_features':.8,'seed':SEED},'monthly_release_policy':'V54 period_end+30d',
  'hourly_label_release_lag_assumption_days':30,'release_audits':audits,'input_sha256':input_hashes,
  'protocol_sha256':sha(PROTOCOL),'script_sha256':sha(Path(__file__)),'summary':summary.to_dict(orient='records'),
  'limitations':['One held-out region only; no region-level bootstrap or generalization claim.','Hourly label first-release vintage unavailable; 30-day lag is assumed.',
   'Fixed-EF positive-generation proxy, not full-grid consumption CIF.']}
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2));print(json.dumps({'monthly':detail.to_dict(orient='records'),'summary':payload['summary']},indent=2))
if __name__=='__main__':main()
