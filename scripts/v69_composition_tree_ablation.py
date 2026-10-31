"""Fixed-capacity feature ablation for V68 cross-region tail-rate forecasting."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v54_unseen_local_ef_confirmation as source_data  # noqa: E402
import v68_coarse_mix_tail_frequency_forecast as v68  # noqa: E402
PROTOCOL = ROOT / "docs/V69_COMPOSITION_TREE_ABLATION_PROTOCOL.md"
OUT = ROOT / "results/v69-composition-tree-ablation-r2"
TAUS = (300.0, 600.0)
SEED = 68027
ARMS = {
    "latest_ef_calendar": np.array([16, 18, 19]),
    "latest_mix_calendar": np.array([*range(8), 18, 19]),
    "latest_plus_3m_mix": np.array([*range(16), 18, 19]),
}
def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def equal_region_weights(regions: pd.Series) -> np.ndarray:
    counts=regions.value_counts()
    w=regions.map(lambda r:1.0/counts[r]).to_numpy(float)
    return w/w.mean()
def bootstrap(d: pd.DataFrame) -> tuple[float,float]:
    regions=sorted(d.region.unique()); rng=np.random.default_rng(SEED); vals=[]
    for _ in range(5000):
        chosen=rng.choice(regions,len(regions),replace=True)
        vals.append(pd.concat([d.loc[d.region.eq(r)] for r in chosen]).delta.mean())
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))
def main():
    hourly, monthly, release_audits, input_hashes=source_data.prepare_regions()
    labels=v68.monthly_labels(hourly)
    rows=[]
    for _,lab in labels.iterrows():
        x=v68.month_features(monthly,str(lab.region),lab.period_start)
        if x is None: continue
        rows.append({"region":lab.region,"month":lab.month,"period_start":lab.period_start,
          "period_end":lab.period_end,"label_available_at":lab.label_available_at,"features":x,
          **{k:lab[k] for k in lab.index if k.startswith(("rate_tau_","any_tau_"))}})
    panel=pd.DataFrame(rows); preds=[]
    for month in v68.TARGET_MONTHS:
        origin=pd.Timestamp(f"{month}-01",tz="UTC")
        for region in sorted(hourly):
            q=panel.loc[panel.region.eq(region)&panel.month.eq(month)]
            qx=v68.month_features(monthly,region,origin)
            if q.empty or qx is None: continue
            train=panel.loc[panel.region.ne(region)&panel.period_end.le(origin)&panel.label_available_at.le(origin)]
            if train.region.nunique()<8 or len(train)<100: raise ValueError(f"insufficient training {region}/{month}")
            X=np.stack(train.features.to_list()); w=equal_region_weights(train.region)
            for tau in TAUS:
                target=f"rate_tau_{int(tau)}"; y=train[target].to_numpy(float)
                actual=float(q.iloc[0][target])
                for arm,cols in ARMS.items():
                    model=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=12,max_features=.8,
                                              random_state=SEED,n_jobs=1)
                    model.fit(X[:,cols],y,sample_weight=w)
                    p=float(np.clip(model.predict(qx[cols].reshape(1,-1))[0],0,1))
                    preds.append({"region":region,"month":month,"tau":int(tau),"arm":arm,
                                  "actual":actual,"prediction":p,"abs_error":abs(p-actual),
                                  "positive":int(q.iloc[0][f"any_tau_{int(tau)}"]),
                                  "training_regions":train.region.nunique(),"training_rows":len(train)})
    detail=pd.DataFrame(preds)
    eligible_pairs=panel.loc[panel.month.isin(v68.TARGET_MONTHS),["region","month"]].drop_duplicates()
    expected=len(eligible_pairs)*len(TAUS)*len(ARMS)
    if len(detail)!=expected:
        raise ValueError(f"unexpected prediction rows: {len(detail)} != {expected}")
    OUT.mkdir(parents=True,exist_ok=False); detail.to_csv(OUT/'paired_predictions.csv',index=False)
    summary=(detail.groupby(['tau','arm']).agg(region_months=('month','count'),regions=('region','nunique'),
      positive_region_months=('positive','sum'),mean_actual=('actual','mean'),mae=('abs_error','mean'),
      rmse=('abs_error',lambda s:float(np.sqrt(np.mean(s*s))))).reset_index())
    summary.to_csv(OUT/'summary.csv',index=False)
    pairs=[]
    for tau in TAUS:
      pivot=detail.loc[detail.tau.eq(tau)].pivot(index=['region','month'],columns='arm',values='abs_error').reset_index()
      for left,right in [('latest_mix_calendar','latest_ef_calendar'),('latest_plus_3m_mix','latest_mix_calendar'),('latest_plus_3m_mix','latest_ef_calendar')]:
        d=pivot[['region','month',left,right]].copy(); d['delta']=d[left]-d[right]
        lo,hi=bootstrap(d[['region','delta']])
        pairs.append({'tau':tau,'candidate':left,'reference':right,'mae_delta':float(d.delta.mean()),
          'reference_mae':float(d[right].mean()),'candidate_mae':float(d[left].mean()),
          'improvement_fraction':float((d[left]<d[right]).mean()),'cluster_bootstrap_95_low':lo,
          'cluster_bootstrap_95_high':hi,'relative_improvement':float((d[right].mean()-d[left].mean())/d[right].mean())})
    pair=pd.DataFrame(pairs); pair.to_csv(OUT/'paired_comparisons.csv',index=False)
    payload={'status':'DEVELOPMENT_ABLATION_NOT_INDEPENDENT_CONFIRMATION','regions':sorted(detail.region.unique()),
      'target_months':list(v68.TARGET_MONTHS),'taus':list(TAUS),'rows':len(detail),
      'arms':{k:v.tolist() for k,v in ARMS.items()},'estimator':{'n_estimators':100,'min_samples_leaf':12,
      'max_features':.8,'random_state':SEED,'n_jobs':1},'hourly_label_release_lag_assumption_days':30,
      'release_audits':release_audits,'input_sha256':input_hashes,
      'protocol_sha256':sha256(PROTOCOL),'script_sha256':sha256(Path(__file__)),
      'summary':summary.to_dict(orient='records'),'paired_comparisons':pair.to_dict(orient='records'),
      'limitations':['Hourly label first-release vintage is not archived; 30-day lag is an assumption.',
       'All 17 regions were previously opened; this is development, not blind region confirmation.',
       'Target is a fixed-factor positive-generation proxy, not full consumption-side CIF.']}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps({'summary':payload['summary'],'paired_comparisons':payload['paired_comparisons']},indent=2))
if __name__=='__main__': main()
