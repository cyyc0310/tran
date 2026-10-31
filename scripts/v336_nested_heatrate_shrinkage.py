"""V336: nested selection of local coal-rate transfer strength."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import v300_official_provincial_ef_mix_transfer as v300
import v327_nbs_mix_transfer as v327
import v329_china_region_block_coldstart as v329
import v330_spatial_nested_ridge_transfer as v330
import v334_local_thermal_heatrate_transfer as v334

ROOT=Path(__file__).resolve().parents[1]
FACTORS=ROOT/"data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
MONTHLY=ROOT/"results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
EMBER=ROOT/"results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
RATES=ROOT/"data/china/cec_province_thermal_coal_consumption_rates_2020_2022.csv"
PROTOCOL=ROOT/"docs/V336_NESTED_HEATRATE_SHRINKAGE_PROTOCOL.md"
OUT=ROOT/"results/v336-nested-heatrate-shrinkage-r2"
GAMMAS=(0,.25,.5,.75,1,1.25); LAMBDAS=tuple(v330.LAMBDAS)

def score(y,p):
 e=np.asarray(p)-np.asarray(y)
 return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e**2))),"p90_absolute_error":float(np.quantile(np.abs(e),.9)),"mean_signed_error":float(np.mean(e))}

def feature_frame(df, source, groups, rate_col, gamma):
 cols=list(groups)
 x=df[cols].to_numpy(float).copy()
 x[:,0] *= 1 + gamma*(df[f"{rate_col}_gce_kwh"].to_numpy(float)/300-1)
 return x

def inner_select(train, year, source, groups, rate_col, gammas=GAMMAS):
 val_year=2021 if year==2022 else 2022
 validation=train.loc[train.target_year.eq(val_year)]
 fit_pool=train.loc[train.target_year.lt(val_year)]
 if val_year==2021: fit_pool=validation
 losses={}
 for gamma in gammas:
  for lam in LAMBDAS:
   errs=[]
   for region in sorted(train.region.unique()):
    va=validation.loc[validation.region.eq(region)]
    fit=fit_pool.loc[fit_pool.region.ne(region)]
    if va.empty or fit.empty: continue
    x=feature_frame(fit,source,groups,rate_col,gamma); xv=feature_frame(va,source,groups,rate_col,gamma)
    beta=v330.fit_ridge_nnls(x,fit.actual.to_numpy(float),lam)
    errs.extend(np.abs(va.actual.to_numpy(float)-xv@beta).tolist())
   if errs: losses[(gamma,lam)]=float(np.mean(errs))
 if not losses: raise RuntimeError("No inner spatial validation predictions")
 best=min(losses,key=lambda k:(losses[k],k[0],k[1]))
 return best, {f"{g:g}|{l:g}":v for (g,l),v in losses.items()}

def main():
 if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
 factors=pd.read_csv(FACTORS).set_index("province"); provinces=sorted(factors.index.astype(str))
 nbs_mix,_=v327.nbs_annual_mix(MONTHLY,provinces); ember_mix=v300.monthly_mix(EMBER)
 rows=v334.add_rows(factors,provinces,nbs_mix,ember_mix,pd.read_csv(RATES))
 schemas={"nbs5":tuple(f"nbs5_{g}" for g in v327.GROUPS),"ember5":tuple(f"ember5_{g}" for g in v300.COARSE)}
 predictions=[]; tunings=[]
 for held,members in v329.REGIONS.items():
  for year in (2022,2023):
   train=rows.loc[rows.target_year.lt(year)&rows.region.ne(held)]
   target=rows.loc[rows.region.eq(held)&rows.target_year.eq(year)]
   for source,groups in schemas.items():
    for rate_col in v334.RATE_COLS:
     (gamma,lam),losses=inner_select(train,year,source,groups,rate_col)
     (mix_gamma,mix_lam),_=inner_select(train,year,source,groups,rate_col,(0,))
     (full_gamma,full_lam),_=inner_select(train,year,source,groups,rate_col,(1,))
     arms={"nested":(gamma,lam),"mix_only":(mix_gamma,mix_lam),"full_rate":(full_gamma,full_lam)}
     for arm,(g,l) in arms.items():
      x=feature_frame(train,source,groups,rate_col,g); xt=feature_frame(target,source,groups,rate_col,g)
      beta=v330.fit_ridge_nnls(x,train.actual.to_numpy(float),l); pp=xt@beta
      tunings.append({"held_region":held,"target_year":year,"source":source,"rate_col":rate_col,"arm":arm,"selected_gamma":gamma,"selected_lambda":lam,"inner_loss_grid":json.dumps(losses)})
      for j,(_,r) in enumerate(target.iterrows()):
       predictions.append({"province":r.province,"held_region":held,"target_year":year,"source":source,"rate_col":rate_col,"arm":arm,"selected_gamma":gamma,"selected_lambda":lam,"actual":float(r.actual),"prediction":float(pp[j])})
 pred=pd.DataFrame(predictions); pred["abs_error"]=(pred.prediction-pred.actual).abs()
 score_rows=[]
 for evalname,sub in [("pooled",pred),*((str(y),pred.loc[pred.target_year.eq(y)]) for y in (2022,2023))]:
  for keys,g in sub.groupby(["source","rate_col","arm"],sort=False):
   score_rows.append({"evaluation":evalname,"source":keys[0],"rate_col":keys[1],"arm":keys[2],"weighting":"province",**score(g.actual,g.prediction)})
   rg=[score(h.actual,h.prediction) for _,h in g.groupby("held_region")]
   score_rows.append({"evaluation":evalname,"source":keys[0],"rate_col":keys[1],"arm":keys[2],"weighting":"region_equal",**{k:float(np.mean([z[k] for z in rg])) for k in rg[0]}})
 region_scores=pred.groupby(["source","rate_col","arm","held_region"]).abs_error.mean().unstack("arm")
 rng=np.random.default_rng(336); comparisons={}
 for (source,rate),g in region_scores.groupby(level=[0,1]):
  for arm in ("nested","full_rate"):
   d=(g[arm]-g.mix_only).to_numpy(float); boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(50000)])
   comparisons[f"{source}_{rate}_{arm}_minus_mix"]={"difference":float(d.mean()),"ci95":np.quantile(boots,[.025,.975]).tolist(),"regions_better":int((d<0).sum()),"per_region":dict(zip(g.index.get_level_values("held_region"),d.tolist()))}
 OUT.mkdir(parents=True)
 pred.to_csv(OUT/"predictions.csv",index=False); pd.DataFrame(score_rows).to_csv(OUT/"model_scores.csv",index=False); pd.DataFrame(tunings).to_csv(OUT/"tuning.csv",index=False)
 summary={"protocol":str(PROTOCOL.relative_to(ROOT)),"gamma_candidates":GAMMAS,"lambda_candidates":LAMBDAS,"comparisons":comparisons,"scores":score_rows,"selected_gamma_counts":pd.DataFrame(tunings).query("arm == 'nested'").groupby(["source","rate_col","selected_gamma"]).size().rename("n").reset_index().to_dict("records"),"limitations":["Annual MEE consumption-side labels; not hourly CIF.","Only seven region clusters and two outer years.","CEC yearbook rate release/vintage alignment remains unverified.","Heat rates are efficiency proxies, not carbon emission factors."]}
 summary["sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (FACTORS,MONTHLY,EMBER,RATES,PROTOCOL,Path(__file__))}
 (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
 print(pd.DataFrame(score_rows).to_string(index=False)); print(json.dumps(comparisons,indent=2,ensure_ascii=False)); print("gamma counts",summary["selected_gamma_counts"])
if __name__=="__main__": main()
