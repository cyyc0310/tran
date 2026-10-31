"""V337: test whether monthly seasonal technology shape adds to annual mix."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
import v300_official_provincial_ef_mix_transfer as v300
import v327_nbs_mix_transfer as v327
import v329_china_region_block_coldstart as v329
import v330_spatial_nested_ridge_transfer as v330
import v334_local_thermal_heatrate_transfer as v334
ROOT=Path(__file__).resolve().parents[1]
MONTHLY=ROOT/"results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
EMBER=ROOT/"results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
FACTORS=ROOT/"data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
RATES=ROOT/"data/china/cec_province_thermal_coal_consumption_rates_2020_2022.csv"
PROTOCOL=ROOT/"docs/V337_MONTHLY_SHAPE_LEVEL_TRANSFER_PROTOCOL.md"
OUT=ROOT/"results/v337-monthly-shape-level-transfer-r1"
LAMBDAS=tuple(v330.LAMBDAS)

def harmonic_features(month_share, groups):
 out={}
 for g in groups:
  vals=month_share[g]
  months=np.array(list(vals),float); y=np.array(list(vals.values()),float)
  xx=2*np.pi*(months-1)/12
  design=np.column_stack([np.ones(len(xx)),np.sin(xx),np.cos(xx)])
  coef=np.linalg.lstsq(design,y,rcond=None)[0]
  out[f"shape_{g}_sin1"]=float(coef[1]); out[f"shape_{g}_cos1"]=float(coef[2])
 return out

def nbs_shape(path, provinces):
 raw=pd.read_csv(path); raw=raw.loc[raw.province.isin(provinces)&raw.year.isin([2020,2021,2022])&raw.month.between(3,12)].copy()
 out={}; techs=("total","thermal","hydro","nuclear","wind","solar")
 for (p,y),g in raw.groupby(["province","year"]):
  by={t:g.loc[g.technology.eq(t)].set_index("month").generation_gwh.astype(float) for t in techs}
  monthly={}
  for m in range(3,13):
   total=by["total"].get(m,np.nan)
   if not np.isfinite(total) or total<=0: continue
   hydro=by["hydro"].get(m,np.nan)
   if not np.isfinite(hydro): hydro=max(total-sum(by[t].get(m,0) for t in ("thermal","nuclear","wind","solar")),0)
   amounts=[by["thermal"].get(m,np.nan),hydro,by["nuclear"].get(m,np.nan),by["wind"].get(m,np.nan),by["solar"].get(m,np.nan)]
   if not np.isfinite(amounts).all() or sum(amounts)<=0: continue
   shares=np.array(amounts,float)/sum(amounts)
   monthly[m]=dict(zip(("thermal","hydro_or_remainder","nuclear","wind","solar"),shares))
  if len(monthly)<8: raise ValueError(f"Too few monthly NBS observations: {p} {y} {len(monthly)}")
  groups=("thermal","hydro_or_remainder","nuclear","wind","solar")
  by_group={name:{month:shares[name] for month,shares in monthly.items()} for name in groups}
  out[(p,int(y))]=harmonic_features(by_group,groups)
 return out

def ember_shape(path):
 raw=pd.read_csv(path); raw=raw.loc[raw["Province code"].notna()&raw.Province.ne("China (total)")].copy(); raw["year"]=pd.to_datetime(raw.Date).dt.year;raw["month"]=pd.to_datetime(raw.Date).dt.month
 src=raw.pivot_table(index=["Province","year","month"],columns="Electricity source",values="Generation (TWh)",aggfunc="first")
 sources=("Coal","Gas","Other fossil","Bioenergy","Hydro","Nuclear","Wind","Solar")
 out={}
 for (p,y),g in src.groupby(level=["Province","year"]):
  data={s:{} for s in ("Thermal","Hydro","Nuclear","Wind","Solar")}
  for (_,_,m),row in g.iterrows():
   total=float(row[list(sources)].sum())
   if total<=0: continue
   vals={"Thermal":sum(float(row[s]) for s in sources[:4])/total,"Hydro":float(row.Hydro)/total,"Nuclear":float(row.Nuclear)/total,"Wind":float(row.Wind)/total,"Solar":float(row.Solar)/total}
   for s,v in vals.items(): data[s][int(m)]=v
  out[(p,int(y))]=harmonic_features(data,tuple(data))
 return out

def build_base():
 factors=pd.read_csv(FACTORS).set_index("province");provinces=sorted(factors.index.astype(str))
 nbs,_=v327.nbs_annual_mix(MONTHLY,provinces);ember=v300.monthly_mix(EMBER)
 rows=v334.add_rows(factors,provinces,nbs,ember,pd.read_csv(RATES))
 ns=nbs_shape(MONTHLY,provinces);es=ember_shape(EMBER)
 shape={}
 for (p,y),f in ns.items():shape[(p,y,"nbs5")]={f"shape_{k}":v for k,v in f.items()}
 for (p,y),f in es.items():shape[(p,y,"ember5")]={f"shape_{k}":v for k,v in f.items()}
 for i,row in rows.iterrows():
  source="nbs5" if "nbs5_thermal" in rows.columns else "nbs5"
  f=shape.get((row.province,int(row.feature_year),source),{})
  for k,v in f.items(): rows.loc[i,k]=v
  f=shape.get((row.province,int(row.feature_year),"ember5"),{})
  for k,v in f.items(): rows.loc[i,k.replace("shape_","ember_shape_")]=v
 return rows

def score(y,p):
 e=np.asarray(p)-np.asarray(y);return {"mae":float(np.mean(abs(e))),"rmse":float(np.sqrt(np.mean(e**2))),"p90_absolute_error":float(np.quantile(abs(e),.9)),"mean_signed_error":float(np.mean(e))}
def tune(train,year,cols):
 vy=2021 if year==2022 else 2022;valid=train.loc[train.target_year.eq(vy)];pool=train.loc[train.target_year.lt(vy)]
 if vy==2021:pool=valid
 loss={l:[] for l in LAMBDAS}
 for region in sorted(train.region.unique()):
  va=valid.loc[valid.region.eq(region)];fit=pool.loc[pool.region.ne(region)]
  if va.empty or fit.empty:continue
  for l in LAMBDAS:
   b=v330.fit_ridge_nnls(fit[list(cols)].to_numpy(float),fit.actual.to_numpy(float),l)
   loss[l].extend(np.abs(va.actual.to_numpy(float)-va[list(cols)].to_numpy(float)@b).tolist())
 means={l:np.mean(v) for l,v in loss.items() if v};best=min(means,key=lambda l:(means[l],l));return best,{str(k):float(v) for k,v in means.items()}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 rows=build_base();schemas={"nbs5":(tuple(f"nbs5_{g}" for g in v327.GROUPS),tuple(f"nbs5_{g}" for g in v327.GROUPS)+tuple(c for c in rows if c.startswith("shape_") and not c.startswith("ember_shape_"))),"ember5":(tuple(f"ember5_{g}" for g in v300.COARSE),tuple(f"ember5_{g}" for g in v300.COARSE)+tuple(c for c in rows if c.startswith("ember_shape_")))}
 preds=[];tune_rows=[]
 for held in v329.REGIONS:
  for year in (2022,2023):
   train=rows.loc[rows.target_year.lt(year)&rows.region.ne(held)];target=rows.loc[rows.region.eq(held)&rows.target_year.eq(year)]
   for source,(base_cols,shape_cols) in schemas.items():
    for arm,cols in (("mean_only",base_cols),("mean_plus_shape",shape_cols)):
     lam,loss=tune(train,year,cols);beta=v330.fit_ridge_nnls(train[list(cols)].to_numpy(float),train.actual.to_numpy(float),lam);pp=target[list(cols)].to_numpy(float)@beta
     tune_rows.append({"held_region":held,"target_year":year,"source":source,"arm":arm,"lambda":lam,"inner_losses":json.dumps(loss)})
     for j,(_,r) in enumerate(target.iterrows()):preds.append({"province":r.province,"held_region":held,"target_year":year,"source":source,"arm":arm,"actual":float(r.actual),"prediction":float(pp[j])})
 pred=pd.DataFrame(preds);pred["abs_error"]=(pred.prediction-pred.actual).abs();scores=[]
 for label,sub in [("pooled",pred),*((str(y),pred.loc[pred.target_year.eq(y)]) for y in (2022,2023))]:
  for (source,arm),g in sub.groupby(["source","arm"]):
   scores.append({"evaluation":label,"source":source,"arm":arm,"weighting":"province",**score(g.actual,g.prediction)})
   rr=[score(a.actual,a.prediction) for _,a in g.groupby("held_region")];scores.append({"evaluation":label,"source":source,"arm":arm,"weighting":"region_equal",**{k:float(np.mean([x[k] for x in rr])) for k in rr[0]}})
 region=pred.groupby(["source","arm","held_region"]).abs_error.mean().unstack("arm");rng=np.random.default_rng(337);comps={}
 for source,g in region.groupby(level=0):
  d=(g.mean_plus_shape-g.mean_only).to_numpy(float);boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(50000)]);comps[source]={"delta":float(d.mean()),"ci95":np.quantile(boot,[.025,.975]).tolist(),"regions_better":int((d<0).sum()),"per_region":dict(zip(g.index.get_level_values("held_region"),d.tolist()))}
 OUT.mkdir(parents=True);pred.to_csv(OUT/"predictions.csv",index=False);pd.DataFrame(scores).to_csv(OUT/"model_scores.csv",index=False);pd.DataFrame(tune_rows).to_csv(OUT/"tuning.csv",index=False)
 files=(MONTHLY,EMBER,FACTORS,RATES,PROTOCOL,Path(__file__));summary={"comparisons":comps,"scores":scores,"limitations":["Annual MEE target, not hourly CIF.","NBS seasonal features exclude Feb cumulative observation; annual means retain prior V335 construction.","Seven regions and two target years; low-power inference.","Input vintage and user-table schema not yet audited."]};summary["sha256"]={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files};(OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
 print(pd.DataFrame(scores).query("weighting == 'region_equal'").to_string(index=False));print(json.dumps(comps,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
