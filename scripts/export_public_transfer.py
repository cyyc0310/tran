"""Train/export all-foreign development references, without reading China inputs."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from transcif.research.multicountry import (
    FACTORS,FINE,HISTORY,METHODS,coarse_values,fit,load_countries,predict,public_features,public_months,weights,
)
from transcif.research.monthly_adaptation import weighted_quantile
from run_cross_country import ROOT,digest


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'src/transcif/research/multicountry.py',ROOT/'src/transcif/data/smard.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    raw,audit=load_countries(ROOT);frames={}
    for r,f in raw.items():
        v=public_features(public_months(f),f.clock).dropna(subset=list(HISTORY))
        v['raw_row']=v.index;v['country']=f.country.loc[v.index];v['region']=r
        frames[r]=v
    def data(year,first_half=False):
        selected={r:f.loc[f.clock.dt.year.eq(year)&(f.clock.dt.month.le(6) if first_half else True)] for r,f in frames.items()}
        joined=pd.concat(selected.values(),ignore_index=True)
        fuel=np.concatenate([raw[r].loc[f.raw_row,list(FINE)].to_numpy() for r,f in selected.items()])
        coarse=np.concatenate([coarse_values(raw[r].loc[f.raw_row]) for r,f in selected.items()])
        return joined,fuel,coarse
    train,fuel,coarse=data(2022)
    models=fit(train,fuel,coarse)
    cal,cal_fuel,_=data(2023,True);y=cal_fuel@FACTORS/cal_fuel.sum(axis=1)
    point=predict(models,cal);w=weights(cal)
    radii={m:weighted_quantile(np.abs(y-point[m]),w,.9) for m in METHODS}
    metadata={'source_countries':['DE','GB','US'],'train_year':2022,'calibration_period':'2023H1',
              'training_hours':len(train),'calibration_hours':len(cal),'research_baselines_only':True,
              'innovation_gate_passed':False,'current_production_accuracy_certified':False}
    package={'models':models,'feature_columns':list(HISTORY),'interval_radii':radii,
             'feature_support':{c:[float(train[c].min()),float(train[c].max())] for c in HISTORY},'metadata':metadata}
    joblib.dump(package,out/'foreign_models.joblib')
    restored=joblib.load(out/'foreign_models.joblib')
    for m,v in predict(restored['models'],cal.iloc[:24]).items():
        np.testing.assert_array_equal(v,point[m][:24])
    (out/'model_metadata.json').write_text(json.dumps({**metadata,'interval_radii':radii},indent=2))
    if any(digest(ROOT/k)!=v for k,v in hashes.items()):raise RuntimeError('code changed during export')
    for item in audit.values():
        if 'path' in item:hashes[item['path']]=digest(ROOT/item['path'])
        for name in item.get('manifests',[]):hashes[name]=digest(ROOT/name)
    hashes['foreign_models.joblib']=digest(out/'foreign_models.joblib')
    (out/'provenance.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(metadata))


if __name__=='__main__':main()
