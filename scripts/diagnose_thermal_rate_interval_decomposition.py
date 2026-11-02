#!/usr/bin/env python3
"""Post-hoc split of thermal-rate center and source-calibrated interval radius."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import run_official_intensity_transfer as official
RUN=ROOT/'results/thermal-rate-official-increment-r3'
METHODS=('coarse_hgb_M','coarse_hgb_MH')

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def main()->None:
    capacity=official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    _,prepared,_,_=official.prepare(ROOT,capacity)
    query=prepared['GB'].loc[prepared['GB'].year.eq(2024)].copy().reset_index(drop=True)
    observed=official.aligned_labels(query,official.source_labels('GB',years=(2024,)))
    predictions=pd.read_csv(RUN/'US_to_GB_predictions.csv.gz')
    calibrations=json.loads((RUN/'US_to_GB_calibration.json').read_text())
    records=[]
    for center in METHODS:
        for radius in METHODS:
            c=calibrations[center];r=calibrations[radius]
            scored=official.score(observed,predictions[center].to_numpy(float),c['bias'],r['sigma'],query.clock.reset_index(drop=True))
            records.append({'center_model':center,'bias_from':center,'source_radius_model':radius,'source_radius':r['sigma'],'target_year':2024,**scored})
    result=pd.DataFrame(records)
    result.to_csv(RUN/'interval_decomposition.csv',index=False)
    manifest={'status':'POST_HOC_DEVELOPMENT_DIAGNOSTIC','target_labels_previously_viewed':True,'uses_target_labels_for_selection':False,'interpretation':'factorial decomposition only; no candidate selected','sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),RUN/'US_to_GB_predictions.csv.gz',RUN/'US_to_GB_calibration.json',RUN/'startup_hashes.json',RUN/'prediction_seal.json']}}
    (RUN/'interval_decomposition_audit.json').write_text(json.dumps(manifest,indent=2))
    print(result.to_string(index=False))
if __name__=='__main__':main()
