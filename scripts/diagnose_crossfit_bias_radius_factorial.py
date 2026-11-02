#!/usr/bin/env python3
"""Post-hoc factorial diagnostic of cross-fitted bias and interval radius."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import run_official_intensity_transfer as official
RUN=ROOT/'results/crossfit-domain-risk-calibration-r2'
OUT=RUN/'bias_radius_factorial.csv'
METHODS=('coarse_hgb_M','coarse_hgb_MH','fixed_ef_recomposition')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(y,p,r,clock):
    keep=np.isfinite(y)&np.isfinite(p);y=y[keep];p=p[keep];clock=pd.to_datetime(clock[keep]).reset_index(drop=True);e=y-p
    d=pd.DataFrame({'day':clock.dt.strftime('%Y-%m-%d'),'y':y,'p':p});c=d[['y','p']]-d.groupby('day')[['y','p']].transform('mean')
    return {'hours':int(len(y)),'mae':float(abs(e).mean()),'shape_mae':float(abs(c.y-c.p).mean()),'coverage90':float((abs(e)<=r).mean()),'width90':float(2*r),'interval_score90':float(np.mean(2*r+20*np.maximum(abs(e)-r,0)))}
def main():
    cap=official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH);_,prepared,_,_=official.prepare(ROOT,cap)
    q=prepared['GB'].loc[prepared['GB'].year.eq(2024)].copy().reset_index(drop=True);y=official.aligned_labels(q,official.source_labels('GB',years=(2024,)))
    pred=pd.read_csv(RUN/'US_to_GB_predictions.csv.gz');cal=json.loads((RUN/'source_calibration.json').read_text());rows=[]
    for center in METHODS:
        for bname,bkey in [('time_only','time_only'),('loo_oof','loo_oof')]:
            bias=cal[center][bkey]['bias']
            radius_specs=[('time_only_same_model',cal[center]['time_only']['pooled_equal_region_q90']),('loo_oof_pooled_same_model',cal[center]['loo_oof']['pooled_equal_region_q90']),('loo_oof_worstBA_same_model',cal[center]['loo_oof']['max_region_q90'])]
            if center=='coarse_hgb_MH':
                radius_specs.extend([('time_only_M_radius',cal['coarse_hgb_M']['time_only']['pooled_equal_region_q90']),('loo_oof_M_radius',cal['coarse_hgb_M']['loo_oof']['pooled_equal_region_q90'])])
            for rname,radius in radius_specs:
                rows.append({'center_model':center,'bias_source':bname,'radius_source':rname,'bias':bias,'radius':float(radius),**metric(y,pred[center].to_numpy(float)+bias,float(radius),q.clock)})
    frame=pd.DataFrame(rows);frame.to_csv(OUT,index=False)
    audit={'status':'POST_HOC_FACTORIAL_DEVELOPMENT_DIAGNOSTIC','target_label_previously_viewed':True,'no_candidate_selection':True,'target_file_hash':sha(official.TARGET_LABEL_ROOT/'GB_consumption_2024_halfhour.csv.gz'),'prediction_hash':sha(RUN/'US_to_GB_predictions.csv.gz'),'source_calibration_hash':sha(RUN/'source_calibration.json'),'output_hash':sha(OUT)}
    (RUN/'bias_radius_factorial_audit.json').write_text(json.dumps(audit,indent=2))
    print(frame.to_string(index=False))
if __name__=='__main__':main()
