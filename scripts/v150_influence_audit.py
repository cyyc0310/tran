"""Post-hoc fixed-teacher fit influence; not independent region validation."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from v139_physical_reserve_calibration import sha

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/v150-severity-support-r1'
C=ROOT/'results/v146-critical-composition-calibration-r1'
OUT=ROOT/'results/v150-support-analysis-r1'

def main():
    if OUT.exists():raise FileExistsError(OUT)
    summary=json.loads((R/'summary.json').read_text())
    for name,h in summary['input_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError(name)
    for name,h in summary['output_hashes'].items():
        if sha(R/name)!=h:raise ValueError(name)
    influence=pd.read_csv(R/'fixed_teacher_influence.csv')
    ci=pd.read_csv(C/'curve_index.csv')
    curves=np.load(C/'curves.npz')['curves']
    inverse=pd.read_csv(C/'calibration_inverse_scores.csv')
    panel=pd.read_csv(ROOT/'results/v144-canonical-profile-cache-r1/canonical_monthly_panel.csv')
    metadata=panel.loc[panel.tau.eq(600)].set_index('profile_id')
    joined=influence.merge(ci[['target_region','seed','profile_id','role']],on=['target_region','seed','profile_id'],validate='many_to_one')
    if len(joined)!=len(influence):raise ValueError('influence rows dropped')
    joined['clipped_scale']=np.clip(np.maximum(0,np.expm1(np.clip(joined.log_prediction,-30,20))/256),1/256,1)
    span=joined.groupby(['target_region','seed','family','model','profile_id','role']).agg(
        min_log=('log_prediction','min'),max_log=('log_prediction','max'),
        min_scale=('clipped_scale','min'),max_scale=('clipped_scale','max')).reset_index()
    span['log_span']=span.max_log-span.min_log
    span['scale_ratio']=span.max_scale/span.min_scale
    rows=[]
    for (target,seed,family,model,omitted),group in joined.groupby(['target_region','seed','family','model','omitted_region']):
        scale=group.set_index('profile_id').clipped_scale
        cal=inverse.loc[inverse.target_region.eq(target)&inverse.seed.eq(seed)&inverse.method.eq(family)]
        if not cal.supported.all():raise ValueError('unexpected unsupported calibration month')
        scores=pd.DataFrame({'region':cal.region.to_numpy(),'score':cal.delta.to_numpy()/scale.loc[cal.profile_id].to_numpy()}).groupby('region').score.max()
        if len(scores)!=12:raise ValueError('calibration region count')
        radius=float(scores.max()) # alpha .10 finite-sample rank=12
        fold=ci.loc[ci.target_region.eq(target)&ci.seed.eq(seed)].set_index('profile_id')
        for obs in panel.loc[panel.region.eq(target)].itertuples(index=False):
            product=radius*scale.loc[obs.profile_id]
            if product>1:lo,hi=int(obs.physical_lower),int(obs.physical_upper)
            else:
                idx=min(int(np.searchsorted(np.linspace(0,1,257),product,side='left')),256)
                a,b=curves[int(fold.loc[obs.profile_id,'curve_id']),0 if family.startswith('horizontal') else 1,(600,900).index(obs.tau),idx]
                lo=max(int(obs.physical_lower),math.floor(a*obs.hours+1e-9));hi=min(int(obs.physical_upper),math.ceil(b*obs.hours-1e-9))
            if not obs.physical_lower<=lo<=hi<=obs.physical_upper:raise ValueError('physical containment')
            rows.append({'region':target,'seed':seed,'family':family,'model':model,'omitted_region':omitted,
                'month':obs.month,'tau':obs.tau,'lower':lo,'upper':hi,'width':hi-lo,
                'covered':lo<=obs.factor_box_K_lower and hi>=obs.factor_box_K_upper})
    frame=pd.DataFrame(rows)
    frontier=frame.groupby(['region','seed','family','model','omitted_region','tau']).agg(
        coverage=('covered','mean'),width=('width','mean'),months=('month','size')).reset_index()
    for key,g in frontier.groupby(['region','seed','family','model','tau']):
        expected=metadata.loc[metadata.region.eq(key[0])].shape[0]
        if not g.months.eq(expected).all():raise ValueError('omission target denominator')
    OUT.mkdir(parents=True)
    span.to_csv(OUT/'clipped_scale_influence.csv',index=False)
    frontier.to_csv(OUT/'omission_frontier.csv',index=False)
    frame.to_csv(OUT/'omission_rows.csv',index=False)
    audit={'assertion_failures':0,'omission_target_rows':len(frame),'source_summary_sha256':sha(R/'summary.json'),
        'script_sha256':sha(Path(__file__)),'scope':'Post-hoc fixed-teacher fit sensitivity, not independently held-out source data.',
        'output_hashes':{p.name:sha(p) for p in sorted(OUT.iterdir())}}
    (OUT/'summary.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(frontier.loc[frontier.region.eq('LGEE')&frontier.family.str.startswith('horizontal')].groupby(['seed','tau','model']).agg(
        min_coverage=('coverage','min'),max_coverage=('coverage','max'),min_width=('width','min'),max_width=('width','max')).to_string())

if __name__=='__main__':main()
