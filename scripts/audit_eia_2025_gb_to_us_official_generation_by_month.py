#!/usr/bin/env python3
"""Post-score BA-month decomposition for the frozen 2025 GB→US generation run."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'docs/evidence/gb-to-us-2025-official-generation-r2'
LABELS=ROOT/'docs/evidence/eia930-official-co2-2025-r1/eia930_official_co2_2025.csv'
LB_TO_G=453.59237

pred=pd.read_csv(RUN/'sealed_predictions.csv.gz',parse_dates=['time'])
labels=pd.read_csv(LABELS)
labels['valid_at']=pd.to_datetime(labels.local_hour_ending,utc=True)-pd.Timedelta(hours=1)
labels['time']=labels.valid_at
labels['y']=labels.co2_generated_intensity_lb_per_kwh*LB_TO_G
labels=labels.loc[labels.valid_at.dt.year.eq(2025),['ba','time','y']].dropna(subset=['y'])
cal=json.loads((RUN/'calibration.json').read_text())
d=pred.merge(labels,on=['ba','time'],how='inner',validate='one_to_one')
rows=[]
methods=list(cal)
for (ba,month),g in d.groupby(['ba','period'],sort=True):
    for method in methods:
        c=cal[method]; p=g[method].to_numpy(float)+c['bias_g_per_kwh']; y=g.y.to_numpy(float)
        err=y-p; radius=c['split_conformal_90_radius_g_per_kwh']
        rows.append({'ba':ba,'month':month,'method':method,'hours':len(g),
                     'mae_g_per_kwh':float(np.abs(err).mean()),'bias_g_per_kwh':float(err.mean()),
                     'coverage90':float((np.abs(err)<=radius).mean()),'width90_g_per_kwh':2*radius,
                     'interval_score90':float((2*radius+20*np.maximum(np.abs(err)-radius,0)).mean())})
monthly=pd.DataFrame(rows)
monthly.to_csv(RUN/'ba_month_metrics.csv',index=False)
summary=monthly.groupby('method').agg(ba_months=('ba','size'),hours=('hours','sum'),
    mae_equal_ba_month=('mae_g_per_kwh','mean'),bias_equal_ba_month=('bias_g_per_kwh','mean'),
    coverage90_equal_ba_month=('coverage90','mean'),width90_equal_ba_month=('width90_g_per_kwh','mean'),
    interval_score90_equal_ba_month=('interval_score90','mean')).reset_index()
summary.to_csv(RUN/'ba_month_macro_metrics.csv',index=False)
pivot=monthly.pivot(index=['ba','month'],columns='method',values='mae_g_per_kwh')
summary_pair={}
for method in methods:
    if method=='anchor700': continue
    diff=pivot[method]-pivot['anchor700']
    summary_pair[method]={'ba_month_cells':int(diff.notna().sum()),
        'wins_vs_anchor700':int((diff<0).sum()),'ties_vs_anchor700':int((diff==0).sum()),
        'losses_vs_anchor700':int((diff>0).sum()),'mean_mae_delta_g_per_kwh':float(diff.mean()),
        'median_mae_delta_g_per_kwh':float(diff.median())}
(RUN/'ba_month_pairwise_vs_anchor700.json').write_text(json.dumps(summary_pair,indent=2)+'\n')
print(summary.to_string(index=False))
print(json.dumps(summary_pair,indent=2))
