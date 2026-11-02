#!/usr/bin/env python3
"""Post-hoc decomposition: does local Ridge improve five-group month forecasts?"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_dual_track_monthly_adaptation as dual
import run_eia923_pseudotarget_transfer as eia_monthly
import run_official_intensity_transfer as official
from run_cross_country import ROOT
from transcif.research import monthly_adaptation


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    mix=eia_monthly.parse_monthly(tuple(range(2020,2025)),official.US_REGIONS)
    tables=dual.monthly_tables(mix)
    rows=[]
    for region in official.US_REGIONS:
        table=tables[region]
        for period in pd.period_range('2024-01','2024-12',freq='M'):
            if period not in table.index or period-2 not in table.index:continue
            truth=table.loc[period,list(official.COARSE)].to_numpy(float)
            observed=monthly_adaptation.forecast(table,period,None,'persistence')
            local=monthly_adaptation.forecast(table,period,None,'local_ridge')
            for method,pred in [('observed_m',observed),('local_ridge',local)]:
                delta=pred-truth
                rows.append({'ba':region,'month':int(period.month),'method':method,
                    'share_mae':float(np.abs(delta).mean()),
                    'share_l1':float(np.abs(delta).sum()),
                    'thermal_abs_error':float(abs(delta[0])),
                    'thermal_signed_error':float(delta[0]),
                    'hydro_abs_error':float(abs(delta[1])),
                    'nuclear_abs_error':float(abs(delta[2])),
                    'wind_abs_error':float(abs(delta[3])),
                    'solar_abs_error':float(abs(delta[4]))})
    monthly=pd.DataFrame(rows);monthly.to_csv(out/'monthly_mix_errors.csv',index=False)
    fields=[c for c in monthly.columns if c.endswith('_error') or c in ('share_mae','share_l1')]
    ba=monthly.groupby(['ba','method'])[fields].mean().reset_index();ba.to_csv(out/'ba_metrics.csv',index=False)
    macro=ba.groupby('method')[fields].mean().reset_index();macro.to_csv(out/'macro_metrics.csv',index=False)
    code=[Path(__file__),ROOT/'scripts/run_dual_track_monthly_adaptation.py',
          ROOT/'scripts/run_eia923_pseudotarget_transfer.py',
          ROOT/'scripts/run_official_intensity_transfer.py',
          ROOT/'src/transcif/research/monthly_adaptation.py']
    data=[eia_monthly.eia_archive(y) for y in range(2020,2025)]
    hashes={'code':{str(p):sha(p) for p in code},'input':{str(p):sha(p) for p in data}}
    (out/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    verify={'code_mismatches':[p for p,v in hashes['code'].items() if sha(Path(p))!=v],
        'input_mismatches':[p for p,v in hashes['input'].items() if sha(Path(p))!=v],
        'scored_ba_months':len(monthly[monthly.method.eq('local_ridge')]),
        'target_response_labels_used':False,
        'target_month_composition_used_only_for_posthoc_scoring':True}
    (out/'verification.json').write_text(json.dumps(verify,indent=2)+'\n')
    print(macro.to_string(index=False))


if __name__=='__main__':
    main()
