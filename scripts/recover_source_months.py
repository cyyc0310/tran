"""Rebuild only LGEE 2023H1 proxy from original native submissions, without imputation."""
import argparse
import json
from pathlib import Path

import pandas as pd

from transcif.data.eia930 import normalize_balance
from transcif.research.proxy_recovery import month_local_proxy
from run_cross_country import ROOT,digest


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    historical=ROOT.parent/'tran-stable/data_2023/raw_eia930'
    paths=[historical/'gen_jul_dec_2022.csv',historical/'gen_jan_jun_2023.csv']
    sources=[];schemas={};blocks=[]
    for snapshot,path in enumerate(paths):
        original=pd.read_csv(path);original=original.loc[original['Balancing Authority'].eq('LGEE')]
        normalized=normalize_balance(original);normalized['source_snapshot']=snapshot
        schemas[str(snapshot)]={'fuel_columns':[c for c in normalized if c.startswith('generation_')]}
        blocks.append(normalized);sources.append({'path':str(path),'sha256':digest(path)})
    rows=pd.concat(blocks,ignore_index=True)
    start,stop=pd.Timestamp('2023-01-01T00:00Z'),pd.Timestamp('2023-07-01T00:00Z')
    rows=rows.loc[rows.valid_at.ge(start)&rows.valid_at.lt(stop)].copy()
    rows.index=pd.DatetimeIndex(rows.valid_at)
    repaired,audits=month_local_proxy(rows,'LGEE',schemas)
    old_path=ROOT/'results/external-v6-data/LGEE_hourly.csv'
    old=pd.read_csv(old_path);old_time=pd.to_datetime(old.time,utc=True)
    new_time=pd.to_datetime(repaired.time,utc=True)
    if not set(new_time)==set(old_time.loc[old_time.ge(start)&old_time.lt(stop)]):
        raise ValueError('repair changes source timestamp population')
    replacement=repaired.copy();replacement['time']=new_time
    replacement=replacement.set_index('time')
    new=old.copy();mask=old_time.ge(start)&old_time.lt(stop)
    columns=[c for c in old if c!='time']
    new.loc[mask,columns]=replacement.loc[old_time.loc[mask],columns].to_numpy()
    pd.testing.assert_frame_equal(old.loc[~mask],new.loc[~mask],check_dtype=False)
    old_valid=old.valid.astype(str).str.lower().eq('true')
    new_valid=new.valid.astype(str).str.lower().eq('true')
    provenance={'status':'MONTH_LOCAL_REPORTED_PROXY_NOT_METERED_TRUTH','source_paths':sources,
                'original_proxy_sha256':digest(old_path),'window_start':start.isoformat(),'window_stop':stop.isoformat(),
                'old_valid_hours':int((old_valid&mask).sum()),'new_valid_hours':int((new_valid&mask).sum()),
                'added_valid_hours':int((~old_valid&new_valid&mask).sum()),
                'outside_window_rows_exactly_equal':True,'monthly_audits':audits,
                'unobserved_fuels':'excluded from reported proxy; never certified physically zero',
                'script_sha256':digest(Path(__file__)),
                'helper_sha256':digest(ROOT/'src/transcif/research/proxy_recovery.py')}
    for source in sources:
        if digest(Path(source['path']))!=source['sha256']:raise RuntimeError('native source changed')
    new.to_csv(out/'LGEE_hourly.csv',index=False)
    provenance['repaired_proxy_sha256']=digest(out/'LGEE_hourly.csv')
    (out/'audit.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({k:provenance[k] for k in ['old_valid_hours','new_valid_hours','added_valid_hours']}))


if __name__=='__main__':main()
