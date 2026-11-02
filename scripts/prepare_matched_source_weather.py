"""Match expanded source observed weather to the same issued-weather source days."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from collect_weather_transfer_pilot import ROOT, digest
from run_resource_experiment import validate_observed_source_rows
from transcif.research import multicountry as mc


def main() -> None:
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();parent=args.source_training.resolve()
    manifest=json.loads((parent/'manifest.json').read_text())
    if not manifest['complete'] or manifest['role']!='source_observed_weather_training':
        raise ValueError('proper observed source history required')
    seals={**manifest['source_files'],**{str(parent/name):h for name,h in manifest['files'].items()}}
    for p in [Path(__file__),parent/'manifest.json',ROOT/'results/weather-transfer-seasonal-r1/cohort.csv']:
        seals[str(p.resolve())]=digest(p)
    for name,expected in seals.items():
        if digest(Path(name))!=expected:raise ValueError('matched source input changed')
    observed=pd.read_csv(parent/'cohort.csv',parse_dates=['clock','valid_at'])
    issued=pd.read_csv(ROOT/'results/weather-transfer-seasonal-r1/cohort.csv',parse_dates=['clock','valid_at'])
    issued=issued.loc[issued.split.eq('train')&issued.clock.dt.month.le(9)]
    keys=['country','region','valid_at','label_row']
    matched=issued[keys].merge(observed,on=keys,how='left',validate='one_to_one')
    if len(matched)!=len(issued) or matched.clock.isna().any():
        raise ValueError('observed weather cannot exactly match all issued source hours')
    raw,_=mc.load_countries(ROOT);validate_observed_source_rows(matched,raw)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    matched.to_csv(out/'cohort.csv',index=False)
    for name,expected in seals.items():
        if digest(Path(name))!=expected:raise ValueError('matched source input/producer changed')
    (out/'manifest.json').write_text(json.dumps(dict(complete=True,role='source_observed_weather_training',
        files={'cohort.csv':digest(out/'cohort.csv')},source_files=seals,
        eligible_for_issued_public=False,available_at=None,source_hours=len(matched),source_days=len(matched)//24,
        information='observed source ERA5 on exactly the same Jan–Sep pilot source days as issued-NWP control',
        target_period_weather_used=False),indent=2)+'\n')
    print('COMPLETE',len(matched),'matched observed source hours',flush=True)


if __name__=='__main__':main()
