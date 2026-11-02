"""Derive a fixed wind-power-curve reference from already sealed climate records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.data.weather_reference import YEARS, acquire, digest, parameters
from transcif.research.resource_controls import wind_curve


def main() -> None:
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--climate-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cache=args.climate_cache.resolve()
    manifest=json.loads((cache/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if digest(cache/name)!=expected:raise ValueError('historical reference seal changed')
    source={str(cache/name):value for name,value in manifest['files'].items()}
    source.update(manifest['code'])
    source[str(cache/'manifest.json')]=digest(cache/'manifest.json')
    for p in [Path(__file__),Path(__file__).resolve().parents[1]/'src/transcif/research/resource_controls.py']:
        source[str(p.resolve())]=digest(p)
    for name,expected in source.items():
        if digest(Path(name))!=expected:raise ValueError('reference input/producer differs')
    plan=json.loads((cache/'plan.json').read_text());rows=[]
    for region,(lat,lon) in plan['points'].items():
        for year in YEARS:
            directory=cache/f'{region}/{year}'
            if not directory.exists():raise ValueError('offline reference must already exist')
            frame,_=acquire(directory,parameters(lat,lon,year))
            local=frame.instant_at.dt.tz_convert(plan['zones'][region])
            curve=wind_curve(frame.wind_speed_100m.to_numpy())
            for month in range(1,13):
                selected=local.dt.year.eq(year)&local.dt.month.eq(month)
                start=pd.Timestamp(year=year,month=month,day=1,tz=plan['zones'][region])
                expected=pd.date_range(start,start+pd.offsets.MonthBegin(1),freq='h',inclusive='left').tz_convert('UTC')
                np.testing.assert_array_equal(pd.DatetimeIndex(frame.loc[selected,'instant_at']).asi8,expected.asi8)
                rows.append(dict(region=region,year=year,month=month,wind_curve_resource=float(curve[selected].mean())))
    annual=pd.DataFrame(rows)
    curve=annual.groupby(['region','month']).wind_curve_resource.mean().reset_index()
    original=pd.read_csv(cache/'climatology.csv')
    combined=original.merge(curve,on=['region','month'],validate='one_to_one')
    if len(combined)!=len(original):raise ValueError('curve reference rows differ')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    annual.to_csv(out/'annual_curve.csv',index=False);combined.to_csv(out/'climatology.csv',index=False)
    for name,expected in source.items():
        if digest(Path(name))!=expected:raise ValueError('reference input/producer changed during aggregation')
    (out/'manifest.json').write_text(json.dumps(dict(complete=True,source_files=source,
        files={name:digest(out/name) for name in ['annual_curve.csv','climatology.csv']},
        curve=dict(cutin=3.,rated=12.,cutout=25.,height=100),target_tuning=False,
        eligible_for_issued_public=False,available_at=None),indent=2)+'\n')


if __name__=='__main__':main()
