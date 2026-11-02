"""Prepare foreign source-2022 observed weather, never target-period observations."""
from __future__ import annotations

import argparse
from http.client import IncompleteRead, RemoteDisconnected
import json
from pathlib import Path
import ssl
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from collect_weather_transfer_pilot import ROOT, COORDINATES
from run_resource_experiment import validate_observed_source_rows
from transcif.data.weather_reference import API, FIELDS, UNITS, digest
from transcif.research import multicountry as mc


def source_parameters(latitude: float, longitude: float) -> dict:
    """Fixed foreign source observation period with time-zone/solar padding."""
    if not np.isfinite([latitude,longitude]).all() or not (-90<=latitude<=90 and -180<=longitude<=180):
        raise ValueError('invalid source coordinates')
    return dict(latitude=latitude, longitude=longitude, start_date='2021-12-31',end_date='2022-10-02',
        hourly=','.join(FIELDS),models='era5',timezone='UTC',timeformat='unixtime',
        wind_speed_unit='ms',temperature_unit='celsius',elevation='nan',cell_selection='nearest')


def decode_source(payload: bytes, params: dict) -> pd.DataFrame:
    if params!=source_parameters(params['latitude'],params['longitude']):
        raise ValueError('source observations outside fixed January–September 2022 contract')
    record=json.loads(payload)
    if record.get('utc_offset_seconds')!=0 or record.get('hourly_units')!=UNITS:
        raise ValueError('source weather units or UTC clock differ')
    point=np.array([record.get('latitude'),record.get('longitude')],float)
    if not np.isfinite(point).all() or np.max(np.abs(point-[params['latitude'],params['longitude']]))>.26:
        raise ValueError('source weather grid point differs')
    expected=pd.date_range('2021-12-31','2022-10-03',freq='h',inclusive='left',tz='UTC')
    frame=pd.DataFrame(record['hourly'])
    if not np.array_equal(frame.time.to_numpy(float),expected.asi8//10**9):
        raise ValueError('source weather hours missing or unordered')
    values=frame[list(FIELDS)].to_numpy(float)
    if not np.isfinite(values).all() or (values[:,1:]<0).any():
        raise ValueError('source weather missing or negative resources')
    out=pd.DataFrame(dict(valid_at=expected,temperature_c=values[:,0],wind_speed_ms=values[:,1]))
    solar=pd.DataFrame(dict(valid_at=expected-pd.Timedelta(hours=1),shortwave_w_per_m2=values[:,2]))
    return out.merge(solar,on='valid_at',how='inner',validate='one_to_one')


def acquire_source(folder: Path, params: dict) -> pd.DataFrame:
    """Immutable request/response checkpoints, finite retries for transport errors."""
    url=API+'?'+urlencode(params)
    if folder.exists():
        receipt=json.loads((folder/'receipt.json').read_text())
        if (receipt.get('parameters')!=params or receipt.get('url')!=url
            or receipt.get('role')!='source_observed_weather_training'
            or receipt.get('eligible_for_issued_public') is not False
            or 'available_at' not in receipt or receipt['available_at'] is not None
            or receipt.get('units')!=UNITS
            or receipt.get('vintage')!='revised ERA5 archive, not original public-release certification'
            or digest(folder/'response.json')!=receipt.get('sha256')):
            raise ValueError('source weather checkpoint differs')
        payload=(folder/'response.json').read_bytes();record=json.loads(payload)
        if (receipt.get('returned_point')!={k:record[k] for k in ('latitude','longitude','elevation')}
            or pd.Timestamp(receipt['retrieved_at']).tzinfo is None or 'headers' not in receipt):
            raise ValueError('source receipt metadata differs')
        return decode_source(payload,params)
    for attempt in range(3):
        try:
            with urlopen(url,timeout=40) as response:
                payload=response.read(2_000_001);headers=dict(response.headers.items())
            break
        except HTTPError as error:
            if error.code not in (429,500,502,503,504) or attempt==2:raise
            time.sleep(2*(attempt+1))
        except (URLError,TimeoutError,IncompleteRead,RemoteDisconnected,ConnectionResetError,ssl.SSLError):
            if attempt==2:raise
            time.sleep(2*(attempt+1))
    if len(payload)>2_000_000:raise ValueError('single source point response too large')
    frame=decode_source(payload,params);record=json.loads(payload)
    folder.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.source-weather-',dir=folder.parent) as temporary:
        stage=Path(temporary)/'complete';stage.mkdir();(stage/'response.json').write_bytes(payload)
        receipt=dict(parameters=params,url=url,sha256=digest(stage/'response.json'),headers=headers,
            retrieved_at=pd.Timestamp.now(tz='UTC').isoformat(),units=record['hourly_units'],
            returned_point={k:record[k] for k in ('latitude','longitude','elevation')},
            role='source_observed_weather_training',eligible_for_issued_public=False,available_at=None,
            vintage='revised ERA5 archive, not original public-release certification')
        (stage/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');stage.rename(folder)
    return frame


def source_cohort(raw: dict, weather: dict) -> tuple[pd.DataFrame,list]:
    frames,audits=[],[]
    for region,source in raw.items():
        selected=source.loc[source.clock.dt.year.eq(2022)&source.clock.dt.month.le(9)].copy()
        selected['label_row']=selected.index;selected=selected.reset_index(drop=True)
        public=mc.public_features(mc.public_months(source),selected.clock)
        for column in mc.HISTORY:selected[column]=public[column].to_numpy()
        selected['valid_at']=pd.to_datetime(selected.time,utc=True)
        selected=selected.merge(weather[region],on='valid_at',how='left',validate='one_to_one')
        selected['region']=region;selected['day']=selected.clock.dt.floor('D')
        for day,group in selected.groupby('day',sort=True):
            group=group.sort_values('clock')
            record=dict(region=region,day=str(day),hours=len(group),eligible=False)
            if len(group)!=24 or group.clock.dt.hour.tolist()!=list(range(24)):
                record['reason']='incomplete or DST-variable local day'
            elif not np.isfinite(group[list(mc.HISTORY)+['temperature_c','wind_speed_ms','shortwave_w_per_m2']].to_numpy(float)).all():
                record['reason']='public past month or weather input unavailable'
            else:
                group=group.copy();group['origin']=str(group.valid_at.iloc[0])
                group['forecast_origin_local']=str(day);group['latest_allowed_period']=str(day.to_period('M')-2)
                group['split']='train';group['weather_kind']='source_historical_reanalysis'
                columns=['clock',*mc.HISTORY,'valid_at','country','region','label_row','origin',
                    'forecast_origin_local','latest_allowed_period','split','weather_kind',
                    'temperature_c','wind_speed_ms','shortwave_w_per_m2']
                frames.append(group[columns]);record['eligible']=True
            audits.append(record)
    result=pd.concat(frames,ignore_index=True)
    validate_observed_source_rows(result,raw)
    return result,audits


def main() -> None:
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    inherited=json.loads((ROOT/'results/resource-amplitudes-r1/input_code_seal.json').read_text())
    paths=[Path(__file__),ROOT/'scripts/run_resource_experiment.py',ROOT/'tests/test_source_weather_history.py']
    seals={**inherited,**{str(p.resolve()):digest(p) for p in paths}}
    for name,expected in seals.items():
        if digest(Path(name))!=expected:raise ValueError('source producer/input changed')
    raw,_=mc.load_countries(ROOT);weather={}
    plan=dict(regions=sorted(raw),period='2022-01-01 through 2022-09-30 local dates',
        role='source_observed_weather_training',target_inference='unchanged pre-issued GFS, no actual target-period weather',
        earliest_source_calibration_origin='2023-01-15T00:00:00Z',
        availability='revised archive; original ERA5 vintage not certified',
        fold_rule='only countries in proper source tuple may enter fitting; target country excluded')
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
        raise ValueError('existing source history plan differs')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for region in raw:
        lat,lon=COORDINATES[region]
        weather[region]=acquire_source(out/'weather'/region,source_parameters(lat,lon))
        print(region,len(weather[region]),'source historical weather hours verified',flush=True)
    cohort,audit=source_cohort(raw,weather)
    cohort.to_csv(out/'cohort.csv',index=False);(out/'cohort_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    for name,expected in seals.items():
        if digest(Path(name))!=expected:raise ValueError('source input/producer changed during preparation')
    files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (out/'manifest.json').write_text(json.dumps(dict(complete=True,role='source_observed_weather_training',
        files=files,source_files=seals,eligible_for_issued_public=False,available_at=None,
        source_hours=len(cohort),source_days=len(cohort)//24,
        country_hours=cohort.groupby('country').size().to_dict(),region_hours=cohort.groupby('region').size().to_dict(),
        last_observation=str(cohort.valid_at.max()),target_labels_used_for_preparation=False,
        note='Source observed weather is additional paired training information; not an issued forecast, and not an unknown-country test.'),indent=2)+'\n')
    print('COMPLETE',len(cohort),'source hours',len(cohort)//24,'source days',flush=True)


if __name__=='__main__':main()
