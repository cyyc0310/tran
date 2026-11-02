#!/usr/bin/env python3
"""Acquire NESO national half-hourly estimated actual carbon intensity history."""
from __future__ import annotations
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

BASE='https://api.carbonintensity.org.uk/intensity'

def fetch_with_retries(url: str, attempts: int = 5):
 for attempt in range(attempts):
  try:
   response=requests.get(url,timeout=40)
   if response.status_code == 429 or response.status_code >= 500:
    response.raise_for_status()
   response.raise_for_status()
   return response
  except requests.RequestException:
   if attempt + 1 == attempts:
    raise
   time.sleep(min(2 ** attempt, 16))
 raise RuntimeError(f'failed to fetch after {attempts} attempts: {url}')

def main():
 p=argparse.ArgumentParser();p.add_argument('--start',default='2022-01-01T00:00Z');p.add_argument('--end',default='2025-01-01T00:00Z');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 start=pd.Timestamp(a.start);end=pd.Timestamp(a.end)
 if start.tzinfo is None or end.tzinfo is None or end<=start: raise ValueError('UTC ISO range required')
 a.output_dir.mkdir(parents=True,exist_ok=True); records=[];requests_meta=[];cursor=start
 while cursor<end:
  stop=min(cursor+pd.Timedelta(days=30),end)
  u=f"{BASE}/{cursor.strftime('%Y-%m-%dT%H:%MZ')}/{stop.strftime('%Y-%m-%dT%H:%MZ')}"
  r=fetch_with_retries(u);j=r.json(); rows=j.get('data',[])
  if not rows: raise ValueError(f'empty NESO response {u}')
  records.extend(rows);requests_meta.append({'url':u,'status':r.status_code,'records':len(rows),'sha256':hashlib.sha256(r.content).hexdigest()})
  cursor=stop
  time.sleep(.05)
 frame=pd.DataFrame([{'from':x['from'],'to':x['to'],'actual_gco2_per_kwh':(x.get('intensity') or {}).get('actual'),'forecast_gco2_per_kwh':(x.get('intensity') or {}).get('forecast'),'index':(x.get('intensity') or {}).get('index')} for x in records])
 frame=frame.drop_duplicates(['from','to']).sort_values('from')
 start_utc=pd.Timestamp(a.start);end_utc=pd.Timestamp(a.end)
 frame=frame[(pd.to_datetime(frame['from'],utc=True)>=start_utc)&(pd.to_datetime(frame['from'],utc=True)<end_utc)].copy()
 expected=pd.date_range(start_utc,end_utc,freq='30min',inclusive='left')
 absent=expected.difference(pd.DatetimeIndex(pd.to_datetime(frame['from'],utc=True)))
 gap_blocks=[]
 if len(absent):
  gap_start=previous=absent[0]
  for stamp in absent[1:]:
   if stamp-previous!=pd.Timedelta('30min'):
    gap_blocks.append({'start':gap_start.isoformat(),'end':(previous+pd.Timedelta('30min')).isoformat(),'missing_intervals':int((previous-gap_start)/pd.Timedelta('30min'))+1})
    gap_start=stamp
   previous=stamp
  gap_blocks.append({'start':gap_start.isoformat(),'end':(previous+pd.Timedelta('30min')).isoformat(),'missing_intervals':int((previous-gap_start)/pd.Timedelta('30min'))+1})
 out=a.output_dir/'neso_national_intensity_history.csv';frame.to_csv(out,index=False)
 manifest={'provider':'NESO official Carbon Intensity API','metric':'national intensity.actual (estimated from metered generation per NESO metadata; not a direct meter of emissions)','period_start':str(frame['from'].min()),'period_end':str(frame['to'].max()),'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'requests':requests_meta,'rows':len(frame),'missing_actual':int(frame.actual_gco2_per_kwh.isna().sum()),'expected_half_hour_intervals':int(len(expected)),'missing_interval_timestamps':int(len(absent)),'missing_interval_blocks':gap_blocks,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
