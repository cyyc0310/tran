"""Worst-case relative-error sensitivity of V74 risk-targeted certificates."""
from __future__ import annotations
import csv,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];V20=ROOT/'results/tail-count-v20'
PROTOCOL=ROOT/'docs/V75_ORDER_STATISTIC_MEASUREMENT_ROBUSTNESS_PROTOCOL.md'
OUT=ROOT/'results/v75-order-statistic-robustness-r1'
SOURCES=('BPAT','CISO','ERCO','FPL','ISNE','MISO','NYIS');TAUS=(100,300,600,900)
EPS=(F(0),F(1,1000),F(1,100),F(1,20),F(1,10));B=F(1000)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canon(x):return pd.Timestamp(x).isoformat()
def robust(hrs,tau,e):
 ef=(1000,400,800,0,0,0,0,400);out={'zero':True,'le4':True,'ge8':True}
 for start in range(0,24,6):
  part=hrs[start:start+6];g=[sum(x,F(0)) for x in part];h=[sum((v*z for v,z in zip(x,ef)),F(0)) for x in part]
  total=sum(g,F(0));H=sum(h,F(0));ordered=sorted(g);m=ordered[0];s2=sum(ordered[:2],F(0));l1=ordered[-1]
  out['zero'] &= H*(1+e)<=F(tau)*m*(1-e)
  out['le4'] &= H*(1+e)<=F(tau)*s2*(1-e)
  out['ge8'] &= H*(1-e)>F(tau)*total*(1+e)+(B-F(tau))*l1*(1+e)
 return out
def main():
 ph=sha(PROTOCOL);completed=json.loads((V20/'completed.json').read_text());freeze=json.loads((V20/'freeze.json').read_text())
 for name,digest in completed['output_hashes'].items():
  if sha(Path(name))!=digest:raise ValueError('V20 result changed: '+name)
 day=pd.read_csv(V20/'day_counts.csv');allowed=set((r,canon(d)) for r,d in day[['region','day']].drop_duplicates().itertuples(index=False,name=None))
 source=Path(freeze['source']);fuels=('coal','gas','oil','nuclear','hydro','wind','solar','other');hours={}
 for region in SOURCES:
  with (source/f'{region}_hourly.csv').open(newline='') as f:
   for row in csv.DictReader(f):
    if not ('2024-07'<=row['time']<'2025-01') or row['valid']!='True':continue
    stamp=pd.Timestamp(row['time']);key=(region,stamp.floor('D').isoformat())
    if key not in allowed:continue
    vals=[F(row['generation_'+z]) for z in fuels]
    if any(x<0 for x in vals) or sum(vals)<=0:raise ValueError('invalid source hour')
    hours.setdefault(key,[]).append((stamp,vals))
 rows=[]
 for key,obs in sorted(hours.items()):
  obs.sort(key=lambda x:x[0])
  if len(obs)!=24 or [x[0] for x in obs]!=list(pd.date_range(pd.Timestamp(key[1]),periods=24,freq='h')):raise ValueError('incomplete day')
  fuel=[x[1] for x in obs]
  for tau in TAUS:
   for eps in EPS:
    x=robust(fuel,tau,eps)
    rows.append({'region':key[0],'day':key[1],'tau':tau,'relative_error':float(eps),
      'certified_K_eq_0':bool(x['zero']),'certified_K_le_4':bool(x['le4']),'certified_K_ge_8':bool(x['ge8'])})
 d=pd.DataFrame(rows)
 if len(d)!=1000*len(TAUS)*len(EPS):raise ValueError(f'row count {len(d)}')
 OUT.mkdir(parents=True,exist_ok=False);d.to_csv(OUT/'day_certificates.csv',index=False)
 cols=['certified_K_eq_0','certified_K_le_4','certified_K_ge_8']
 region=d.groupby(['tau','relative_error','region'])[cols].mean().reset_index()
 macro=region.groupby(['tau','relative_error'])[cols].mean().reset_index()
 pooled=d.groupby(['tau','relative_error'])[cols].mean().reset_index()
 macro.to_csv(OUT/'source_macro_summary.csv',index=False);pooled.to_csv(OUT/'pooled_summary.csv',index=False)
 # Each target certificate must only be lost as error bounds widen.
 for _,g in d.groupby(['region','day','tau']):
  g=g.sort_values('relative_error')
  for c in cols:
   if g[c].astype(int).diff().fillna(0).gt(0).any():raise ValueError('certificate increased with error')
 payload={'status':'BOUNDED_RELATIVE_ERROR_SENSITIVITY_NOT_SENSOR_ACCURACY',
  'regions':list(SOURCES),'source_days':1000,'rows':len(d),'thresholds':list(TAUS),
  'relative_error_levels':[float(x) for x in EPS],'source_macro_summary':macro.to_dict(orient='records'),
  'pooled_summary':pooled.to_dict(orient='records'),
  'input_sha256':{str(p):sha(p) for p in [V20/'freeze.json',V20/'completed.json',V20/'day_counts.csv']},
  'raw_source_hashes':freeze['input_hashes'],'protocol_sha256':ph,'script_sha256':sha(Path(__file__)),
  'limitations':['Error percentages are assumed stress levels, not measured device precision.',
   'Independent worst-case box is conservative despite shared G-derived summaries.',
   'Historical source-only identification, not a forecast or China deployment test.']}
 if sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False));print(macro.to_string(index=False))
if __name__=='__main__':main()
