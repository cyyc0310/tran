"""Four-scalar, risk-targeted order-statistic certificates on the V20 cohort."""
from __future__ import annotations
import csv,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];V20=ROOT/'results/tail-count-v20'
PROTOCOL=ROOT/'docs/V74_ORDER_STATISTIC_RISK_MEASUREMENT_PROTOCOL.md'
OUT=ROOT/'results/v74-order-statistic-risk-measurements-r1'
SOURCES=('BPAT','CISO','ERCO','FPL','ISNE','MISO','NYIS');TAUS=(100,300,600,900);B=F(1000)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canon(x):return pd.Timestamp(x).isoformat()
def parse_counts(x):return tuple(int(v) for v in str(x).split('|'))
def cert(hrs,tau):
 gs=[sum(x,F(0)) for x in hrs];hs=[sum((v*e for v,e in zip(x,(1000,400,800,0,0,0,0,400))),F(0)) for x in hrs]
 zeros=[];le1=[];ge2=[]
 for start in range(0,24,6):
  g=gs[start:start+6];h=sum(hs[start:start+6],F(0));total=sum(g,F(0));ordered=sorted(g)
  zeros.append(h<=F(tau)*ordered[0])
  le1.append(h<=F(tau)*sum(ordered[:2],F(0)))
  ge2.append(h>F(tau)*total+(B-F(tau))*ordered[-1])
 return all(zeros),all(le1),all(ge2),sum(h>F(tau)*g for h,g in zip(hs,gs))
def main():
 ph=sha(PROTOCOL);completed=json.loads((V20/'completed.json').read_text());freeze=json.loads((V20/'freeze.json').read_text())
 if completed['region_days']!=1000:raise ValueError('V20 cohort mismatch')
 for name,digest in completed['output_hashes'].items():
  if sha(Path(name))!=digest:raise ValueError('V20 output hash mismatch '+name)
 day=pd.read_csv(V20/'day_counts.csv');day['day_key']=day.day.map(canon)
 source=Path(freeze['source']);fuels=('coal','gas','oil','nuclear','hydro','wind','solar','other')
 allowed=set((r,canon(d)) for r,d in day[['region','day']].drop_duplicates().itertuples(index=False,name=None));hours={}
 for region in SOURCES:
  with (source/f'{region}_hourly.csv').open(newline='') as f:
   for row in csv.DictReader(f):
    if not ('2024-07'<=row['time']<'2025-01') or row['valid']!='True':continue
    stamp=pd.Timestamp(row['time']);key=(region,stamp.floor('D').isoformat())
    if key not in allowed:continue
    vals=[F(row['generation_'+z]) for z in fuels]
    if any(v<0 for v in vals) or sum(vals)<=0:raise ValueError('bad generation row')
    hours.setdefault(key,[]).append((stamp,vals))
 rows=[]
 for key,values in sorted(hours.items()):
  values.sort(key=lambda z:z[0])
  if len(values)!=24 or [z[0] for z in values]!=list(pd.date_range(pd.Timestamp(key[1]),periods=24,freq='h')):
   raise ValueError('incomplete registered V20 day')
  for tau in TAUS:
   z,l,g,actual=cert([x[1] for x in values],tau)
   ref=day.loc[day.region.eq(key[0])&day.day_key.eq(key[1])&day.threshold.eq(tau)]
   if len(ref)!=3:raise ValueError('missing V20 budget rows')
   sets={x.budget:parse_counts(x.counts) for x in ref.itertuples(index=False)}
   if any(actual not in x for x in sets.values()):raise ValueError('actual K omitted by parent set')
   criteria={'K_eq_0':z,'K_le_4':l,'K_ge_8':g}
   certified={
    'T8_Gmin_K_eq_0':z,
    'T8_G2_K_eq_0':max(sets['T8_G2'])==0,
    'T8_G24_K_eq_0':max(sets['T8_G24'])==0,
    'T8_S2_K_le_4':l,
    'T8_G2_K_le_4':max(sets['T8_G2'])<=4,
    'T8_G24_K_le_4':max(sets['T8_G24'])<=4,
    'T8_L1_K_ge_8':g,
    'T8_G2_K_ge_8':min(sets['T8_G2'])>=8,
    'T8_G24_K_ge_8':min(sets['T8_G24'])>=8}
   rows.append({'region':key[0],'day':key[1],'tau':tau,'actual_K':actual,**criteria,**certified})
 d=pd.DataFrame(rows)
 if len(d)!=4000:raise ValueError(f'row count {len(d)}')
 OUT.mkdir(parents=True,exist_ok=False);d.to_csv(OUT/'day_certificates.csv',index=False)
 certcols=[c for c in d if c.startswith('T8_')]
 region=d.groupby(['tau','region'])[[*criteria.keys(),*certcols]].mean().reset_index()
 region.to_csv(OUT/'region_summary.csv',index=False)
 macro=region.groupby('tau')[[*criteria.keys(),*certcols]].mean().reset_index();macro.to_csv(OUT/'macro_summary.csv',index=False)
 # Check each affirmative targeted certificate against the exact realized event count.
 false_cert=int(((d.K_eq_0)&(d.actual_K!=0)).sum()+((d.K_le_4)&(d.actual_K>4)).sum()+((d.K_ge_8)&(d.actual_K<8)).sum())
 if false_cert:raise ValueError(f'false logical certificates: {false_cert}')
 payload={'status':'SOURCE_ONLY_RISK_TARGETED_IDENTIFICATION_FOUR_SCALAR_SUMMARIES',
  'regions':list(SOURCES),'source_days':1000,'rows':len(d),'thresholds':list(TAUS),
  'targeted_certificate_summary':macro.to_dict(orient='records'),'false_certificates':false_cert,
  'input_sha256':{str(p):sha(p) for p in [V20/'freeze.json',V20/'completed.json',V20/'day_counts.csv']},
  'raw_source_hashes':freeze['input_hashes'],'protocol_sha256':ph,'script_sha256':sha(Path(__file__)),
  'limitations':['Four scalar count is not equal measurement cost.','T8_G2 is a necessary outer set, not sharp.',
   'Historical oracle set identification only, no forecast or China validation.']}
 if sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed during execution')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
 print(macro.to_string(index=False));print('false_certificates=',false_cert)
if __name__=='__main__':main()
