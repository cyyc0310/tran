"""Temporal replication of V74 risk-targeted G summaries on 2025H1/H2."""
from __future__ import annotations
import csv,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import tail_count_v20 as model  # noqa:E402
ROOT=Path(__file__).resolve().parents[1];V20=ROOT/'results/tail-count-v20'
PROTOCOL=ROOT/'docs/V76_ORDER_STATISTIC_TIME_REPLICATION_PROTOCOL.md'
OUT=ROOT/'results/v76-order-statistic-time-replication-r3'
SOURCES=('BPAT','MISO','NYIS');TAUS=(100,300,600,900);B=F(1000)
PERIODS=(('2024H2','2024-07-01','2025-01-01'),('2025H1','2025-01-01','2025-07-01'),('2025H2','2025-07-01','2026-01-01'))
EF=(1000,400,800,0,0,0,0,400);FUELS=('coal','gas','oil','nuclear','hydro','wind','solar','other')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def cert(hrs,tau):
 gs=[sum(x,F(0)) for x in hrs];hs=[sum((v*e for v,e in zip(x,EF)),F(0)) for x in hrs]
 zero=True;le4=True;ge8=True;sets={'T8':[],'T8_G2':[],'T8_G24':[]}
 for start in range(0,24,6):
  g=gs[start:start+6];h=sum(hs[start:start+6],F(0));total=sum(g,F(0));order=sorted(g)
  zero &= h<=F(tau)*order[0]
  le4 &= h<=F(tau)*sum(order[:2],F(0))
  ge8 &= h>F(tau)*total+(B-F(tau))*order[-1]
  sq=sum((x*x for x in g),F(0));actual=sum(hv>F(tau)*gv for hv,gv in zip(hs[start:start+6],gs[start:start+6]))
  cs=(model.coarse_counts(6,total,h,F(tau)),model.moment_counts(6,total,sq,h,F(tau)),model.full_g_counts(g,h,F(tau)))
  if not set(cs[2]).issubset(cs[1]) or not set(cs[1]).issubset(cs[0]):raise ValueError('block nesting failed')
  for name,c in zip(('T8','T8_G2','T8_G24'),cs):
   if actual not in c:raise ValueError('actual block K outside feasible set')
   sets[name].append(c)
 daily={k:model.combine_blocks(v) for k,v in sets.items()}
 if not set(daily['T8_G24']).issubset(daily['T8_G2']) or not set(daily['T8_G2']).issubset(daily['T8']):raise ValueError('daily nesting failed')
 actual=sum(h>F(tau)*g for h,g in zip(hs,gs))
 if any(actual not in x for x in daily.values()):raise ValueError('actual K outside daily set')
 return {'actual_K':actual,'minG_K0':bool(zero),'S2_K_le4':bool(le4),'L1_K_ge8':bool(ge8),
  **{f'{k}_lower':min(v) for k,v in daily.items()},**{f'{k}_upper':max(v) for k,v in daily.items()},
  **{f'{k}_span':max(v)-min(v) for k,v in daily.items()},
  **{f'{k}_K0':max(v)==0 for k,v in daily.items()},
  **{f'{k}_Kle4':max(v)<=4 for k,v in daily.items()},
  **{f'{k}_Kge8':min(v)>=8 for k,v in daily.items()}}
def main():
 ph=sha(PROTOCOL);frozen=json.loads((V20/'freeze.json').read_text());completed=json.loads((V20/'completed.json').read_text())
 for name,digest in completed['output_hashes'].items():
  if sha(Path(name))!=digest:raise ValueError('V20 completed output hash mismatch')
 source=Path(frozen['source']);rows=[]
 for label,start,stop in PERIODS:
  for region in SOURCES:
   with (source/f'{region}_hourly.csv').open(newline='') as f:
    grouped={}
    for row in csv.DictReader(f):
     if not(start<=row['time']<stop) or row['valid']!='True':continue
     ts=pd.Timestamp(row['time']);day=ts.floor('D').isoformat()
     vals=[F(row['generation_'+z]) for z in FUELS]
     if any(x<0 for x in vals) or sum(vals)<=0:raise ValueError('invalid generation row')
     grouped.setdefault(day,[]).append((ts,vals))
   for day,hh in sorted(grouped.items()):
    hh.sort(key=lambda z:z[0])
    if len(hh)!=24 or [x[0] for x in hh]!=list(pd.date_range(pd.Timestamp(day),periods=24,freq='h')):continue
    for tau in TAUS:
     rows.append({'period':label,'region':region,'day':day,'tau':tau,**cert([x[1] for x in hh],tau)})
 d=pd.DataFrame(rows)
 if d.empty:raise ValueError('no complete-day data')
 # Enforce the frozen 80% complete-day floor per matched source and half-year.
 valid_days=d.groupby(['period','region']).day.nunique()
 for label,start,stop in PERIODS:
  expected=len(pd.date_range(pd.Timestamp(start),pd.Timestamp(stop)-pd.Timedelta(days=1),freq='D'))
  for region in SOURCES:
   if valid_days.loc[(label,region)]<0.8*expected:raise ValueError(f'quality floor failed {label}/{region}')
 OUT.mkdir(parents=True,exist_ok=False);d.to_csv(OUT/'day_certificates.csv',index=False)
 metrics=['minG_K0','T8_G2_K0','T8_G24_K0','S2_K_le4','T8_G2_Kle4','T8_G24_Kle4','L1_K_ge8','T8_G2_Kge8','T8_G24_Kge8']
 reg=d.groupby(['period','tau','region'])[metrics].mean().reset_index();reg.to_csv(OUT/'region_summary.csv',index=False)
 macro=reg.groupby(['period','tau'])[metrics].mean().reset_index();macro=macro.merge(d.groupby(['period','tau']).size().rename('day_threshold_rows').reset_index(),on=['period','tau'])
 macro.to_csv(OUT/'macro_summary.csv',index=False)
 bad=int(((d.minG_K0)&(d.actual_K!=0)).sum()+((d.S2_K_le4)&(d.actual_K>4)).sum()+((d.L1_K_ge8)&(d.actual_K<8)).sum())
 if bad:raise ValueError(f'false certificates {bad}')
 payload={'status':'SAME_REGION_TEMPORAL_REPLICATION_NOT_UNSEEN_REGION_CONFIRMATION','periods':[x[0] for x in PERIODS],
  'regions':list(SOURCES),'rows':len(d),'valid_days_by_period_region':d.groupby(['period','region']).day.nunique().unstack().to_dict('index'),
  'macro_summary':macro.to_dict(orient='records'),'false_certificates':bad,
  'input_sha256':{str(p):sha(p) for p in [V20/'freeze.json',V20/'completed.json']},
  'raw_source_hashes':frozen['input_hashes'],'protocol_sha256':ph,'script_sha256':sha(Path(__file__)),
  'limitations':['Temporal replication only; the same seven regions were previously opened.',
   'T8_G2 is an outer set, not sharp.','Historical oracle identification, not forecast coverage or China deployment.']}
 if sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed during run')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False));print(macro.to_string(index=False));print('rows',len(d),'false_certificates',bad)
if __name__=='__main__':main()
