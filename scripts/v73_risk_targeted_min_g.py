"""Exact V20-cohort comparison of block min(G) and sum(G^2) summaries."""
from __future__ import annotations
import csv,hashlib,json,math
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/V73_RISK_TARGETED_MIN_G_PROTOCOL.md'
V20=ROOT/'results/tail-count-v20';OUT=ROOT/'results/v73-risk-targeted-min-g-r2'
SOURCES=('BPAT','CISO','ERCO','FPL','ISNE','MISO','NYIS');TAUS=(100,300,600,900);B=F(1000)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def parse_counts(s):return tuple(int(x) for x in str(s).split('|'))
def combine(blocks):
 possible={0}
 for bs in blocks:possible={a+b for a in possible for b in bs}
 return tuple(sorted(possible))
def min_g_counts(n,total,h,m,tau):
 """Sharp K set given fixed H, sum G and block minimum m, with G_h>0."""
 if n<1 or total<=0 or m<=0 or F(n)*m>total or h<0 or h>B*total:
  raise ValueError('incompatible block summary')
 result=[]
 if h<=F(tau)*total:result.append(0)
 for k in range(1,n+1):
  lo=max(F(k)*m,(h-F(tau)*total)/(B-F(tau)))
  hi=min(total-F(n-k)*m,h/F(tau))
  # s=H/tau is an open endpoint from strict C>tau; the complement bound is closed.
  open_hi=hi==h/F(tau)
  if lo<hi or (lo==hi and not open_hi):result.append(k)
 return tuple(result)
def canonical_day(s):return pd.Timestamp(s).isoformat()
def main():
 protocol_hash=sha(PROTOCOL);completed=json.loads((V20/'completed.json').read_text())
 freeze=json.loads((V20/'freeze.json').read_text())
 if completed['region_days']!=1000 or completed['day_rows']!=12000 or completed['block_rows']!=48000:
  raise ValueError('V20 completed cohort mismatch')
 for name,digest in completed['output_hashes'].items():
  p=Path(name)
  if sha(p)!=digest:raise ValueError('V20 parent output changed: '+str(p))
 day_all=pd.read_csv(V20/'day_counts.csv');block=pd.read_csv(V20/'block_counts.csv')
 day_all['day_key']=day_all.day.map(canonical_day);block['day_key']=block.day.map(canonical_day)
 day=day_all.loc[day_all.budget.eq('T8_G2')].copy()
 # Read and validate registered source days from the frozen V20 CSV inputs.
 source=Path(freeze['source']);fuels=('coal','gas','oil','nuclear','hydro','wind','solar','other');ef=(1000,400,800,0,0,0,0,400)
 allowed=set((r,canonical_day(d)) for r,d in day[['region','day']].drop_duplicates().itertuples(index=False,name=None))
 records={}
 for region in SOURCES:
  p=source/f'{region}_hourly.csv'
  with p.open(newline='') as f:
   for row in csv.DictReader(f):
    if not ('2024-07'<=row['time']<'2025-01') or row['valid']!='True':continue
    stamp=pd.Timestamp(row['time']);dk=stamp.floor('D').isoformat();key=(region,dk)
    if key not in allowed:continue
    hour=[F(row['generation_'+fuel]) for fuel in fuels]
    if any(x<0 for x in hour) or sum(hour)<=0:raise ValueError('invalid source fuel row')
    records.setdefault(key,[]).append((stamp,hour))
 day_rows=[]
 for key,hrs in sorted(records.items()):
  hrs.sort(key=lambda x:x[0])
  if len(hrs)!=24 or [t for t,_ in hrs]!=list(pd.date_range(pd.Timestamp(key[1]),periods=24,freq='h')):
   raise ValueError('registered source day is not 24h complete')
  fuel= [x[1] for x in hrs]
  gs=[sum(v,F(0)) for v in fuel];hs=[sum((v*e for v,e in zip(row,ef)),F(0)) for row in fuel]
  for tau in TAUS:
   sets=[];actual=sum(h>F(tau)*g for h,g in zip(hs,gs))
   for start in range(0,24,6):
    g=gs[start:start+6];h=sum(hs[start:start+6],F(0));total=sum(g,F(0));m=min(g)
    sets.append(min_g_counts(6,total,h,m,tau))
   counts=combine(sets)
   rows=day_all.loc[day_all.region.eq(key[0])&day_all.day_key.eq(key[1])&day_all.threshold.eq(tau)]
   if len(rows)!=3:raise ValueError('missing V20 comparison rows')
   old={r.budget:parse_counts(r.counts) for r in rows.itertuples(index=False)}
   if actual not in counts or actual not in old['T8_G2'] or actual not in old['T8_G24']:
    raise ValueError('actual tail count not contained')
   if not set(old['T8_G24']).issubset(counts):raise ValueError('G24 sharp set not nested in min-G sharp set')
   day_rows.append({'region':key[0],'day':key[1],'threshold':tau,'actual_count':actual,
    't8_gmin_counts':'|'.join(map(str,counts)),'t8_gmin_lower':min(counts),'t8_gmin_upper':max(counts),
    't8_gmin_span':max(counts)-min(counts),'t8_gmin_cardinality':len(counts),
    't8_gmin_no_event_certified':max(counts)==0,'t8_gmin_any_event_forced':min(counts)>0,
    **{f'{b.lower()}_span':max(v)-min(v) for b,v in old.items()},
    **{f'{b.lower()}_no_event_certified':max(v)==0 for b,v in old.items()},
    **{f'{b.lower()}_any_event_forced':min(v)>0 for b,v in old.items()}})
 detail=pd.DataFrame(day_rows)
 if len(detail)!=1000*len(TAUS):raise ValueError(f'row count {len(detail)}')
 OUT.mkdir(parents=True,exist_ok=False);detail.to_csv(OUT/'paired_day_bounds.csv',index=False)
 metrics=[]
 for budget,span_col,zero_col,any_col in [('T8_Gmin','t8_gmin_span','t8_gmin_no_event_certified','t8_gmin_any_event_forced'),
  ('T8_G2','t8_g2_span','t8_g2_no_event_certified','t8_g2_any_event_forced'),
  ('T8_G24','t8_g24_span','t8_g24_no_event_certified','t8_g24_any_event_forced')]:
  region=(detail.groupby(['threshold','region']).agg(span=(span_col,'mean'),zero=(zero_col,'mean'),any_event=(any_col,'mean')).reset_index())
  macro=region.groupby('threshold').agg(mean_span=('span','mean'),zero_event_certified=('zero','mean'),
   any_event_forced=('any_event','mean')).reset_index()
  macro['budget']=budget;metrics.extend(macro.to_dict(orient='records'))
 summary=pd.DataFrame(metrics);summary.to_csv(OUT/'macro_summary.csv',index=False)
 comparisons=[]
 for tau,g in detail.groupby('threshold'):
  for other in ('t8_g2_span','t8_g24_span','t8_span'):
   delta=g['t8_gmin_span']-g[other]
   comparisons.append({'threshold':int(tau),'reference':other,'gmin_narrower_days':int((delta<0).sum()),
    'equal_days':int((delta==0).sum()),'gmin_wider_days':int((delta>0).sum()),
    'mean_paired_span_delta':float(delta.mean()),'median_paired_span_delta':float(delta.median()),
    'gmin_zero_certifications':int(g.t8_gmin_no_event_certified.sum()),
    'reference_zero_certifications':int(g[other.replace('_span','_no_event_certified')].sum()) if other!='t8_span' else None})
 pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv',index=False)
 payload={'status':'SOURCE_ONLY_HISTORICAL_IDENTIFICATION_SAME_FOUR_SCALAR_SUMMARY_BUDGET',
  'regions':list(SOURCES),'source_days':1000,'day_threshold_rows':len(detail),'thresholds':list(TAUS),
  'summaries':summary.to_dict(orient='records'),'comparisons':comparisons,
  'input_sha256':{str(p):sha(p) for p in [V20/'freeze.json',V20/'completed.json',V20/'day_counts.csv',V20/'block_counts.csv']},
  'raw_source_hashes':freeze['input_hashes'],'protocol_sha256':protocol_hash,'script_sha256':sha(Path(__file__)),
  'limitations':['T8_G2 is a necessary outer set, not proven sharp; span differences are not pure information-value estimates.',
   'Four block minima and four block second moments have different acquisition costs despite equal scalar count.',
   'Historical oracle identification only; not forecast gain, empirical coverage, or China validation.']}
 if sha(PROTOCOL)!=protocol_hash:raise RuntimeError('protocol changed during run')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
 print(summary.to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False))
if __name__=='__main__':main()
