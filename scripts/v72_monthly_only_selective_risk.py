"""Selective decision frontier implied by exact monthly-only sharp K bounds."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'results/v66-monthly-margin-tail-identifiability-r2/province_month_tail_bounds.csv'
PROTOCOL=ROOT/'docs/V72_MONTHLY_ONLY_SELECTIVE_RISK_PROTOCOL.md'
OUT=ROOT/'results/v72-monthly-only-selective-risk-r1'
RATES=(.01,.05,.10,.25,.50,.90,1.0)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def summarize(frame):
 out=[]
 for (tau,goal),g in frame.groupby(['tau_g_kwh','decision_goal'],sort=True):
  n=len(g);yes=int(g.certified_event.sum());no=int(g.certified_no_event.sum());abst=int(g.abstain.sum())
  out.append({'tau_g_kwh':int(tau),'decision_goal':goal,'province_months':n,
   'certified_event':yes,'certified_no_event':no,'abstain':abst,
   'decision_coverage':(yes+no)/n,'abstention_rate':abst/n,
   'mean_sharp_fraction_width':float((g.fraction_max-g.fraction_min).mean()),
   'median_sharp_fraction_width':float((g.fraction_max-g.fraction_min).median()),
   'mean_required_hours':float(g.required_hours.mean())})
 return pd.DataFrame(out)
def main():
 protocol_hash=sha(PROTOCOL);source_hash=sha(INPUT)
 d=pd.read_csv(INPUT)
 required={'province','month','hours','tau_g_kwh','sharp_k_min','sharp_k_max','fraction_min','fraction_max'}
 if required-set(d.columns):raise ValueError(f'missing columns: {required-set(d.columns)}')
 if d[['province','month','tau_g_kwh']].duplicated().any():raise ValueError('duplicate province-month-threshold rows')
 rows=[]
 for x in d.itertuples(index=False):
  for goal,r in [('any_event',None),*((f'frequency_ge_{int(rate*100):02d}pct',rate) for rate in RATES)]:
   kreq=1 if r is None else math.ceil(r*x.hours-1e-12)
   yes=x.sharp_k_min>=kreq; no=x.sharp_k_max<kreq
   if yes and no:raise AssertionError('inconsistent sharp set decision')
   rows.append({'province':x.province,'month':x.month,'tau_g_kwh':int(x.tau_g_kwh),'decision_goal':goal,
    'required_hours':kreq,'hours':int(x.hours),'sharp_k_min':int(x.sharp_k_min),'sharp_k_max':int(x.sharp_k_max),
    'fraction_min':float(x.fraction_min),'fraction_max':float(x.fraction_max),
    'certified_event':bool(yes),'certified_no_event':bool(no),'abstain':not(yes or no)})
 detail=pd.DataFrame(rows);summary=summarize(detail)
 OUT.mkdir(parents=True,exist_ok=False)
 detail.to_csv(OUT/'province_month_decision_frontier.csv',index=False)
 summary.to_csv(OUT/'decision_frontier_summary.csv',index=False)
 payload={'status':'POSTHOC_SHARP_SET_SELECTIVE_DECISION_FRONTIER_NOT_FORECAST',
  'province_months':int(d[['province','month']].drop_duplicates().shape[0]),'provinces':sorted(d.province.unique()),
  'thresholds_g_kwh':sorted(int(x) for x in d.tau_g_kwh.unique()),
  'decision_goals':['any_event',*[f'frequency_ge_{int(r*100):02d}pct' for r in RATES]],
  'input_sha256':source_hash,'protocol_sha256':protocol_hash,'script_sha256':sha(Path(__file__)),
  'summary':summary.to_dict(orient='records'),
  'limitations':['Retrospective monthly Ember fuel margins with illustrative fixed EFs.',
   'Logical decisions are conditional on the V66 compatibility assumptions; no hourly labels or empirical error rates.',
   'Province-month rows are descriptive, not independent samples.']}
 if sha(PROTOCOL)!=protocol_hash:raise RuntimeError('protocol changed during execution')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
 print(summary.to_string(index=False))
if __name__=='__main__':main()
