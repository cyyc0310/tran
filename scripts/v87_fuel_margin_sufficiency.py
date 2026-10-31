"""V87: verify target-aware fuel-margin compression against V66 bounds."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'results/v66-monthly-margin-tail-identifiability-r2/province_month_tail_bounds.csv'
PROTOCOL=ROOT/'docs/V87_FUEL_MARGIN_SUFFICIENCY_PROTOCOL.md'
OUT=ROOT/'results/v87-fuel-margin-sufficiency-r1'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def derive(row):
 t=int(row.hours);p=int(row.positive_ef_support)>0;n=int(row.negative_ef_support)>0;z=int(row.neutral_ef_support)>0
 sign=row.monthly_excess_sign
 if not p:return (0,0,'no_positive_support')
 if n:
  if sign=='positive':return (1,t,'mixed_positive_excess')
  return (0,t-1,'mixed_nonpositive_excess')
 if z:return (1,t,'positive_neutral_support')
 return (t,t,'positive_only_support')

def main():
 protocol_hash=sha(PROTOCOL);d=pd.read_csv(INPUT)
 required={'province','month','hours','tau_g_kwh','sharp_k_min','sharp_k_max','sharp_set','positive_ef_support','negative_ef_support','neutral_ef_support','monthly_excess_sign'}
 if required-set(d):raise ValueError(f'missing {required-set(d)}')
 rows=[]
 for r in d.itertuples(index=False):
  lo,hi,case=derive(r)
  label={'no_positive_support':'K=0','mixed_positive_excess':'K=1..T',
         'mixed_nonpositive_excess':'K=0..T-1','positive_neutral_support':'K=1..T',
         'positive_only_support':'K=T'}[case]
  ok=(lo==int(r.sharp_k_min) and hi==int(r.sharp_k_max) and label==r.sharp_set)
  rows.append({'province':r.province,'month':r.month,'tau_g_kwh':int(r.tau_g_kwh),'case':case,'reconstructed_set':label,'v66_set':r.sharp_set,'matches':ok})
 out=pd.DataFrame(rows)
 if not out.matches.all():raise AssertionError(out.loc[~out.matches].head().to_dict('records'))
 OUT.mkdir(parents=True,exist_ok=False);out.to_csv(OUT/'rowwise_equivalence.csv',index=False)
 summary=(out.groupby(['tau_g_kwh','case']).agg(rows=('matches','size'),matches=('matches','sum')).reset_index())
 summary.to_csv(OUT/'case_equivalence.csv',index=False)
 payload={'status':'EXACT_COMPRESSION_VERIFIED_ON_V66_NOT_NEW_HOURLY_EVIDENCE','province_month_threshold_rows':len(out),'province_months':int(out[['province','month']].drop_duplicates().shape[0]),'mismatch_rows':int((~out.matches).sum()),'by_threshold_case':summary.to_dict('records'),'input_sha256':sha(INPUT),'protocol_sha256':protocol_hash,'script_sha256':sha(Path(__file__)),'limitations':['Reconstruction uses V66 exact support/sign diagnostics, not raw-hour observations.','The theorem assumes no hourly capacities, ramping, demand, or positive minimum-output constraints.','This is a result about information sufficiency under the declared idealized allocation model, not forecast skill or a new hourly method.']}
 if sha(PROTOCOL)!=protocol_hash:raise RuntimeError('protocol changed during run')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 print(summary.to_string(index=False));print(f'rows={len(out)} mismatches={(~out.matches).sum()}')
if __name__=='__main__':main()
