"""V86: same-sample monthly to daily temporal-information bridge."""
from __future__ import annotations
import hashlib, json, sys
from fractions import Fraction as F
from math import ceil
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import v54_unseen_local_ef_confirmation as source_data  # noqa:E402
sys.path.insert(0,str(ROOT))
from scripts import tail_count_v20 as model  # noqa:E402
PROTOCOL=ROOT/'docs/V86_MONTHLY_TO_DAILY_INFO_BRIDGE_PROTOCOL.md'
OUT=ROOT/'results/v86-monthly-daily-info-bridge-r2'
CENSUS=ROOT/'results/eia-ba-quality-census-v52-r1/quality_census.json'
REGIONS=('AVA','BANC','BPAT','CHPD','DOPD','EPE','FMPP','FPL','GCPD','GRID','GVL','IID','JEA','LDWP','MISO','NWMT','NYIS','PSEI','SC','SCEG','SCL','SEC','TEC','TIDC','TPWR')
EF=(1000,400,800,0,0,0,0,400)
TAUS=(100,300);FREQUENCIES=(.01,.05,.10)
START=pd.Timestamp('2024-07-01',tz='UTC');STOP=pd.Timestamp('2026-01-01',tz='UTC')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def choose(counts,actual,k):
 if not counts or actual not in counts:raise ValueError('observed count not in identified/outer set')
 lo,hi=min(counts),max(counts)
 pos=lo>=k;neg=hi<k;abstain=not pos and not neg
 if (pos and actual<k) or (neg and actual>=k):raise ValueError('incorrect monthly decision')
 return {'lower':lo,'upper':hi,'positive':pos,'negative':neg,'abstain':abstain}

def main():
 census=json.loads(CENSUS.read_text())
 if tuple(census['quality_qualified_regions_alphabetical'])!=REGIONS:raise ValueError('census roster mismatch')
 source_data.ALL_REGIONS=tuple(dict.fromkeys(source_data.ALL_REGIONS+REGIONS))
 hourly,monthly,audits,input_hashes=source_data.prepare_regions()
 monthly=monthly[monthly.region.isin(REGIONS)].copy()
 rows=[];quality=[]
 cols=[f'generation_{f}' for f in source_data.FUELS]
 for region in REGIONS:
  frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
  frame=frame.loc[times.ge(START)&times.lt(STOP)&frame.valid.astype(bool)].copy()
  frame['time']=pd.to_datetime(frame.time,utc=True);frame['day']=frame.time.dt.floor('D');frame['month']=pd.to_datetime(frame.time.dt.strftime('%Y-%m-01'),utc=True)
  for month,mg in frame.groupby('month',sort=True):
   expected=month.days_in_month
   valid_days=[]
   for day,dg in mg.groupby('day',sort=True):
    dg=dg.sort_values('time')
    if len(dg)==24 and list(dg.time)==list(pd.date_range(day,periods=24,freq='h')):valid_days.append((day,dg))
   quality.append({'region':region,'month':month.isoformat(),'complete_days':len(valid_days),'expected_days':expected,'coverage':len(valid_days)/expected})
   if not valid_days:continue
   mr=monthly[(monthly.region==region)&(monthly.period_start==month)]
   if len(mr)>1:raise ValueError('duplicate monthly fuel support')
   if len(mr)==0:M=F(1000)
   else:
    active=[i for i,f in enumerate(source_data.FUELS) if float(mr.iloc[0][f'fuelshare_{f}'])>0]
    M=F(max(EF[i] for i in active)) if active else F(0)
   day_parts=[];all_g=[];all_pairs=[];total_h=F(0);total_g=F(0)
   for day,dg in valid_days:
    hs=[];gs=[]
    for _,row in dg.iterrows():
     vals=[F(str(float(row[c]))) for c in cols]
     if any(v<0 for v in vals) or sum(vals)<=0:raise ValueError('invalid hourly generation')
     g=sum(vals,F(0));h=sum((v*ef for v,ef in zip(vals,EF)),F(0))
     gs.append(g);hs.append(h);all_g.append(g);all_pairs.append((g,h));total_g+=g;total_h+=h
    # daily actuals and aggregate inputs cached in compact tuples
    day_parts.append((gs,sum(hs,F(0)),sum(gs,F(0)),sum((g*g for g in gs),F(0)),sum(int(h>0) for h in [])))
   n=len(all_g)
   for tau in TAUS:
    actual=sum(int(h>F(tau)*g) for g,h in all_pairs)
    monthly_set=(0,) if M<=tau else model.coarse_counts(n,total_g,total_h,F(tau),M)
    daily_coarse=[];daily_moment=[]
    for gs,Hd,Gd,Qd,_ in day_parts:
     if M<=tau:
      cset=(0,);qset=(0,)
     else:
      cset=model.coarse_counts(24,Gd,Hd,F(tau),M)
      qset=model.moment_counts(24,Gd,Qd,Hd,F(tau),M)
     daily_coarse.append(cset);daily_moment.append(qset)
    daily_set=model.combine_blocks(daily_coarse);moment_set=model.combine_blocks(daily_moment)
    if not set(moment_set).issubset(daily_set) or not set(daily_set).issubset(monthly_set):raise ValueError('information-set nesting failed')
    if actual not in monthly_set or actual not in daily_set or actual not in moment_set:raise ValueError('actual K outside information set')
    for freq in FREQUENCIES:
     required=ceil(freq*n)
     methods={'monthly':monthly_set,'daily_hg':daily_set,'daily_hgq':moment_set}
     decisions={m:choose(counts,actual,required) for m,counts in methods.items()}
     row={'region':region,'month':month.isoformat(),'tau':tau,'M':int(M),'n_hours':n,'complete_days':len(valid_days),'coverage':len(valid_days)/expected,'frequency_target':freq,'required_hours':required,'actual_K':actual,'monthly_fields':2,'daily_HG_fields':2*len(valid_days),'daily_Q_fields':len(valid_days)}
     for method,d in decisions.items():
      row.update({f'{method}_{k}':v for k,v in d.items()})
     row['monthly_width']=max(monthly_set)-min(monthly_set);row['daily_hg_width']=max(daily_set)-min(daily_set);row['daily_hgq_width']=max(moment_set)-min(moment_set)
     rows.append(row)
 d=pd.DataFrame(rows);q=pd.DataFrame(quality)
 if d.empty:raise ValueError('empty V86 result')
 OUT.mkdir(parents=True,exist_ok=False);d.to_csv(OUT/'month_identification.csv',index=False);q.to_csv(OUT/'quality_audit.csv',index=False)
 metric_cols=[f'{method}_{metric}' for method in ('monthly','daily_hg','daily_hgq') for metric in ('positive','negative','abstain')]
 rp=d.groupby(['region','tau','frequency_target','month'])[metric_cols].mean().reset_index()
 reg=rp.groupby(['region','tau','frequency_target'])[metric_cols].mean().reset_index()
 macro=reg.groupby(['tau','frequency_target'])[metric_cols].mean().reset_index()
 macro.to_csv(OUT/'region_macro_frontier.csv',index=False)
 q['month']=pd.to_datetime(q.month,utc=True)
 d['month']=pd.to_datetime(d.month,utc=True)
 qpairs={(r,m) for r,m,c in q[['region','month','coverage']].itertuples(index=False,name=None) if c>=.8}
 dq=d[d.apply(lambda x:(x.region,pd.Timestamp(x.month)) in qpairs,axis=1)]
 rqp=dq.groupby(['region','tau','frequency_target','month'])[metric_cols].mean().reset_index()
 rq=rqp.groupby(['region','tau','frequency_target'])[metric_cols].mean().reset_index()
 qm=rq.groupby(['tau','frequency_target'])[metric_cols].mean().reset_index();qm.to_csv(OUT/'quality_gated_frontier.csv',index=False)
 payload={'status':'SAME_SAMPLE_MONTHLY_TO_DAILY_PARTIAL_IDENTIFICATION_NOT_FORECAST','regions':list(REGIONS),'tau':list(TAUS),'frequency_targets':list(FREQUENCIES),'months':int(d.month.nunique()),'region_months':int(d[['region','month']].drop_duplicates().shape[0]),'rows':len(d),'quality':quality,'frontier':macro.to_dict(orient='records'),'quality_gated_frontier':qm.to_dict(orient='records'),'input_sha256':input_hashes,'quality_census_sha256':sha(CENSUS),'protocol_sha256':sha(PROTOCOL),'script_sha256':sha(Path(__file__)),'source_audits':audits,'limitations':['Monthly totals are aggregated from the scored complete-day hourly sample; they are an oracle aggregation, not the official monthly release.','All regions have been opened in prior work; no blind-region confirmation.','Fixed-EF contemporaneous hourly proxy is not actual CIF or forecast coverage.','Daily Q is an extra summary derived from hourly generation and its real publication/acquisition cost is not priced.']}
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False))
 print(macro.to_string(index=False));print('quality-gated');print(qm.to_string(index=False));print('quality range',q.coverage.min(),q.coverage.max())
if __name__=='__main__':main()
