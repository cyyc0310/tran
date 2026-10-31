"""V88: audit and test annual generator-capacity envelopes for monthly tail sets."""
from __future__ import annotations
import hashlib,json,sys,zipfile
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import v54_unseen_local_ef_confirmation as source_data  # noqa:E402
sys.path.insert(0,str(ROOT))
from scripts import tail_count_v20 as model  # noqa:E402
PROTOCOL=ROOT/'docs/V88_CAPACITY_CONSTRAINED_IDENTIFICATION_PROTOCOL.md'
OUT=ROOT/'results/v88-capacity-constrained-identification-r2'
DATA=ROOT/'data/research-v87'
REGIONS=('AVA','BANC','BPAT','CHPD','DOPD','EPE','FMPP','FPL','GCPD','GRID','GVL','IID','JEA','LDWP','MISO','NWMT','NYIS','PSEI','SC','SCEG','SCL','SEC','TEC','TIDC','TPWR')
FUELS=source_data.FUELS;EF=dict(zip(FUELS,(1000,400,800,0,0,0,0,400)))
TAUS=(100,300);RATES=(.01,.05,.10)
COAL={'ANT','BIT','LIG','SUB','WC','RC'};GAS={'NG','OG','PG','BFG','SG','SGC'};OIL={'DFO','RFO','JF','KER','WO'}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bucket(code):
 c=str(code).strip().upper()
 if c in COAL:return 'coal'
 if c in GAS:return 'gas'
 if c in OIL:return 'oil'
 if c=='NUC':return 'nuclear'
 if c in {'WAT','MWH'}:return 'hydro'
 if c=='WND':return 'wind'
 if c=='SUN':return 'solar'
 return 'other'
def capacities(year):
 zpath=DATA/f'eia860{year}.zip';z=zipfile.ZipFile(zpath)
 with z.open(f'2___Plant_Y{year}.xlsx') as f:p=pd.read_excel(f,sheet_name='Plant',header=1)
 with z.open(f'3_1_Generator_Y{year}.xlsx') as f:g=pd.read_excel(f,sheet_name='Operable',header=1)
 p=p[['Plant Code','Balancing Authority Code']].dropna().drop_duplicates('Plant Code')
 g=g[g.Status.isin(['OP','OS','SB'])].merge(p,on='Plant Code',how='left')
 g=g[g['Balancing Authority Code'].isin(REGIONS)]
 rows=[]
 capcols=['Nameplate Capacity (MW)','Summer Capacity (MW)','Winter Capacity (MW)']
 fcols=[f'Energy Source {i}' for i in range(1,7)]
 for row in g.itertuples(index=False,name=None):
  x=dict(zip(g.columns,row)); vals=[pd.to_numeric(x[c],errors='coerce') for c in capcols]
  vals=[v for v in vals if pd.notna(v) and v>0]
  if not vals:continue
  cap=max(vals); active={bucket(x[c]) for c in fcols if pd.notna(x[c])} or {'other'}
  for fuel in active:rows.append((x['Balancing Authority Code'],fuel,float(cap)))
 d=pd.DataFrame(rows,columns=['region','fuel','capacity_mw']).groupby(['region','fuel']).capacity_mw.sum().unstack(fill_value=0)
 return d,sha(zpath)
def v66_set(margins,tau,T):
 pos=any(margins[f]>0 and EF[f]>tau for f in FUELS);neg=any(margins[f]>0 and EF[f]<tau for f in FUELS);neu=any(margins[f]>0 and EF[f]==tau for f in FUELS)
 if not pos:return tuple([0])
 D=sum((F(EF[f]-tau)*margins[f] for f in FUELS),F(0))
 if neg:return tuple(range(1,T+1)) if D>0 else tuple(range(T))
 if neu:return tuple(range(1,T+1))
 return (T,)
def capacity_outer(margins,caps,tau,T):
 q=[F(EF[f]-tau) for f in FUELS];M=[margins[f] for f in FUELS];C=[F(str(float(caps.get(f,0)))) for f in FUELS]
 tol=F(1,10**8)
 if any(M[i]>T*C[i]+tol for i in range(len(FUELS))):return None
 D=sum((q[i]*M[i] for i in range(len(FUELS))),F(0));out=[]
 for k in range(T+1):
  if k==0:
   if D<=0:out.append(k)
   continue
  if k==T:
   if D>0:out.append(k)
   continue
  lo=[max(F(0),M[i]-k*C[i]) for i in range(len(FUELS))]
  hi=[min(M[i],(T-k)*C[i]) for i in range(len(FUELS))]
  if any(lo[i]>hi[i]+tol for i in range(len(FUELS))):continue
  ymin=[lo[i] if q[i]>=0 else hi[i] for i in range(len(FUELS))]
  qmin=sum((q[i]*ymin[i] for i in range(len(FUELS))),F(0))
  if D>0:
   if qmin<0 or (qmin==0 and (sum(ymin,F(0))>0 or any(q[i]==0 and hi[i]>lo[i] for i in range(len(FUELS))))):out.append(k)
  elif qmin<D:out.append(k)
 return tuple(out)
def decide(counts,actual,req):
 if not counts or actual not in counts:raise ValueError('observed K not contained')
 lo,hi=min(counts),max(counts);pos=lo>=req;neg=hi<req
 return (pos,neg,not(pos or neg),hi-lo)
def main():
 ph=sha(PROTOCOL);census=json.loads((ROOT/'results/eia-ba-quality-census-v52-r1/quality_census.json').read_text())
 if tuple(census['quality_qualified_regions_alphabetical'])!=REGIONS:raise ValueError('frozen roster mismatch')
 caps={};data_hashes={}
 for year in (2024,2025):caps[year],data_hashes[str(year)]=capacities(year)
 source_data.ALL_REGIONS=tuple(dict.fromkeys(source_data.ALL_REGIONS+REGIONS))
 hourly,monthly,audits,input_hashes=source_data.prepare_regions()
 monthly=monthly[monthly.region.isin(REGIONS)].copy();rows=[];qrows=[];capacity_rows=[]
 for region in REGIONS:
  frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
  frame=frame.loc[times.ge('2024-01-01')&times.lt('2026-01-01')&frame.valid.astype(bool)].copy()
  frame['time']=pd.to_datetime(frame.time,utc=True);frame['month']=pd.to_datetime(frame.time.dt.strftime('%Y-%m-01'),utc=True)
  for month,g in frame.groupby('month',sort=True):
   year=month.year;T=len(g);expected=month.days_in_month*24;coverage=T/expected
   if T==0:continue
   gcols=[f'generation_{f}' for f in FUELS];margins={f:sum((F(str(float(x))) for x in g[f'generation_{f}']),F(0)) for f in FUELS}
   cs=caps[year].loc[region] if region in caps[year].index else pd.Series(dtype=float)
   obs_peak={f:float(g[f'generation_{f}'].max()) for f in FUELS};cap={f:float(cs.get(f,0)) for f in FUELS}
   peak_ok=all(obs_peak[f]<=cap[f]+1e-6 for f in FUELS)
   energy_ok=all(float(margins[f])<=T*cap[f]+1e-6 for f in FUELS)
   qrows.append({'region':region,'month':month.isoformat(),'year':year,'hours':T,'coverage':coverage,'peak_all_fuels_within_capacity':peak_ok,'monthly_energy_within_capacity':energy_ok})
   if coverage<.8:continue
   for f in FUELS:capacity_rows.append({'region':region,'month':month.isoformat(),'year':year,'fuel':f,'observed_peak_mw':obs_peak[f],'capacity_mw':cap[f],'peak_ratio':obs_peak[f]/cap[f] if cap[f]>0 else (0 if obs_peak[f]==0 else None),'capacity_violation':obs_peak[f]>cap[f]+1e-6})
   if not peak_ok or not energy_ok:continue
   actuals=[]
   for _,r in g.iterrows():
    vals=[F(str(float(r[f'generation_{f}']))) for f in FUELS];gg=sum(vals,F(0));hh=sum((vals[i]*EF[FUELS[i]] for i in range(len(FUELS))),F(0));actuals.append((gg,hh))
   for tau in TAUS:
    actual=sum(h>F(tau)*gen for gen,h in actuals);base=v66_set(margins,tau,T)
    narrowed=capacity_outer(margins,cap,tau,T)
    if narrowed is None:continue
    if actual not in narrowed or not set(narrowed).issubset(base):
     raise ValueError(f'capacity outer bound not safely nested: {region} {month} tau={tau} actual={actual} base=({min(base)},{max(base)}) cap={narrowed}')
    for rate in RATES:
     req=int((F(str(rate))*T).__ceil__());bp,bn,ba,bw=decide(base,actual,req);cp,cn,ca,cw=decide(narrowed,actual,req)
     rows.append({'region':region,'month':month.isoformat(),'tau':tau,'rate':rate,'hours':T,'required_hours':req,'actual_K':actual,'baseline_lower':min(base),'baseline_upper':max(base),'capacity_lower':min(narrowed),'capacity_upper':max(narrowed),'baseline_positive':bp,'baseline_negative':bn,'baseline_abstain':ba,'capacity_positive':cp,'capacity_negative':cn,'capacity_abstain':ca,'baseline_width':bw,'capacity_width':cw,'peak_ok':peak_ok,'coverage':coverage})
 OUT.mkdir(parents=True,exist_ok=False);d=pd.DataFrame(rows);q=pd.DataFrame(qrows);c=pd.DataFrame(capacity_rows)
 d.to_csv(OUT/'capacity_frontier.csv',index=False);q.to_csv(OUT/'capacity_quality_audit.csv',index=False);c.to_csv(OUT/'fuel_capacity_audit.csv',index=False)
 if d.empty:raise ValueError('no capacity-feasible scored rows')
 metric=['baseline_positive','baseline_negative','baseline_abstain','capacity_positive','capacity_negative','capacity_abstain','baseline_width','capacity_width']
 rp=d.groupby(['region','tau','rate','month'])[metric].mean().reset_index();rg=rp.groupby(['region','tau','rate'])[metric].mean().reset_index();macro=rg.groupby(['tau','rate'])[metric].mean().reset_index();macro.to_csv(OUT/'region_macro_frontier.csv',index=False)
 payload={'status':'RETROSPECTIVE_CAPACITY_CONSTRAINED_OUTER_SET_NOT_FORECAST','regions':list(REGIONS),'capacity_feasible_region_months':int(d[['region','month']].drop_duplicates().shape[0]),'scored_rows':len(d),'capacity_feasibility_audit':q.groupby(['year','peak_all_fuels_within_capacity','monthly_energy_within_capacity']).size().rename('region_months').reset_index().to_dict('records'),'frontier':macro.to_dict('records'),'capacity_source_sha256':data_hashes,'hourly_source_sha256':input_hashes,'protocol_sha256':ph,'script_sha256':sha(Path(__file__)),'source_audits':audits,'limitations':['Capacity uses year-end generator inventory and broad fuel-code mapping; all hourly generation peaks must be within reported capacity for scored region-months.','Feasibility scoring excludes failures using contemporaneous hourly observations and is retrospective mechanism analysis, not deployable screening.','The capacity set is a conservative necessary outer set with relaxed group allocations, not established sharp.','All 25 regions were previously opened; labels are fixed-EF proxy and not actual CIF.']}
 if sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 print('capacity audit',payload['capacity_feasibility_audit']);print(macro.to_string(index=False))
if __name__=='__main__':main()
