"""V90: test technology-group (China-style) capacity bounds for hourly tail sets."""
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
PROTOCOL=ROOT/'docs/V90_COARSE_CAPACITY_IDENTIFICATION_PROTOCOL.md'
OUT=ROOT/'results/v90-coarse-capacity-identification-r5'
DATA=ROOT/'data/research-v87'
REGIONS=('AVA','BANC','BPAT','CHPD','DOPD','EPE','FMPP','FPL','GCPD','GRID','GVL','IID','JEA','LDWP','MISO','NWMT','NYIS','PSEI','SC','SCEG','SCL','SEC','TEC','TIDC','TPWR')
FUELS=source_data.FUELS;EF=dict(zip(FUELS,(1000,400,800,0,0,0,0,400)))
TAUS=(100,300);RATES=(.01,.05,.10)
GROUPS={'thermal':('coal','gas','oil','other'),'nuclear':('nuclear',),'hydro':('hydro',),'wind':('wind',),'solar':('solar',)}
FINE_GROUPS={f:(f,) for f in FUELS}
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
 z=zipfile.ZipFile(DATA/f'eia860{year}.zip')
 with z.open(f'2___Plant_Y{year}.xlsx') as f:p=pd.read_excel(f,sheet_name='Plant',header=1)
 with z.open(f'3_1_Generator_Y{year}.xlsx') as f:g=pd.read_excel(f,sheet_name='Operable',header=1)
 p=p[['Plant Code','Balancing Authority Code']].dropna().drop_duplicates('Plant Code')
 g=g[g.Status.isin(['OP','OS','SB'])].merge(p,on='Plant Code',how='left')
 g=g[g['Balancing Authority Code'].isin(REGIONS)];rows=[]
 capcols=['Nameplate Capacity (MW)','Summer Capacity (MW)','Winter Capacity (MW)'];fcols=[f'Energy Source {i}' for i in range(1,7)]
 for row in g.itertuples(index=False,name=None):
  x=dict(zip(g.columns,row));vals=[pd.to_numeric(x[c],errors='coerce') for c in capcols];vals=[v for v in vals if pd.notna(v) and v>0]
  if not vals:continue
  cap=max(vals);active={bucket(x[c]) for c in fcols if pd.notna(x[c])} or {'other'}
  for fuel in active:rows.append((x['Balancing Authority Code'],fuel,float(cap)))
 d=pd.DataFrame(rows,columns=['region','fuel','capacity_mw']).groupby(['region','fuel']).capacity_mw.sum().unstack(fill_value=0)
 return d,sha(DATA/f'eia860{year}.zip')
def v66_set(margins,tau,T):
 pos=any(margins[f]>0 and EF[f]>tau for f in FUELS);neg=any(margins[f]>0 and EF[f]<tau for f in FUELS);neu=any(margins[f]>0 and EF[f]==tau for f in FUELS)
 if not pos:return (0,)
 D=sum((F(EF[f]-tau)*margins[f] for f in FUELS),F(0))
 if neg:return tuple(range(1,T+1)) if D>0 else tuple(range(T))
 if neu:return tuple(range(1,T+1))
 return (T,)
def capacity_group_set(margins,caps,tau,T,groups):
 """Conservative outer set using exact groupwise linear optimization."""
 fuels=list(FUELS);q=[float(EF[f]-tau) for f in fuels];M=[float(margins[f]) for f in fuels]
 D=sum(qi*mi for qi,mi in zip(q,M));total=sum(M);out=[]
 group_members={g:[fuels.index(f) for f in fs] for g,fs in groups.items()}
 group_month={g:sum(M[i] for i in ix) for g,ix in group_members.items()}
 group_cap={g:float(caps.get(g,0)) for g in groups}
 if any(group_month[g]>T*group_cap[g]+1e-6 for g in groups):return None
 for k in range(T+1):
  if k==0:
   if D<=0:out.append(k)
   continue
  if k==T:
   if D>0:out.append(k)
   continue
  qmax=0.0;feasible=True
  for g,ix in group_members.items():
   cap=group_cap[g];mg=group_month[g]
   lo=max(0.0,mg-(T-k)*cap);hi=min(mg,k*cap)
   if lo>hi+1e-7:
    feasible=False;break
   items=sorted(((q[i],M[i]) for i in ix),reverse=True)
   positive_mass=sum(mass for qi,mass in items if qi>0)
   # At fixed total event allocation x, the maximizing allocation fills fuels
   # in descending q order. The concave piecewise-linear value peaks at an
   # endpoint or at the cumulative positive-q breakpoint.
   candidates={lo,hi,min(max(positive_mass,lo),hi)}
   def value(x):
    rem=x;v=0.0
    for qi,mass in items:
     take=min(rem,mass);v+=qi*take;rem-=take
     if rem<=1e-9:break
    return v
   qmax+=max(value(x) for x in candidates)
  if feasible:
   eps=1e-8*max(1.0,abs(D),sum(abs(qi*mi) for qi,mi in zip(q,M)))
   if qmax>0 and qmax>=D-eps:out.append(k)
 return tuple(out)
def decide(counts,actual,req):
 if not counts or actual not in counts:raise ValueError('observed K not contained')
 lo,hi=min(counts),max(counts);pos=lo>=req;neg=hi<req
 return (pos,neg,not(pos or neg),hi-lo)
def main():
 ph=sha(PROTOCOL);census=json.loads((ROOT/'results/eia-ba-quality-census-v52-r1/quality_census.json').read_text())
 if tuple(census['quality_qualified_regions_alphabetical'])!=REGIONS:raise ValueError('frozen roster mismatch')
 caps={};data_hashes={}
 for year in (2023,2024):caps[year],data_hashes[str(year)]=capacities(year)
 source_data.ALL_REGIONS=tuple(dict.fromkeys(source_data.ALL_REGIONS+REGIONS))
 hourly,monthly,audits,input_hashes=source_data.prepare_regions();monthly=monthly[monthly.region.isin(REGIONS)].copy();rows=[];qrows=[]
 for region in REGIONS:
  frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
  frame=frame.loc[times.ge('2024-01-01')&times.lt('2026-01-01')&frame.valid.astype(bool)].copy()
  frame['time']=pd.to_datetime(frame.time,utc=True);frame['month']=pd.to_datetime(frame.time.dt.strftime('%Y-%m-01'),utc=True)
  for month,g in frame.groupby('month',sort=True):
   year=month.year;cy=year-1;T=len(g);expected=month.days_in_month*24;coverage=T/expected
   if T==0:continue
   margins={f:sum((F(str(float(x))) for x in g[f'generation_{f}']),F(0)) for f in FUELS}
   cs=caps[cy].loc[region] if region in caps[cy].index else pd.Series(dtype=float)
   cap={f:float(cs.get(f,0)) for f in ('coal','gas','oil','nuclear','hydro','wind','solar','other')}
   fine_cap={f:float(cs.get(f,0)) for f in FUELS}
   coarse_cap={group:sum(cap[f] for f in members) for group,members in GROUPS.items()}
   peak_ok=True
   for group,members in GROUPS.items():
    observed=g[[f'generation_{f}' for f in members]].sum(axis=1).max();peak_ok &= observed<=coarse_cap[group]+1e-6
   energy_ok=all(sum(float(margins[f]) for f in members)<=T*coarse_cap[group]+1e-6 for group,members in GROUPS.items())
   fine_peak_ok=all(float(g[f'generation_{f}'].max())<=fine_cap[f]+1e-6 for f in FUELS)
   fine_energy_ok=all(float(margins[f])<=T*fine_cap[f]+1e-6 for f in FUELS)
   fine_ok=fine_peak_ok and fine_energy_ok
   qrows.append({'region':region,'month':month.isoformat(),'year':year,'capacity_vintage':cy,'hours':T,'coverage':coverage,'group_peak_within_capacity':bool(peak_ok),'group_energy_within_capacity':bool(energy_ok),'fine_peak_within_capacity':bool(fine_peak_ok),'fine_energy_within_capacity':bool(fine_energy_ok)})
   if coverage<.8 or not peak_ok or not energy_ok:continue
   vals=[]
   for _,r in g.iterrows():
    v=[F(str(float(r[f'generation_{f}'])) ) for f in FUELS];gen=sum(v,F(0));hh=sum((v[i]*EF[FUELS[i]] for i in range(len(FUELS))),F(0));vals.append((gen,hh))
   for tau in TAUS:
    actual=sum(h>F(tau)*gen for gen,h in vals);base=v66_set(margins,tau,T);candidate=capacity_group_set(margins,coarse_cap,tau,T,GROUPS)
    if candidate is None:continue
    # The relaxed group allocation may allow a zero-generation complement.
    # Intersecting with the sharp monthly-margin set removes such impossible K.
    bounded=tuple(k for k in candidate if k in base)
    if actual not in bounded or not set(bounded).issubset(base):raise ValueError(f'unsafe coarse outer set: {region} {month} tau={tau} actual={actual} bounded={bounded}')
    fine_bounds=None
    if fine_ok:
     fine_candidate=capacity_group_set(margins,fine_cap,tau,T,FINE_GROUPS)
     fine_bounds=tuple(k for k in fine_candidate if k in base) if fine_candidate is not None else None
     if not fine_bounds or actual not in fine_bounds:raise ValueError(f'unsafe fine-capacity outer set: {region} {month} tau={tau} actual={actual}')
    for rate in RATES:
     req=int((F(str(rate))*T).__ceil__());bp,bn,ba,bw=decide(base,actual,req);cp,cn,ca,cw=decide(bounded,actual,req)
     if fine_bounds:
      fp,fn,fa,fw=decide(fine_bounds,actual,req);flo,fhi=min(fine_bounds),max(fine_bounds)
     else:fp=fn=fa=fw=flo=fhi=None
     rows.append({'region':region,'month':month.isoformat(),'tau':tau,'rate':rate,'hours':T,'required_hours':req,'actual_K':actual,'baseline_lower':min(base),'baseline_upper':max(base),'coarse_lower':min(bounded),'coarse_upper':max(bounded),'fine_lower':flo,'fine_upper':fhi,'baseline_positive':bp,'baseline_negative':bn,'baseline_abstain':ba,'coarse_positive':cp,'coarse_negative':cn,'coarse_abstain':ca,'fine_positive':fp,'fine_negative':fn,'fine_abstain':fa,'baseline_width':bw,'coarse_width':cw,'fine_width':fw,'fine_available':bool(fine_bounds),'peak_ok':peak_ok,'coverage':coverage})
 OUT.mkdir(parents=True,exist_ok=False);d=pd.DataFrame(rows);q=pd.DataFrame(qrows);d.to_csv(OUT/'coarse_capacity_frontier.csv',index=False);q.to_csv(OUT/'coarse_capacity_quality_audit.csv',index=False)
 if d.empty:raise ValueError('no coarse-capacity feasible rows')
 metric=['baseline_positive','baseline_negative','baseline_abstain','coarse_positive','coarse_negative','coarse_abstain','baseline_width','coarse_width']
 rg=d.groupby(['region','tau','rate','month'])[metric].mean().reset_index().groupby(['region','tau','rate'])[metric].mean().reset_index();macro=rg.groupby(['tau','rate'])[metric].mean().reset_index();macro.to_csv(OUT/'region_macro_frontier.csv',index=False)
 paired=d[d.fine_available].copy();paired_metrics=['baseline_positive','baseline_negative','baseline_abstain','fine_positive','fine_negative','fine_abstain','coarse_positive','coarse_negative','coarse_abstain','baseline_width','fine_width','coarse_width']
 pr=paired.groupby(['region','tau','rate','month'])[paired_metrics].mean().reset_index().groupby(['region','tau','rate'])[paired_metrics].mean().reset_index();pm=pr.groupby(['tau','rate'])[paired_metrics].mean().reset_index();pm.to_csv(OUT/'paired_fuel_vs_coarse_frontier.csv',index=False)
 payload={'status':'LAGGED_COARSE_TECH_CAPACITY_RETROSPECTIVE_OUTER_SET_NOT_FORECAST','regions':list(REGIONS),'capacity_feasible_region_months':int(d[['region','month']].drop_duplicates().shape[0]),'fine_capacity_feasible_region_months':int(d[d.fine_available][['region','month']].drop_duplicates().shape[0]),'scored_rows':len(d),'quality_audit':q.groupby(['year','group_peak_within_capacity','group_energy_within_capacity','fine_peak_within_capacity','fine_energy_within_capacity']).size().rename('region_months').reset_index().to_dict('records'),'frontier':macro.to_dict('records'),'paired_frontier':pm.to_dict('records'),'capacity_source_sha256':data_hashes,'hourly_source_sha256':input_hashes,'protocol_sha256':ph,'script_sha256':sha(Path(__file__)),'source_audits':audits,'limitations':['Technology groups are thermal/nuclear/hydro/wind/solar; thermal includes coal/gas/oil/other and group capacity is shared without fuel-specific allocation.','Lagged year-end EIA capacity is only an analogue of annual coarse public statistics, not a verified Chinese source schema.','Feasibility uses contemporaneous hourly labels and is retrospective mechanism screening.','The maximizing allocation calculation is groupwise exact for the stated linear relaxation; intersecting V66 yields a conservative outer set, not a proven sharp capacity-constrained set.','All 25 regions were previously opened; fixed-EF proxy is not actual CIF.']}
 if sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print('quality audit',payload['quality_audit']);print(macro.to_string(index=False))
if __name__=='__main__':main()
