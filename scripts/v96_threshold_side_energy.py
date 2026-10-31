"""V96: test a threshold-side monthly energy summary as minimum local information."""
from __future__ import annotations
import json
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
import v90_coarse_capacity_identification as v90
import v93_china_budget_coarsening as v93

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/V96_THRESHOLD_SIDE_ENERGY_PROTOCOL.md'
OUT=ROOT/'results/v96-threshold-side-energy-r1'
COHORT=ROOT/'results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv'
TAUS=(600,900);RATES=(.01,.05,.10)

def threshold_side_set(margins,high_amount,high_ef_range,low_ef_range,caps,tau,T):
    mt=sum((margins[f] for f in v93.THERMAL),F(0));mc=sum((margins[f] for f in v93.CLEAN),F(0));total=mt+mc
    mh=F(high_amount);ml=mt-mh
    if mh<0 or ml<0:raise ValueError('invalid threshold-side margin')
    hmin,hmax=(F(x-tau) for x in high_ef_range);lmin,lmax=(F(x-tau) for x in low_ef_range);t=F(tau)
    if not(hmin>0 and lmax<0):raise ValueError('thermal side ranges do not straddle threshold as specified')
    if total<=0:return (0,)
    out=[]
    if hmin*mh+lmin*ml-t*mc<=0:out.append(0)
    if hmax*mh+lmax*ml-t*mc>0:out.append(T)
    A=-lmin;diff=hmin-lmin
    for k in range(1,T):
        bounds=v93.group_bounds(margins,caps,T,k);xlo,xhi=bounds['thermal'];ylo,yhi=bounds['clean']
        # Event high-side amount z is maximized at min(M_high,x); complement
        # non-event condition imposes z >= (A*x-R)/diff.
        base=hmin*mh+lmin*ml-t*mc
        R=-base-t*ylo
        xlo=max(xlo,R/(A-diff))  # A-diff == -hmin < 0
        xhi=min(xhi,(R+diff*mh)/A)
        complement_limit=total-ylo
        if xlo>xhi or xlo>=complement_limit:continue
        xhi=min(xhi,complement_limit)
        candidates={xlo,xhi}
        if xlo<=mh<=xhi:candidates.add(mh)
        qmax=max(lmax*x+(hmax-lmax)*min(mh,x)-t*ylo for x in candidates)
        if qmax>0:out.append(k)
    return tuple(sorted(set(out)))

def main():
    ph=v90.sha(PROTOCOL);frozen=pd.read_csv(COHORT).drop_duplicates(['region','month'])
    cohort={(str(r.region),str(r.month)[:10]) for r in frozen.itertuples(index=False)}
    data=v90.source_data;data.ALL_REGIONS=tuple(dict.fromkeys(data.ALL_REGIONS+v90.REGIONS))
    hourly,_,audits,input_hashes=data.prepare_regions();caps={};cap_hashes={}
    for year in (2023,2024):caps[year],cap_hashes[str(year)]=v90.capacities(year)
    rows=[]
    for region in v90.REGIONS:
        frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
        frame=frame.loc[times.ge('2024-01-01')&times.lt('2026-01-01')&frame.valid.astype(bool)].copy()
        frame['time']=pd.to_datetime(frame.time,utc=True);frame['month']=frame.time.dt.strftime('%Y-%m-01')
        for month,g in frame.groupby('month',sort=True):
            if (region,month) not in cohort:continue
            T=len(g);year=int(month[:4]);cy=year-1
            margins={f:sum((F(str(float(x))) for x in g[f'generation_{f}']),F(0)) for f in v90.FUELS}
            raw=caps[cy].loc[region] if region in caps[cy].index else pd.Series(dtype=float)
            cap_f={f:float(raw.get(f,0)) for f in ('coal','gas','oil','nuclear','hydro','wind','solar','other')}
            cap_group={'thermal':F(str(sum(cap_f[f] for f in v93.THERMAL))),**{f:F(str(cap_f[f])) for f in v93.CLEAN}}
            no_cap={'thermal':sum((margins[f] for f in v93.THERMAL),F(0)),**{f:margins[f] for f in v93.CLEAN}}
            vals=[]
            for _,r in g.iterrows():
                v=[F(str(float(r[f'generation_{f}']))) for f in v90.FUELS]
                gen=sum(v,F(0));carbon=sum((v[i]*v90.EF[v90.FUELS[i]] for i in range(len(v))),F(0));vals.append((gen,carbon))
            for tau in TAUS:
                if tau==600:high=margins['coal']+margins['oil'];hrange=(800,1000);lrange=(400,400)
                else:high=margins['coal'];hrange=(1000,1000);lrange=(400,800)
                actual=sum(c>F(tau)*gen for gen,c in vals)
                coarse_no=v93.coarse_budget_set(margins,no_cap,tau,T);coarse_cap=v93.coarse_budget_set(margins,cap_group,tau,T)
                side_no=threshold_side_set(margins,high,hrange,lrange,no_cap,tau,T)
                side_cap=threshold_side_set(margins,high,hrange,lrange,cap_group,tau,T)
                detailed=v90.v66_set(margins,tau,T)
                fine=v90.capacity_group_set(margins,cap_f,tau,T,v90.FINE_GROUPS)
                fine_ok=all(float(g[f'generation_{f}'].max())<=cap_f[f]+1e-6 and float(margins[f])<=T*cap_f[f]+1e-6 for f in v90.FUELS)
                if not fine_ok:fine=None
                if any(actual not in s for s in (coarse_no,coarse_cap,side_no,side_cap,detailed)):
                    raise ValueError(f'actual K excluded {region} {month} tau={tau}')
                if not set(detailed).issubset(side_no) or not set(side_no).issubset(coarse_no):raise ValueError('monthly information sets are not nested')
                if not set(side_cap).issubset(coarse_cap):raise ValueError('threshold-side capacity set widened coarse set')
                if fine is not None and (actual not in fine or not set(fine).issubset(side_cap)):
                    raise ValueError('threshold-side capacity set fails to contain audited detailed-capacity set')
                for rate in RATES:
                    req=int((F(str(rate))*T).__ceil__());record={'region':region,'month':month,'tau':tau,'rate':rate,'hours':T,'actual_K':actual,'required_hours':req,'fine_capacity_available':fine is not None}
                    sets={'coarse_month':coarse_no,'threshold_side_month':side_no,'coarse_capacity':coarse_cap,'threshold_side_capacity':side_cap,'detailed_month':detailed}
                    for name,ks in sets.items():
                        pos,neg,abst,width=v90.decide(ks,actual,req);record.update({f'{name}_lower':min(ks),f'{name}_upper':max(ks),f'{name}_positive':pos,f'{name}_negative':neg,f'{name}_abstain':abst,f'{name}_width':width})
                    if fine is not None:
                        fp,fn,fa,fw=v90.decide(fine,actual,req);record.update({'fine_capacity_lower':min(fine),'fine_capacity_upper':max(fine),'fine_capacity_positive':fp,'fine_capacity_negative':fn,'fine_capacity_abstain':fa,'fine_capacity_width':fw})
                    else:record.update({'fine_capacity_lower':None,'fine_capacity_upper':None,'fine_capacity_positive':None,'fine_capacity_negative':None,'fine_capacity_abstain':None,'fine_capacity_width':None})
                    rows.append(record)
    OUT.mkdir(parents=True,exist_ok=False);d=pd.DataFrame(rows);d.to_csv(OUT/'threshold_side_frontier.csv',index=False)
    metrics=[c for c in d if c.endswith(('_positive','_negative','_abstain','_width'))]
    rm=d.groupby(['region','tau','rate','month'])[metrics].mean().reset_index().groupby(['region','tau','rate'])[metrics].mean().reset_index();macro=rm.groupby(['tau','rate'])[metrics].mean().reset_index();macro.to_csv(OUT/'region_macro_frontier.csv',index=False)
    paired=d[d.fine_capacity_available].copy();pm=paired.groupby(['region','tau','rate','month'])[metrics].mean().reset_index().groupby(['region','tau','rate'])[metrics].mean().reset_index().groupby(['tau','rate'])[metrics].mean().reset_index();pm.to_csv(OUT/'fine_capacity_paired_frontier.csv',index=False)
    summary={'status':'THRESHOLD_SIDE_MONTHLY_ENERGY_RETROSPECTIVE_PROXY','region_months':int(d[['region','month']].drop_duplicates().shape[0]),'fine_capacity_months':int(paired[['region','month']].drop_duplicates().shape[0]),'scored_rows':len(d),'frontier':macro.to_dict('records'),'fine_capacity_paired_frontier':pm.to_dict('records'),'capacity_source_sha256':cap_hashes,'hourly_source_sha256':input_hashes,'source_audits':audits,'protocol_sha256':ph,'script_sha256':v90.sha(Path(__file__)),'limitations':['Threshold-side aggregates are oracle-derived from EIA monthly fuel records, not an ingested Chinese public dataset.','EF endpoints are V54 proxy assumptions; no actual CIF or blind-region validation.','Cohort uses retrospective capacity feasibility checks.']}
    if v90.sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print(macro.to_string(index=False))

if __name__=='__main__':main()
