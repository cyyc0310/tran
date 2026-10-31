"""V95: test whether one local monthly coal-generation field restores high-tail identification."""
from __future__ import annotations
import json
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
import v90_coarse_capacity_identification as v90
import v93_china_budget_coarsening as v93

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/V95_SINGLE_MONTHLY_COAL_FIELD_PROTOCOL.md'
OUT=ROOT/'results/v95-single-monthly-coal-field-r2'
COHORT=ROOT/'results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv'
TAUS=(600,900);RATES=(.01,.05,.10)

def coal_split_set(margins,coal_amount,caps,tau,T):
    mt=sum((margins[f] for f in v93.THERMAL),F(0));mc=sum((margins[f] for f in v93.CLEAN),F(0));total=mt+mc
    mcoal=F(coal_amount);mnon=mt-mcoal
    if mcoal<0 or mnon<0:raise ValueError('invalid coal margin')
    qcoal=F(1000-tau);qnon_lo=F(400-tau);qnon_hi=F(800-tau);t=F(tau)
    if total<=0:return (0,)
    result=[]
    if qcoal*mcoal+qnon_lo*mnon-t*mc<=0:result.append(0)
    if qcoal*mcoal+qnon_hi*mnon-t*mc>0:result.append(T)
    for k in range(1,T):
        bounds=v93.group_bounds(margins,caps,T,k);xlo,xhi=bounds['thermal'];ylo,yhi=bounds['clean']
        # For fixed thermal event total x, event excess increases with event coal
        # output z, so z*=min(Mcoal,x). The complement non-event inequality gives
        # a lower bound on z; requiring it not exceed z* reduces feasibility to
        # one interval in x. Event excess is piecewise linear with a single kink.
        A=t-F(400)
        rhs=-(qcoal*mcoal+qnon_lo*mnon-t*mc)-t*ylo
        if A<=0 or A>=600:raise ValueError('unexpected tau for coal-field interval reduction')
        xlo=max(xlo,rhs/(A-F(600)))
        xhi=min(xhi,(rhs+F(600)*mcoal)/A,mcoal+mnon)
        complement_limit=total-ylo
        if xlo>xhi or xlo>=complement_limit:continue
        xhi=min(xhi,complement_limit)
        candidates={xlo,xhi}
        if xlo<=mcoal<=xhi:candidates.add(mcoal)
        max_event=max(qnon_hi*x+F(200)*min(mcoal,x)-t*ylo for x in candidates)
        if max_event>0:result.append(k)
    return tuple(sorted(set(result)))

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
            coal=margins['coal'];raw=caps[cy].loc[region] if region in caps[cy].index else pd.Series(dtype=float)
            caps_float={f:float(raw.get(f,0)) for f in ('coal','gas','oil','nuclear','hydro','wind','solar','other')}
            coarse_cap={'thermal':F(str(sum(caps_float[f] for f in v93.THERMAL))),**{f:F(str(caps_float[f])) for f in v93.CLEAN}}
            no_cap={'thermal':sum((margins[f] for f in v93.THERMAL),F(0)),**{f:margins[f] for f in v93.CLEAN}}
            vals=[]
            for _,r in g.iterrows():
                v=[F(str(float(r[f'generation_{f}']))) for f in v90.FUELS]
                gen=sum(v,F(0));carbon=sum((v[i]*v90.EF[v90.FUELS[i]] for i in range(len(v))),F(0));vals.append((gen,carbon))
            for tau in TAUS:
                actual=sum(c>F(tau)*gen for gen,c in vals)
                coarse_no=v93.coarse_budget_set(margins,no_cap,tau,T)
                coarse_set=v93.coarse_budget_set(margins,coarse_cap,tau,T)
                coal_no=coal_split_set(margins,coal,no_cap,tau,T)
                coal_cap=coal_split_set(margins,coal,coarse_cap,tau,T)
                detailed=v90.v66_set(margins,tau,T)
                if any(actual not in ks for ks in (coarse_no,coal_no,coarse_set,coal_cap,detailed)):
                    raise ValueError(f'candidate set excluded observed K: {region} {month} tau={tau} K={actual}')
                if not set(coal_no).issubset(coarse_no) or not set(coal_cap).issubset(coarse_set):
                    raise ValueError('adding monthly coal field widened identified set')
                fine_cap_set=v90.capacity_group_set(margins,caps_float,tau,T,v90.FINE_GROUPS)
                fine_ok=all(float(g[f'generation_{f}'].max())<=caps_float[f]+1e-6 and float(margins[f])<=T*caps_float[f]+1e-6 for f in v90.FUELS)
                if not fine_ok:fine_cap_set=None
                if fine_cap_set is not None and (actual not in fine_cap_set or not set(fine_cap_set).issubset(coal_cap)):
                    raise ValueError('coal-field capacity set failed to contain fine-capacity set')
                for rate in RATES:
                    req=int((F(str(rate))*T).__ceil__())
                    sets={'coarse_month':coarse_no,'coal_month':coal_no,'coarse_capacity':coarse_set,'coal_capacity':coal_cap,'detailed_month':detailed}
                    record={'region':region,'month':month,'tau':tau,'rate':rate,'hours':T,'actual_K':actual,'required_hours':req,'fine_capacity_available':fine_cap_set is not None}
                    for name,ks in sets.items():
                        pos,neg,abst,width=v90.decide(ks,actual,req);record.update({f'{name}_lower':min(ks),f'{name}_upper':max(ks),f'{name}_positive':pos,f'{name}_negative':neg,f'{name}_abstain':abst,f'{name}_width':width})
                    if fine_cap_set is not None:
                        fp,fn,fa,fw=v90.decide(fine_cap_set,actual,req);record.update({'fine_capacity_lower':min(fine_cap_set),'fine_capacity_upper':max(fine_cap_set),'fine_capacity_positive':fp,'fine_capacity_negative':fn,'fine_capacity_abstain':fa,'fine_capacity_width':fw})
                    else:
                        record.update({'fine_capacity_lower':None,'fine_capacity_upper':None,'fine_capacity_positive':None,'fine_capacity_negative':None,'fine_capacity_abstain':None,'fine_capacity_width':None})
                    rows.append(record)
    OUT.mkdir(parents=True,exist_ok=False);d=pd.DataFrame(rows);d.to_csv(OUT/'coal_field_frontier.csv',index=False)
    metrics=[c for c in d if c.endswith(('_positive','_negative','_abstain','_width'))]
    rm=d.groupby(['region','tau','rate','month'])[metrics].mean().reset_index().groupby(['region','tau','rate'])[metrics].mean().reset_index();macro=rm.groupby(['tau','rate'])[metrics].mean().reset_index();macro.to_csv(OUT/'region_macro_frontier.csv',index=False)
    paired=d[d.fine_capacity_available].copy();pm=paired.groupby(['region','tau','rate','month'])[metrics].mean().reset_index().groupby(['region','tau','rate'])[metrics].mean().reset_index().groupby(['tau','rate'])[metrics].mean().reset_index();pm.to_csv(OUT/'fine_capacity_paired_frontier.csv',index=False)
    summary={'status':'ONE_MONTHLY_COAL_FIELD_RETROSPECTIVE_PROXY','region_months':int(d[['region','month']].drop_duplicates().shape[0]),'fine_capacity_months':int(paired[['region','month']].drop_duplicates().shape[0]),'scored_rows':len(d),'frontier':macro.to_dict('records'),'fine_capacity_paired_frontier':pm.to_dict('records'),'capacity_source_sha256':cap_hashes,'hourly_source_sha256':input_hashes,'source_audits':audits,'protocol_sha256':ph,'script_sha256':v90.sha(Path(__file__)),'limitations':['Cohort is the retrospectively screened V90 358-month set.','Monthly coal MWh is oracle-aggregated from EIA hourly records; no Chinese public file was ingested.','Noncoal thermal EF is allowed to vary within [400,800] and independently across event/complement blocks.','Fixed EF proxy labels are not actual CIF; no unseen-region forecast validation.']}
    if v90.sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print(macro.to_string(index=False))

if __name__=='__main__':main()
