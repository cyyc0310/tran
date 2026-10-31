"""V92: sharpen grouped-capacity tail identification with positive output in both blocks."""
from __future__ import annotations
import json
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
import v90_coarse_capacity_identification as v90

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/V92_SHARP_COARSE_CAPACITY_PROTOCOL.md'
OUT=ROOT/'results/v92-sharp-coarse-capacity-r1'
COARSE=ROOT/'results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv'

def sharp_group_set(margins,caps,tau,T):
    """Exact feasible K set for group caps and positive generation in both blocks.

    For a candidate event block of k hours, event group fuel totals y_f satisfy
    0<=y_f<=M_f and group totals are between the event and complement capacity
    envelopes. Carbon excess over the event block is maximized groupwise. A split
    exists iff the maximum is positive and >= the monthly excess D; at the equality
    boundary, require a maximizing allocation with positive complement generation.
    Return None when monthly energy is incompatible with the capacity contract.
    """
    fuels=list(v90.FUELS)
    q={f:F(v90.EF[f]-tau) for f in fuels}
    M={f:F(margins[f]) for f in fuels}
    total=sum(M.values(),F(0))
    D=sum((q[f]*M[f] for f in fuels),F(0))
    groups=v90.GROUPS
    caps={g:F(str(float(caps.get(g,0)))) for g in groups}
    # This condition applies to every K, including the endpoint shortcuts below.
    # Without it, an infeasible month can incorrectly produce (0,) or (T,).
    if T < 0 or any(value < 0 for value in M.values()) or any(value < 0 for value in caps.values()):
        raise ValueError('hours, fuel margins and capacities must be nonnegative')
    if any(sum((M[f] for f in fs),F(0)) > T*caps[group] for group,fs in groups.items()):
        return None
    base=v90.v66_set(margins,tau,T)
    out=[]
    for k in range(T+1):
        if k in (0,T):
            if k in base:out.append(k)
            continue
        qmax=F(0);minimum_event_total=F(0);feasible=True
        for group,fs in groups.items():
            mg=sum((M[f] for f in fs),F(0));cap=caps[group]
            lo=max(F(0),mg-(T-k)*cap);hi=min(mg,k*cap)
            if lo>hi:
                feasible=False;break
            positive_mass=sum((M[f] for f in fs if q[f]>0),F(0))
            x=min(max(positive_mass,lo),hi)
            remaining=x;value=F(0)
            for f in sorted(fs,key=lambda fuel:q[fuel],reverse=True):
                allocated=min(remaining,M[f]);value+=q[f]*allocated;remaining-=allocated
                if remaining==0:break
            if remaining>0:raise ArithmeticError('group allocation failed')
            qmax+=value;minimum_event_total+=x
        if not feasible:continue
        # Strict event excess; non-event excess <=0. If the maximum is exactly
        # D, only retain K when its maximizing face permits a positive complement.
        valid=qmax>0 and qmax>=D
        if valid and qmax==D and minimum_event_total>=total:valid=False
        if valid and k in base:out.append(k)
    return tuple(out)

def main():
    ph=v90.sha(PROTOCOL)
    cohort=pd.read_csv(COARSE).drop_duplicates(['region','month'])
    cohort=cohort[['region','month']]
    data=v90.source_data
    data.ALL_REGIONS=tuple(dict.fromkeys(data.ALL_REGIONS+v90.REGIONS))
    hourly,_,audits,input_hashes=data.prepare_regions()
    caps={}
    hashes={}
    for year in (2023,2024):caps[year],hashes[str(year)]=v90.capacities(year)
    source=pd.read_csv(COARSE)
    source['month_key']=pd.to_datetime(source.month,utc=True).dt.strftime('%Y-%m-01')
    source=source.drop_duplicates(['region','month_key','tau','rate'])
    rows=[]
    for region in v90.REGIONS:
        frame=hourly[region].copy();times=pd.to_datetime(frame.time,utc=True)
        frame=frame.loc[times.ge('2024-01-01')&times.lt('2026-01-01')&frame.valid.astype(bool)].copy()
        frame['time']=pd.to_datetime(frame.time,utc=True)
        frame['month_key']=frame.time.dt.strftime('%Y-%m-01')
        for month,g in frame.groupby('month_key',sort=True):
            if not ((cohort.region==region)&(cohort.month.str.slice(0,10)==month)).any():continue
            year=int(month[:4]);T=len(g);cy=year-1
            margins={f:sum((F(str(float(x))) for x in g[f'generation_{f}']),F(0)) for f in v90.FUELS}
            cs=caps[cy].loc[region] if region in caps[cy].index else pd.Series(dtype=float)
            fine={f:float(cs.get(f,0)) for f in ('coal','gas','oil','nuclear','hydro','wind','solar','other')}
            coarse={group:sum(fine[f] for f in members) for group,members in v90.GROUPS.items()}
            vals=[]
            for _,r in g.iterrows():
                v=[F(str(float(r[f'generation_{f}']))) for f in v90.FUELS]
                gen=sum(v,F(0));carbon=sum((v[i]*v90.EF[v90.FUELS[i]] for i in range(len(v))),F(0));vals.append((gen,carbon))
            for tau in v90.TAUS:
                actual=sum(carbon>F(tau)*gen for gen,carbon in vals)
                base=v90.v66_set(margins,tau,T)
                sharp=sharp_group_set(margins,coarse,tau,T)
                prior=source[(source.region==region)&(source.month_key==month)&(source.tau==tau)]
                if len(prior)!=len(v90.RATES):raise ValueError(f'missing frozen V90 rows {region} {month} tau={tau}')
                coarse_lo=int(prior.coarse_lower.iloc[0]);coarse_hi=int(prior.coarse_upper.iloc[0])
                if actual not in sharp or not set(sharp).issubset(base):
                    raise ValueError(f'sharp set failed truth/nesting: {region} {month} tau={tau} K={actual} set={sharp}')
                if min(sharp)<coarse_lo or max(sharp)>coarse_hi:
                    raise ValueError(f'sharp set not inside V90 outer endpoints: {region} {month} tau={tau}')
                for rate in v90.RATES:
                    req=int((F(str(rate))*T).__ceil__());bp,bn,ba,bw=v90.decide(base,actual,req);sp,sn,sa,sw=v90.decide(sharp,actual,req)
                    rows.append({'region':region,'month':month,'tau':tau,'rate':rate,'hours':T,'actual_K':actual,'required_hours':req,'baseline_lower':min(base),'baseline_upper':max(base),'coarse_outer_lower':coarse_lo,'coarse_outer_upper':coarse_hi,'sharp_lower':min(sharp),'sharp_upper':max(sharp),'baseline_positive':bp,'baseline_negative':bn,'baseline_abstain':ba,'outer_positive':bool(int(coarse_lo)>=req),'outer_negative':bool(int(coarse_hi)<req),'outer_abstain':not(int(coarse_lo)>=req or int(coarse_hi)<req),'sharp_positive':sp,'sharp_negative':sn,'sharp_abstain':sa,'baseline_width':bw,'outer_width':coarse_hi-coarse_lo,'sharp_width':sw})
    OUT.mkdir(parents=True,exist_ok=False)
    d=pd.DataFrame(rows);d.to_csv(OUT/'sharp_coarse_frontier.csv',index=False)
    metrics=['baseline_positive','baseline_negative','baseline_abstain','outer_positive','outer_negative','outer_abstain','sharp_positive','sharp_negative','sharp_abstain','baseline_width','outer_width','sharp_width']
    region=d.groupby(['region','tau','rate','month'])[metrics].mean().reset_index().groupby(['region','tau','rate'])[metrics].mean().reset_index()
    macro=region.groupby(['tau','rate'])[metrics].mean().reset_index();macro.to_csv(OUT/'region_macro_frontier.csv',index=False)
    summary={'status':'SHARP_GROUP_CAPACITY_K_SET_RETROSPECTIVE_PROXY_NOT_FORECAST','regions':list(v90.REGIONS),'region_months':int(d[['region','month']].drop_duplicates().shape[0]),'scored_rows':len(d),'frontier':macro.to_dict('records'),'eia_capacity_sha256':hashes,'hourly_source_sha256':input_hashes,'source_audits':audits,'protocol_sha256':ph,'script_sha256':v90.sha(Path(__file__)),'limitations':['Retrospective 358-month V90 cohort; inclusion depended on target-hour group peak and energy audit.','The bound is sharp only under fixed EF, exact monthly fuel margins, shared group hourly caps, nonnegative fuel generation, no cross-hour constraints, and positive generation in both event and complement hours.','This is EIA proxy mechanism evidence, not China data validation or future coverage.']}
    if v90.sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed during run')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(macro.to_string(index=False))

if __name__=='__main__':main()
