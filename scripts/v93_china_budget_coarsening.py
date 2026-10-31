"""V93: test the China-style monthly technology-margin information budget."""
from __future__ import annotations
import json
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
import v90_coarse_capacity_identification as v90

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/V93_CHINA_BUDGET_COARSENING_PROTOCOL.md'
OUT=ROOT/'results/v93-china-budget-coarsening-r1'
COARSE=ROOT/'results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv'
THERMAL=('coal','gas','oil','other')
CLEAN=('nuclear','hydro','wind','solar')

def group_bounds(margins,caps,T,k):
    mt=sum((margins[f] for f in THERMAL),F(0))
    thermal=(max(F(0),mt-(T-k)*caps['thermal']),min(mt,k*caps['thermal']))
    clean_lo=F(0);clean_hi=F(0)
    for fuel in CLEAN:
        mf=margins[fuel];cap=caps[fuel]
        clean_lo+=max(F(0),mf-(T-k)*cap);clean_hi+=min(mf,k*cap)
    return {'thermal':thermal,'clean':(clean_lo,clean_hi)}

def polygon_vertices(xlo,xhi,ylo,yhi,a,tau,rhs):
    """Vertices of a rectangle clipped by -a*x+tau*y <= rhs."""
    candidates={(xlo,ylo),(xlo,yhi),(xhi,ylo),(xhi,yhi)}
    for x in (xlo,xhi):
        y=(rhs+a*x)/tau
        if ylo<=y<=yhi:candidates.add((x,y))
    for y in (ylo,yhi):
        x=(tau*y-rhs)/a
        if xlo<=x<=xhi:candidates.add((x,y))
    return [(x,y) for x,y in candidates if -a*x+tau*y<=rhs]

def coarse_budget_set(margins,caps,tau,T):
    """Outer K set with monthly thermal/clean totals, unknown thermal fuel mix.

    Thermal EF is allowed to vary from 400 to 1000 g/kWh independently in the
    event and complement blocks; all clean-group proxy EFs are 0. This exactly
    represents the V54 proxy support while withholding fuel-specific month totals.
    """
    mt=sum((margins[f] for f in THERMAL),F(0));mc=sum((margins[f] for f in CLEAN),F(0));total=mt+mc
    a=F(400-tau);b=F(1000-tau);t=F(tau);out=[]
    if total<=0:return (0,)
    # Endpoints have only one active block; the unknown thermal mix uses its
    # lowest EF for K=0 and highest EF for K=T.
    if a*mt-t*mc<=0:out.append(0)
    if b*mt-t*mc>0:out.append(T)
    for k in range(1,T):
        bounds=group_bounds(margins,caps,T,k)
        xlo,xhi=bounds['thermal'];ylo,yhi=bounds['clean']
        if xlo>xhi or ylo>yhi:continue
        # Complement can be non-event using the lowest thermal EF (400).
        # q_non<=0 is -a*x + tau*y <= tau*Mc-a*Mt.
        rhs=t*mc-a*mt
        vertices=polygon_vertices(xlo,xhi,ylo,yhi,a,t,rhs)
        if not vertices:continue
        max_event=max(b*x-t*y for x,y in vertices)
        min_event_total=min(x+y for x,y in vertices)
        # If an event-positive point and a positive-output complement point both
        # exist in the feasible convex polygon, their mixture preserves both.
        if max_event>0 and min_event_total<total:out.append(k)
    return tuple(sorted(set(out)))

def main():
    ph=v90.sha(PROTOCOL)
    comparison=pd.read_csv(COARSE)
    frozen=comparison.drop_duplicates(['region','month'])
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
            year=int(month[:4]);T=len(g);cy=year-1
            margins={f:sum((F(str(float(x))) for x in g[f'generation_{f}']),F(0)) for f in v90.FUELS}
            raw=caps[cy].loc[region] if region in caps[cy].index else pd.Series(dtype=float)
            fine={f:F(str(float(raw.get(f,0)))) for f in ('coal','gas','oil','nuclear','hydro','wind','solar','other')}
            cg={group:sum((fine[f] for f in members),F(0)) for group,members in v90.GROUPS.items()}
            cap_group={'thermal':cg['thermal'],**{f:cg[f] for f in CLEAN}}
            no_cap={'thermal':sum((margins[f] for f in THERMAL),F(0)),**{f:margins[f] for f in CLEAN}}
            vals=[]
            for _,r in g.iterrows():
                v=[F(str(float(r[f'generation_{f}']))) for f in v90.FUELS]
                gen=sum(v,F(0));carbon=sum((v[i]*v90.EF[v90.FUELS[i]] for i in range(len(v))),F(0));vals.append((gen,carbon))
            for tau in v90.TAUS:
                actual=sum(carbon>F(tau)*gen for gen,carbon in vals)
                no_capacity=coarse_budget_set(margins,no_cap,tau,T)
                with_capacity=coarse_budget_set(margins,cap_group,tau,T)
                if actual not in no_capacity or actual not in with_capacity:
                    raise ValueError(f'coarse information set excludes observed K: {region} {month} tau={tau} K={actual}')
                if not set(with_capacity).issubset(no_capacity):
                    raise ValueError(f'capacity failed to nest within no-capacity set: {region} {month} tau={tau}')
                prior_tau=comparison[(comparison.region==region)&(comparison.month.str[:10]==month)&(comparison.tau==tau)]
                if prior_tau.empty:raise ValueError('missing V90 comparison rows')
                detailed_width=int(prior_tau.baseline_upper.iloc[0]-prior_tau.baseline_lower.iloc[0])
                capacity_width=int(prior_tau.coarse_upper.iloc[0]-prior_tau.coarse_lower.iloc[0])
                for rate in v90.RATES:
                    req=int((F(str(rate))*T).__ceil__())
                    np,nneg,na,nw=v90.decide(no_capacity,actual,req);cp,cn,ca,cw=v90.decide(with_capacity,actual,req)
                    rows.append({'region':region,'month':month,'tau':tau,'rate':rate,'hours':T,'required_hours':req,'actual_K':actual,'no_capacity_lower':min(no_capacity),'no_capacity_upper':max(no_capacity),'capacity_lower':min(with_capacity),'capacity_upper':max(with_capacity),'detailed_month_lower':int(prior_tau.baseline_lower.iloc[0]),'detailed_month_upper':int(prior_tau.baseline_upper.iloc[0]),'v90_detailed_capacity_lower':int(prior_tau.coarse_lower.iloc[0]),'v90_detailed_capacity_upper':int(prior_tau.coarse_upper.iloc[0]),'no_capacity_positive':np,'no_capacity_negative':nneg,'no_capacity_abstain':na,'capacity_positive':cp,'capacity_negative':cn,'capacity_abstain':ca,'no_capacity_width':nw,'capacity_width':cw,'detailed_month_width':detailed_width,'v90_detailed_capacity_width':capacity_width})
    OUT.mkdir(parents=True,exist_ok=False)
    d=pd.DataFrame(rows);d.to_csv(OUT/'coarse_monthly_budget_frontier.csv',index=False)
    metrics=['no_capacity_positive','no_capacity_negative','no_capacity_abstain','capacity_positive','capacity_negative','capacity_abstain','no_capacity_width','capacity_width','detailed_month_width','v90_detailed_capacity_width']
    rm=d.groupby(['region','tau','rate','month'])[metrics].mean().reset_index().groupby(['region','tau','rate'])[metrics].mean().reset_index()
    macro=rm.groupby(['tau','rate'])[metrics].mean().reset_index();macro.to_csv(OUT/'region_macro_frontier.csv',index=False)
    summary={'status':'CHINA_STYLE_COARSE_MONTHLY_GENERATION_BUDGET_RETROSPECTIVE_PROXY','region_months':int(d[['region','month']].drop_duplicates().shape[0]),'scored_rows':len(d),'frontier':macro.to_dict('records'),'hourly_source_sha256':input_hashes,'capacity_source_sha256':cap_hashes,'source_audits':audits,'protocol_sha256':ph,'script_sha256':v90.sha(Path(__file__)),'limitations':['Monthly coarse generation margins are oracle-aggregated from EIA hourly fuel records, not Chinese public source rows.','Thermal subfuel EF composition is left unknown in [400,1000] and may vary between event and complement blocks.','The cohort is retrospectively selected by V90 target-hour capacity audits.','Fixed EF proxy is not actual CIF; no unseen-region or prospective coverage validation.']}
    if v90.sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(macro.to_string(index=False))

if __name__=='__main__':main()
