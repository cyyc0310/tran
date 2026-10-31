"""V98: uncertainty-set stress test for the threshold-side monthly energy statistic."""
from __future__ import annotations
import json
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
import v90_coarse_capacity_identification as v90
import v93_china_budget_coarsening as v93
import v96_threshold_side_energy as v96

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/V98_SIDE_ENERGY_MEASUREMENT_ERROR_PROTOCOL.md'
OUT=ROOT/'results/v98-side-energy-measurement-error-r2'
COHORT=ROOT/'results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv'
PAIRED=ROOT/'results/v97-side-vs-detail-same-capacity-r1/paired_frontier.csv'
ERRORS=(0,.01,.02,.05,.10)

def uncertain_side_set(margins,observed_high,high_range,low_range,caps,tau,T,error):
    mt=sum((margins[f] for f in v93.THERMAL),F(0));mc=sum((margins[f] for f in v93.CLEAN),F(0))
    h=F(observed_high);e=F(str(error));hlo=max(F(0),h*(1-e));hhi=min(mt,h*(1+e))
    qhmin=F(high_range[0]-tau);qlmin=F(low_range[0]-tau);qhmax=F(high_range[1]-tau);qlmax=F(low_range[1]-tau);t=F(tau)
    # Any feasible allocation with event high-side amount z admits a minimal
    # monthly high-side total H=max(Hlo,z): lowering H increases complement low
    # fuel and relaxes its non-event condition, until the measurement lower bound
    # or H=z is reached. Thus the union is the fixed-Hlo case plus the H=z case.
    ks=set(v96.threshold_side_set(margins,hlo,high_range,low_range,caps,tau,T))
    # Case H=z: all monthly high-side generation is allocated to the event block.
    # The complement then contains only below-threshold thermal and clean output.
    for k in range(1,T):
        bounds=v93.group_bounds(margins,caps,T,k);xlo,xhi=bounds['thermal'];ylo,yhi=bounds['clean']
        xlo=max(xlo,hlo);limit=sum(margins.values(),F(0))-ylo
        if xlo>=limit or xlo>xhi:continue
        xhi=min(xhi,limit)
        candidates={xlo,xhi}
        if xlo<=hhi<=xhi:candidates.add(hhi)
        qmax=max(qlmax*x+(qhmax-qlmax)*min(hhi,x)-t*ylo for x in candidates)
        if qmax>0:ks.add(k)
    if qhmax*hhi+qlmax*(mt-hhi)-t*mc>0:ks.add(T)
    return tuple(sorted(ks))

def main():
    ph=v90.sha(PROTOCOL);frozen=pd.read_csv(COHORT).drop_duplicates(['region','month'])
    cohort={(str(r.region),str(r.month)[:10]) for r in frozen.itertuples(index=False)}
    reference=pd.read_csv(PAIRED).drop_duplicates(['region','month','tau','rate'])
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
            T=len(g);cy=int(month[:4])-1
            margins={f:sum((F(str(float(x))) for x in g[f'generation_{f}']),F(0)) for f in v90.FUELS}
            raw=caps[cy].loc[region] if region in caps[cy].index else pd.Series(dtype=float)
            capf={f:F(str(float(raw.get(f,0)))) for f in ('coal','gas','oil','nuclear','hydro','wind','solar','other')}
            capgroup={group:sum((capf[f] for f in fuels),F(0)) for group,fuels in v90.GROUPS.items()}
            capcats={'thermal':capgroup['thermal'],**{f:capgroup[f] for f in v93.CLEAN}}
            no_cap={'thermal':sum((margins[f] for f in v93.THERMAL),F(0)),**{f:margins[f] for f in v93.CLEAN}}
            vals=[]
            for _,r in g.iterrows():
                v=[F(str(float(r[f'generation_{f}']))) for f in v90.FUELS];gen=sum(v,F(0));c=sum((v[i]*v90.EF[v90.FUELS[i]] for i in range(len(v))),F(0));vals.append((gen,c))
            for tau in v96.TAUS:
                if tau==600:high=margins['coal']+margins['oil'];hr=(800,1000);lr=(400,400)
                else:high=margins['coal'];hr=(1000,1000);lr=(400,800)
                actual=sum(c>F(tau)*gen for gen,c in vals)
                coarse=v93.coarse_budget_set(margins,capcats,tau,T)
                exact_side=v96.threshold_side_set(margins,high,hr,lr,capcats,tau,T)
                detailcap=v90.capacity_group_set(margins,capcats,tau,T,v90.GROUPS)
                if detailcap is None:raise ValueError('frozen coarse-capacity cohort unexpectedly infeasible')
                prior=reference[(reference.region==region)&(reference.month.str[:10]==month)&(reference.tau==tau)]
                if len(prior)!=len(v96.RATES):raise ValueError('missing V97 paired rows')
                for err in ERRORS:
                    ks=uncertain_side_set(margins,high,hr,lr,capcats,tau,T,err)
                    if actual not in ks or not set(exact_side).issubset(ks) or not set(ks).issubset(coarse):
                        raise ValueError(f'uncertain set nesting failure {region} {month} tau={tau} e={err}')
                    if err==0 and ks!=exact_side:raise ValueError('zero-error set differs from V96 exact side set')
                    for rate in v96.RATES:
                        req=int((F(str(rate))*T).__ceil__());p,n,a,w=v90.decide(ks,actual,req);dp,dn,da,dw=v90.decide(detailcap,actual,req)
                        rows.append({'region':region,'month':month,'tau':tau,'rate':rate,'relative_error':err,'hours':T,'actual_K':actual,'side_lower':min(ks),'side_upper':max(ks),'side_positive':p,'side_negative':n,'side_abstain':a,'side_width':w,'detailed_plus_same_coarse_cap_lower':min(detailcap),'detailed_plus_same_coarse_cap_upper':max(detailcap),'detailed_positive':dp,'detailed_negative':dn,'detailed_abstain':da,'detailed_width':dw,'exact_side_width':max(int(prior.threshold_side_plus_coarse_cap_upper.iloc[0]-prior.threshold_side_plus_coarse_cap_lower.iloc[0]),0)})
    OUT.mkdir(parents=True,exist_ok=False);d=pd.DataFrame(rows);d.to_csv(OUT/'measurement_error_frontier.csv',index=False)
    metrics=['side_positive','side_negative','side_abstain','side_width','detailed_positive','detailed_negative','detailed_abstain','detailed_width']
    rm=d.groupby(['relative_error','region','tau','rate','month'])[metrics].mean().reset_index().groupby(['relative_error','region','tau','rate'])[metrics].mean().reset_index();macro=rm.groupby(['relative_error','tau','rate'])[metrics].mean().reset_index();macro.to_csv(OUT/'region_macro_frontier.csv',index=False)
    summary={'status':'SIDE_ENERGY_MEASUREMENT_UNCERTAINTY_RETROSPECTIVE_PROXY','region_months':int(d[['region','month']].drop_duplicates().shape[0]),'scored_rows':len(d),'frontier':macro.to_dict('records'),'capacity_source_sha256':cap_hashes,'hourly_source_sha256':input_hashes,'source_audits':audits,'protocol_sha256':ph,'script_sha256':v90.sha(Path(__file__)),'limitations':['Error interval is relative uncertainty in the threshold-side MWh, while thermal total is treated exact.','The exact error-union derivation splits feasible allocations by whether event high-side output is below or above the measurement lower endpoint.','Oracle EIA source, fixed EF ranges, retrospectively selected cohort, no China validation.']}
    if v90.sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print(macro.to_string(index=False))

if __name__=='__main__':main()
