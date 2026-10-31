"""V94: stress the China-style coarse monthly information budget at high thresholds."""
from __future__ import annotations
import json
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
import v90_coarse_capacity_identification as v90
import v93_china_budget_coarsening as v93

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/V94_HIGH_THRESHOLD_COARSENING_PROTOCOL.md'
OUT=ROOT/'results/v94-high-threshold-coarsening-r2'
COARSE=ROOT/'results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv'
TAUS=(600,900);RATES=(.01,.05,.10)

def main():
    ph=v90.sha(PROTOCOL);frozen=pd.read_csv(COARSE).drop_duplicates(['region','month'])
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
            fine={f:float(raw.get(f,0)) for f in ('coal','gas','oil','nuclear','hydro','wind','solar','other')}
            fine_caps={f:fine[f] for f in v90.FUELS}
            coarse_fuel={f:F(str(v)) for f,v in fine.items()}
            cat_caps={'thermal':sum((coarse_fuel[f] for f in v93.THERMAL),F(0)),**{f:coarse_fuel[f] for f in v93.CLEAN}}
            no_caps={'thermal':sum((margins[f] for f in v93.THERMAL),F(0)),**{f:margins[f] for f in v93.CLEAN}}
            vals=[]
            for _,r in g.iterrows():
                v=[F(str(float(r[f'generation_{f}']))) for f in v90.FUELS]
                gen=sum(v,F(0));carbon=sum((v[i]*v90.EF[v90.FUELS[i]] for i in range(len(v))),F(0));vals.append((gen,carbon))
            for tau in TAUS:
                actual=sum(carbon>F(tau)*gen for gen,carbon in vals)
                coarse_no=v93.coarse_budget_set(margins,no_caps,tau,T)
                coarse_cap=v93.coarse_budget_set(margins,cat_caps,tau,T)
                detailed=v90.v66_set(margins,tau,T)
                fine_cap_set=v90.capacity_group_set(margins,fine_caps,tau,T,v90.FINE_GROUPS)
                fine_peak_ok=all(float(g[f'generation_{f}'].max())<=fine[f]+1e-6 for f in v90.FUELS)
                fine_energy_ok=all(float(margins[f])<=T*fine[f]+1e-6 for f in v90.FUELS)
                if not (fine_peak_ok and fine_energy_ok):fine_cap_set=None
                if actual not in coarse_no or actual not in coarse_cap or actual not in detailed:
                    raise ValueError(f'actual K excluded: {region} {month} tau={tau} K={actual}')
                if not set(coarse_cap).issubset(coarse_no):raise ValueError('capacity failed no-capacity nesting')
                if fine_cap_set is not None and (actual not in fine_cap_set or not set(fine_cap_set).issubset(coarse_cap)):
                    raise ValueError('coarse capacity set failed to contain feasible detailed-fuel set')
                for rate in RATES:
                    req=int((F(str(rate))*T).__ceil__())
                    dm,dmn,dma,dmw=v90.decide(detailed,actual,req)
                    cn,cnn,cna,cnw=v90.decide(coarse_no,actual,req)
                    cc,ccn,cca,ccw=v90.decide(coarse_cap,actual,req)
                    if fine_cap_set is not None:
                        fc,fcn,fca,fcw=v90.decide(fine_cap_set,actual,req);flo,fhi=min(fine_cap_set),max(fine_cap_set)
                    else:fc=fcn=fca=fcw=flo=fhi=None
                    rows.append({'region':region,'month':month,'tau':tau,'rate':rate,'hours':T,'actual_K':actual,'required_hours':req,'detailed_month_lower':min(detailed),'detailed_month_upper':max(detailed),'coarse_month_lower':min(coarse_no),'coarse_month_upper':max(coarse_no),'coarse_capacity_lower':min(coarse_cap),'coarse_capacity_upper':max(coarse_cap),'fine_capacity_lower':flo,'fine_capacity_upper':fhi,'fine_capacity_available':fine_cap_set is not None,'detailed_month_positive':dm,'detailed_month_negative':dmn,'detailed_month_abstain':dma,'coarse_month_positive':cn,'coarse_month_negative':cnn,'coarse_month_abstain':cna,'coarse_capacity_positive':cc,'coarse_capacity_negative':ccn,'coarse_capacity_abstain':cca,'fine_capacity_positive':fc,'fine_capacity_negative':fcn,'fine_capacity_abstain':fca,'detailed_month_width':dmw,'coarse_month_width':cnw,'coarse_capacity_width':ccw,'fine_capacity_width':fcw})
    OUT.mkdir(parents=True,exist_ok=False);d=pd.DataFrame(rows);d.to_csv(OUT/'high_threshold_frontier.csv',index=False)
    metrics=['detailed_month_positive','detailed_month_negative','detailed_month_abstain','coarse_month_positive','coarse_month_negative','coarse_month_abstain','coarse_capacity_positive','coarse_capacity_negative','coarse_capacity_abstain','fine_capacity_positive','fine_capacity_negative','fine_capacity_abstain','detailed_month_width','coarse_month_width','coarse_capacity_width','fine_capacity_width']
    rm=d.groupby(['region','tau','rate','month'])[metrics].mean().reset_index().groupby(['region','tau','rate'])[metrics].mean().reset_index();macro=rm.groupby(['tau','rate'])[metrics].mean().reset_index();macro.to_csv(OUT/'region_macro_frontier.csv',index=False)
    fine=d[d.fine_capacity_available].copy();fine_metrics=['detailed_month_positive','coarse_month_positive','coarse_capacity_positive','fine_capacity_positive','detailed_month_width','coarse_month_width','coarse_capacity_width','fine_capacity_width']
    fm=fine.groupby(['region','tau','rate','month'])[fine_metrics].mean().reset_index().groupby(['region','tau','rate'])[fine_metrics].mean().reset_index();fine_macro=fm.groupby(['tau','rate'])[fine_metrics].mean().reset_index();fine_macro.to_csv(OUT/'fine_capacity_paired_frontier.csv',index=False)
    summary={'status':'HIGH_THRESHOLD_COARSE_BUDGET_RETROSPECTIVE_PROXY','region_months':int(d[['region','month']].drop_duplicates().shape[0]),'fine_capacity_region_months':int(d[d.fine_capacity_available][['region','month']].drop_duplicates().shape[0]),'scored_rows':len(d),'frontier':macro.to_dict('records'),'fine_capacity_paired_frontier':fine_macro.to_dict('records'),'capacity_source_sha256':cap_hashes,'hourly_source_sha256':input_hashes,'source_audits':audits,'protocol_sha256':ph,'script_sha256':v90.sha(Path(__file__)),'limitations':['Coarse-budget frontier uses the retrospectively audited V90 358-month sample; fuel-specific capacity comparison is limited to its 326-month feasible intersection.','Thermal subfuel EF composition is withheld and allowed to vary within the V54 proxy support [400,1000].','EIA hourly generation is used to create oracle monthly coarse categories and evaluate actual K; this is not China source data or prospective forecast coverage.','Fixed EFs are proxy labels, not actual CIF.']}
    if v90.sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print(macro.to_string(index=False))

if __name__=='__main__':main()
