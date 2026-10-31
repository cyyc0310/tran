"""V97: compare threshold-side and detailed monthly information under identical coarse capacities."""
from __future__ import annotations
import json
from fractions import Fraction as F
from pathlib import Path
import pandas as pd
import v90_coarse_capacity_identification as v90
import v93_china_budget_coarsening as v93
import v96_threshold_side_energy as v96

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/V97_SIDE_VS_DETAIL_SAME_CAPACITY_PROTOCOL.md'
OUT=ROOT/'results/v97-side-vs-detail-same-capacity-r1'
COHORT=ROOT/'results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv'

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
            cap_f={f:F(str(float(raw.get(f,0)))) for f in ('coal','gas','oil','nuclear','hydro','wind','solar','other')}
            cap_group={group:sum((cap_f[f] for f in fuels),F(0)) for group,fuels in v90.GROUPS.items()}
            no_cap={'thermal':sum((margins[f] for f in v93.THERMAL),F(0)),**{f:margins[f] for f in v93.CLEAN}}
            cap_categories={'thermal':cap_group['thermal'],**{f:cap_group[f] for f in v93.CLEAN}}
            vals=[]
            for _,r in g.iterrows():
                v=[F(str(float(r[f'generation_{f}']))) for f in v90.FUELS]
                gen=sum(v,F(0));carbon=sum((v[i]*v90.EF[v90.FUELS[i]] for i in range(len(v))),F(0));vals.append((gen,carbon))
            for tau in v96.TAUS:
                if tau==600:high=margins['coal']+margins['oil'];hrange=(800,1000);lrange=(400,400)
                else:high=margins['coal'];hrange=(1000,1000);lrange=(400,800)
                actual=sum(c>F(tau)*gen for gen,c in vals)
                detailed=v90.v66_set(margins,tau,T)
                side_no=v96.threshold_side_set(margins,high,hrange,lrange,no_cap,tau,T)
                side_cap=v96.threshold_side_set(margins,high,hrange,lrange,cap_categories,tau,T)
                detail_cap=v90.capacity_group_set(margins,cap_categories,tau,T,v90.GROUPS)
                if detail_cap is None:raise ValueError(f'detailed + coarse capacity infeasible in frozen cohort {region} {month}')
                coarse_no=v93.coarse_budget_set(margins,no_cap,tau,T)
                coarse_cap=v93.coarse_budget_set(margins,cap_categories,tau,T)
                if any(actual not in s for s in (detailed,side_no,side_cap,detail_cap,coarse_no,coarse_cap)):
                    raise ValueError(f'observed K excluded {region} {month} tau={tau}')
                if not set(detailed).issubset(side_no):raise ValueError('side monthly set failed to contain detailed set')
                if not set(detail_cap).issubset(side_cap):raise ValueError('side+same-capacity set failed to contain detailed+same-capacity set')
                if not set(side_no).issubset(coarse_no) or not set(side_cap).issubset(coarse_cap):raise ValueError('side information set not nested in coarsest budget')
                for rate in v96.RATES:
                    req=int((F(str(rate))*T).__ceil__());rec={'region':region,'month':month,'tau':tau,'rate':rate,'hours':T,'actual_K':actual,'required_hours':req}
                    sets={'detailed_month':detailed,'threshold_side_month':side_no,'detailed_month_plus_coarse_cap':detail_cap,'threshold_side_plus_coarse_cap':side_cap,'coarse_month':coarse_no,'coarse_month_plus_capacity':coarse_cap}
                    for name,ks in sets.items():
                        p,n,a,w=v90.decide(ks,actual,req);rec.update({f'{name}_lower':min(ks),f'{name}_upper':max(ks),f'{name}_positive':p,f'{name}_negative':n,f'{name}_abstain':a,f'{name}_width':w})
                    rows.append(rec)
    OUT.mkdir(parents=True,exist_ok=False);d=pd.DataFrame(rows);d.to_csv(OUT/'paired_frontier.csv',index=False)
    metrics=[c for c in d if c.endswith(('_positive','_negative','_abstain','_width'))]
    rm=d.groupby(['region','tau','rate','month'])[metrics].mean().reset_index().groupby(['region','tau','rate'])[metrics].mean().reset_index();macro=rm.groupby(['tau','rate'])[metrics].mean().reset_index();macro.to_csv(OUT/'region_macro_frontier.csv',index=False)
    summary={'status':'SAME_COARSE_CAPACITY_SIDE_VS_DETAIL_RETROSPECTIVE_PROXY','region_months':int(d[['region','month']].drop_duplicates().shape[0]),'scored_rows':len(d),'frontier':macro.to_dict('records'),'capacity_source_sha256':cap_hashes,'hourly_source_sha256':input_hashes,'source_audits':audits,'protocol_sha256':ph,'script_sha256':v90.sha(Path(__file__)),'limitations':['Threshold-side energy is oracle-aggregated from EIA hourly fuel data.','The fixed 358-month cohort was selected using retrospective target-hour capacity audits.','EF intervals and source-specific mapping uncertainty are not validated for China.']}
    if v90.sha(PROTOCOL)!=ph:raise RuntimeError('protocol changed')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print(macro.to_string(index=False))

if __name__=='__main__':main()
