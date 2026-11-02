#!/usr/bin/env python3
"""Paired annual fossil heat-rate information test on US↔GB official intensity."""
from __future__ import annotations
import argparse, calendar, hashlib, json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_eia923_pseudotarget_transfer as pseudo
import run_official_intensity_transfer as official
from run_cross_country import ROOT

PROTOCOL=ROOT/'docs/THERMAL_HEATRATE_OFFICIAL_INCREMENT_PROTOCOL_20261101.md'
DUKES=ROOT/'docs/evidence/thermal-rate-official-increment-r1/input/DUKES_5.6_2026.xlsx'
OUT_DEFAULT=ROOT/'results/thermal-rate-official-increment-r1'
BA=('BPAT','CISO','ERCO','FPL','ISNE','MISO','NYIS')
FEATURES=tuple(official.FEATURES)
METHODS=('coarse_ridge_M','coarse_ridge_MH','coarse_hgb_M','coarse_hgb_MH','fixed_ef_recomposition')

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def eia_rates()->tuple[pd.DataFrame,list[Path]]:
    rows=[]; files=[]
    for year in (2020,2021,2022):
        root=ROOT/('docs/evidence/eia923-monthly-pseudotarget-r1/input' if year<2022 else 'docs/evidence/eia923-ba-mapping-r1')
        path=root/f'f923_{year}.zip';files.append(path)
        with ZipFile(path) as z:
            member=next(n for n in z.namelist() if n.startswith('EIA923_Schedules_2_3_4_5_M_'))
            wb=load_workbook(BytesIO(z.read(member)),read_only=True,data_only=True)
            try:
                ws=wb['Page 1 Generation and Fuel Data']
                header=next(ws.iter_rows(min_row=6,max_row=6,values_only=True))
                col={str(v).replace('\n',' ').strip():i for i,v in enumerate(header) if v is not None}
                fuel_col='MER Fuel Type Code' if year>=2022 else 'AER Fuel Type Code'
                monthly={m:(col[f'Netgen {calendar.month_name[m]}'],col[f'Elec_MMBtu {calendar.month_name[m]}']) for m in range(1,13)}
                totals={b:[0.0,0.0] for b in BA}
                for r in ws.iter_rows(min_row=7,values_only=True):
                    ba=str(r[col['Balancing Authority Code']] or '').strip()
                    if ba not in totals:continue
                    fuel=str(r[col[fuel_col]] or '').strip()
                    if fuel not in {'COL','NG','DFO','RFO','PC'}:continue
                    for gc,hc in monthly.values():
                        try:g=float(r[gc]);h=float(r[hc])
                        except (TypeError,ValueError):continue
                        if np.isfinite(g) and np.isfinite(h) and g>0 and h>=0:
                            totals[ba][0]+=g;totals[ba][1]+=h
                for ba,(gen,heat) in totals.items():
                    rows.append({'region':ba,'country':'US','stat_year':year,'fossil_netgen_mwh':gen,'fossil_elec_mmbtu':heat,'fossil_heat_rate_mmbtu_mwh':heat/gen if gen>0 else np.nan})
            finally:wb.close()
    return pd.DataFrame(rows),files

def gb_rates()->pd.DataFrame:
    wb=load_workbook(DUKES,data_only=True,read_only=True);ws=wb['5.6 Annual summaries']; rows=[]
    try:
        vals=list(ws.iter_rows(values_only=True))
        for year in (2020,2021,2022):
            start=next(i for i,r in enumerate(vals) if r[0]==f'Table 5.6.J {year} summary of electricity fuel use, generation and supply, GWh')
            header=vals[start+1]
            rows_for={}
            for r in vals[start+2:start+30]:
                if len(r)>1 and r[0]=='All generating companies' and r[1] in ('Fuel used','Generation '): rows_for[r[1]]=r
            fuel_row=rows_for['Fuel used']; gen_row=rows_for['Generation ']
            fossil=('Coal (thermal)','Oil (thermal)','Gas (thermal)')
            ix={str(v):j for j,v in enumerate(header) if v is not None}
            fuel=sum(float(fuel_row[ix[x]]) for x in fossil); gen=sum(float(gen_row[ix[x]]) for x in fossil)
            rows.append({'region':'GB','country':'GB','stat_year':year,'fossil_gen_gwh':gen,'fossil_fuel_input_gwh':fuel,'fossil_heat_rate_mmbtu_mwh':fuel/gen*3.412141633 if gen>0 else np.nan})
    finally:wb.close()
    return pd.DataFrame(rows)

def weighted(frame:pd.DataFrame)->np.ndarray:
    n=frame.groupby('region').region.transform('size').to_numpy(float);return 1/(n*frame.region.nunique())

def train_models(frame:pd.DataFrame,y:np.ndarray)->dict:
    good=np.isfinite(y);x=frame.loc[good].reset_index(drop=True);yy=y[good];w=weighted(x)
    models={}
    for name,cols in [('coarse_ridge_M',FEATURES),('coarse_ridge_MH',FEATURES+('fossil_heat_rate_mmbtu_mwh',))]:
        model=make_pipeline(StandardScaler(),Ridge(alpha=10.))
        model.fit(x[list(cols)],yy,standardscaler__sample_weight=w,ridge__sample_weight=w*len(w));models[name]=(model,cols)
    for name,cols in [('coarse_hgb_M',FEATURES),('coarse_hgb_MH',FEATURES+('fossil_heat_rate_mmbtu_mwh',))]:
        model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=10.,random_state=412,early_stopping=False)
        model.fit(x[list(cols)],yy,sample_weight=w*len(w));models[name]=(model,cols)
    return models

def predict(models:dict,proxy:dict,frame:pd.DataFrame)->dict[str,np.ndarray]:
    out={}
    for name,(model,cols) in models.items():out[name]=np.clip(model.predict(frame[list(cols)]),0,1000)
    out['fixed_ef_recomposition']=np.clip(official.predict_models(proxy,frame)['coarse_recomposition'],0,1000)
    return out

def calibrate(parts:dict[str,list[np.ndarray]])->dict:
    out={}
    for name,vs in parts.items():
        bias=float(np.mean([v.mean() for v in vs]));sigma=float(np.sqrt(np.mean([np.mean((v-bias)**2) for v in vs])))
        out[name]={'bias':bias,'sigma':max(sigma,1.),'regions':len(vs),'calibration_hours':int(sum(map(len,vs)))}
    return out

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT_DEFAULT);a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    capacity=official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH);raw,base,labels,audit=official.prepare(ROOT,capacity)
    monthly=pseudo.parse_monthly(tuple(range(2020,2025)),official.US_REGIONS);prepared_by_arm,_=pseudo.replace_source_features(base,monthly);prepared=prepared_by_arm['eia923_monthly']
    er,ezips=eia_rates();gr=gb_rates();rates=pd.concat([er,gr],ignore_index=True)
    if rates.fossil_heat_rate_mmbtu_mwh.isna().any() or not rates.fossil_heat_rate_mmbtu_mwh.between(3,25).all():raise ValueError('unexpected/invalid annual fossil heat rate')
    rates.to_csv(out/'annual_fossil_heat_rates.csv',index=False)
    for region,frame in prepared.items():
        q=frame.year.map(lambda y: rates.loc[(rates.region.eq(region))&(rates.stat_year.eq(int(y)-2)),'fossil_heat_rate_mmbtu_mwh'].iloc[0] if ((rates.region.eq(region))&(rates.stat_year.eq(int(y)-2))).any() else np.nan)
        frame['fossil_heat_rate_mmbtu_mwh']=q.to_numpy(float)
        if frame.fossil_heat_rate_mmbtu_mwh.isna().any():raise ValueError(f'missing lagged rate {region}')
    code=[Path(__file__),PROTOCOL,DUKES,ROOT/'scripts/run_official_intensity_transfer.py',ROOT/'scripts/run_eia923_pseudotarget_transfer.py',ROOT/'scripts/prepare_official_intensity_panel.py',ROOT/'scripts/run_cross_country.py',ROOT/'src/transcif/research/cross_country.py']
    data=ezips+[DUKES,official.EIA_ROOT/'acquisition_manifest.json',official.EIA_ROOT/'coverage_audit_2022_2024.json']
    data += [official.SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz' for r in BA]
    data += [official.SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz',official.EIA_ROOT/'feature_inputs/GB_generation_2022_2024.csv.gz']
    data += [ROOT/f'results/research/data/{r}_hourly.csv' for r in BA]
    hashes={'code':{str(p):sha(p) for p in dict.fromkeys(code)},'data':{str(p):sha(p) for p in dict.fromkeys(data)}}
    (out/'startup_hashes.json').write_text(json.dumps(hashes,indent=2))
    (out/'heat_rate_audit.json').write_text(json.dumps({'source':'EIA-923 (US) and DUKES 5.6 (GB)','feature_lag_years':2,'target_year_to_stat_year':{str(y):y-2 for y in (2022,2023,2024)},'rates':rates.to_dict('records'),'eia_vintage':'current Final archives','gb_vintage':'DUKES 2026 current revised values; not historical as-issued'},indent=2))
    seals=[];pending=[]
    for src,tgt in [('US','GB'),('GB','US')]:
        sources=[r for r,f in prepared.items() if f.country.iloc[0]==src];targets=[r for r,f in prepared.items() if f.country.iloc[0]==tgt]
        train_parts=[];label_parts=[];fuel_parts=[]
        for r in sources:
            q=prepared[r].loc[prepared[r].year.eq(2022)].copy()
            yy=official.aligned_labels(q,labels[r]);good=np.isfinite(yy)
            q=q.loc[good].copy();yy=yy[good]
            train_parts.append(q);label_parts.append(yy)
            fuel_parts.append(raw[r].iloc[q.raw_row.to_numpy()][list(official.FUELS)].to_numpy(float))
        train=pd.concat(train_parts,ignore_index=True);yy=np.concatenate(label_parts);fuel=np.concatenate(fuel_parts)
        models=train_models(train,yy)
        proxy=official.fit_models(train, fuel)
        residuals={m:[] for m in METHODS}
        for r in sources:
            cal=prepared[r].loc[prepared[r].year.eq(2023)].copy();y=official.aligned_labels(cal,labels[r]);good=np.isfinite(y);cal=cal.loc[good];y=y[good]
            pp=predict(models,proxy,cal)
            for m in METHODS:residuals[m].append(y-pp[m])
        calibration=calibrate(residuals);(out/f'{src}_to_{tgt}_calibration.json').write_text(json.dumps(calibration,indent=2))
        for r in targets:
            q=prepared[r].loc[prepared[r].year.eq(2024)].copy();pp=predict(models,proxy,q);path=out/f'{src}_to_{r}_predictions.csv.gz'
            pd.DataFrame({'time_utc':pd.to_datetime(q.time,utc=True).astype(str),'observed_month':q.observed_month.to_numpy(),**pp}).to_csv(path,index=False,compression='gzip')
            seals.append({'source':src,'target':r,'path':path.name,'sha256':sha(path),'hours_predicted':len(q)});pending.append((src,tgt,r,q,path,calibration));print(src,'to',r,'sealed',len(q),flush=True)
    (out/'prediction_seal.json').write_text(json.dumps(seals,indent=2))
    if any(sha(Path(p))!=d for p,d in hashes['code'].items()) or any(sha(Path(p))!=d for p,d in hashes['data'].items()):raise RuntimeError('startup input/code hash changed during run')
    scores=[]
    for src,tgt,r,q,path,cal in pending:
        target=official.source_labels(r,years=(2024,));y=official.aligned_labels(q,target);pred=pd.read_csv(path)
        for month in sorted(q.clock.dt.month.unique()):
            mask=q.clock.dt.month.eq(month).to_numpy()
            for method in METHODS:
                m=cal[method];z=official.score(y[mask],pred.loc[mask,method].to_numpy(float),m['bias'],m['sigma'],q.loc[mask,'clock'].reset_index(drop=True));scores.append({'source':src,'target':tgt,'region':r,'method':method,'month':int(month),**z})
    metrics=pd.DataFrame(scores);metrics.to_csv(out/'monthly_metrics.csv',index=False);cols=['mae','shape_mae','crps_gaussian','coverage90','width90','interval_score90']
    reg=metrics.groupby(['source','target','region','method'])[cols].mean().reset_index();reg.to_csv(out/'region_metrics.csv',index=False)
    country=reg.groupby(['source','target','method'])[cols].mean().reset_index();country.to_csv(out/'country_metrics.csv',index=False)
    (out/'target_scoring_hashes.json').write_text(json.dumps({r:sha(official.TARGET_LABEL_ROOT/('GB_consumption_2024_halfhour.csv.gz' if r=='GB' else f'{r}_consumption_2024.csv.gz')) for _,_,r,_,_,_ in pending},indent=2))
    (out/'status.json').write_text(json.dumps({'status':'DEVELOPMENT_HEATRATE_INFORMATION_INCREMENT','source_train_year':2022,'source_calibration_year':2023,'target_year':2024,'target_labels_previously_viewed':True,'target_labels_read_after_prediction_seal':True,'annual_rate_input_lag_years':2,'m_plus_h':'M plus one annual aggregate fossil fuel heat rate','blind_confirmation':False},indent=2))
    print(country.to_string(index=False))
if __name__=='__main__':main()
