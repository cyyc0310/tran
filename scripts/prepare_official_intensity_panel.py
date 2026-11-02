"""Separate hourly generation features from official source/target labels."""
from __future__ import annotations

import hashlib
import json
import argparse
from openpyxl import load_workbook
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EIA_ROOT = ROOT/'data/eia930-hourly-co2-r1'
FEATURE_ROOT = EIA_ROOT/'feature_inputs'
SOURCE_LABEL_ROOT = EIA_ROOT/'source_labels_2022_2023'
TARGET_LABEL_ROOT = EIA_ROOT/'sealed_target_labels_2024'
SOURCE_HOURLY_ROOT = ROOT/'results/research/data'
NESO_PATH = ROOT/'data/neso-gb/df_fuel_ckan.csv'
REGIONS = ('BPAT','CISO','ERCO','FPL','ISNE','MISO','NYIS')
FUELS = ('coal','gas','oil','other','nuclear','hydro','wind','solar')
EIA_TO_GRAMS = 453.59237


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frame(frame: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression='gzip')
    return {'path':str(path.relative_to(ROOT)), 'rows':len(frame), 'sha256':sha(path)}


def eia_labels(region: str, years: tuple[int, ...]) -> tuple[pd.DataFrame, dict]:
    path=EIA_ROOT/f'{region}.xlsx'
    workbook=load_workbook(path,read_only=True,data_only=True)
    sheet=workbook['Published Hourly Data']
    headers=[str(c.value).strip() if c.value is not None else ''
             for c in next(sheet.iter_rows(min_row=1,max_row=1))]
    names=['UTC time','Time zone','CO2 Emissions Intensity for Consumed Electricity',
           'CO2 Emissions Consumed','Consumed Electricity']
    indexes={name:headers.index(name) for name in names}
    rows=[]
    for row in sheet.iter_rows(min_row=2,values_only=False):
        stamp=pd.to_datetime(row[indexes['UTC time']].value,utc=True,errors='coerce')
        if pd.isna(stamp) or stamp.year not in years:
            continue
        # Access response cells only for the requested source or sealed target year.
        rows.append((stamp,str(row[indexes['Time zone']].value or ''),
                     row[indexes[names[2]]].value,row[indexes[names[3]]].value,
                     row[indexes[names[4]]].value))
    workbook.close()
    data=pd.DataFrame(rows,columns=['time_utc','timezone','intensity','emissions','energy'])
    intensity=pd.to_numeric(data.intensity,errors='coerce')
    consumed_emissions=pd.to_numeric(data.emissions,errors='coerce')
    consumed_energy=pd.to_numeric(data.energy,errors='coerce')
    valid=intensity.notna()&consumed_emissions.notna()&consumed_energy.gt(0)
    value_g=intensity*EIA_TO_GRAMS
    implied_g=consumed_emissions/consumed_energy*1000.
    error=np.abs(value_g[valid]-implied_g[valid]).to_numpy(float)
    if not len(error) or not np.isfinite(error).all() or float(error.max())>.01:
        raise ValueError(f'EIA unit identity failed for {region}')
    out=pd.DataFrame({'time_utc':data.time_utc,'label_g_kwh':value_g})
    out=out.dropna(subset=['label_g_kwh'])
    if out.time_utc.duplicated().any():raise ValueError(f'duplicate EIA labels: {region}')
    timezone_values=sorted(data.timezone.dropna().astype(str).unique().tolist())
    unit_audit={'timezone_values':timezone_values,'label_hours':len(out),
                'unit_identity_rows':int(valid.sum()),
                'unit_identity_mean_abs_g_kwh':float(error.mean()),
                'unit_identity_p99_abs_g_kwh':float(np.quantile(error,.99)),
                'unit_identity_max_abs_g_kwh':float(error.max())}
    return out,unit_audit


def main() -> None:
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--target-labels',action='store_true',
                        help='extract held-out 2024 responses; run only after prediction seal')
    args=parser.parse_args()
    FEATURE_ROOT.mkdir(parents=True,exist_ok=True)
    SOURCE_LABEL_ROOT.mkdir(parents=True,exist_ok=True)
    TARGET_LABEL_ROOT.mkdir(parents=True,exist_ok=True)
    outputs={};unit_audit={}

    if args.target_labels:
        for region in REGIONS:
            labels,audit=eia_labels(region,(2024,));unit_audit[region]=audit
            outputs[f'{region}_target_labels']=write_frame(
                labels,TARGET_LABEL_ROOT/f'{region}_consumption_2024.csv.gz')
        # NESO target responses are opened only in this post-seal invocation.
        import csv
        rows=[]
        with NESO_PATH.open(newline='') as handle:
            reader=csv.DictReader(handle)
            for row in reader:
                stamp=pd.to_datetime(row['DATETIME'],utc=True,errors='raise')
                if stamp.year==2024:
                    value=pd.to_numeric(row['CARBON_INTENSITY'],errors='coerce')
                    if pd.notna(value):rows.append((stamp,float(value)))
        gb_labels=pd.DataFrame(rows,columns=['time_utc','label_g_kwh'])
        gb_hourly=gb_labels.groupby(gb_labels.time_utc.dt.floor('h')).size()
        if not gb_hourly.eq(2).all():raise ValueError('NESO target needs two half-hour labels per UTC hour')
        outputs['GB_target_labels']=write_frame(
            gb_labels,TARGET_LABEL_ROOT/'GB_consumption_2024_halfhour.csv.gz')
        print(json.dumps({'mode':'post_seal_target_labels','outputs':outputs,
                          'eia_unit_audit':unit_audit},indent=2))
        return

    # The EIA generation feature files are the model's source for power mix.
    for region in REGIONS:
        source=SOURCE_HOURLY_ROOT/f'{region}_hourly.csv'
        cols=['time',*(f'generation_{fuel}' for fuel in FUELS),'valid']
        raw=pd.read_csv(source,usecols=cols)
        data=raw[cols].copy()
        data['time_utc']=pd.to_datetime(data.time,utc=True,errors='raise')
        values=data[[f'generation_{fuel}' for fuel in FUELS]].to_numpy(float)
        valid=(data.valid.astype(str).str.lower().eq('true').to_numpy()&
               np.isfinite(values).all(axis=1)&(values>=0).all(axis=1)&(values.sum(axis=1)>0))
        data=data.loc[valid,['time_utc',*(f'generation_{fuel}' for fuel in FUELS)]]
        outputs[f'{region}_features']=write_frame(data,FEATURE_ROOT/f'{region}_generation_2022_2024.csv.gz')
        labels,audit=eia_labels(region,(2022,2023));unit_audit[region]=audit
        outputs[f'{region}_source_labels']=write_frame(
            labels,
            SOURCE_LABEL_ROOT/f'{region}_consumption_2022_2023.csv.gz')

    generation_columns=['DATETIME','GAS','COAL','NUCLEAR','WIND','WIND_EMB','HYDRO',
                        'BIOMASS','OTHER','SOLAR']
    generation=pd.read_csv(NESO_PATH,usecols=generation_columns)
    stamp=pd.to_datetime(generation.DATETIME,utc=True,errors='raise')
    generation['time_utc']=stamp
    generation['coal']=generation.COAL
    generation['gas']=generation.GAS
    generation['oil']=0.
    generation['other']=generation.BIOMASS+generation.OTHER
    generation['nuclear']=generation.NUCLEAR
    generation['hydro']=generation.HYDRO
    generation['wind']=generation.WIND+generation.WIND_EMB
    generation['solar']=generation.SOLAR
    gb_features=generation[['time_utc',*FUELS]].copy()
    gb_features=gb_features.loc[gb_features.time_utc.dt.year.between(2022,2024)]
    outputs['GB_features']=write_frame(gb_features,FEATURE_ROOT/'GB_generation_2022_2024.csv.gz')

    # Read the official intensity column separately, then put held-out-year
    # responses in a different file the runner opens only after prediction seal.
    import csv
    rows=[]
    with NESO_PATH.open(newline='') as handle:
        reader=csv.DictReader(handle)
        for row in reader:
            stamp=pd.to_datetime(row['DATETIME'],utc=True,errors='raise')
            if stamp.year in (2022,2023):
                value=pd.to_numeric(row['CARBON_INTENSITY'],errors='coerce')
                if pd.notna(value):rows.append((stamp,float(value)))
    gb_labels=pd.DataFrame(rows,columns=['time_utc','label_g_kwh'])
    if gb_labels.time_utc.duplicated().any():raise ValueError('duplicate NESO labels')
    gb_hourly=gb_labels.groupby(gb_labels.time_utc.dt.floor('h')).size()
    if not gb_hourly.eq(2).all():raise ValueError('NESO official intensity needs two half-hour labels per UTC hour')
    outputs['GB_source_labels']=write_frame(
        gb_labels.loc[gb_labels.time_utc.dt.year.isin([2022,2023])],
        SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz')
    manifest={'source_urls':{
        'EIA':'https://www.eia.gov/electricity/930-content/about.html',
        'NESO':'https://www.neso.energy/data-portal/historic-generation-mix'},
        'upstream':{},'outputs':outputs,'eia_unit_audit':unit_audit}
    upstream=[EIA_ROOT/'acquisition_manifest.json',EIA_ROOT/'coverage_audit_2022_2024.json']
    upstream += [SOURCE_HOURLY_ROOT/f'{r}_hourly.csv' for r in REGIONS]
    manifest['upstream']={str(p.relative_to(ROOT)):sha(p) for p in upstream}
    out=EIA_ROOT/'official_intensity_panel_manifest.json'
    out.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'mode':'features_and_source_labels','outputs':len(outputs),'manifest':str(out),
                      'eia_unit_audit':unit_audit},indent=2))


if __name__=='__main__':main()
