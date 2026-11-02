"""Label-matched direct CIF/fraction and conditional interval comparisons."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from collect_weather_transfer_pilot import ROOT, digest
from run_weather_transfer_pilot import joined_data
from run_resource_experiment import validate_observed_source_rows
from transcif.research import hourly_transfer_baselines as direct
from transcif.research import multicountry as mc
from transcif.research import coarse_trajectories as ct
from transcif.research import resource_controls as controls
from transcif.research import resource_controls_daily as daily
from transcif.research.resource_amplitudes import scale_paths
from transcif.research.capacity_transfer import load_capacity
from transcif.research.monthly_adaptation import weighted_quantile


def canonical_fuel(frame: pd.DataFrame, raw: dict) -> np.ndarray:
    fuel=np.empty((len(frame),len(mc.FINE)))
    for region,positions in frame.groupby('region').indices.items():
        part=frame.iloc[positions];label=raw[region].loc[part.label_row.astype(int)]
        for actual,declared in [(pd.to_datetime(label.time,utc=True).to_numpy(),part.valid_at.to_numpy()),
                                (label.clock.to_numpy(),part.clock.to_numpy()),(label.country.to_numpy(),part.country.to_numpy())]:
            if not np.array_equal(actual,declared):raise ValueError('canonical label identity differs')
        fuel[positions]=label[list(mc.FINE)].to_numpy()
    return fuel


def paired_recomposition(bank: dict, fitted: dict, public: pd.DataFrame, climate: pd.DataFrame) -> dict:
    """Only compute the three required R4 references; no model is retrained."""
    features,_,_=daily.day_inputs(public,climate)
    factors={name:controls.predict_amplitudes(fitted,features,'hgb',kind)
             for name,kind in [('paired_hgb_no_weather','B'),('paired_hgb_weather','W')]}
    result={name:np.empty((len(public),128)) for name in ('paired_joint',*factors)}
    for d,ix in enumerate(controls.day_groups(public)):
        part=public.iloc[ix];base,_=ct.predict(bank,part);result['paired_joint'][ix]=base['joint']
        for name,values in factors.items():
            modified={**bank,'q':scale_paths(bank['q'],bank['support'],values[d])}
            predicted,_=ct.predict(modified,part);result[name][ix]=predicted['joint']
    return result


def main() -> None:
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--weather-cache',type=Path,required=True)
    parser.add_argument('--climate-cache',type=Path,required=True)
    parser.add_argument('--source-training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();parent=ROOT/'results/resource-daily-clean-r4'
    inherited=json.loads((parent/'input_code_seal.json').read_text())
    for name,expected in inherited.items():
        if digest(Path(name))!=expected:raise ValueError('previous source/code seal changed')
    prior_outputs={}
    for seal in ['model_seal.json','prediction_seal.json']:
        for name,expected in json.loads((parent/seal).read_text()).items():
            if digest(parent/name)!=expected:raise ValueError('prior model/prediction seal changed')
            prior_outputs[str(parent/name)]=expected
    training=args.source_training.resolve();train_manifest=json.loads((training/'manifest.json').read_text())
    if (not train_manifest['complete'] or train_manifest['role']!='source_observed_weather_training'
        or train_manifest['eligible_for_issued_public'] is not False or train_manifest['available_at'] is not None):
        raise ValueError('explicit uncertified source-observed weather budget required')
    r4_cohort=[value for name,value in inherited.items() if name.endswith('/source-observed-weather-2022-r1/cohort.csv')]
    if len(r4_cohort)!=1 or digest(training/'cohort.csv')!=r4_cohort[0]:
        raise ValueError('paired training cohort must exactly match the frozen R4 amplitude training cohort')
    extra={str(training/name):h for name,h in train_manifest['files'].items()}
    extra.update(train_manifest['source_files'])
    climate_cache=args.climate_cache.resolve();climate_manifest=json.loads((climate_cache/'manifest.json').read_text())
    if not climate_manifest['complete'] or climate_manifest['eligible_for_issued_public'] is not False:
        raise ValueError('complete historical climate reference required')
    extra.update({str(climate_cache/name):h for name,h in climate_manifest['files'].items()})
    extra.update(climate_manifest['source_files'])
    for name,h in extra.items():
        if digest(Path(name))!=h:raise ValueError('source/climate input or producer changed')
    data,weather_seals=joined_data(args.weather_cache.resolve());raw,_=mc.load_countries(ROOT)
    train_all=pd.read_csv(training/'cohort.csv',parse_dates=['clock','valid_at'])
    validate_observed_source_rows(train_all,raw)
    climate=pd.read_csv(climate_cache/'climatology.csv')
    cap_path=ROOT/'results/public-capacity-input-r1/capacity.json';capacity=load_capacity(cap_path)
    for frame in (train_all,data):
        for region,group in frame.groupby('region'):
            c=capacity['regions'][region]
            frame.loc[group.index,'capacity_factor']=400.+600.*c['coal_mw']/(c['coal_mw']+c['gas_mw'])
    train_fuel=canonical_fuel(train_all,raw);eval_fuel=canonical_fuel(data,raw)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir()
    paths=[Path(__file__),ROOT/'src/transcif/research/hourly_transfer_baselines.py',
           ROOT/'tests/test_hourly_transfer_baselines.py',ROOT/'tests/test_hourly_transfer_runner.py',
           ROOT/'docs/HOURLY_DIRECT_BASELINES_PROTOCOL.md',training/'manifest.json',climate_cache/'manifest.json',
           cap_path,parent/'model_seal.json',parent/'prediction_seal.json',parent/'predictions.csv',
           ROOT/'scripts/run_resource_experiment.py']
    seals={**inherited,**prior_outputs,**extra,**weather_seals,**{str(p.resolve()):digest(p) for p in paths}}
    (out/'input_code_seal.json').write_text(json.dumps(seals,indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(),**{n:version(n) for n in ['numpy','pandas','scikit-learn','joblib']}),indent=2)+'\n')
    (out/'execution_plan.json').write_text(json.dumps(dict(source_labels='strict same 35448 paired hours; public past aggregates shared',
        source_weather='observed ERA5; source calibration and target inference issued GFS',
        reference_bank='new paired-label bank; old R4 full-source predictions extra-label reference only',
        risks=direct.RISKS,units=dict(cif='g/kWh native',thermal='permille fraction times EF/1000'),
        options=direct.OPTIONS,quantiles=[.05,.5,.95],source_calendar_blocks=[[3,4],[5,6],[7,8,9]],
        target_model_selection=False,new_country_opened=False,adaptive_development=True),indent=2)+'\n')
    predictions=[];calibrations=[];model_seals={};member_records=[]
    prior_pred=pd.read_csv(parent/'predictions.csv')

    def append_rows(query,tag,budget,method,risk,prediction,track='paired_labels'):
        rows=query[['country','region','origin','valid_at','label_row']].copy()
        rows['configuration'],rows['budget'],rows['method'],rows['risk'],rows['track']=tag,budget,method,risk,track
        for field in ['point','median','lower','upper']:rows[field]=prediction[field]
        predictions.append(rows)

    for target in ('DE','GB','US'):
        others=[c for c in ('DE','GB','US') if c!=target]
        for source in [(others[0],),(others[1],),tuple(others)]:
            tag='+'.join(source)+'_to_'+target;source_key='+'.join(sorted(source))
            train=train_all.loc[train_all.country.isin(source)].copy()
            cal=data.loc[data.country.isin(source)&data.split.eq('calibration')].copy()
            query=data.loc[data.country.eq(target)&data.split.eq('development')].copy()
            fuel=train_fuel[train.index];total=fuel.sum(axis=1)
            source_y=dict(cif=fuel@mc.FACTORS/total,thermal=1000.*fuel[:,:4].sum(axis=1)/total)
            cal_y=eval_fuel[cal.index]@mc.FACTORS/eval_fuel[cal.index].sum(axis=1);wcal=mc.weights(cal)
            bank_path=out/'models'/f'{source_key}_paired_bank.joblib'
            if not bank_path.exists():
                bank=ct.fit(train,fuel);joblib.dump(bank,bank_path,compress=3);model_seals[str(bank_path.relative_to(out))]=digest(bank_path)
            else:
                if digest(bank_path)!=model_seals[str(bank_path.relative_to(out))]:raise ValueError('cached paired bank changed')
                bank=joblib.load(bank_path)
                if bank['audit']['countries']!=sorted(source) or bank['audit']['eligible_hours']!=len(train):
                    raise ValueError('cached paired bank source cohort differs')
            r4_path=parent/f'{tag}_heads.joblib'
            fitted=joblib.load(r4_path)  # own parent artifact, verified and sealed above
            if fitted['audit']['source_countries']!=sorted(source) or fitted['audit']['source_hours']!=len(train):
                raise ValueError('parent amplitude source cohort differs')
            refs_cal=paired_recomposition(bank,fitted,cal,climate);refs_query=paired_recomposition(bank,fitted,query,climate)
            for budget in ('monthly','capacity'):
                cs=cal.capacity_factor.to_numpy() if budget=='capacity' else np.full(len(cal),bank['macro_ef'])
                cq=query.capacity_factor.to_numpy() if budget=='capacity' else np.full(len(query),bank['macro_ef'])
                for method,array in refs_query.items():
                    array=array*cq[:,None];source_array=refs_cal[method]*cs[:,None]
                    slo,shi=np.quantile(source_array,[.05,.95],axis=1)
                    signed=weighted_quantile(np.maximum(slo-cal_y,cal_y-shi),wcal,.9)
                    expansion=max(signed,0.)
                    calibrations.append(dict(configuration=tag,budget=budget,method=method,source_hours=len(cal),
                        signed_expansion=signed,nonnegative_expansion=expansion,paired_bank_macro_ef=bank['macro_ef']))
                    lo,med,hi=np.quantile(array,[.05,.5,.95],axis=1);point=array.mean(axis=1)
                    for risk,q in [('raw',0.),('calibrated',expansion),('calibrated_signed',signed)]:
                        append_rows(query,tag,budget,method,risk,dict(point=point,median=med,
                            lower=np.clip(np.minimum(lo-q,med),0.,1000.),upper=np.clip(np.maximum(hi+q,med),0.,1000.)))
                    p=out/f'{tag}_{budget}_{method}_members.npz';np.savez_compressed(p,members=array,row=query.index.to_numpy())
                    member_records.append(dict(file=p.name,sha256=digest(p),configuration=tag,budget=budget,method=method))
                xtrain={wx:direct.design(train,climate,weather=wx,capacity=budget=='capacity') for wx in (False,True)}
                xcal={wx:direct.design(cal,climate,weather=wx,capacity=budget=='capacity') for wx in (False,True)}
                xquery={wx:direct.design(query,climate,weather=wx,capacity=budget=='capacity') for wx in (False,True)}
                for target_kind in ('cif','thermal'):
                    conversion_cal=np.ones(len(cal)) if target_kind=='cif' else cs/1000.
                    conversion_query=np.ones(len(query)) if target_kind=='cif' else cq/1000.
                    for wx in (False,True):
                        method=target_kind+('_weather' if wx else '_no_weather')
                        model_path=out/'models'/f'{source_key}_{budget}_{method}.joblib'
                        oof_path=model_path.with_suffix('.oof.npz')
                        if not model_path.exists():
                            model,oof=direct.fit(train,source_y[target_kind],xtrain[wx],allowed_countries=source,
                                progress=lambda step:print(tag,budget,method,step,flush=True))
                            model['audit'].update(response=target_kind,weather=wx,budget=budget,
                                response_units='native g/kWh' if target_kind=='cif' else '1000*thermal generation/total generation')
                            joblib.dump(model,model_path,compress=3)
                            np.savez_compressed(oof_path,row=train.index.to_numpy(),response=source_y[target_kind],oof=oof,weights=mc.weights(train))
                            model_seals[str(model_path.relative_to(out))]=digest(model_path)
                            model_seals[str(oof_path.relative_to(out))]=digest(oof_path)
                        else:
                            for p in (model_path,oof_path):
                                if digest(p)!=model_seals[str(p.relative_to(out))]:raise ValueError('cached source model changed')
                            model=joblib.load(model_path)
                            if model['audit']['source_countries']!=sorted(source) or model['audit']['source_hours']!=len(train):
                                raise ValueError('cached direct model source cohort differs')
                            with np.load(oof_path,allow_pickle=False) as archive:
                                for field,expected in [('row',train.index.to_numpy()),('response',source_y[target_kind]),('weights',mc.weights(train))]:
                                    np.testing.assert_array_equal(archive[field],expected)
                        restored=joblib.load(model_path)
                        source_forecast=direct.forecast(model,xcal[wx],conversion_cal)
                        target_forecast=direct.forecast(model,xquery[wx],conversion_query)
                        for field,values in direct.forecast(restored,xquery[wx],conversion_query).items():
                            np.testing.assert_array_equal(values,target_forecast[field])
                        calibration=direct.calibrate(source_forecast,cal_y,wcal)
                        calibrations.append(dict(configuration=tag,budget=budget,method=method,source_hours=len(cal),**calibration))
                        for risk,prediction in direct.intervals(target_forecast,calibration).items():
                            append_rows(query,tag,budget,method,risk,prediction)
                # Previously frozen full-year banks used extra fine source labels.
                for old_method in ('joint_original','cube_weather_wind_solar','hgb_weather'):
                    member_path=parent/f'{tag}_full_source_extra_labels_{budget}_{old_method}_members.npz'
                    with np.load(member_path,allow_pickle=False) as a:
                        np.testing.assert_array_equal(a['row'],query.index.to_numpy());median=np.median(a['members'],axis=1)
                    for risk in ('raw','calibrated'):
                        old=prior_pred.loc[prior_pred.configuration.eq(tag)&prior_pred.track.eq('full_source_extra_labels')&prior_pred.budget.eq(budget)&prior_pred.method.eq(old_method)&prior_pred.risk.eq(risk)]
                        np.testing.assert_array_equal(old.valid_at.to_numpy(),query.valid_at.astype(str).to_numpy())
                        append_rows(query,tag,budget,'extra_full_'+old_method,risk,
                            dict(point=old.point.to_numpy(),median=median,lower=old.lower.to_numpy(),upper=old.upper.to_numpy()),'extra_full_year_reference')
            print(tag,'all source models and target predictions ready',flush=True)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    (out/'calibration.json').write_text(json.dumps(calibrations,indent=2)+'\n')
    (out/'model_seal.json').write_text(json.dumps(model_seals,indent=2)+'\n')
    output_seals={r['file']:r['sha256'] for r in member_records}
    output_seals.update({name:digest(out/name) for name in ['predictions.csv','calibration.json','runtime.json','execution_plan.json']})
    (out/'prediction_seal.json').write_text(json.dumps(output_seals,indent=2)+'\n')
    # Only now are target labels consumed for evaluation, after every method is sealed.
    pred=pd.read_csv(out/'predictions.csv');scores=[]
    for key,group in pred.groupby(['configuration','country','region','track','budget','method','risk'],sort=False):
        label=raw[key[2]].loc[group.label_row.astype(int)];g=label[list(mc.FINE)].to_numpy()
        y=g@mc.FACTORS/g.sum(axis=1);point=group.point.to_numpy()
        day=pd.DataFrame(dict(day=group.origin.to_numpy(),y=y,point=point))
        centered=day[['y','point']]-day.groupby('day')[['y','point']].transform('mean')
        scores.append(dict(zip(['configuration','country','region','track','budget','method','risk'],key))|
            dict(hours=len(group),shape_mae=float(np.abs(centered.y-centered.point).mean()),
                 **direct.score_intervals(y,point,group['median'].to_numpy(),group.lower.to_numpy(),group.upper.to_numpy())))
    metrics=pd.DataFrame(scores);metrics.to_csv(out/'region_metrics.csv',index=False)
    configs=metrics.groupby(['configuration','country','track','budget','method','risk']).mean(numeric_only=True).reset_index()
    configs=configs.rename(columns={'hours':'mean_hours_per_region'})
    configs.to_csv(out/'configuration_metrics.csv',index=False)
    configs.groupby(['track','budget','method','risk']).mean(numeric_only=True).reset_index().to_csv(out/'summary.csv',index=False)
    for name,h in seals.items():
        if digest(Path(name))!=h:raise RuntimeError('input/code changed during experiment')
    for name,h in {**output_seals,**model_seals}.items():
        if digest(out/name)!=h:raise RuntimeError('sealed model or forecast changed')
    (out/'completion.json').write_text(json.dumps(dict(complete=True,source_train_hours=len(train_all),unique_target_hours=int(data.split.eq('development').sum()),
        adaptive_development=True,target_calibration=False,new_country_opened=False,novelty_established=False,
        strict_paired_source_label_budget=True,probability_comparison='WIS90 and intervals; no full CRPS inferred from three quantiles'),indent=2)+'\n')


if __name__=='__main__':main()
