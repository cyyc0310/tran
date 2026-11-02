"""Label-matched hourly CIF/fraction, heteroscedastic and quantile references."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from transcif.research import resource_controls as resources
from transcif.research.multicountry import GROUPS, HISTORY, weights
from transcif.research.monthly_adaptation import weighted_quantile
from transcif.research.conditional_risk import interval_metrics

OPTIONS = dict(max_iter=200, max_leaf_nodes=15, min_samples_leaf=64,
               learning_rate=.05, l2_regularization=10., early_stopping=False, random_state=416)
RISKS = ('global', 'hetero', 'cqr_signed', 'cqr_nonnegative')


def design(frame: pd.DataFrame, climate: pd.DataFrame, *, weather: bool, capacity: bool) -> np.ndarray:
    """All hourly methods share public features; labels/IDs never enter the design."""
    public=frame
    if not weather:
        # The reused aggregator validates these fields; neutral placeholders are
        # discarded, so a no-issued-weather design requires no forecast values.
        public=frame.copy()
        public.loc[:,['temperature_c','wind_speed_ms','shortwave_w_per_m2']]=0.
    daily, _, _ = resources.day_inputs(public, climate)
    known = ['log_seasonal_wind_resource', 'log_seasonal_solar_resource',
             'log_seasonal_curve_wind', 'wind_resource_supported', 'solar_resource_supported',
             'curve_wind_supported', 'climate_temperature_now', 'climate_temperature_lag']
    issued = ['log_weather_wind_resource', 'log_weather_solar_resource', 'log_weather_curve_wind',
              'forecast_temperature', 'forecast_heating', 'forecast_cooling']
    extra = np.empty((len(frame), len(known)+(len(issued)+9 if weather else 0)))
    reference = climate.set_index(['region','month'])
    for d, ix in enumerate(resources.day_groups(frame)):
        extra[ix, :len(known)] = daily.iloc[d][known].to_numpy(float)
        if weather:
            extra[ix, len(known):len(known)+len(issued)] = daily.iloc[d][issued].to_numpy(float)
            part = frame.iloc[ix]; first = part.iloc[0]
            key = (first.region, pd.Period(first.latest_allowed_period,freq='M').month)
            old = reference.loc[key]
            temperature = part.temperature_c.to_numpy(float)
            wind = part.wind_speed_ms.to_numpy(float)
            solar = part.shortwave_w_per_m2.to_numpy(float)
            denominators = [float(old.wind_resource),float(old.solar_resource),float(old.wind_curve_resource)]
            numerators = [(np.minimum(wind,25.)**3),solar,resources.wind_curve(wind)]
            ratios = np.column_stack([v/b if np.isfinite(b) and b>0 else np.ones(len(ix))
                                      for v,b in zip(numerators,denominators)])
            row = np.column_stack([temperature/20., wind/10., solar/500.,
                np.maximum(18.-temperature,0.)/20.,np.maximum(temperature-22.,0.)/20.,
                np.log1p(ratios),
                # Expected renewable contributions in the same public monthly units.
                part.wind.to_numpy()*ratios[:,0]+part.solar.to_numpy()*ratios[:,1]])
            extra[ix, len(known)+len(issued):] = row
    clock = frame.clock.dt.normalize()
    month_end = frame.latest_allowed_period.map(lambda x:pd.Period(x,freq='M').end_time.normalize())
    age_days = (clock-month_end).dt.total_seconds().to_numpy()/86400.
    x = np.column_stack([frame[list(HISTORY)].to_numpy(float),age_days/90.,extra])
    if capacity:
        value = frame.capacity_factor.to_numpy(float)
        if not np.isfinite(value).all() or (value<=0).any():
            raise ValueError('positive public capacity factor required for this budget')
        x = np.column_stack([x,value/1000.])
    if not np.isfinite(x).all():
        raise ValueError('nonfinite hourly public design')
    return x


def fit(frame: pd.DataFrame, response: np.ndarray, x: np.ndarray, *, allowed_countries: tuple,
        progress: Callable[[str],None] | None = None) -> tuple[dict,np.ndarray]:
    """Three calendar-block OOF residuals train scale; source calibration is separate."""
    if (frame.empty or not set(frame.country)<=set(allowed_countries)
        or not frame.split.eq('train').all() or not frame.clock.dt.year.eq(2022).all()):
        raise ValueError('proper source-2022 training rows required')
    y = np.asarray(response,float); x = np.asarray(x,float)
    if y.shape!=(len(frame),) or x.ndim!=2 or len(x)!=len(frame) or not np.isfinite(x).all() or not np.isfinite(y).all() or (y<0).any() or (y>1000.).any():
        raise ValueError('aligned finite source response in 0–1000 units required')
    resources.day_groups(frame)
    w = weights(frame)
    block = np.where(frame.clock.dt.month<=4,0,np.where(frame.clock.dt.month<=6,1,2))
    if set(block)!={0,1,2}:
        raise ValueError('three nonempty source calendar blocks required')
    oof = np.empty(len(frame));folds=[]
    for held_block in (0,1,2):
        held = block==held_block; proper = ~held
        if set(frame.loc[proper,'country'])!=set(frame.country):
            raise ValueError('every source country needs proper rows in each fold')
        wf = weights(frame.loc[proper])
        model = HistGradientBoostingRegressor(**OPTIONS).fit(x[proper],y[proper],sample_weight=wf*len(wf))
        oof[held] = np.clip(model.predict(x[held]),0.,1000.)
        folds.append(dict(held_block=held_block,proper_hours=int(proper.sum()),held_hours=int(held.sum()),
                          proper_months=sorted(frame.loc[proper,'clock'].dt.month.unique().astype(int).tolist()),
                          held_months=sorted(frame.loc[held,'clock'].dt.month.unique().astype(int).tolist())))
        if progress:progress('OOF block '+str(held_block))
    mean = HistGradientBoostingRegressor(**OPTIONS).fit(x,y,sample_weight=w*len(w))
    if progress:progress('final mean')
    scale = HistGradientBoostingRegressor(loss='absolute_error',**OPTIONS).fit(x,np.abs(y-oof),sample_weight=w*len(w))
    if progress:progress('OOF absolute-error scale')
    quantiles=[]
    for alpha in (.05,.5,.95):
        quantiles.append(HistGradientBoostingRegressor(loss='quantile',quantile=alpha,**OPTIONS).fit(x,y,sample_weight=w*len(w)))
        if progress:progress('quantile '+str(alpha))
    return dict(mean=mean,scale=scale,quantiles=quantiles,
        audit=dict(source_countries=sorted(set(frame.country)),source_hours=len(frame),source_year=2022,
                   feature_count=x.shape[1],source_region_hours=frame.groupby('region').size().to_dict(),
                   source_country_weight={c:float(w[frame.country.eq(c)].sum()) for c in sorted(frame.country.unique())},
                   scale_training='absolute held-calendar-block prediction error; all regions held together',
                   folds=folds,options=OPTIONS,quantile_levels=[.05,.5,.95],target_fit=False)),oof


def forecast(fitted: dict, x: np.ndarray, factor: np.ndarray) -> dict:
    """Native CIF factor=1; per-mille thermal fraction factor=public EF/1000."""
    x = np.asarray(x,float);factor=np.asarray(factor,float)
    if factor.shape!=(len(x),) or not np.isfinite(factor).all() or (factor<=0).any() or (factor>1.).any():
        raise ValueError('aligned positive unit conversion factors required')
    point = np.clip(fitted['mean'].predict(x),0.,1000.)*factor
    scale = np.maximum(fitted['scale'].predict(x)*factor,1.)
    quantiles = np.sort(np.clip(np.column_stack([m.predict(x) for m in fitted['quantiles']]),0.,1000.),axis=1)*factor[:,None]
    return dict(point=point,scale=scale,lower=quantiles[:,0],median=quantiles[:,1],upper=quantiles[:,2])


def calibrate(prediction: dict, truth: np.ndarray, w: np.ndarray) -> dict:
    y=np.asarray(truth,float);w=np.asarray(w,float)
    if y.shape!=prediction['point'].shape or w.shape!=y.shape or not np.isfinite(y).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('aligned source calibration truth and weights required')
    score=np.maximum(prediction['lower']-y,y-prediction['upper'])
    return dict(global_radius=weighted_quantile(np.abs(y-prediction['point']),w,.9),
                hetero_multiplier=weighted_quantile(np.abs(y-prediction['point'])/prediction['scale'],w,.9),
                cqr_signed=weighted_quantile(score,w,.9),
                cqr_nonnegative=weighted_quantile(np.maximum(score,0.),w,.9))


def intervals(prediction: dict, calibration: dict) -> dict:
    p=prediction; result={}
    for risk in RISKS:
        if risk in ('global','hetero'):
            radius=calibration['global_radius'] if risk=='global' else calibration['hetero_multiplier']*p['scale']
            point=median=p['point'];lo,hi=point-radius,point+radius
        else:
            q=calibration[risk];point=median=p['median']
            # Median-envelope repair also enlarges nonempty sets excluding q50.
            lo=np.minimum(p['lower']-q,median);hi=np.maximum(p['upper']+q,median)
        result[risk]=dict(point=point,median=median,lower=np.clip(lo,0.,1000.),upper=np.clip(hi,0.,1000.))
    return result


def score_intervals(y: np.ndarray, point: np.ndarray, median: np.ndarray,
                    lower: np.ndarray, upper: np.ndarray) -> dict:
    y=np.asarray(y); median=np.asarray(median)
    metrics=interval_metrics(y,np.asarray(lower),np.asarray(upper))
    median_mae=float(np.abs(y-median).mean())
    return dict(mae=float(np.abs(y-point).mean()),median_mae=median_mae,
                wis90=(.5*median_mae+.05*metrics['interval_score90'])/1.5,**metrics)
