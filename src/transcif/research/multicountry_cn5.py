"""Five-group public-input contract with nine-category source-label proxy."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transcif.data.smard import load_native
from transcif.research.cross_country import FUELS, EF, load_inputs

FINE = (*FUELS, 'other_renewable')
FACTORS = np.r_[EF, 0.]
GROUPS = ('thermal', 'hydro', 'nuclear', 'wind', 'solar')
LATEST = tuple([f'cal{i}' for i in range(6)]+list(GROUPS)+
               [f'{g}_h{s}' for g in GROUPS for s in ('sin','cos')])
HISTORY = (*LATEST, 'thermal_mean', 'thermal_std', 'thermal_change', 'total_cv', 'observations')
METHODS = ('anchor700','macro_recomposition','ratio_recomposition','ridge_history',
           'direct_latest','direct_history','factor_latest','factor_history')


def load_countries(root: Path) -> tuple[dict,dict]:
    raw,audit=load_inputs(root)
    for frame in raw.values():
        # This is a bookkeeping split: legacy "other" remains unresolved.
        frame['other_renewable']=0.
    manifests=[root/f'results/smard-de-{year}-full.json' for year in (2022,2023)]
    de,meta=load_native(manifests,root)
    for year in meta['years'].values():
        if year['hours']!=year['expected_hours'] or year['nonfinite'] or year['negative']:
            raise ValueError('German source quality gate failed')
    frame=pd.DataFrame(index=de.index)
    frame['coal']=de.lignite+de.hard_coal
    frame['gas'],frame['oil']=de.gas,0.
    frame['other']=de.biomass+de.other_conventional
    frame['nuclear'],frame['hydro']=de.nuclear,de.hydro
    frame['wind'],frame['solar']=de.wind_onshore+de.wind_offshore,de.solar
    frame['other_renewable']=de.other_renewable
    expected=de.drop(columns='clock').sum(axis=1)-de.pumped_storage
    np.testing.assert_allclose(frame[list(FINE)].sum(axis=1),expected,atol=1e-6)
    frame['time']=de.index
    frame['clock']=de.clock
    frame['country'],frame['region']='DE','DE'
    periods=de.clock.dt.to_period('M')
    hours={p: ((p+1).start_time.tz_localize('Europe/Berlin')-
               p.start_time.tz_localize('Europe/Berlin')).total_seconds()/3600 for p in periods.unique()}
    frame['expected_hours']=periods.map(hours)
    raw['DE']=frame.reset_index(drop=True)
    audit['DE']={'manifests':[str(p.relative_to(root)) for p in manifests], 'native_audit':meta,
                 'oil':'not separately resolved; retained in other_conventional proxy',
                 'storage':'excluded from primary generation denominator'}
    return raw,audit


def coarse_values(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([frame[list(FUELS[:4])].sum(axis=1), frame.hydro,frame.nuclear,
                             frame.wind,frame.solar])


def public_months(frame: pd.DataFrame) -> pd.DataFrame:
    period=frame.clock.dt.to_period('M')
    g=pd.DataFrame(coarse_values(frame),columns=GROUPS,index=frame.index)
    g['period']=period
    sums=g.groupby('period')[list(GROUPS)].sum()
    coverage=g.groupby('period').size()/frame.groupby(period).expected_hours.first()
    return sums.loc[coverage>=.99]


def public_features(months: pd.DataFrame, clock: pd.Series) -> pd.DataFrame:
    """No hourly generation/labels accepted; exact t-2, older months only."""
    out=pd.DataFrame({'clock':clock})
    out['period']=clock.dt.to_period('M')
    for i,(v,denom) in enumerate([(clock.dt.hour,24),(clock.dt.dayofweek,7),(clock.dt.month,12)]):
        out[f'cal{2*i}']=np.sin(v*2*np.pi/denom)
        out[f'cal{2*i+1}']=np.cos(v*2*np.pi/denom)
    records={}
    for p in out.period.unique():
        history=months.loc[(months.index<=p-2)&(months.index>=p-7)]
        if p-2 not in history.index:
            continue
        shares=history.div(history.sum(axis=1),axis=0)
        latest=shares.loc[p-2]
        values=latest.to_dict()
        values.update(thermal_mean=float(shares.thermal.mean()),thermal_std=float(shares.thermal.std(ddof=0)),
                      thermal_change=float(latest.thermal-shares.thermal.iloc[0]),
                      total_cv=float(history.sum(axis=1).std(ddof=0)/history.sum(axis=1).mean()),
                      observations=len(history)/6.)
        records[p]=values
    mapped=pd.DataFrame.from_dict(records,orient='index')
    out=out.join(mapped,on='period')
    for g in GROUPS:
        out[f'{g}_hsin']=out[g]*out.cal0
        out[f'{g}_hcos']=out[g]*out.cal1
    return out


def weights(frame: pd.DataFrame) -> np.ndarray:
    # Country equal, then region equal, then each observed hour equal.
    region_n=frame.groupby(['country','region']).region.transform('size').to_numpy(float)
    regions_per_country=frame.groupby('country').region.transform('nunique').to_numpy(float)
    w=1/(region_n*regions_per_country)
    return w/w.sum()


def fit(frame: pd.DataFrame, fuel: np.ndarray, coarse: np.ndarray) -> dict:
    w=weights(frame); y=fuel@FACTORS/fuel.sum(axis=1)
    thermal=fuel[:,:4].sum(axis=1)
    thermal_y=np.divide(fuel@FACTORS,thermal,out=np.full(len(fuel),400.),where=thermal>0)
    regional_ef=[];regional_profiles=[];country=[]
    for (c,r),group in frame.groupby(['country','region'],sort=False):
        mask=(frame.country.eq(c)&frame.region.eq(r)).to_numpy()
        regional_ef.append(float(np.sum(fuel[mask]@FACTORS)/max(thermal[mask].sum(),1e-9)))
        data=pd.DataFrame(coarse[mask],columns=GROUPS)
        data['hour']=group.clock.dt.hour.to_numpy()
        profile=data.groupby('hour')[list(GROUPS)].mean().reindex(range(24)).fillna(0.)
        profile=profile.div(profile.mean().replace(0.,np.nan)).fillna(1.).to_numpy()
        regional_profiles.append(profile);country.append(c)
    profile=np.mean([np.mean([p for p,c in zip(regional_profiles,country) if c==k],axis=0)
                     for k in sorted(set(country))],axis=0)
    macro_ef=float(np.mean([np.mean([e for e,c in zip(regional_ef,country) if c==k])
                            for k in sorted(set(country))]))
    models={'profile':profile,'macro_ef':macro_ef,
            'ratio_ef':float(np.sum(w*(fuel@FACTORS))/np.sum(w*thermal))}
    scaler=StandardScaler().fit(frame[list(HISTORY)],sample_weight=w)
    ridge=Ridge(alpha=10.).fit(scaler.transform(frame[list(HISTORY)]),y,sample_weight=w*len(w))
    models['ridge_history']=(scaler,ridge)
    options=dict(max_iter=80,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=10.,
                 random_state=416,early_stopping=False)
    for name,columns in [('direct_latest',LATEST),('direct_history',HISTORY),
                         ('factor_latest',LATEST),('factor_history',HISTORY)]:
        active=thermal>0 if name.startswith('factor') else np.ones(len(frame),dtype=bool)
        target=thermal_y if name.startswith('factor') else y
        model=HistGradientBoostingRegressor(**options)
        model.fit(frame.loc[active,list(columns)],target[active],sample_weight=w[active]*len(w))
        models[name]=model
    return models


def predict(models: dict, public: pd.DataFrame) -> dict:
    q=public[list(GROUPS)].to_numpy()*models['profile'][public.clock.dt.hour.to_numpy()]
    hourly_thermal=q[:,0]/q.sum(axis=1)
    scaler,ridge=models['ridge_history']
    out={'anchor700':700*public.thermal.to_numpy(),
         'macro_recomposition':models['macro_ef']*hourly_thermal,
         'ratio_recomposition':models['ratio_ef']*hourly_thermal,
         'ridge_history':np.clip(ridge.predict(scaler.transform(public[list(HISTORY)])),0.,1000.)}
    for name,columns in [('direct_latest',LATEST),('direct_history',HISTORY),
                         ('factor_latest',LATEST),('factor_history',HISTORY)]:
        value=models[name].predict(public[list(columns)])
        out[name]=np.clip(value,400.,1000.)*hourly_thermal if name.startswith('factor') else np.clip(value,0.,1000.)
    return out


def add_capacity_features(features: pd.DataFrame, capacities: pd.DataFrame) -> pd.DataFrame:
    """Join annual thermal capacity with the same conservative two-year lag."""
    out = features.copy()
    lookup = capacities.set_index(['country', 'stat_year'])['thermal_capacity_share']
    key = pd.MultiIndex.from_arrays(
        [out.country, out.year.astype(int) - 2], names=['country', 'stat_year'])
    out['capacity_thermal_share'] = lookup.reindex(key).to_numpy()
    if out.capacity_thermal_share.isna().any():
        missing = out.loc[out.capacity_thermal_share.isna(), ['country', 'year']].drop_duplicates()
        raise ValueError(f'missing two-year-lagged capacity rows: {missing.to_dict("records")}')
    return out


def fit_capacity_heads(frame: pd.DataFrame, fuel: np.ndarray) -> dict:
    """Fit M+C heads using only strict five-group histories and capacity share."""
    columns = [*HISTORY, 'capacity_thermal_share']
    w = weights(frame)
    y = fuel @ FACTORS / fuel.sum(axis=1)
    thermal = fuel[:, :4].sum(axis=1)
    thermal_y = np.divide(fuel @ FACTORS, thermal, out=np.full(len(fuel), 400.), where=thermal > 0)
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10., random_state=416, early_stopping=False)
    direct = HistGradientBoostingRegressor(**options).fit(
        frame[columns], y, sample_weight=w * len(w))
    scaler = StandardScaler().fit(frame[columns], sample_weight=w)
    ridge = Ridge(alpha=10.).fit(
        scaler.transform(frame[columns]), y, sample_weight=w * len(w))
    active = thermal > 0
    factor = HistGradientBoostingRegressor(**options).fit(
        frame.loc[active, columns], thermal_y[active], sample_weight=w[active] * len(w))
    return {'columns': columns, 'direct': direct, 'ridge': (scaler, ridge), 'factor': factor}


def predict_capacity_heads(models: dict, public: pd.DataFrame, reference: dict) -> dict:
    """Predict direct intensity and factor-scaled five-group thermal shape."""
    columns = models['columns']
    factor = np.clip(models['factor'].predict(public[columns]), 400., 1000.)
    scaler, ridge = models['ridge']
    q = public[list(GROUPS)].to_numpy() * reference['profile'][public.clock.dt.hour.to_numpy()]
    hourly_thermal = q[:, 0] / q.sum(axis=1)
    return {
        'capacity_direct': np.clip(models['direct'].predict(public[columns]), 0., 1000.),
        'capacity_ridge': np.clip(ridge.predict(scaler.transform(public[columns])), 0., 1000.),
        'capacity_factor': factor * hourly_thermal,
    }
