"""Standard source-trained predictors under explicit lagged local energy packets.

Packet dimension is not Shannon information or acquisition cost. This module
contains no target access, forecasting guarantee, or algorithm novelty claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SOURCES = ('BPAT','CISO','ERCO','FPL','ISNE','MISO','NYIS')
FUELS = ('coal','gas','oil','nuclear','hydro','wind','solar','other')
EF = np.array([1000.,400.,800.,0.,0.,0.,0.,400.])
PACKETS = {'D2':2,'D8':8,'T8':8,'G5':5}
MODELS = ('ridge_global','ET_global','ET_hetero','ET_QRF')
SEED, SAMPLES = 17023,128
POINT_TREES, POINT_LEAF, RISK_TREES, RISK_LEAF = 150,10,128,20
PERIODS = {'proper':('2024-01-25','2024-07-01'),
           'riskfit':('2024-07-01','2024-10-01'), 'riskpool':('2024-10-01','2025-01-01')}


def packets_from_day(generation: np.ndarray) -> dict[str,np.ndarray]:
    """Isolated aggregation: positive-proxy energy, not signed BA telemetry."""
    a=np.asarray(generation,float)
    if a.shape!=(24,8) or not np.isfinite(a).all() or np.any(a<0) or np.any(a.sum(1)<=0):
        raise ValueError('One complete positive-total UTC generation day required')
    fuel=a.sum(0); blocks=a.reshape(4,6,8).sum(1)
    h,g=float(fuel@(EF/1000)),float(fuel.sum())
    bh,bg=blocks@(EF/1000),blocks.sum(1)
    return {'D2':np.array([h,g]),'D8':fuel,
            'T8':np.column_stack((bh,bg)).ravel(),'G5':np.r_[h,bg]}


def channels(values: np.ndarray, packet: str) -> np.ndarray:
    """Invertible intensive coordinates expose every supplied packet field."""
    a=np.asarray(values,float)
    if packet not in PACKETS or a.shape[-1:]!=(PACKETS[packet],) or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('Finite nonnegative registered packet required')
    if packet=='D8':
        g=a.sum(-1); h=a@(EF/1000)
    elif packet=='T8':
        bh,bg=a[...,::2],a[...,1::2]
        if np.any(bg<=0) or np.any(bh>bg+1e-8):raise ValueError('Invalid block H/G')
        h,g=bh.sum(-1),bg.sum(-1)
    elif packet=='G5':
        h,bg=a[...,0],a[...,1:];g=bg.sum(-1)
        if np.any(bg<=0):raise ValueError('Positive block denominators required')
    else:h,g=a[...,0],a[...,1]
    if np.any(g<=0) or np.any(h>g+1e-8):raise ValueError('Positive total energy and 0<=H<=G required')
    anchor=1000*h/g;level=np.log1p(g)
    if packet=='D8':return np.concatenate((a/g[...,None],level[...,None],anchor[...,None]),-1)
    if packet=='T8':return np.concatenate((1000*bh/bg,bg/g[...,None],level[...,None],anchor[...,None]),-1)
    if packet=='G5':return np.concatenate((bg/g[...,None],level[...,None],anchor[...,None]),-1)
    return np.stack((level,anchor),-1)


def calendar(origin: pd.Timestamp) -> np.ndarray:
    p=2*np.pi*(origin.dayofyear-1)/365.25;w=2*np.pi*origin.dayofweek/7
    return np.array([np.sin(p),np.cos(p),np.sin(2*p),np.cos(2*p),np.sin(w),np.cos(w)])


@dataclass
class PacketHistory:
    table: pd.DataFrame
    packet: str
    lag: int
    _cache: dict[int,tuple[np.ndarray,float,dict]] = field(default_factory=dict,init=False)

    def __post_init__(self) -> None:
        if self.packet not in PACKETS or self.lag not in (7,30):raise ValueError('Frozen packet/lag required')
        expected={'region','day',*[f'v{i}' for i in range(PACKETS[self.packet])]}
        if set(self.table)!=expected:raise ValueError('Packet schema must contain only declared fields')
        self.table=self.table.sort_values('day').reset_index(drop=True).copy()
        if len(self.table) and (self.table.region.nunique()!=1 or not set(self.table.region)<=set(SOURCES) or self.table.day.duplicated().any()):
            raise ValueError('One unique-day local region per history')
        self.days=pd.DatetimeIndex(pd.to_datetime(self.table.day,utc=True,errors='raise'))
        if self.days.hasnans or not self.days.equals(self.days.normalize()):raise ValueError('UTC day starts required')
        self.release=self.days+pd.Timedelta(days=1+self.lag)

    def features(self,origin: pd.Timestamp) -> tuple[np.ndarray,float,dict[str,Any]]:
        at=pd.Timestamp(origin)
        if at.tzinfo is None or at.tz_convert('UTC')!=at.tz_convert('UTC').normalize():
            raise ValueError('Timezone-aware UTC midnight origin required')
        at=at.tz_convert('UTC')
        if at.value in self._cache:
            x,a,d=self._cache[at.value];return x.copy(),a,dict(d)
        eligible=np.flatnonzero(self.release<=at)
        columns=[f'v{i}' for i in range(PACKETS[self.packet])]
        if len(eligible):
            latest=self.days[eligible[-1]]
            keep=eligible[self.days[eligible]>=latest-pd.Timedelta(days=27)]
            raw=self.table.loc[keep,columns].to_numpy(float) # Numeric gate follows time selection.
            days=self.days[keep];age=np.asarray((latest-days).total_seconds()/86400)
            weights=np.exp(-age/14);weights/=weights.sum()
            states=channels(raw,self.packet)
            ema=channels(weights@raw,self.packet)
            center=weights@states
            std=np.sqrt(np.maximum(weights@((states-center)**2),0))
            same=days.dayofweek==at.dayofweek
            sw=weights[same]/weights[same].sum() if same.any() else weights
            weekday=channels(sw@raw[same] if same.any() else sw@raw,self.packet)
            anchor=float(ema[-1]);availability=np.array([len(keep),same.sum(),(at-self.release[keep[-1]]).days,0.])
            summaries=np.concatenate((states[-1],ema,std,weekday))
            audit=dict(history_days=len(keep),same_weekday_days=int(same.sum()),missing=False,
                latest_day=latest.isoformat(),max_release=self.release[keep[-1]].isoformat(),
                history_start=days[0].isoformat())
        else:
            # All packets share the same explicit missing-history reference.
            nominal={'D2':np.array([.5,1.]),'D8':np.full(8,1/8),
                     'T8':np.tile([.125,.25],4),'G5':np.r_[.5,np.full(4,.25)]}[self.packet]
            neutral=channels(nominal,self.packet);neutral[-1]=500.
            summaries=np.concatenate((neutral,neutral,np.zeros_like(neutral),neutral))
            anchor=500.;availability=np.array([0.,0.,999.,1.])
            audit=dict(history_days=0,same_weekday_days=0,missing=True,latest_day=None,max_release=None,history_start=None)
        x=np.r_[anchor,summaries,calendar(at),availability]
        if not np.isfinite(x).all():raise ValueError('Nonfinite packet features')
        self._cache[at.value]=(x.copy(),anchor,dict(audit))
        return x,anchor,audit


def balanced(regions: np.ndarray) -> np.ndarray:
    labels=np.asarray(regions)
    if labels.ndim!=1 or not len(labels) or not set(labels)<=set(SOURCES):raise ValueError('Known source rows required')
    _,inv,counts=np.unique(labels,return_inverse=True,return_counts=True)
    w=1/counts[inv];return w/w.sum()


def validate_training(x: np.ndarray,y: np.ndarray,regions: np.ndarray,days: pd.DatetimeIndex,
                      period: str,exclude: str) -> None:
    if exclude not in SOURCES or exclude in set(regions) or len(set(regions))<2 or not set(regions)<=set(SOURCES):
        raise ValueError('At least two actual source regions, excluding held region, required')
    d=pd.DatetimeIndex(pd.to_datetime(days,utc=True));start,stop=PERIODS[period]
    if (x.ndim!=2 or y.shape!=(len(x),24) or len(regions)!=len(x) or len(d)!=len(x)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or d.hasnans or np.any(d<pd.Timestamp(start,tz='UTC')) or np.any(d>=pd.Timestamp(stop,tz='UTC'))
            or pd.MultiIndex.from_arrays([regions,d]).has_duplicates):
        raise ValueError('Finite source-only arrays within the fixed training period required')


class PointModel:
    def __init__(self,kind: str) -> None:
        if kind not in ('ridge','ET'):raise ValueError('Registered point model required')
        self.kind=kind

    def fit(self,x: np.ndarray,y: np.ndarray,anchors: np.ndarray,regions: np.ndarray,
            days: pd.DatetimeIndex,*,exclude: str) -> PointModel:
        validate_training(x,y,regions,days,'proper',exclude)
        if anchors.shape!=(len(x),) or not np.isfinite(anchors).all() or np.any(y<0) or np.any(y>1000):
            raise ValueError('Bounded fine labels and aligned legal anchors required')
        w=balanced(regions)*len(x)
        self.scaler=StandardScaler().fit(x,sample_weight=w)
        self.model=(Ridge(alpha=30.) if self.kind=='ridge' else ExtraTreesRegressor(
            n_estimators=POINT_TREES,min_samples_leaf=POINT_LEAF,random_state=SEED,n_jobs=1))
        self.model.fit(self.scaler.transform(x),y-anchors[:,None],sample_weight=w)
        self.regions=tuple(sorted(set(regions)))
        return self

    def predict(self,x: np.ndarray,anchors: np.ndarray) -> np.ndarray:
        if x.ndim!=2 or anchors.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(anchors).all():
            raise ValueError('Finite query features and anchors required')
        return np.clip(anchors[:,None]+self.model.predict(self.scaler.transform(x)),0,1000)


def cdf_ids(weights: np.ndarray,uniforms: np.ndarray) -> np.ndarray:
    w,u=np.asarray(weights,float),np.asarray(uniforms,float)
    if (w.ndim!=2 or u.ndim!=2 or len(w)!=len(u) or not np.isfinite(w).all() or np.any(w<0)
            or not np.allclose(w.sum(1),1,rtol=0,atol=1e-12) or not np.isfinite(u).all()
            or np.any(u<0) or np.any(u>=1)):raise ValueError('Normalized day weights and [0,1) uniforms required')
    ids=[]
    for row,v in zip(w,u):
        last=np.flatnonzero(row>0)[-1];cdf=np.cumsum(row);cdf[last:]=1.
        ids.append(np.minimum(np.searchsorted(cdf,v,side='right'),last))
    return np.asarray(ids,dtype=np.int64)


class RiskModel:
    """Disjoint riskfit learning and riskpool whole-day calibration sampling."""
    def __init__(self,kind: str) -> None:
        if kind not in ('global','hetero','QRF'):raise ValueError('Registered risk model required')
        self.kind=kind

    def fit(self,fit_x: np.ndarray,fit_errors: np.ndarray,fit_regions: np.ndarray,fit_days: pd.DatetimeIndex,
            pool_x: np.ndarray,pool_errors: np.ndarray,pool_regions: np.ndarray,pool_days: pd.DatetimeIndex,
            *,exclude: str) -> RiskModel:
        validate_training(fit_x,fit_errors,fit_regions,fit_days,'riskfit',exclude)
        validate_training(pool_x,pool_errors,pool_regions,pool_days,'riskpool',exclude)
        if fit_x.shape[1]!=pool_x.shape[1] or np.max(abs(fit_errors))>1000+1e-8 or np.max(abs(pool_errors))>1000+1e-8:
            raise ValueError('Aligned feature dimensions and bounded residuals required')
        self.pool_errors=pool_errors.copy();self.pool_regions=np.asarray(pool_regions).copy()
        self.pool_days=pd.DatetimeIndex(pool_days).as_unit('ns').asi8
        self.global_weights=balanced(pool_regions)
        w=balanced(fit_regions)*len(fit_regions)
        self.scaler=StandardScaler().fit(fit_x,sample_weight=w)
        fx,px=self.scaler.transform(fit_x),self.scaler.transform(pool_x)
        if self.kind=='hetero':
            self.model=ExtraTreesRegressor(n_estimators=POINT_TREES,min_samples_leaf=POINT_LEAF,random_state=SEED,n_jobs=1)
            self.model.fit(fx,np.log(fit_errors**2+25.),sample_weight=w)
            self.pool_scales=np.exp(.5*self.model.predict(px))
        elif self.kind=='QRF':
            self.model=RandomForestRegressor(n_estimators=RISK_TREES,min_samples_leaf=RISK_LEAF,
                max_features=.7,bootstrap=True,random_state=SEED,n_jobs=1)
            self.model.fit(fx,fit_errors,sample_weight=w)
            self.pool_leaves=self.model.apply(px)
        self.fit_regions=tuple(sorted(set(fit_regions)));self.regions=tuple(sorted(set(pool_regions)))
        return self

    def weights(self,x: np.ndarray) -> tuple[np.ndarray,np.ndarray | None,int]:
        if x.ndim!=2 or x.shape[1]!=self.scaler.n_features_in_ or not np.isfinite(x).all():
            raise ValueError('Finite aligned query features required')
        if self.kind!='QRF':return np.tile(self.global_weights,(len(x),1)),None,0
        leaves=self.model.apply(self.scaler.transform(x));weights=np.zeros((len(x),len(self.pool_errors)));empty=0
        for i,row in enumerate(leaves):
            for tree,leaf in enumerate(row):
                members=np.flatnonzero(self.pool_leaves[:,tree]==leaf)
                if len(members):weights[i,members]+=balanced(self.pool_regions[members])
                else:weights[i]+=self.global_weights;empty+=1
        weights/=self.model.n_estimators
        weights/=weights.sum(1,keepdims=True)
        return weights,leaves,empty

    def draw(self,center: np.ndarray,x: np.ndarray,uniforms: np.ndarray) -> dict[str,Any]:
        if center.shape!=(len(x),24) or not np.isfinite(center).all() or uniforms.shape!=(len(x),SAMPLES):
            raise ValueError('Aligned centers and128whole-day uniforms required')
        weights,leaves,empty=self.weights(x);ids=cdf_ids(weights,uniforms)
        if self.kind=='hetero':
            scales=np.exp(.5*self.model.predict(self.scaler.transform(x)))
            raw=center[:,None]+(self.pool_errors/self.pool_scales)[ids]*scales[:,None]
        else:scales=np.ones_like(center);raw=center[:,None]+self.pool_errors[ids]
        if not np.isfinite(raw).all():raise ValueError('Nonfinite residual scenarios')
        result=dict(raw=raw,residual_indices=ids,conditional_weights=weights,query_scales=scales,
            pool_scales=self.pool_scales if self.kind=='hetero' else np.ones_like(self.pool_errors),
            empty_leaf_tree_queries=empty)
        if leaves is not None:result.update(query_leaf_ids=leaves,pool_leaf_ids=self.pool_leaves)
        return result
