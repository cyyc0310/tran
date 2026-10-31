"""Finite-support source/event balancing and exact one-dimensional CDF impact."""
from __future__ import annotations
import numpy as np


def weights_checked(weights: np.ndarray) -> np.ndarray:
    w=np.asarray(weights,float)
    if w.ndim!=1 or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('invalid source law weights')
    return w/w.sum()


def concentration_gamma(masses: np.ndarray) -> float:
    p=weights_checked(masses);k=len(p)
    if k<2:return 0.
    value=float(np.clip((p@p-1/k)/(1-1/k),0,1))
    return 0. if value<=64*np.finfo(float).eps else value


def transport_shift(values: np.ndarray, before: np.ndarray, after: np.ndarray, order: np.ndarray | None=None) -> float:
    x=np.asarray(values,float)
    if not np.isfinite(x).all() or x.ndim!=1 or x.shape!=before.shape or x.shape!=after.shape:raise ValueError('transport axes/domain')
    ids=np.argsort(x,kind='stable') if order is None else order
    return float(np.diff(x[ids])@abs(np.cumsum(after[ids]-before[ids])[:-1]))


def source_equalization(weights: np.ndarray, sources: np.ndarray, take: np.ndarray):
    w=weights_checked(weights);take=np.asarray(take,bool)
    if take.shape!=w.shape or sources.shape!=w.shape:raise ValueError('source layer dimensions')
    p=float(w[take].sum());out=w.copy()
    mass=np.bincount(sources[take],weights=w[take],minlength=int(sources.max())+1)
    supported=np.flatnonzero(mass>0);k=len(supported)
    if p<=0 or k<2:return out,{'mass':p,'sources':k,'concentration':1. if k else None,'gamma':0.,'status':'absent' if k==0 else 'single_source'}
    r=mass[supported]/p;gamma=concentration_gamma(r)
    equal=w[take]*p/(k*mass[sources[take]])
    out[take]=(1-gamma)*w[take]+gamma*equal
    if (not np.isfinite(out).all() or (out<0).any() or abs(out.sum()-1)>1e-12
        or abs(out[take].sum()-p)>1e-12 or not np.array_equal(out[~take],w[~take])):raise ValueError('event mass preservation')
    return out,{'mass':p,'sources':k,'concentration':float(r@r),'gamma':gamma,'status':'supported'}


def event_updates(values: np.ndarray, weights: np.ndarray, sources: np.ndarray, layers: np.ndarray):
    w=weights_checked(weights);order=np.argsort(values,kind='stable');records=[];updated=[]
    if layers.shape!=(16,len(w)):raise ValueError('sixteen source event layer bank')
    for j,take in enumerate(layers):
        candidate,d=source_equalization(w,sources,take)
        impact=transport_shift(values,w,candidate,order)
        if impact<=64*np.finfo(float).eps*max(1.,float(np.max(abs(values)))):impact=0.
        records.append({'layer':j,'fuel_index':j//2,'class':'zero' if j%2==0 else 'nonzero',**d,
                        'transport_impact':impact,'mass_concentration_score':d['mass']*d['gamma']})
        updated.append(candidate)
    risk=int(np.argmax([r['transport_impact'] for r in records]));mass=int(np.argmax([r['mass_concentration_score'] for r in records]))
    risk_weights=updated[risk] if records[risk]['transport_impact']>0 else w
    mass_weights=updated[mass] if records[mass]['mass_concentration_score']>0 else w
    return {'event_risk_half':.5*w+.5*risk_weights,'event_risk_full':risk_weights,
            'event_mass_full':mass_weights,'oil_nonzero_full':updated[5]},records,{'risk_layer':risk if records[risk]['transport_impact']>0 else None,'mass_layer':mass if records[mass]['mass_concentration_score']>0 else None}


def global_updates(values: np.ndarray, weights: np.ndarray, sources: np.ndarray):
    w=weights_checked(weights);mass=np.bincount(sources,weights=w);active=np.flatnonzero(mass>0);k=len(active)
    if k<2 or np.max(mass)>=1:raise ValueError('source deletion law requires multiple positive sources')
    uniform=w/(k*mass[sources]);inverse=1/(1-mass[active])
    deleted=w*(inverse.sum()-1/(1-mass[sources]))/k
    mean=float(w@values);var=float(w@(values-mean)**2)
    source_means=np.bincount(sources,weights=w*values)[active]/mass[active]
    spread=float(np.std(source_means));scale=float(np.sqrt(max(0.,var)))
    return {'source_uniform':weights_checked(uniform),'source_delete_mix':weights_checked(deleted)}, {
        'effective_sources':float(1/(mass@mass)),'ess_mix':concentration_gamma(mass[active]),
        'source_mean_spread':spread,'spread_mix':spread/(scale+spread) if scale+spread>0 else 0.}
