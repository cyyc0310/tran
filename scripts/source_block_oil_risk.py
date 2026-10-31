"""Ordinary fixed-budget oil risk controls after deterministic source deletion."""
from __future__ import annotations
import numpy as np
from monthly_risk_features import feature_matrices
from monthly_history_zero_risk import array_hash
from two_scale_zero import probability_forest,checked_probability
from v154_hourly_strong_baselines import balanced

OIL=2
HOURLY=('direct43','direct84','persistence_clock','persistence_cond43','persistence_cond84')
COMPONENTS=('clock','cond43','cond84')


class SourceBlockOil:
    def __init__(self,arrays,states,deleted:str):
        self.deleted=deleted
        removed=arrays['region']==deleted
        positive=arrays['positive'][:,OIL];events=arrays['zero'][:,OIL]
        w=balanced(arrays['region'][positive]);total_event_mass=float(w@events[positive])
        self.removed={'hours':int(removed.sum()),'positive_month_hours':int((removed&positive).sum()),
                      'zero_event_hours':int((removed&positive&events).sum()),
                      'nonzero_hours':int((removed&positive&~events).sum()),
                      'baseline_positive_weight_mass':float(w[removed[positive]].sum()/w.sum()),
                      'baseline_zero_event_weight_share':float(w[removed[positive]]@events[positive][removed[positive]]/total_event_mass) if total_event_mass>0 else None}
        keep=~removed
        self.raw_ids=arrays['raw_id'][keep]
        self.profiles=arrays['profile_id'][keep]
        self.regions=arrays['region'][keep]
        self.X={name:arrays['X_'+name][keep] for name in ('direct43','direct84')}
        self.zero=arrays['zero'][keep,OIL]
        self.positive=arrays['positive'][keep,OIL]
        if not len(self.raw_ids) or not self.positive.any():raise ValueError('oil risk is not estimable after fixed source deletion')
        self.models={}
        for name,X in self.X.items():
            self.models[name]=probability_forest(X,self.zero.astype(float),self.regions,64)
            self.models['cond'+name.removeprefix('direct')]=probability_forest(X[self.positive],self.zero[self.positive].astype(float),self.regions[self.positive],64)
        positive_states=states[self.raw_ids[self.positive]]
        weights=balanced(self.regions[self.positive]);self.clock=np.empty(48)
        for h in range(48):
            take=positive_states==h
            if not take.any():take=np.ones(len(positive_states),bool)
            self.clock[h]=np.average(self.zero[self.positive][take],weights=weights[take])

    def predict(self,short,long,q):
        features,contexts,_=feature_matrices(short,long,q)
        probabilities={name:checked_probability(model.predict(features['long'+name.removeprefix('direct').removeprefix('cond')])) for name,model in self.models.items()}
        components={'clock':self.clock.copy(),'cond43':probabilities['cond43'],'cond84':probabilities['cond84']}
        m=float(long.M[contexts[1]['local'],OIL]==0)
        hourly={name:probabilities[name] for name in ('direct43','direct84')}
        hourly.update({'persistence_'+name:checked_probability(m+(1-m)*p) for name,p in components.items()})
        return hourly,components,contexts[1]

    def fingerprint(self):
        fields=('children_left','children_right','feature','threshold','value','weighted_n_node_samples')
        return {name:[{key:array_hash(getattr(t.tree_,key)) for key in fields} for t in model.estimators_] for name,model in self.models.items()}

    def provenance(self):
        w=balanced(self.regions[self.positive]);y=self.zero[self.positive]
        return {'deleted':self.deleted,'removed':self.removed,'hours':len(self.raw_ids),'regions':sorted(set(self.regions)),
                'positive_month_hours':int(self.positive.sum()),'positive_month_regions':sorted(set(self.regions[self.positive])),
                'zero_event_hours':int(y.sum()),'nonzero_hours':int((~y).sum()),
                'zero_event_regions':sorted(set(self.regions[self.positive][y])),
                'nonzero_regions':sorted(set(self.regions[self.positive][~y])),
                'weighted_zero_event_probability':float(w@y/w.sum()),'single_class':bool(not y.any() or y.all()),
                'raw_index_sha256':array_hash(self.raw_ids),'positive_index_sha256':array_hash(self.raw_ids[self.positive])}


def guard(model,data,short,long,q):
    before=model.predict(short,long,q)[:2];saved=[];raw=data.raw
    for container in (data,short,long):
        ids=container.groups[container.index.region.iloc[q]]
        ids=ids[(container.release.iloc[ids]>container.dates.iloc[q]).to_numpy()]
        saved.append((container,ids,container.M[ids].copy(),container.shares[ids].copy()))
    try:
        for container,ids,M,_ in saved:
            changed=np.random.default_rng(q+189).uniform(1,1000,size=M.shape)
            container.M[ids]=changed;container.shares[ids]=changed/changed.sum(axis=1,keepdims=True)
        data.raw=None;after=model.predict(short,long,q)[:2]
    finally:
        data.raw=raw
        for container,ids,M,S in saved:container.M[ids],container.shares[ids]=M,S
    if any(not np.array_equal(v,b[k]) for a,b in zip(before,after) for k,v in a.items()):raise ValueError('future/raw guard')
