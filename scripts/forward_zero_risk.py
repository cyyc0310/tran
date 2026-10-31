"""Fixed-cutoff, proper-region ordinary zero-risk controls for V188."""
from __future__ import annotations
import json
import numpy as np
from monthly_history_features import monthly_context
from monthly_risk_features import feature_matrices, volatility_features
from monthly_history_zero_risk import array_hash
from two_scale_zero import probability_forest, checked_probability, hurdle_probability
from v154_hourly_strong_baselines import balanced

MONTH_HEADS={'month_base':'base13','month_history':'history36','month_volatility':'history77'}
COMPONENTS=('clock','cond43','cond84')
HURDLES={f'{m}_{r}':(m,r) for m in (*MONTH_HEADS,'persistence') for r in COMPONENTS}
METHODS=('direct43','direct84',*HURDLES)
MONTH_METHODS=(*MONTH_HEADS,'persistence')


class ForwardZeroRisk:
    def __init__(self,data,short,long,fold,source_rows,selections,queries,cutoff):
        self.data,self.short,self.long=data,short,long
        self.fold,self.cutoff=fold,cutoff
        self.fit=sorted(source_rows.loc[source_rows.fold.eq(fold),'region'])
        self.queries=queries.loc[queries.fold.eq(fold)].copy()
        self.allowed=set(self.queries.profile_id)
        if set(self.queries.region)&set(self.fit):raise ValueError('source/query roles')
        chosen=[selections[f'{fold}:{r}'] for r in self.fit]
        self.raw_ids=np.concatenate([np.asarray(s['raw_ids'],int) for s in chosen])
        self.profiles=np.concatenate([np.asarray(s['profiles'],int) for s in chosen])
        self.month_ids=np.concatenate([np.asarray(s['monthly_ids'],int) for s in chosen])
        self.regions=data.index.region.to_numpy()[self.profiles]
        self.month_regions=long.index.region.to_numpy()[self.month_ids]
        self.fine_fit=sorted(set(self.regions))
        if (not set(self.regions).issubset(self.fit) or not set(self.month_regions)==set(self.fit)
            or not (data.release.iloc[self.profiles]<=cutoff).all()
            or not (long.release.iloc[self.month_ids]<=cutoff).all()):raise ValueError('training role/cutoff')
        self.contexts={int(q):feature_matrices(short,long,int(q)) for q in np.unique(self.profiles)}
        self.month_contexts={int(q):monthly_context(long,int(q)) for q in self.month_ids}
        if any(c is None for c in self.month_contexts.values()):raise ValueError('training prefix missing')
        self.X={name:np.asarray([self.contexts[int(q)][0][feature][h] for q,h in zip(self.profiles,data.states[self.raw_ids])])
                for name,feature in [('direct43','long43'),('direct84','long84')]}
        self.month_X={feature:np.asarray([self.month_contexts[int(q)][key] for q in self.month_ids])
                      for feature,key in [('base13','base'),('history36','history')]}
        self.month_X['history77']=np.column_stack([self.month_X['history36'],np.asarray([
            volatility_features(long,self.month_contexts[int(q)]['prefix']) for q in self.month_ids])])
        self.zero=data.raw[self.raw_ids]==0
        self.positive=long.M[self.profiles]>0
        self.month_zero=long.M[self.month_ids]==0
        self.models={name:[] for name in (*self.X,'cond43','cond84',*MONTH_HEADS)}
        self.clock=np.empty((48,8))
        states=data.states[self.raw_ids]
        for fuel in range(8):
            take=self.positive[:,fuel]
            if not take.any():raise ValueError('no positive-month fine supervision')
            for name,X in self.X.items():
                self.models[name].append(probability_forest(X,self.zero[:,fuel].astype(float),self.regions,64))
                conditional='cond'+name.removeprefix('direct')
                self.models[conditional].append(probability_forest(X[take],self.zero[take,fuel].astype(float),self.regions[take],64))
            for name,feature in MONTH_HEADS.items():
                self.models[name].append(probability_forest(self.month_X[feature],self.month_zero[:,fuel].astype(float),self.month_regions,4))
            ids=np.flatnonzero(take);weights=balanced(self.regions[take])
            for h in range(48):
                mask=states[ids]==h
                if not mask.any():mask=np.ones(len(ids),bool)
                self.clock[h,fuel]=np.average(self.zero[ids[mask],fuel],weights=weights[mask])

    def predict(self,q:int):
        if q not in self.allowed or self.long.index.region.iloc[q] in self.fit:raise ValueError('query whitelist')
        features,contexts,extra=feature_matrices(self.short,self.long,q)
        c=contexts[1]
        X={'base13':c['base'][None],'history36':c['history'][None],'history77':np.r_[c['history'],extra][None]}
        hourly={name:checked_probability(np.column_stack([m.predict(features['long'+name.removeprefix('direct')]) for m in self.models[name]])) for name in self.X}
        components={'clock':self.clock.copy()}
        for name in ('cond43','cond84'):
            components[name]=checked_probability(np.column_stack([m.predict(features['long'+name.removeprefix('cond')]) for m in self.models[name]]))
        monthly={name:checked_probability(np.asarray([m.predict(X[feature])[0] for m in self.models[name]])) for name,feature in MONTH_HEADS.items()}
        monthly['persistence']=(self.long.M[c['local']]==0).astype(float)
        for name,(month,conditional) in HURDLES.items():hourly[name]=checked_probability(hurdle_probability(monthly[month],components[conditional]))
        return hourly,monthly,components,c

    def arrays(self):
        return {'raw_id':self.raw_ids,'profile_id':self.profiles,'region':np.asarray(self.regions,str),
                'zero':self.zero,'positive':self.positive,'month_profile_id':self.month_ids,
                'month_region':np.asarray(self.month_regions,str),'month_zero':self.month_zero,
                **{'X_'+k:v for k,v in self.X.items()},**{'month_X_'+k:v for k,v in self.month_X.items()}}

    def prefixes(self):
        rows=[]
        for kind,container,contexts in [('monthly',self.long,self.month_contexts),
                                       ('fine_short',self.short,{q:c[1][0] for q,c in self.contexts.items()}),
                                       ('fine_long',self.long,{q:c[1][1] for q,c in self.contexts.items()})]:
            for q,c in contexts.items():
                rows.append({'kind':kind,'profile_id':q,'region':container.index.region.iloc[q],
                             'local':int(c['local']),'prefix':json.dumps([int(p) for p in c['prefix']])})
        return rows

    def fingerprint(self):
        fields=('children_left','children_right','feature','threshold','value','weighted_n_node_samples')
        return {name:[[{key:array_hash(getattr(t.tree_,key)) for key in fields} for t in model.estimators_]
                      for model in models] for name,models in self.models.items()}


def check_inputs(model,q):
    before=model.predict(q)[:3];saved=[]
    for data in (model.data,model.short,model.long):
        ids=data.groups[data.index.region.iloc[q]]
        ids=ids[(data.release.iloc[ids]>data.dates.iloc[q]).to_numpy()]
        saved.append((data,ids,data.M[ids].copy(),data.shares[ids].copy()))
    raw=model.data.raw
    try:
        for data,ids,M,_ in saved:
            changed=np.random.default_rng(188+q).uniform(1,1000,size=M.shape)
            data.M[ids]=changed;data.shares[ids]=changed/changed.sum(axis=1,keepdims=True)
        model.data.raw=None;after=model.predict(q)[:3]
    finally:
        model.data.raw=raw
        for data,ids,M,S in saved:data.M[ids],data.shares[ids]=M,S
    if any(not np.array_equal(v,b[k]) for a,b in zip(before,after) for k,v in a.items()):raise ValueError('future query input/raw guard')
