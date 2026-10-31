"""Fixed-July source-event stabilization of physical CIF laws, without test labels."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.ensemble import HistGradientBoostingRegressor
from fuel_activity_forest import IndexedLeafMixture,carrier,response_values
from monthly_risk_features import feature_matrices
from monthly_history_zero_risk import array_hash
from source_event_transport import event_updates,global_updates
from v151_hourly_law import EF,Law
from v154_hourly_strong_baselines import CompressedLaw,balanced

BASELINES=('anchor_calendar','direct20','direct84','hetero84','physical84_leaf','physical84_bg','uniform_range')
MODIFIED=('source_uniform','source_delete_mix','event_mass_full','event_risk_half','event_risk_full',
          'oil_nonzero_full','uniform_mix10','ess_mix_anchor','source_spread_mix_anchor')
METHODS=BASELINES+MODIFIED


def hgb(loss='squared_error'):
    return HistGradientBoostingRegressor(loss=loss,max_iter=100,learning_rate=.1,max_leaf_nodes=15,
        min_samples_leaf=64,l2_regularization=1,early_stopping=False,random_state=20261028)


def law_digest(law):
    digest=hashlib.sha256()
    digest.update(np.ascontiguousarray(law.x).tobytes());digest.update(np.ascontiguousarray(law.w).tobytes())
    return digest.hexdigest()


def mixture(a,b,fraction):
    if not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('invalid mixture strength')
    if fraction==0:return a
    if fraction==1:return b
    return CompressedLaw(np.r_[a.x,b.x],np.r_[(1-fraction)*a.w,fraction*b.w])


class SourceEventCIF:
    def __init__(self,data,short,long,arrays,proper,queries,root):
        self.data,self.short,self.long=data,short,long
        self.raw_ids=arrays['raw_id'];self.profiles=arrays['profile_id'];self.regions=arrays['region'];self.proper=proper
        self.allowed=set(queries.profile_id);self.queries=queries.copy()
        if set(self.regions)-set(proper) or set(queries.region)&set(proper):raise ValueError('proper/cal/test roles')
        if not (data.release.iloc[self.profiles]<=pd.Timestamp('2024-07-01')).all():raise ValueError('source current-M timing')
        cutoff=pd.Timestamp('2024-07-01')
        cal_ids=queries.loc[queries.role.eq('calibration'),'profile_id'].to_numpy(int)
        if not len(cal_ids) or not (data.release.iloc[cal_ids]<=cutoff).all():raise ValueError('cal current month not released by cutoff')
        with np.load(root/'results/v153-calendar-times-r1/timestamps.npz',allow_pickle=False) as packed:
            unix=packed['unix_seconds']
        cal_raw=np.concatenate([np.arange(int(data.index.offset.iloc[q]),int(data.index.stop.iloc[q])) for q in cal_ids])
        if not len(cal_raw) or not np.all(unix[cal_raw]<int(cutoff.tz_localize('UTC').timestamp())):raise ValueError('cal hourly labels cross cutoff')
        if not np.all(unix[self.raw_ids]<int(cutoff.tz_localize('UTC').timestamp())):raise ValueError('proper hourly labels cross cutoff')
        self.cal_inventory=[]
        for q in cal_ids:
            row=data.index.iloc[q];ids=np.arange(int(row.offset),int(row.stop))
            self.cal_inventory.append({'profile_id':int(q),'region':row.region,'month':row.month,'hours':len(ids),
                'synthetic_month_release':str(data.release.iloc[q]),'min_utc_seconds':int(unix[ids].min()),'max_utc_seconds':int(unix[ids].max())})


        self.X={'20':arrays['X_direct84'][:,:20].copy(),'84':arrays['X_direct84']}
        self.source_features={int(q):feature_matrices(short,long,int(q)) for q in np.unique(self.profiles)}
        rebuilt=np.asarray([self.source_features[int(q)][0]['long84'][h] for q,h in zip(self.profiles,data.states[self.raw_ids])])
        if not np.array_equal(rebuilt,self.X['84']):raise ValueError('archived source features differ')

        self.local={int(q):long.local(int(q),'asof_30d') for q in np.unique(self.profiles)}
        local=np.asarray([self.local[int(q)] for q in self.profiles],int)
        if not (long.release.iloc[local].to_numpy()<=long.dates.iloc[self.profiles].to_numpy()).all():raise ValueError('source released prefix')
        self.source_anchor=np.asarray([long.shares[p]@EF for p in local])
        raw=data.raw[self.raw_ids];total=raw.sum(axis=1)
        if (total<=0).any():raise ValueError('source CIF undefined')
        self.y=raw@EF/total-self.source_anchor;fit_weights=balanced(self.regions);self.weights=fit_weights/fit_weights.sum()
        self.source_names=sorted(set(self.regions));self.source_ids=np.searchsorted(self.source_names,self.regions)
        prior=np.mean([long.shares[np.unique(local[self.regions==r])].mean(axis=0) for r in self.source_names],axis=0)+1e-6
        self.prior=prior/prior.sum()
        scale=long.M[local].sum(axis=1)/data.index.hours.to_numpy()[self.profiles]
        source_base=.99*long.shares[local]+.01*self.prior
        self.activity=raw/scale[:,None]/source_base
        if not np.isfinite(self.activity).all() or (self.activity<0).any() or (self.activity.sum(axis=1)<=0).any():raise ValueError('source activity')
        positive=long.M[self.profiles]>0
        self.layers=np.asarray([positive[:,f] & ((raw[:,f]==0) if cls==0 else (raw[:,f]>0)) for f in range(8) for cls in (0,1)])
        self.kernels={name:IndexedLeafMixture(X,self.y,fit_weights) for name,X in self.X.items()}
        for kernel in self.kernels.values():kernel.forest.n_jobs=1
        ordered=sorted(self.source_names,key=lambda r:hashlib.sha256(f'20261028/{r}'.encode()).hexdigest())
        self.inner={r:j%5 for j,r in enumerate(ordered)};self.oof=np.full(len(self.y),np.nan);self.oof_models=[]
        for fold in range(5):
            take=np.asarray([self.inner[r]==fold for r in self.regions]);model=hgb()
            model.fit(self.X['84'][~take],self.y[~take],sample_weight=balanced(self.regions[~take]));self.oof[take]=model.predict(self.X['84'][take]);self.oof_models.append(model)
        if not np.isfinite(self.oof).all():raise ValueError('proper group OOF')
        self.mean_model=hgb().fit(self.X['84'],self.y,sample_weight=balanced(self.regions))
        self.variance_model=hgb('poisson').fit(self.X['84'],(self.y-self.oof)**2+25,sample_weight=balanced(self.regions))
        self.z=ndtri((np.arange(1025)+.5)/1025);self.uniform=CompressedLaw((np.arange(1025)+.5)*1000/1025,np.ones(1025))
        self.calendar=[];states=data.states[self.raw_ids]
        for h in range(48):
            take=states==h
            if not take.any():take=np.ones(len(states),bool)
            self.calendar.append((self.y[take],self.weights[take]))

    def build(self,q):
        if q not in self.allowed or self.long.index.region.iloc[q] in self.proper:raise ValueError('query whitelist/role')
        features,contexts,_=feature_matrices(self.short,self.long,q);c=contexts[1];anchor=float(c['anchor'])
        X={'20':features['direct20'],'84':features['long84']};signatures={k:m.forest.apply(X[k]) for k,m in self.kernels.items()}
        base=carrier(self.long.shares[c['local']],'asof_30d',self.prior)
        values,positive=response_values(base,self.activity)
        if not positive.all():raise ValueError('positive carrier lost generation support')
        support,inverse=np.unique(values,return_inverse=True)
        def physical_law(w):return Law(support,np.bincount(inverse,weights=w,minlength=len(support)))
        mu=np.clip(anchor+self.mean_model.predict(X['84']),0,1000)
        sigma=np.clip(np.sqrt(np.maximum(self.variance_model.predict(X['84'])-25,1)),1,500)
        laws={m:[] for m in METHODS};diagnostics=[];states=[]
        for h in range(48):
            yy,ww=self.calendar[h];anchor_law=CompressedLaw(np.clip(anchor+yy,0,1000),ww)
            laws['anchor_calendar'].append(anchor_law)
            selected={k:self.kernels[k].atoms(signatures[k][h]) for k in self.kernels}
            for name in ('20','84'):
                ids,w=selected[name];laws['direct'+name].append(CompressedLaw(np.clip(anchor+self.y[ids],0,1000),w))
            laws['hetero84'].append(CompressedLaw(np.clip(mu[h]+sigma[h]*self.z,0,1000),np.ones(1025)))
            ids,w=selected['84'];laws['physical84_leaf'].append(CompressedLaw(values[ids],w))
            weight=.05*self.weights.copy();weight[ids]+=.95*w;weight/=weight.sum()
            physical=physical_law(weight);laws['physical84_bg'].append(physical);laws['uniform_range'].append(self.uniform)
            event,records,choice=event_updates(values,weight,self.source_ids,self.layers)
            global_weight,global_stats=global_updates(values,weight,self.source_ids)
            for name,wnew in (event|global_weight).items():
                law=physical_law(wnew)
                if not np.array_equal(law.x,physical.x):raise ValueError('event/source update changed common CIF support')
                laws[name].append(law)
            laws['uniform_mix10'].append(mixture(physical,self.uniform,.1))
            laws['ess_mix_anchor'].append(mixture(physical,anchor_law,global_stats['ess_mix']))
            laws['source_spread_mix_anchor'].append(mixture(physical,anchor_law,global_stats['spread_mix']))
            diagnostics.extend({'profile_id':q,'clock':h,**record} for record in records)
            states.append({'profile_id':q,'clock':h,**choice,**global_stats,'anchor':anchor,'physical_mean':physical.mean})
        if any(len(v)!=48 for v in laws.values()):raise ValueError('law clock inventory')
        return laws,diagnostics,states,c

    def source_prefixes(self):
        rows=[]
        for q,(_,contexts,_) in self.source_features.items():
            for kind,container,c in zip(('short','long'),(self.short,self.long),contexts):
                prefix=[int(p) for p in c['prefix']]
                if q in prefix or not (container.release.iloc[prefix]<=container.dates.iloc[q]).all() or any(container.index.region.iloc[p]!=container.index.region.iloc[q] for p in prefix):raise ValueError('source prefix legality')
                rows.append({'kind':kind,'profile_id':q,'region':container.index.region.iloc[q],'month':container.index.month.iloc[q],
                             'local':int(c['local']),'prefix':json.dumps(prefix)})
        return rows

    def fingerprint(self):
        fields=('children_left','children_right','feature','threshold','value','weighted_n_node_samples')
        result={f'kernel{k}':[{key:array_hash(getattr(t.tree_,key)) for key in fields} for t in m.forest.estimators_] for k,m in self.kernels.items()}
        for name,model in [(f'oof{i}',m) for i,m in enumerate(self.oof_models)]+[('mean',self.mean_model),('variance',self.variance_model)]:
            predictors=[]
            for stage in model._predictors:
                for p in stage:
                    fields={}
                    for field in p.nodes.dtype.names:
                        values=np.ascontiguousarray(p.nodes[field])
                        if field=='num_threshold':
                            values=values.copy();values[values==0.]=0.
                        fields[field]=array_hash(values)
                    predictors.append({'node_count':len(p.nodes),'nodes_dtype':str(p.nodes.dtype),'fields':fields,
                        'raw_left_cat_bitsets':array_hash(p.raw_left_cat_bitsets),'binned_left_cat_bitsets':array_hash(p.binned_left_cat_bitsets)})
            result[name]={'n_iter':model.n_iter_,'baseline':array_hash(model._baseline_prediction),'predictors':predictors}
        return result


def law_hashes(laws):
    return {name:[law_digest(law) for law in cells] for name,cells in laws.items()}


def input_guard(model,q):
    before=law_hashes(model.build(q)[0]);saved=[];raw=model.data.raw
    for data in (model.data,model.short,model.long):
        ids=data.groups[data.index.region.iloc[q]];ids=ids[(data.release.iloc[ids]>data.dates.iloc[q]).to_numpy()]
        saved.append((data,ids,data.M[ids].copy(),data.shares[ids].copy()))
    try:
        for data,ids,M,_ in saved:
            changed=np.random.default_rng(q+190).uniform(1,1000,size=M.shape);data.M[ids]=changed;data.shares[ids]=changed/changed.sum(axis=1,keepdims=True)
        model.data.raw=None;after=law_hashes(model.build(q)[0])
    finally:
        model.data.raw=raw
        for data,ids,M,S in saved:data.M[ids],data.shares[ids]=M,S
    if before!=after:raise ValueError('future/raw input changed CIF law')
