"""Shared legal monthly/calendar features and sampled joint fuel supervision."""
from fractions import Fraction
import hashlib
import numpy as np
import pandas as pd
from v151_hourly_law import EF

class HourlyFuelData:
    def __init__(self,root):
        p=root/'results/v144-canonical-profile-cache-r1'
        self.index=pd.read_csv(p/'profile_index.csv');packed=np.load(p/'profiles.npz',allow_pickle=False)
        self.raw=packed['generation'];self.M=np.asarray([[float(Fraction(str(v))) for v in row] for row in packed['margin_fractions']])
        self.shares=self.M/self.M.sum(axis=1,keepdims=True);self.dates=pd.to_datetime(self.index.month)
        self.release=self.dates+pd.offsets.MonthBegin(1)+pd.Timedelta(days=30)
        times=pd.DatetimeIndex(pd.to_datetime(np.load(root/'results/v153-calendar-times-r1/timestamps.npz')['unix_seconds'],unit='s',utc=True))
        self.states=times.hour.to_numpy()+24*(times.dayofweek.to_numpy()>=5)
        self.groups={r:np.asarray(ids,int) for r,ids in self.index.groupby('region').indices.items()};self.cache={}
    def local(self,q,budget):
        if budget=='same_month':return q
        ids=self.groups[self.index.region.iloc[q]];eligible=ids[(self.release.iloc[ids]<=self.dates.iloc[q]).to_numpy()]
        return int(eligible[np.argmax(self.dates.iloc[eligible].to_numpy())]) if len(eligible) else None
    def features(self,q,local,cell):
        hour=cell%24;sin=np.sin(2*np.pi*hour/24);cos=np.cos(2*np.pi*hour/24);mix=self.shares[local]
        mn=int(self.index.month.iloc[q][5:7]);age=(self.dates.iloc[q]-self.dates.iloc[local]).days/365
        return np.column_stack([np.tile(mix,(len(cell),1)),np.full(len(cell),np.log1p(self.M[local].sum())),np.full(len(cell),age),
            np.full(len(cell),mix@EF/1000),np.full(len(cell),np.sin(2*np.pi*mn/12)),np.full(len(cell),np.cos(2*np.pi*mn/12)),
            sin,cos,cell//24,mix[0]*sin,mix[0]*cos,mix[6]*sin,mix[6]*cos])
    def source(self,region,budget,allowed):
        allowed=np.asarray(allowed,int);key=(region,budget)
        digest=hashlib.sha256(allowed.tobytes()).hexdigest()
        if key in self.cache:
            old_digest,result=self.cache[key]
            if digest!=old_digest:raise ValueError('source samples changed across target folds')
            return result
        xs=[];scalar=[];joint=[];ordered=[];input_ids=[]
        for q in self.groups[region]:
            row=self.index.iloc[q];pick=allowed[(allowed>=row.offset)&(allowed<row.stop)]
            if not len(pick):continue
            if not row.month.startswith('2024'):raise ValueError('source year leakage')
            local=self.local(q,budget)
            if local is None:raise ValueError('unavailable source input')
            g=self.raw[pick];total=g.sum(axis=1);s=g/total[:,None]
            xs.append(self.features(q,local,self.states[pick]));scalar.extend(g@EF/total-self.shares[local]@EF)
            joint.append(s-self.shares[local]);ordered.extend(pick);input_ids.append(local)
        d=np.concatenate(joint);y=np.asarray(scalar)
        if not np.array_equal(np.asarray(ordered),allowed) or not np.allclose(d.sum(axis=1),0,atol=1e-12):raise ValueError('source order/simplex')
        if not np.allclose(d@EF,y,atol=1e-9,rtol=0):raise ValueError('joint/scalar response identity')
        result=(np.concatenate(xs),y,d,np.unique(input_ids))
        self.cache[key]=(digest,result);return result
    def truth(self,q):
        row=self.index.iloc[q];g=self.raw[int(row.offset):int(row.stop)]
        return g@EF/g.sum(axis=1)

def repair_response(composition,innovations,hard_support=False):
    fuel=np.maximum(composition+innovations,0)
    if hard_support:fuel[:,composition==0]=0
    total=fuel.sum(axis=1);fallback=total<=1e-12
    fuel=np.divide(fuel,total[:,None],out=np.tile(composition,(len(fuel),1)),where=~fallback[:,None])
    if not np.allclose(fuel.sum(axis=1),1,atol=1e-12) or (fuel<0).any():raise ValueError('invalid fuel repair')
    if hard_support and np.any(fuel[:,composition==0]!=0):raise ValueError('hard support repair')
    return fuel@EF,int(fallback.sum())
