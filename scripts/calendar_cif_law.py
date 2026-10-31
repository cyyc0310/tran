"""Calendar-stratified source laws and expected fuel-margin reconciliation."""
import numpy as np
import pandas as pd
from v151_hourly_law import Law, EF

def cells(times,weekend=False,shift=0):
    return (times.hour.to_numpy()+shift)%24+(24*(times.dayofweek.to_numpy()>=5) if weekend else 0)

class CalendarLaw:
    def __init__(self,laws,query_cells):
        self.laws=laws;self.query_cells=query_cells
        self.mean=np.asarray([laws[int(s)].mean for s in query_cells])
    def quantile(self,p):
        values=np.asarray([law.quantile(p) for law in self.laws])
        return values[self.query_cells]
    def crps(self,y):
        result=np.empty(len(y))
        for s in np.unique(self.query_cells):
            mask=self.query_cells==s;result[mask]=self.laws[int(s)].crps(y[mask])
        return result

class CalendarSource:
    def __init__(self,index,raw,margins,unix):
        self.index=index;self.raw=raw;self.M=margins
        self.times=pd.DatetimeIndex(pd.to_datetime(unix,unit='s',utc=True));self.cache={}
        self.margin_checks=0;self.max_margin_error=0.;self.empty_cell_fallbacks=0
    def donor(self,d,weekend,shift):
        key=(int(d),weekend,shift)
        if key not in self.cache:
            row=self.index.iloc[d];g=self.raw[int(row.offset):int(row.stop)]
            shape=np.divide(g,self.M[d]/len(g),out=np.ones_like(g),where=self.M[d]>0)
            s=cells(self.times[int(row.offset):int(row.stop)],weekend,shift)
            indices=[];means=[]
            for state in range(48 if weekend else 24):
                ids=np.flatnonzero(s==state)
                if not len(ids):ids=np.arange(len(g));self.empty_cell_fallbacks+=1
                indices.append(ids);means.append(shape[ids].mean(axis=0))
            self.cache[key]=(shape,indices,np.asarray(means))
        return self.cache[key]
    def build(self,q,donors,weights,local,weekend,reconcile,shift=0):
        row=self.index.iloc[q];start=pd.Timestamp(row.month,tz='UTC')
        schedule=pd.date_range(start,start+pd.offsets.MonthBegin(1),freq='h',inclusive='left')
        full_cells=cells(schedule,weekend);n=48 if weekend else 24
        occupancy=np.bincount(full_cells,minlength=n)/len(schedule)
        prepared=[]
        for d in donors:
            shape,indices,means=self.donor(int(d),weekend,shift)
            normalization=occupancy@means if reconcile else np.ones(8)
            if (normalization<=0).any():raise ValueError('invalid fuel normalization')
            if reconcile:
                error=float(np.max(np.abs(occupancy@(means/normalization)-1)))
                self.margin_checks+=1;self.max_margin_error=max(self.max_margin_error,error)
                if error>1e-10:raise ValueError('expected fuel-margin reconciliation failed')
            prepared.append((shape,indices,normalization))
        laws=[]
        for state in range(n):
            values=[];atom_weights=[]
            for j,(shape,indices,normalization) in enumerate(prepared):
                g=shape[indices[state]]/normalization*self.M[local];total=g.sum(axis=1)
                cif=np.divide(g@EF,total,out=np.zeros(len(g)),where=total>0)
                values.extend(cif);atom_weights.extend(np.full(len(g),weights[j]/len(g)))
            laws.append(Law(values,atom_weights))
        requested=cells(self.times[int(row.offset):int(row.stop)],weekend)
        return CalendarLaw(laws,requested)
