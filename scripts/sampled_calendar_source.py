"""Calendar source law restricted to an explicitly supplied source-hour sample."""
import numpy as np
from calendar_cif_law import CalendarSource,cells

class SampledCalendarSource(CalendarSource):
    def __init__(self,index,raw,margins,unix,allowed):
        super().__init__(index,raw,margins,unix)
        self.allowed=np.sort(np.asarray(allowed,dtype=int))
        if len(np.unique(self.allowed))!=len(self.allowed):raise ValueError('duplicate sampled source hour')
    def donor(self,d,weekend,shift):
        key=(int(d),weekend,shift)
        if key not in self.cache:
            row=self.index.iloc[d]
            selected=self.allowed[(self.allowed>=int(row.offset))&(self.allowed<int(row.stop))]
            if not len(selected):raise ValueError('source month absent from matched sample')
            g=self.raw[selected]
            means=self.M[d]/int(row.hours)
            shape=np.divide(g,means,out=np.ones_like(g),where=means>0)
            states=cells(self.times[selected],weekend,shift)
            indices=[];averages=[]
            for s in range(48 if weekend else 24):
                chosen=np.flatnonzero(states==s)
                if not len(chosen):chosen=np.arange(len(selected));self.empty_cell_fallbacks+=1
                indices.append(chosen);averages.append(shape[chosen].mean(axis=0))
            self.cache[key]=(shape,indices,np.asarray(averages))
        return self.cache[key]
