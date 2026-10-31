"""Version-pinned pinball subgradient control for histogram boosting."""
import numpy as np
import sklearn
from sklearn._loss.loss import PinballLoss

class StrictHistogramPinball(PinballLoss):
    def __init__(self,quantile=.5,sample_weight=None):
        if sklearn.__version__!='1.8.0':raise RuntimeError('requires sklearn 1.8.0')
        super().__init__(sample_weight=sample_weight,quantile=quantile);self.q=quantile
    def _correction(self,y_true,raw_prediction,sample_weight):
        mask=np.asarray(y_true).reshape(-1)==np.asarray(raw_prediction).reshape(-1)
        value=1-self.q if sample_weight is None else (1-self.q)*np.asarray(sample_weight).reshape(-1)[mask].copy()
        return mask,value
    def gradient(self,y_true,raw_prediction,sample_weight=None,gradient_out=None,n_threads=1):
        mask,value=self._correction(y_true,raw_prediction,sample_weight)
        out=super().gradient(y_true,raw_prediction,sample_weight,gradient_out,n_threads)
        out[mask]=value;return out
    def gradient_hessian(self,y_true,raw_prediction,sample_weight=None,gradient_out=None,hessian_out=None,n_threads=1):
        mask,value=self._correction(y_true,raw_prediction,sample_weight)
        g,h=super().gradient_hessian(y_true,raw_prediction,sample_weight,gradient_out,hessian_out,n_threads)
        g[mask]=value;return g,h
    def loss_gradient(self,y_true,raw_prediction,sample_weight=None,loss_out=None,gradient_out=None,n_threads=1):
        mask,value=self._correction(y_true,raw_prediction,sample_weight)
        loss,g=super().loss_gradient(y_true,raw_prediction,sample_weight,loss_out,gradient_out,n_threads)
        g[mask]=value;return loss,g
