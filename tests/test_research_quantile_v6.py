import numpy as np
import pandas as pd
import pytest

from transcif.research.quantile_baseline_v6 import QuantileBaseline


def fixture():
    days = pd.DatetimeIndex(["2024-01-10", "2024-03-10", "2024-07-10", "2024-10-10"] * 2)
    regions = np.repeat(["a", "b"], 4)
    point = np.full((8,24), 300.)
    truth = point + np.random.default_rng(1).normal(size=point.shape)*10
    features = np.arange(16).reshape(8,2).astype(float)
    return point, truth, features, regions, days


def test_quantile_temporal_calibration_does_not_train_on_bank():
    p,t,x,r,d = fixture()
    options = dict(days=d, level_scales=np.ones(8), shape_scales=np.ones((8,24)))
    first = QuantileBaseline(["a","b"]).fit(p,t,x,r,**options)
    shifted=t.copy();shifted[d.month>6]+=50
    second = QuantileBaseline(["a","b"]).fit(p,shifted,x,r,**options)
    design=first._design(p,x,options["level_scales"],options["shape_scales"])
    for a,b in zip(first.models_,second.models_):
        np.testing.assert_array_equal(a.predict(design),b.predict(design))
    assert first.offset_ != second.offset_
    lo,hi=first.intervals(p,x,level_scales=options["level_scales"],shape_scales=options["shape_scales"])
    assert lo.shape==hi.shape==p.shape and np.all(lo<=hi)
    assert first.audit_["interval_only"]


def test_quantile_rejects_target_and_wrong_year():
    p,t,x,r,d=fixture()
    options=dict(level_scales=np.ones(8),shape_scales=np.ones((8,24)))
    with pytest.raises(ValueError,match="source"):
        QuantileBaseline(["a","b"]).fit(p,t,x,np.repeat("target",8),days=d,**options)
    with pytest.raises(ValueError,match="year"):
        QuantileBaseline(["a","b"]).fit(p,t,x,r,days=d+pd.DateOffset(years=1),**options)
