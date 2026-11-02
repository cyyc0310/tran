import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from transcif.research.cross_country import COARSE
from transcif.research.monthly_adaptation import (
    METHODS, fit_risks, forecast, risk_bounds, supervised_months, weighted_quantile,
)


def test_all_adapters_ignore_newer_than_release_cutoff():
    index = pd.period_range('2021-01', '2024-12', freq='M')
    rng = np.random.default_rng(10)
    values = rng.dirichlet(np.ones(5), len(index))
    table = pd.DataFrame(values, index=index, columns=COARSE)
    x, y = supervised_months(table.loc[:'2022-12'])
    source = Ridge(alpha=10.).fit(x, y)
    period = pd.Period('2024-05', freq='M')
    changed = table.copy()
    changed.loc[changed.index > period-2] = [.99, .0025, .0025, .0025, .0025]
    for method in METHODS:
        a, b = forecast(table, period, source, method), forecast(changed, period, source, method)
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(a).all() and (a >= 0).all()
        np.testing.assert_allclose(a.sum(), 1.)


def test_local_adapter_needs_no_cif_or_subfuel_columns():
    table = pd.DataFrame([[.5, .1, .1, .2, .1]]*12,
                         index=pd.period_range('2022-01', periods=12, freq='M'), columns=COARSE)
    x, y = supervised_months(table)
    model = Ridge(alpha=10.).fit(x, y)
    for method in METHODS:
        result = forecast(table, pd.Period('2023-02', freq='M'), model, method)
        np.testing.assert_allclose(result, table.iloc[-1], atol=.001)


def test_risk_reference_bounds_and_region_weighted_quantile():
    assert weighted_quantile(np.array([0., 10.]), np.array([.9, .1]), .95) == 10.
    rng = np.random.default_rng(42)
    x = rng.normal(size=(240, 11))
    e = rng.normal(size=240)
    regions = np.array(['A']*80+['B']*160)
    fitted = fit_risks(x, e, regions, x, e, regions)
    bounds = risk_bounds(fitted, x, np.zeros(240))
    for lower, upper in bounds.values():
        assert np.isfinite(lower).all() and np.isfinite(upper).all()
        assert (lower <= upper).all()
