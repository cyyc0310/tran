import numpy as np
import pandas as pd
import torch

from transcif.research.budget_transfer import (
    BudgetNet, INPUTS, budget_views, predict_model, save_model, train_model,
)
from transcif.research.cross_country import FUELS, monthly_inputs


def test_matched_supervision_and_legal_ages():
    clock = pd.date_range('2022-01-01', '2022-08-01', freq='h', inclusive='left')
    raw = pd.DataFrame(np.ones((len(clock), 8)), columns=FUELS)
    raw['clock'], raw['time'] = clock, clock
    raw['country'], raw['region'] = 'GB', 'GB'
    raw['expected_hours'] = clock.days_in_month*24
    views = {lag: monthly_inputs(raw, lag) for lag in (2, 3, 4)}
    a, b = budget_views(views, 2022, False), budget_views(views, 2022, True)
    assert a.raw_row.equals(b.raw_row)
    assert set(a.age_months) == {2}
    assert set(b.age_months) == {2, 3, 4}
    delta = b.period.astype('int64')-pd.PeriodIndex(b.observed_month, freq='M').asi8
    np.testing.assert_array_equal(delta, b.age_months)


def test_mixture_range_and_safe_checkpoint(tmp_path):
    frame = pd.DataFrame(np.zeros((64, len(INPUTS))), columns=INPUTS)
    frame['region'] = 'source'
    fuel = np.ones((64, 8))
    fitted = train_model(frame, fuel, 'fuel_aux', 10, steps=2, batch=16)
    pred = predict_model(fitted, frame)
    assert np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 1000).all()
    path = tmp_path/'model.pt'
    save_model(fitted, path)
    saved = torch.load(path, weights_only=True)
    model = BudgetNet(saved['head'])
    model.load_state_dict(saved['state_dict'])
    restored = {'model': model, 'mean': saved['mean'].numpy(), 'scale': saved['scale'].numpy()}
    np.testing.assert_array_equal(predict_model(restored, frame), pred)
