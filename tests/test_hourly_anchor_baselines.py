"""Response parameterization and OOF-scale unit contracts of ordinary references."""
import numpy as np
import pytest

from tests.test_hourly_transfer_baselines import example
from transcif.research import hourly_anchor_baselines as anchor
from transcif.research import hourly_transfer_baselines as direct


@pytest.mark.parametrize('kind', anchor.KINDS)
def test_response_roundtrip_and_neutral_increment_preserves_public_level(kind):
    prior = np.array([0., .015, .4, .9, 1.])
    response = np.array([0., 20., 900., 450., 1000.])
    encoded = anchor.encode(response, prior, kind)
    expected = np.clip(response, 1., 999.) if kind == 'logit' else response
    np.testing.assert_allclose(anchor.decode(encoded, prior, kind), expected, atol=1e-10)
    expected_prior = np.clip(prior, .001, .999) if kind == 'logit' else prior
    np.testing.assert_allclose(anchor.decode(np.full(5, 500.), prior, kind), expected_prior * 1000.)


@pytest.mark.parametrize('kind', anchor.KINDS)
def test_held_calendar_labels_do_not_enter_their_oof_and_scale_uses_physical_units(monkeypatch, kind):
    f, climate = example(); x = direct.design(f, climate, weather=True, capacity=False)
    original_regressor = anchor.HistGradientBoostingRegressor
    scale_targets = []

    class RecordingRegressor(original_regressor):
        def fit(self, design, response, **kwargs):
            if self.loss == 'absolute_error':
                scale_targets.append(np.asarray(response).copy())
            return super().fit(design, response, **kwargs)

    monkeypatch.setattr(anchor, 'HistGradientBoostingRegressor', RecordingRegressor)
    monkeypatch.setattr(anchor, 'OPTIONS', dict(max_iter=3, max_leaf_nodes=3, min_samples_leaf=2,
        learning_rate=.1, l2_regularization=1., early_stopping=False, random_state=416))
    y = 200. + np.arange(len(f))
    model, first = anchor.fit(f, y, x, kind=kind, allowed_countries=('S',))
    changed = y.copy(); changed[:24] += 100.
    _, second = anchor.fit(f, changed, x, kind=kind, allowed_countries=('S',))
    np.testing.assert_array_equal(first['encoded'][:24], second['encoded'][:24])
    np.testing.assert_array_equal(first['permille'][:24], second['permille'][:24])
    np.testing.assert_array_equal(first['permille'], anchor.decode(first['encoded'], f.thermal.to_numpy(), kind))
    np.testing.assert_array_equal(scale_targets[0], np.abs(y - first['permille']))
    assert not np.allclose(scale_targets[0], np.abs(anchor.encode(y, f.thermal.to_numpy(), kind) - first['encoded']))
    for fold in model['audit']['folds']:
        assert not set(fold['proper_months']) & set(fold['held_months'])
    with pytest.raises(ValueError, match='source-2022'):
        anchor.fit(f, y, x, kind=kind, allowed_countries=('T',))


@pytest.mark.parametrize('kind, bounds', [('additive', [0., 400.]), ('logit', [.4, 399.6])])
def test_decoding_and_scale_conversion_are_in_cif_units(kind, bounds):
    class Constant:
        def __init__(self, value): self.value = value
        def predict(self, x): return np.full(len(x), self.value)

    fitted = dict(kind=kind, epsilon=.001, location=Constant(500.), scale=Constant(20.),
                  quantiles=[Constant(value) for value in [750., 250., 500.]])
    prediction = anchor.forecast(fitted, np.zeros((1, 2)), np.array([.5]), np.array([.4]))
    assert prediction['point'][0] == pytest.approx(200.)
    assert prediction['median'][0] == pytest.approx(200.)
    assert prediction['scale'][0] == pytest.approx(8.)
    assert prediction['lower'][0] == pytest.approx(bounds[0])
    assert prediction['upper'][0] == pytest.approx(bounds[1])


def test_invalid_anchor_or_response_cannot_silently_clip_bad_source_data():
    with pytest.raises(ValueError, match='anchor'):
        anchor.encode(np.array([500.]), np.array([1.1]), 'additive')
    with pytest.raises(ValueError, match='response'):
        anchor.encode(np.array([1001.]), np.array([.5]), 'logit')
