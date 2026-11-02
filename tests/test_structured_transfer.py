import numpy as np
import pandas as pd
import torch

from transcif.research.budget_transfer import INPUTS
from transcif.research.cross_country import COARSE, predict_models
from transcif.research.structured_transfer import (
    HEADS, StructuredNet, load_structured, predict_structured, save_structured, train_structured,
)


def test_zero_increment_identity_and_exact_disabled_path():
    frame = pd.DataFrame(np.zeros((24, len(INPUTS))), columns=INPUTS)
    frame['clock'] = pd.date_range('2024-01-01', periods=24, freq='h')
    frame[list(COARSE)] = [.5, .1, .1, .2, .1]
    profile = np.ones((24, 5))
    profile[:, 4] = np.maximum(0., np.sin(np.arange(24)/24*2*np.pi))
    # predict_models computes all reference heads; dummy models isolate fallback.
    class Zero:
        def predict(self, x):
            return np.zeros(len(x))
    legacy = {'profile': profile, 'thermal_ef': 600.,
              **{k: Zero() for k in ['calendar_ridge', 'coarse_ridge', 'coarse_boosting']}}
    expected = predict_models(legacy, frame)['coarse_recomposition']
    for head in HEADS:
        model = StructuredNet(head, profile, 600., np.array([.3, .5, .1, .1]))
        fitted = {'model': model, 'mean': np.zeros(len(INPUTS)),
                  'scale': np.ones(len(INPUTS)), 'legacy': legacy}
        np.testing.assert_allclose(predict_structured(fitted, frame), expected, atol=1e-4)
        np.testing.assert_array_equal(predict_structured(fitted, frame, enabled=False), expected)


def test_zero_thermal_stays_zero_for_structure_heads():
    for head in ['factor', 'shape', 'joint', 'joint_aux']:
        model = StructuredNet(head, np.ones((24, 5)), 600., np.ones(4)/4)
        with torch.no_grad():
            model.output.bias.fill_(10.)
        value, _ = model(torch.zeros((2, len(INPUTS))),
                         torch.tensor([[0., .5, 0., .5, 0.]]*2), torch.tensor([0, 12]))
        assert torch.equal(value, torch.zeros(2))


def test_structured_checkpoint_preserves_learned_and_native_outputs(tmp_path):
    frame = pd.DataFrame(np.zeros((48, len(INPUTS))), columns=INPUTS)
    frame['clock'] = pd.date_range('2024-01-01', periods=48, freq='h')
    frame[list(COARSE)] = [.5, .1, .1, .2, .1]
    frame['region'] = 'source'
    legacy = {'profile': np.ones((24, 5)), 'thermal_ef': 600.123456789}
    fitted = train_structured(frame, np.ones((48, 8)), 'joint_aux', 10, legacy, steps=2, batch=16)
    path = tmp_path/'structured.pt'
    save_structured(fitted, path)
    saved = torch.load(path, weights_only=True)
    model = StructuredNet(saved['head'], np.ones((24, 5)), 600., np.ones(4)/4)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    restored = {'model': model, 'mean': saved['mean'].numpy(), 'scale': saved['scale'].numpy(),
                'legacy': {'profile': saved['native_profile'].numpy(), 'thermal_ef': saved['native_thermal_ef']}}
    for enabled in (False, True):
        np.testing.assert_array_equal(predict_structured(fitted, frame, enabled),
                                      predict_structured(restored, frame, enabled))
        np.testing.assert_array_equal(predict_structured(fitted, frame, enabled),
                                      predict_structured(load_structured(path), frame, enabled))
