import json
import numpy as np
import pytest
from transcif.research.joint_v3 import compose, select_source_alpha, load_component, sha


def test_composition_residual_identity_and_zero_mean():
    rng=np.random.default_rng(9)
    truth=rng.normal(size=(8,24))
    latest=rng.normal(size=(8,24))
    delta=rng.normal(size=(8,24)); delta-=delta.mean(axis=1,keepdims=True)
    p={'latest':latest,'analog_shape':latest+delta,'seasonal':latest+.3}
    e={k:truth-v for k,v in p.items()}
    for a in (0,.25,.5,1):
        point,residual=compose(p,e,a)
        np.testing.assert_allclose(residual,truth-point,atol=1e-14)
        np.testing.assert_allclose(point.mean(axis=1),p['seasonal'].mean(axis=1))
    # No clipping may destroy the mean identity.
    assert compose(p,e,.5)[0].min()<0


def test_policy_requires_only_source_residuals():
    e={'seasonal':np.ones((8,24)), 'analog_shape':np.zeros((8,24)), 'latest':np.ones((8,24))}
    assert select_source_alpha(e)[0]==1


def test_reject_unaligned_indices(tmp_path):
    p=np.ones((2,24))
    np.savez(tmp_path/'predictions.npz',**{k:p for k in ('latest','analog_shape','seasonal','analog_shape_half','source_policy')},**{'residual_'+k:p for k in ('latest','analog_shape','seasonal','analog_shape_half','source_policy')},origins=np.arange(2))
    seal={'prediction_sha256':sha(tmp_path/'predictions.npz'),'target_labels_read_before_seal':False,'calibration_indices':[1,2],'calibration_indices_by_method':{'seasonal':[2,1]},'policy':{'folds':[{'days':3,'held_source':'a','fit_sources':['b']}]}}
    (tmp_path/'prediction_seal.json').write_text(json.dumps(seal))
    with pytest.raises(ValueError,match='Unaligned source indices'):
        load_component(tmp_path,{})


@pytest.mark.parametrize('changed', ['code.py', 'data.csv'])
def test_prefit_freeze_rejects_fit_mutation_before_labels(tmp_path, changed):
    from transcif.research.joint_v3 import freeze_inputs, verify_frozen_inputs
    (tmp_path/'code.py').write_text('frozen code')
    (tmp_path/'data.csv').write_text('frozen data')
    manifest=freeze_inputs(tmp_path,tmp_path,['code.py'],tmp_path/'run')
    def mock_fit():
        (tmp_path/changed).write_text('changed during fit')
    mock_fit()
    label_opened=False
    with pytest.raises(ValueError,match='Frozen input changed'):
        verify_frozen_inputs(tmp_path,manifest)
        label_opened=True
    assert not label_opened
