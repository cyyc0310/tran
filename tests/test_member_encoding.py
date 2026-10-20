from dataclasses import fields

import pytest
import torch

from transcif.data.dataset import ModelInputs, stack_inputs
from transcif.models.dispatch import ProbabilisticDispatchNet


def differentiable_copy(inputs):
    return ModelInputs(**{f.name: getattr(inputs, f.name).double().detach().requires_grad_()
                          for f in fields(inputs)})


@pytest.mark.parametrize("backbone", ["tcn", "context", "shared", "covariate"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_member_encoding_preserves_paths_and_gradients(bundle, backbone, deterministic):
    _, doc, grids = bundle
    inputs = stack_inputs([grids[name].example(doc["origins"][0])
                           for name in ("source_coal", "source_wind")])
    torch.manual_seed(9)
    efficient = ProbabilisticDispatchNet(hidden=8, backbone=backbone).double()
    # Nonzero heads exercise encoder/input gradients as well as decoder gradients.
    with torch.no_grad():
        efficient.location.weight.normal_(0, .03)
        efficient.scale.weight.add_(torch.randn_like(efficient.scale.weight) * .03)
    reference = ProbabilisticDispatchNet(**(efficient.model_config | {"encode_members": False})).double()
    reference.load_state_dict(efficient.state_dict())
    left, right = differentiable_copy(inputs), differentiable_copy(inputs)
    forecasts = []
    for model, batch in ((efficient, left), (reference, right)):
        result = model(batch, 8, generator=torch.Generator().manual_seed(6), deterministic=deterministic)
        forecasts.append(result)
        (result.cif.square().mean() / 1000 + result.generation.square().mean() / 1e6).backward()
    for f in fields(forecasts[0]):
        torch.testing.assert_close(getattr(forecasts[0], f.name), getattr(forecasts[1], f.name),
                                   atol=1e-8, rtol=1e-8)
    for (_, a), (_, b) in zip(efficient.named_parameters(), reference.named_parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-7)
    for f in fields(left):
        a, b = getattr(left, f.name).grad, getattr(right, f.name).grad
        if a is None or b is None:
            assert a is b is None
        else:
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-7)


def test_member_reuse_never_encodes_more_than_sampled_scenarios(bundle):
    _, doc, grids = bundle
    batch = stack_inputs([grids["source_coal"].example(doc["origins"][0])])
    model = ProbabilisticDispatchNet(hidden=8)
    calls = []
    hook = model.encoder.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[0]))
    model(batch, 16)
    model(batch, 2)
    model(batch, 16, deterministic=True)
    hook.remove()
    assert calls == [batch.weather.shape[1], 2, 1]


def test_shared_policy_has_no_configuration_encoder_and_roundtrips(bundle):
    _, doc, grids = bundle
    batch = stack_inputs([grids["source_coal"].example(doc["origins"][0])])
    model = ProbabilisticDispatchNet(backbone="shared", point_forecast=True)
    assert model.encoder is None
    assert sum(p.numel() for p in model.parameters()) == 22
    restored = ProbabilisticDispatchNet(**model.model_config)
    restored.load_state_dict(model.state_dict())
    forecast = model(batch, 2, deterministic=True)
    torch.testing.assert_close(forecast.cif, restored(batch, 2, deterministic=True).cif)
    torch.testing.assert_close(forecast.generation.sum(-1) + forecast.unserved, forecast.demand,
                               rtol=1e-5, atol=1e-3)
