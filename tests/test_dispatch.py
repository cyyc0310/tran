import numpy as np
import pytest
import torch

from transcif.data.dataset import stack_inputs
from transcif.models.dispatch import ProbabilisticDispatchNet
from transcif.physics.dispatch import capped_dispatch
from transcif.training.losses import dispatch_loss


@pytest.mark.parametrize("load", [0., .4, 1.5, 3., 4.])
def test_dispatch_balance_capacity_and_shortage(load):
    cap = torch.tensor([1., 0., 2.], dtype=torch.double)
    cost = torch.tensor([.2, -.5, .6], dtype=torch.double)
    generation, shortage = capped_dispatch(cap, cost, cap.new_tensor(load))
    assert (generation >= 0).all() and (generation <= cap).all()
    assert generation[1] == 0
    assert generation.sum().item() == pytest.approx(min(load, 3), abs=1e-10)
    assert shortage.item() == pytest.approx(max(load - 3, 0), abs=1e-10)


@pytest.mark.parametrize("capacity,cost,load", [
    ([1., 2., 1.5], [.1, .2, .3], 1.3),
    ([1., 2., 1.5], [-.4, .3, .4], 1.8),
    ([1., 2., 1.5], [-.4, .3, .4], 5.),
])
def test_implicit_dispatch_gradients(capacity, cost, load):
    args = tuple(torch.tensor(x, dtype=torch.double, requires_grad=True)
                 for x in (capacity, cost, load))
    assert torch.autograd.gradcheck(lambda *x: capped_dispatch(*x), args,
                                    eps=1e-6, atol=1e-5, rtol=1e-4)


def test_cost_changes_merit_order():
    cap = torch.ones(2, dtype=torch.double)
    first, _ = capped_dispatch(cap, cap.new_tensor([0., 1.]), cap.new_tensor(.8))
    second, _ = capped_dispatch(cap, cap.new_tensor([1., 0.]), cap.new_tensor(.8))
    assert first[0] > first[1] and second[0] < second[1]


def test_scenarios_obey_physics_are_repeatable_and_trainable(bundle):
    _, doc, grids = bundle
    example = grids["source_coal"].example(doc["origins"][0])
    public = stack_inputs([example])
    model = ProbabilisticDispatchNet()
    def draw():
        return model(public, 128, torch.Generator().manual_seed(42))
    forecast = draw()
    assert torch.equal(forecast.cif, draw().cif)
    assert torch.all(forecast.generation >= 0)
    assert torch.all(forecast.generation <= public.capacity[:, None, None] + 1e-3)
    torch.testing.assert_close(forecast.generation.sum(-1) + forecast.unserved,
                               forecast.demand, atol=1e-3, rtol=1e-5)
    expected = ((forecast.generation * public.ef[:, None, None]).sum(-1)
                / forecast.generation.sum(-1))
    torch.testing.assert_close(forecast.cif, expected)
    assert forecast.cif.std(1).mean() > 0
    # Conditional scenario dependence remains after removing the deterministic daily curve.
    noise = forecast.cif[0].detach().numpy()
    noise = noise - noise.mean(0)
    assert np.corrcoef(noise[:, :-1].ravel(), noise[:, 1:].ravel())[0, 1] > .3
    observed = grids["source_coal"].labels(example)
    loss, _ = dispatch_loss(forecast, {key: torch.tensor(value[None])
                                      for key, value in observed.items()})
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.scale.weight.grad.abs().sum() > 0
    assert model.location.weight.grad[6:9].abs().sum() > 0


def test_no_invented_generation_when_fuel_absent(bundle):
    _, doc, grids = bundle
    example = grids["target_mixed"].example(doc["origins"][0])
    example.values["capacity"][0] = 0
    result = ProbabilisticDispatchNet()(stack_inputs([example]), 8)
    assert torch.count_nonzero(result.generation[..., 0]) == 0
