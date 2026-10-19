"""Differentiable capacity-constrained convex dispatch, with implicit gradients."""

import torch


class _CappedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, capacity, cost, served_load, temperature):
        # min_g sum(cost*g + temperature*g²/(2*capacity)),
        # subject to 0 <= g <= capacity and sum(g) = served_load.
        lower = cost.amin(-1) - temperature
        upper = cost.amax(-1) + temperature
        for _ in range(48):
            price = (lower + upper) / 2
            utilization = ((price[..., None] - cost) / temperature).clamp(0, 1)
            supply = (capacity * utilization).sum(-1)
            lower = torch.where(supply < served_load, price, lower)
            upper = torch.where(supply < served_load, upper, price)
        generation = capacity * (((lower + upper)[..., None] / 2 - cost)
                                 / temperature).clamp(0, 1)
        generation = torch.where(served_load[..., None] <= 0,
                                 torch.zeros_like(generation), generation)
        ctx.save_for_backward(capacity, generation)
        ctx.temperature = temperature
        return generation

    @staticmethod
    def backward(ctx, grad):
        capacity, generation = ctx.saved_tensors
        utilization = torch.where(capacity > 0, generation / capacity.clamp_min(1e-15),
                                  torch.zeros_like(capacity))
        active = (utilization > 1e-7) & (utilization < 1 - 1e-7) & (capacity > 0)
        slope = capacity / ctx.temperature * active
        total = slope.sum(-1, keepdim=True)
        average_grad = torch.where(total > 0, (grad * slope).sum(-1, keepdim=True)
                                   / total.clamp_min(1e-15), torch.zeros_like(total))
        return (utilization * (grad - average_grad),
                slope * (average_grad - grad), average_grad.squeeze(-1), None)


def capped_dispatch(capacity, cost, load, temperature=0.3):
    """Return generation and explicit unserved load; units must be consistent.

    Cost and temperature share units. Capacity and load share units. This
    single-bus relaxation has no ramps, unit commitment, storage or imports.
    """
    if (temperature <= 0 or capacity.shape != cost.shape or capacity.ndim < 1
            or capacity.shape[-1] == 0 or load.shape != capacity.shape[:-1]):
        raise ValueError("aligned capacity/cost (...,F), load (...), positive temperature required")
    if (not torch.isfinite(capacity).all() or not torch.isfinite(cost).all()
            or not torch.isfinite(load).all() or (capacity < 0).any() or (load < 0).any()):
        raise ValueError("finite inputs and nonnegative capacity/load required")
    served = torch.minimum(load, capacity.sum(-1))
    generation = _CappedDispatch.apply(capacity, cost, served, float(temperature))
    return generation, (load - generation.sum(-1)).clamp_min(0)
