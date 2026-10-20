"""Exact capacity-only CIF extrema for a stated single-bus relaxation.

These are conditional feasibility bounds, not probability intervals, and not a
guarantee about real generation, weather, accounting boundaries or target risk.
There is no storage, net interchange, ramping or unit commitment in this set.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CIFEnvelope:
    lower: np.ndarray
    upper: np.ndarray
    lower_generation: np.ndarray
    upper_generation: np.ndarray

    @property
    def midpoint(self):
        """Minimax absolute-error point within this information class only."""
        return (self.lower + self.upper) / 2

    @property
    def radius(self):
        return (self.upper - self.lower) / 2


def dispatch_cif_envelope(capacity, total_generation, ef_lower, ef_upper=None,
                          minimum_generation=None):
    """Extremize EF-weighted generation / fixed positive domestic generation.

    Shapes are capacity (..., F), total_generation (...), broadcastable EF and
    minimum-generation arrays (..., F). Units are MW and gCO2/kWh. Any number
    of nonnegative-output primary-generation categories is supported. Signed
    storage/net-flow measurements cannot be passed as generation capacities.

    Assumptions: each output independently lies in [minimum, capacity], sums
    exactly to total_generation, and each EF independently lies in its stated
    interval. No costs/merit order or learned source restrictions are imposed.
    Missing fuels or EFs must be resolved explicitly; NaNs are never zeroed.
    A domestic-generation forecast is not generally equal to a load forecast.

    The endpoints are attained by the returned witnesses with the respective
    EF endpoints. The construction is the elementary continuous-knapsack LP,
    and should not itself be presented as a novel identification theorem.
    """
    capacity = np.asarray(capacity, dtype=np.float64)
    total = np.asarray(total_generation, dtype=np.float64)
    if capacity.ndim < 1 or capacity.shape[-1] == 0 or total.shape != capacity.shape[:-1]:
        raise ValueError("capacity (...,F) and aligned total_generation (...) required")
    try:
        low_ef = np.broadcast_to(np.asarray(ef_lower, dtype=np.float64), capacity.shape)
        high_ef = np.broadcast_to(np.asarray(ef_lower if ef_upper is None else ef_upper,
                                           dtype=np.float64), capacity.shape)
        minimum = np.broadcast_to(np.asarray(0 if minimum_generation is None else minimum_generation,
                                           dtype=np.float64), capacity.shape)
    except ValueError as error:
        raise ValueError("EF/minimum arrays must broadcast to capacity") from error
    if any(not np.isfinite(x).all() for x in (capacity, total, low_ef, high_ef, minimum)):
        raise ValueError("finite complete generation and EF inputs required")
    if ((minimum < 0).any() or (capacity < minimum).any() or (low_ef < 0).any()
            or (high_ef < low_ef).any() or (total <= 0).any()):
        raise ValueError("nonnegative ordered generation/EF bounds and positive total required")
    if ((total < minimum.sum(-1)).any() or (total > capacity.sum(-1)).any()):
        raise ValueError("infeasible total generation; no shortage or rescaling is silently applied")

    spare, residual = capacity - minimum, total - minimum.sum(-1)

    def extreme(ef, descending=False):
        order = np.argsort(-ef if descending else ef, axis=-1, kind="stable")
        sorted_spare = np.take_along_axis(spare, order, axis=-1)
        # Exclusive prefix avoids cancellation from inclusive_prefix - current
        # when capacity magnitudes differ substantially.
        prefix = np.concatenate([np.zeros_like(sorted_spare[..., :1]),
                                 np.cumsum(sorted_spare[..., :-1], axis=-1)], axis=-1)
        allocation = np.minimum(sorted_spare, np.maximum(0, residual[..., None] - prefix))
        generation = minimum + np.take_along_axis(allocation, np.argsort(order, axis=-1), axis=-1)
        return (generation * ef).sum(-1) / total, generation

    lower, lower_generation = extreme(low_ef)
    upper, upper_generation = extreme(high_ef, descending=True)
    return CIFEnvelope(lower, upper, lower_generation, upper_generation)
