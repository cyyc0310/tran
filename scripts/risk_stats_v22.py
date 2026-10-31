"""Exact historical HG/support tail information, not forecasting or a new method."""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from scripts.tail_count_v20 import coarse_counts, moment_counts, combine_blocks, describe

F = Fraction


def _exact(value: int | F) -> bool:
    return isinstance(value, F) or type(value) is int


def _check(n: int, total_g: int | F, carbon_energy: int | F,
           threshold: int | F, support: int | F) -> None:
    if (type(n) is not int or n < 1
            or not all(_exact(x) for x in (total_g, carbon_energy, threshold, support))
            or total_g <= 0 or not 0 < threshold < support
            or not 0 <= carbon_energy <= support * total_g):
        raise ValueError('Exact positive G, n>=1, supported H and 0<threshold<support required')


def full_g_counts(generation: Sequence[int | F], carbon_energy: int | F,
                  threshold: int | F, support: int | F = F(1000)) -> tuple[int, ...]:
    """Sharp strict-tail count interval for fixed positive hourly G, via sorting.

    H<=tau*G: k is feasible iff k=0 or tau*sum(k smallest G)<H.
    H>tau*G: k is feasible iff H<=tau*G+(B-tau)*sum(k largest G).
    Thus feasible integers form one contiguous interval. No floating arithmetic.
    """
    g = tuple(generation)
    if not g or any(not _exact(x) or x <= 0 for x in g):
        raise ValueError('Nonempty exact strictly positive hourly G required')
    total = sum(g, F(0))
    _check(len(g), total, carbon_energy, threshold, support)
    ordered = sorted(g)
    if carbon_energy <= threshold * total:
        upper, selected = 0, F(0)
        for k, value in enumerate(ordered, 1):
            selected += value
            if threshold * selected < carbon_energy:
                upper = k
        return tuple(range(upper + 1))
    selected = F(0)
    for k, value in enumerate(reversed(ordered), 1):
        selected += value
        if carbon_energy <= threshold * total + (support - threshold) * selected:
            return tuple(range(k, len(g) + 1))
    raise AssertionError('Supported H must admit the all-exceedance assignment')


def minimum_g_counts(n: int, total_g: int | F, minimum_g: int | F,
                     carbon_energy: int | F, threshold: int | F,
                     support: int | F = F(1000)) -> tuple[int, ...]:
    """Sharp count set for exact min(G)=m, total G, H and C support only.

    This information class omits the G second moment, fuel margins and timing.
    Exact min, not a lower bound, is assumed; it is a contemporaneous oracle.
    """
    _check(n, total_g, carbon_energy, threshold, support)
    if (not _exact(minimum_g) or minimum_g <= 0 or n * minimum_g > total_g
            or (n == 1 and minimum_g != total_g)):
        raise ValueError('Exact min G requires 0<m<=G/n and m=G when n=1')
    result = []
    low = carbon_energy <= threshold * total_g
    for k in range(n + 1):
        if k == 0:
            possible = low
        elif k == n:
            possible = not low
        elif low:
            possible = carbon_energy > threshold * k * minimum_g
        else:
            possible = carbon_energy <= (threshold * total_g
                + (support - threshold) * (total_g - (n - k) * minimum_g))
        if possible:
            result.append(k)
    return tuple(result)


def _carbon_for_subset(generation: Sequence[F], selected: Sequence[bool],
                       carbon_energy: F, threshold: F, support: F) -> tuple[F, ...]:
    """Construct a feasible exact carbon vector; strict selected margins stay >0."""
    mass = sum((g for g, flag in zip(generation, selected) if flag), F(0))
    if mass == 0:
        return tuple(carbon_energy / sum(generation) for _ in generation)
    margin = min((support - threshold) / 2, (carbon_energy - threshold * mass) / (2 * mass))
    if margin <= 0:
        raise ValueError('Selected strict exceedances have no feasible positive margin')
    carbon = [threshold + margin if flag else F(0) for flag in selected]
    remaining = carbon_energy - sum((g * c for g, c in zip(generation, carbon)), F(0))
    for i, (g, flag) in enumerate(zip(generation, selected)):
        increment = min(remaining / g, (support if flag else threshold) - carbon[i])
        carbon[i] += increment
        remaining -= increment * g
    if remaining != 0:
        raise AssertionError('Feasible HG construction did not exhaust H')
    return tuple(carbon)


def minimum_g_witness(n: int, total_g: int | F, minimum_g: int | F,
                      carbon_energy: int | F, threshold: int | F, count: int,
                      support: int | F = F(1000)) -> tuple[tuple[F, ...], tuple[F, ...]]:
    """Give exact extremal-allocation witnesses for every admitted Gmin count."""
    allowed = minimum_g_counts(n, total_g, minimum_g, carbon_energy, threshold, support)
    if type(count) is not int or count not in allowed:
        raise ValueError('Requested count is not feasible for these exact observations')
    total_g, minimum_g, carbon_energy, threshold, support = map(
        F, (total_g, minimum_g, carbon_energy, threshold, support))
    g = [minimum_g] * n
    # Place the excess on the complement in the low-H regime, the event set otherwise.
    index = count if carbon_energy <= threshold * total_g and count < n else 0
    g[index] += total_g - n * minimum_g
    selected = [i < count for i in range(n)]
    c = _carbon_for_subset(g, selected, carbon_energy, threshold, support)
    if (min(g) != minimum_g or sum(g) != total_g or sum(a * b for a, b in zip(g, c)) != carbon_energy
            or sum(value > threshold for value in c) != count or any(not 0 <= value <= support for value in c)):
        raise AssertionError('Gmin witness violated its declared observations')
    return tuple(g), c
