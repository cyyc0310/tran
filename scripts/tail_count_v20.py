"""Exact finite-block tail-count diagnostics, never a forecast or new theorem.

Carbon energy is sum(G*C), with C in[0,B] and every G strictly positive.
Fractions represent the declared decimal observations exactly. The CV route
is a necessary outer set; the full-G route is sharp only for HG/support.
"""
from fractions import Fraction
from itertools import product
from typing import Sequence

F = Fraction


def _check(n: int, total_g: F, carbon_energy: F, threshold: F, support: F) -> None:
    if (type(n) is not int or n < 1 or any(not isinstance(x, (int, F)) for x in
            (total_g, carbon_energy, threshold, support)) or total_g <= 0
            or not 0 < threshold < support or not 0 <= carbon_energy <= support*total_g):
        raise ValueError('Positive G, supported carbon energy and0<threshold<support required')


def coarse_counts(n: int, total_g: F, carbon_energy: F, threshold: F,
                  support: F = F(1000)) -> tuple[int, ...]:
    """Sharp count set when G allocation is otherwise arbitrary and positive."""
    _check(n, total_g, carbon_energy, threshold, support)
    mu = F(carbon_energy)/total_g
    if mu == 0:
        return (0,)
    if mu == support:
        return (n,)
    return tuple(k for k in range(n+1)
                 if (k != 0 or mu <= threshold) and (k != n or mu > threshold))


def moment_counts(n: int, total_g: F, sum_g_squared: F, carbon_energy: F,
                  threshold: F, support: F = F(1000)) -> tuple[int, ...]:
    """Necessary integer-count outer set using exact G second moment.

    Intersects coarse feasibility, finite-n mean bounds, the candidate count's
    mean interval (open at k*tau/n for k>0), and the classical Cauchy/bounded
    variance quadratic. Minimizing that convex quadratic requires no sqrt.
    It does NOT establish a joint G/C solution or a sharp moment count set.
    """
    allowed = coarse_counts(n, total_g, carbon_energy, threshold, support)
    if not isinstance(sum_g_squared, (int, F)) or sum_g_squared <= 0:
        raise ValueError('Positive exact G second-moment sum required')
    cv2 = F(n)*sum_g_squared/(F(total_g)**2)-1
    if cv2 < 0 or (n == 1 and cv2 != 0) or (n > 1 and cv2 >= n-1):
        raise ValueError('Second moment incompatible with n strictly positive G values')
    mu = F(carbon_energy)/total_g
    a, b = 1+cv2, 2*mu+cv2*support
    vertex = b/(2*a)
    def q(m: F) -> F:
        return a*m*m-b*m+mu*mu
    result = []
    for k in allowed:
        count_lo = F(k)*threshold/n
        lo = max(count_lo, mu/n)
        hi = min(threshold+(support-threshold)*F(k)/n, (mu+(n-1)*support)/n)
        open_lo = k > 0 and lo == count_lo
        if lo > hi or (lo == hi and open_lo):
            continue
        if vertex <= lo:
            possible = q(lo) < 0 if open_lo else q(lo) <= 0
        else:
            possible = q(min(vertex, hi)) <= 0
        if possible:
            result.append(k)
    return tuple(result)


def full_g_counts(generation: Sequence[F], carbon_energy: F, threshold: F,
                  support: F = F(1000)) -> tuple[int, ...]:
    """Sharp strict tail-count set for fixed positive G and one H constraint.

    A nonempty event set S is feasible iff tau*G_S < H <=
    tau*G_total+(B-tau)*G_S. S empty is feasible iff H<=tau*G_total.
    Enumerates at most4096 subsets, never approximates strict inequalities.
    """
    g = tuple(generation)
    if not g or len(g) > 12 or any(not isinstance(x, (int, F)) or x <= 0 for x in g):
        raise ValueError('One to twelve strictly positive exact G values required')
    total = sum(g, F(0))
    _check(len(g), total, carbon_energy, threshold, support)
    result = set()
    for bits in product((0, 1), repeat=len(g)):
        k = sum(bits)
        selected = sum((x for x, flag in zip(g, bits) if flag), F(0))
        if k == 0:
            possible = carbon_energy <= threshold*total
        else:
            possible = (threshold*selected < carbon_energy
                        <= threshold*total+(support-threshold)*selected)
        if possible:
            result.add(k)
    return tuple(sorted(result))


def combine_blocks(sets: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Exact Minkowski sum, retaining holes rather than only min/max."""
    if not sets:
        raise ValueError('At least one block count set required')
    possible = {0}
    for block in sets:
        if not block or any(type(k) is not int or k < 0 for k in block):
            raise ValueError('Nonempty nonnegative integer count sets required')
        possible = {a+b for a in possible for b in block}
    return tuple(sorted(possible))


def describe(counts: Sequence[int], actual: int) -> dict:
    if not counts or actual not in counts:
        raise ValueError('Actual count must belong to every reported identified/outer set')
    return dict(counts='|'.join(map(str, counts)), lower=min(counts), upper=max(counts),
                span=max(counts)-min(counts), cardinality=len(counts), point_identified=len(counts) == 1,
                no_exceedance_certified=max(counts) == 0, any_exceedance_forced=min(counts) > 0,
                actual_count=actual)
