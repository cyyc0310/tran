"""Exact synthetic identification counterexamples; never reads regional inputs.

This is not a forecast, information-value experiment, new theorem or proof of
any universally minimal observation budget. The only output is a new JSON file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
HOURS, BLOCK_HOURS, THRESHOLD = 24, 6, 600.
OBSERVED_HOURS = (0, 6, 12, 18)
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/risk-information-v19/witnesses.json'


def path(coal: np.ndarray) -> np.ndarray:
    """One synthetic day with strictly positive total G=1 at every hour."""
    x = np.asarray(coal, float)
    if x.shape != (24,) or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
        raise ValueError('Exactly24 finite coal amounts in[0,1] required')
    result = np.zeros((24, 8))
    result[:, 0], result[:, 4] = x, 1-x
    return result


def validate_path(generation: np.ndarray) -> np.ndarray:
    a = np.asarray(generation, float)
    if (a.shape != (24, 8) or not np.isfinite(a).all() or np.any(a < 0)
            or np.any(a[:, [1, 2, 3, 5, 6, 7]] != 0) or not np.array_equal(a.sum(1), np.ones(24))):
        raise ValueError('Synthetic coal/hydro-only hourly generation with G=1 required')
    return a


def observations(generation: np.ndarray) -> dict[str, Any]:
    a = validate_path(generation)
    blocks = a.reshape(4, 6, 8).sum(1)
    h, g = blocks@(EF/1000), blocks.sum(1)
    return dict(D8=a.sum(0).tolist(), T8_H_G=np.column_stack((h, g)).tolist(),
                G24=a.sum(1).tolist(), derived_D2_H_G=[float(h.sum()), float(g.sum())])


def risks(generation: np.ndarray) -> dict[str, Any]:
    a = validate_path(generation)
    cif = a@EF  # G=1, so hourly coal-energy fraction times1000.
    exceed = np.flatnonzero(cif > THRESHOLD)
    ramps = np.abs(np.diff(cif))  # h=0..22 only; no last-to-first wrap.
    peak_hours = np.flatnonzero(cif == cif.max())
    return dict(hourly_cif=cif.tolist(), exceedance_threshold=THRESHOLD,
        exceedance_hours=int(len(exceed)), exceedance_fraction=float(len(exceed)/24),
        exceedance_hour_indices=exceed.tolist(), peak_cif=float(cif.max()),
        peak_hour_indices=peak_hours.tolist(), first_peak_hour=int(peak_hours[0]),
        max_adjacent_absolute_ramp=float(ramps.max()),
        max_ramp_transition_start_hours=np.flatnonzero(ramps == ramps.max()).tolist(),
        nonzero_adjacent_ramp_count=int(np.count_nonzero(ramps)),
        cyclic_boundary_absolute_ramp=float(abs(cif[-1]-cif[0])))


def observation_map(*, labels: bool = False, time_moment: bool = False) -> dict[str, Any]:
    """Full affine coal-to-(D8,T8,G24,optional labels/moment) observation map.

    Hydro=1-coal and all other fuels=0; G24 contributes constant rows. D8 adds
    redundant daily totals to the four independent T8 block coal-energy rows.
    The optional moment is ONE row, supported only on the first UTC6h block.
    """
    rows, offsets, names = [], [], []
    for fuel in FUELS:
        rows.append(np.ones(24) if fuel == 'coal' else -np.ones(24) if fuel == 'hydro' else np.zeros(24))
        offsets.append(24. if fuel == 'hydro' else 0.); names.append('D8_'+fuel)
    for block in range(4):
        row = np.zeros(24);row[6*block:6*(block+1)] = 1
        rows.extend((row, np.zeros(24)));offsets.extend((0., 6.))
        names.extend((f'T8_H_block{block}', f'T8_G_block{block}'))
    for hour in range(24):
        rows.append(np.zeros(24));offsets.append(1.);names.append(f'G24_hour{hour}')
    if labels:
        for hour in OBSERVED_HOURS:
            row = np.zeros(24);row[hour] = 1000
            rows.append(row);offsets.append(0.);names.append(f'local_CIF_label_hour{hour}')
    if time_moment:
        row = np.zeros(24);row[:6] = np.arange(6)
        rows.append(row);offsets.append(0.);names.append('first_block_coal_time_moment')
    matrix = np.stack(rows)
    rank = int(np.linalg.matrix_rank(matrix))
    return dict(matrix=matrix.tolist(), offset=offsets, row_names=names,
                rank=rank, nullity=24-rank, variable_count=24,
                optional_moment_definition='sum(h*coal[h] for h=0..5); one independent first-block row')


def apply_map(coal: np.ndarray, specification: dict) -> np.ndarray:
    return np.asarray(specification['matrix'])@coal+np.asarray(specification['offset'])


def pair(left: np.ndarray, right: np.ndarray, specification: dict) -> dict[str, Any]:
    """Return raw values, observable identity and a checked nonzero null direction."""
    a, b = validate_path(left), validate_path(right)
    oa, ob = observations(a), observations(b)
    if oa != ob:
        raise ValueError('Witness pair must have exactly equal D8/T8/G24 observations')
    x, y = apply_map(a[:, 0], specification), apply_map(b[:, 0], specification)
    delta = b[:, 0]-a[:, 0]
    residual = np.asarray(specification['matrix'])@delta
    if (not np.array_equal(x, y) or not np.any(delta) or not np.array_equal(residual, np.zeros(len(residual)))):
        raise ValueError('Distinct feasible paths must preserve the complete registered observation map')
    return dict(left=dict(generation=left.tolist(), observations=oa, risks=risks(a)),
        right=dict(generation=right.tolist(), observations=ob, risks=risks(b)),
        map_observations=x.tolist(), coal_null_direction=delta.tolist(),
        max_abs_map_residual=float(abs(residual).max()), observations_exactly_equal=True)


def sum_preserving_perturbation(coal: np.ndarray, observed_hours: tuple[int, ...],
                                first: int, second: int, epsilon: float) -> np.ndarray:
    """Optional local witness: two unobserved hours within one block, +/-epsilon.

    This only promises preservation of block sums and selected hourly labels,
    not preservation of an additional time moment or change of any given risk.
    """
    x = path(coal)[:, 0].copy()
    if (not isinstance(first, int) or not isinstance(second, int) or first == second
            or not 0 <= first < 24 or not 0 <= second < 24 or first//6 != second//6
            or first in observed_hours or second in observed_hours
            or any(not isinstance(h, int) or not 0 <= h < 24 for h in observed_hours)
            or not np.isfinite(epsilon) or epsilon == 0):
        raise ValueError('Two different unobserved hours in one block and finite nonzero epsilon required')
    x[first] += epsilon;x[second] -= epsilon
    path(x)  # Feasible nonnegative coal/hydro support, without clipping the witness.
    return x


def capacity_exclusion(coal_cap: float, threshold: float = THRESHOLD) -> dict[str, Any]:
    """In THIS fixed-G two-fuel model, certified coal cap bounds hourly CIF."""
    if not np.isfinite(coal_cap) or not 0 <= coal_cap <= 1 or not np.isfinite(threshold):
        raise ValueError('Finite coal cap in[0,1] and finite threshold required')
    maximum = 1000*coal_cap
    return dict(coal_energy_fraction_cap=coal_cap, hourly_cif_upper_bound=maximum,
                strict_exceedance_threshold=threshold, strict_exceedance_excluded=bool(maximum <= threshold),
                assumptions=['G=1 in every hour', 'only coal and zero-EF hydro',
                             'cap certified, same denominator/time scope, and valid for all relevant future hours'],
                real_capacity_certified=False, sufficient_for_future_prediction=False)


def _quadratic_mean_bounds(mu_energy: float, cv_generation: float, support: float) -> tuple[float, float]:
    """Stable roots of the normalized Cauchy/bounded-variance quadratic."""
    mu, cv, bound = float(mu_energy), float(cv_generation), float(support)
    if (not np.isfinite([mu, cv, bound]).all() or bound <= 0 or not 0 <= mu <= bound or cv < 0):
        raise ValueError('Finite positive support, mean in[0,support] and nonnegative generation CV required')
    if cv == 0:
        return mu, mu
    t = mu/bound
    # Divide by1+cv² before solving, avoiding overflow for arbitrarily large finite CV.
    if cv <= 1:
        cv2 = cv*cv; inverse = 1/(1+cv2); beta = cv2/(1+cv2)
    else:
        inverse_cv2 = (1/cv)**2
        inverse = inverse_cv2/(1+inverse_cv2); beta = 1/(1+inverse_cv2)
    coefficient = 2*t*inverse+beta
    # Algebraically equivalent discriminant, without subtracting near-equal squares.
    discriminant_root = np.sqrt(beta)*np.sqrt(beta+4*inverse*t*(1-t))
    upper = (coefficient+discriminant_root)/2
    lower = (t*t*inverse)/upper if upper else 0.
    # The exact interval always contains mu; preserve that fact at vanishing CV.
    return max(0., min(mu, lower*bound)), min(bound, max(mu, upper*bound))


def mean_outer_bounds(mu_energy: float, cv_generation: float, support: float = 1000.) -> tuple[float, float]:
    """Outer bounds on equal-hour mean given energy mean and contemporaneous G CV.

    Assume strictly positive finite hourly G and0<=C<=support. Cauchy and the
    bounded-variance inequality imply (mu_energy-m)²<=cv²*m*(support-m).
    This classical necessary-condition bound is not claimed sharp. Strict G>0
    additionally forces m=0 or support when mu_energy is that endpoint, so those
    cases are tightened beyond the generic quadratic roots. No future G/CV
    availability is assumed, and this bounds a mean, not tails or temporal risk.
    Floating evaluation adds128*machine_epsilon*support to both sides; this is
    a conservative diagnostic margin, not directed-arithmetic certification of
    the exact roots. Normalization/squaring underflow falls back to[0,support].
    """
    lower, upper = _quadratic_mean_bounds(mu_energy, cv_generation, support)
    if mu_energy == 0 or mu_energy == support:
        return float(mu_energy), float(mu_energy)
    if cv_generation == 0:
        return lower, upper
    t = float(mu_energy)/float(support)
    if t == 0 or t*t == 0 or (cv_generation <= 1 and cv_generation*cv_generation == 0):
        return 0., float(support)
    margin = float(support)*(128*np.finfo(float).eps)
    lo = max(0., lower-margin)
    hi = float(support) if upper >= float(support)-margin else upper+margin
    return lo, hi


def synthetic_mean_checks() -> dict[str, Any]:
    """Fixed-seed1000 positive-G samples, including highly unequal generation."""
    rng = np.random.default_rng(19023)
    maximum_identity_error = 0.; maximum_cv = 0.
    for i in range(1000):
        sigma = (0., .01, .1, 1., 3., 6., 12.)[i % 7]
        generation = np.exp(rng.normal(0, sigma, 24))
        carbon = rng.uniform(0, 1000, 24)
        relative = generation/generation.mean()
        mean = float(carbon.mean()); energy = float(np.mean(relative*carbon))
        cv = float(np.std(relative))
        identity = float(np.mean((relative-1)*(carbon-mean)))
        maximum_identity_error = max(maximum_identity_error, abs((energy-mean)-identity))
        maximum_cv = max(maximum_cv, cv)
        low, high = mean_outer_bounds(energy, cv)
        radius = min(1000., cv*500.)
        simple_low, simple_high = max(0., energy-radius), min(1000., energy+radius)
        if not low-1e-9 <= mean <= high+1e-9 or not simple_low-1e-9 <= mean <= simple_high+1e-9:
            raise ValueError('Synthetic equal-hour mean violated a classical necessary outer bound')
    examples = []
    for energy, cv in ((500., 0.), (500., .25), (500., 1.), (950., .9), (0., 1.), (1000., 1.)):
        examples.append(dict(mu_energy=energy, cv_generation=cv,
            approximate_quadratic_roots=list(_quadratic_mean_bounds(energy, cv, 1000.)),
            guarded_outer_bounds_with_positive_G_endpoints=list(mean_outer_bounds(energy, cv)),
            simple_B_over_two_cv_bounds=[max(0., energy-cv*500), min(1000., energy+cv*500)]))
    return dict(seed=19023, trials=1000, hours_per_sample=24, all_generation_strictly_positive=True,
        lognormal_sigma_grid=[0., .01, .1, 1., 3., 6., 12.], largest_generation_cv=maximum_cv,
        largest_covariance_identity_abs_error=maximum_identity_error, all_outer_bound_checks_passed=True,
        examples=examples, identity='mu_energy-mu_uniform=Cov_uniform(G,C)/mean_uniform(G)',
        quadratic_necessary_condition='(mu_energy-m)^2<=CV_G^2*m*(B-m)',
        simple_outer_bound='abs(mu_energy-mu_uniform)<=CV_G*B/2',
        numerical_margin='128*float64_epsilon*support on each side, not exact interval-arithmetic root certification; '
                         'underflow in normalized mean or squared CV uses[0,support] fallback',
        limit='Classical necessary outer bounds, not sharp or original; contemporaneous positive G/CV '
              'is assumed for this diagnostic, not made available to a future forecaster. '
              'Uniform G identifies the equal-hour mean but still fails to identify tails, peak or ramp.')


def build_witnesses() -> dict[str, Any]:
    """All inputs are hardcoded synthetic constants; this function performs no I/O."""
    flat = path(np.full(24, .5))
    grouped = path(np.tile([1., 1., 1., 0., 0., 0.], 4))
    reversed_blocks = grouped.reshape(4, 6, 8)[:, ::-1].reshape(24, 8).copy()
    alternating = grouped.reshape(4, 6, 8)[:, [0, 3, 1, 4, 2, 5]].reshape(24, 8).copy()
    sparse = path(np.tile([.5, 1., 1., .5, 0., 0.], 4))
    moment_coal = np.full(24, .5);moment_coal[[1, 2, 3]] += [.25, -.5, .25]
    moment = path(moment_coal)
    base = observation_map()
    with_labels = observation_map(labels=True)
    with_moment = observation_map(labels=True, time_moment=True)
    if [(m['rank'], m['nullity']) for m in (base, with_labels, with_moment)] != [(4, 20), (8, 16), (9, 15)]:
        raise ValueError('Unexpected synthetic identification ranks')
    return dict(version=19, kind='exact_synthetic_contemporaneous_oracle_identification',
        inputs=dict(hours=list(range(24)), units='G=1 energy unit per one-hour UTC cell',
                    fuels=list(FUELS), emission_factors=EF.tolist(), observed_label_hours=list(OBSERVED_HOURS)),
        conventions=dict(exceedance='CIF strictly greater than600; equality is not exceedance',
            peak='all maximizing hour indices retained; first_peak_hour is smallest tied index',
            ramp='max abs(CIF[h+1]-CIF[h]) for h=0..22, including6h-block boundaries; no day wrap',
            cyclic_boundary='abs(CIF[23]-CIF[0]) reported separately; excluded from main ramp'),
        maps=dict(D8_T8_G24=base, plus_four_local_labels=with_labels,
                  plus_four_labels_and_one_time_moment=with_moment),
        witnesses=dict(coarse_tail_peak_ramp=pair(flat, grouped, base),
                       coarse_tail_equal_but_time_location_reversed=pair(grouped, reversed_blocks, base),
                       coarse_tail_equal_but_time_location_permuted=pair(grouped, alternating, base),
                       sparse_four_labels_still_not_tail=pair(flat, sparse, with_labels),
                       additional_time_moment_still_not_tail=pair(flat, moment, with_moment)),
        support_cap_example=capacity_exclusion(.6),
        mean_information_bounds=synthetic_mean_checks(),
        limits=dict(all_inputs_synthetic=True, regional_values_read=False, old_scores_read=False,
                    noiseless_exact_contemporaneous_oracle=True, future_forecast=False,
                    new_theorem=False, real_information_value_measured=False,
                    universally_minimal_budget_proved=False, target_or_unseen_opened=False,
                    statement='Counterexamples show insufficiency of these particular observations in this feasible family, '
                              'not that local information is useless or a particular extra statistic is universally minimal.'))


def write_witnesses() -> None:
    """Explicit post-approval entry point; refuses to overwrite any existing output."""
    document = build_witnesses()
    script = Path(__file__).resolve()
    test = ROOT/'tests/test_risk_information_witness_v19.py'
    document['code_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (script, test)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', required=True,
                        help='Write the synthetic artifact only after independent approval')
    parser.parse_args()
    write_witnesses()
