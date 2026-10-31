"""Cache complete-month fuel-recomposition risk responses with exact tie handling."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import v139_physical_reserve_calibration as v139
from v144_prepare_profile_cache import FACTORS

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V144_CANONICAL_PROFILE_RESPONSE_PROTOCOL.md"
INPUT = ROOT / "results/v144-canonical-profile-cache-r1"
OUT = ROOT / "results/v144-composition-response-cache-r1"
TAUS = (600, 900)


def exact_coefficients(query: list[F], donor: list[F], coefficients: np.ndarray, hours: int):
    terms = [F(int(value)) * q / d if d > 0 else F(0)
             for q, d, value in zip(query, donor, coefficients)]
    constant = sum((F(int(value)) * q / hours for q, d, value in zip(query, donor, coefficients) if d == 0), F(0))
    return terms, constant


def rational_sign(row: np.ndarray, terms: list[F], constant: F) -> bool:
    return sum((F(str(float(value))) * term for value, term in zip(row, terms)), constant) > 0


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_hash = v139.sha(PROTOCOL)
    input_summary = json.loads((INPUT / "summary.json").read_text())
    if input_summary["input_hashes"][str(PROTOCOL.relative_to(ROOT))] != protocol_hash:
        raise ValueError("profile cache protocol mismatch")
    if v139.sha(INPUT / "profiles.npz") != input_summary["profiles_sha256"]:
        raise ValueError("raw profile cache changed")
    index = pd.read_csv(INPUT / "profile_index.csv")
    panel = pd.read_csv(INPUT / "canonical_monthly_panel.csv")
    data = np.load(INPUT / "profiles.npz", allow_pickle=False)
    generation, margin_strings = data["generation"], data["margin_fractions"]
    exact_margins = [[F(str(value)) for value in row] for row in margin_strings]
    margins = np.asarray([[float(value) for value in row] for row in exact_margins])
    n = len(index)
    if not index.profile_id.eq(np.arange(n)).all() or n != 936:
        raise ValueError("profile index not canonical")
    low, high = np.zeros((2, n, n), dtype=np.uint16), np.zeros((2, n, n), dtype=np.uint16)
    near_decisions, corrected_decisions = 0, 0
    # A small BLAS thread count avoids oversubscription on these thin matrices.
    with threadpool_limits(limits=2, user_api="blas"):
        for donor in index.itertuples(index=False):
            j = donor.profile_id
            raw = generation[donor.offset:donor.stop]
            positive = margins[j] > 0
            for t, tau in enumerate(TAUS):
                for factor, result in (("low", low), ("high", high)):
                    coefficients = np.asarray(FACTORS[factor], dtype=int) - tau
                    active = np.flatnonzero(((margins > 0) & (coefficients > 0)).any(axis=1))
                    if not len(active):
                        continue
                    scaled = np.divide(margins[active], margins[j], out=np.zeros((len(active), 8)), where=positive)
                    scaled *= coefficients
                    constant = margins[active][:, ~positive] @ coefficients[~positive] / donor.hours
                    constant_abs = margins[active][:, ~positive] @ np.abs(coefficients[~positive]) / donor.hours
                    excess = raw @ scaled.T + constant
                    absolute = raw @ np.abs(scaled).T + constant_abs
                    tolerance = 64 * np.finfo(float).eps * absolute
                    events = excess > 0
                    near = (np.abs(excess) <= tolerance) & (absolute > 0)
                    for column in np.flatnonzero(near.any(axis=0)):
                        query_id = int(active[column])
                        terms, exact_constant = exact_coefficients(exact_margins[query_id], exact_margins[j], coefficients, donor.hours)
                        for h in np.flatnonzero(near[:, column]):
                            exact_event = rational_sign(raw[h], terms, exact_constant)
                            corrected_decisions += bool(events[h, column]) != exact_event
                            events[h, column] = exact_event
                            near_decisions += 1
                    result[t, active, j] = events.sum(axis=0)
            if (j + 1) % 48 == 0:
                print(f"donor responses {j + 1}/{n}; exact boundary decisions {near_decisions}", flush=True)
    identity_failures = []
    for row in panel.itertuples(index=False):
        t = TAUS.index(row.tau)
        j = row.profile_id
        if int(low[t, j, j]) != row.factor_box_K_lower or int(high[t, j, j]) != row.factor_box_K_upper:
            identity_failures.append({"profile_id": j, "tau": row.tau})
    if identity_failures or np.any(low > high) or np.any(high > index.hours.to_numpy()[None, None, :]):
        raise ValueError(f"response count/own-composition identity failure: {identity_failures[:5]}")
    checks = []
    for i in range(24):
        query_id, donor_id = (i * 41) % n, (i * 83 + 17) % n
        if index.region.iloc[query_id] == index.region.iloc[donor_id]:
            donor_id = (donor_id + 47) % n
        donor = index.iloc[donor_id]
        raw = generation[int(donor.offset):int(donor.stop)]
        for t, tau in enumerate(TAUS):
            for factor, result in (("low", low), ("high", high)):
                coefficients = np.asarray(FACTORS[factor], dtype=int) - tau
                terms, constant = exact_coefficients(exact_margins[query_id], exact_margins[donor_id], coefficients, int(donor.hours))
                direct = sum(rational_sign(row, terms, constant) for row in raw)
                actual = int(result[t, query_id, donor_id])
                checks.append({"query_id": query_id, "donor_id": donor_id, "tau": tau, "factor": factor,
                               "cached_count": actual, "rational_count": direct, "passed": actual == direct})
    if not all(check["passed"] for check in checks) or protocol_hash != v139.sha(PROTOCOL):
        raise ValueError("cross-region exact response audit/protocol failure")
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "response_counts.npz", low=low, high=high, taus=np.asarray(TAUS))
    index.to_csv(OUT / "profile_index.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "exact_cross_region_audit.csv", index=False)
    payload = {"status": "OPENED_COHORT_SOURCE_PROFILE_COMPOSITION_RESPONSE_CACHE", "queries": n, "donor_profiles": n,
               "response_count_cells": int(low.size + high.size), "own_composition_identity_checks": len(panel) * 2,
               "cross_region_checks": len(checks), "assertion_failures": 0,
               "near_boundary_exact_decisions": near_decisions, "floating_signs_corrected": corrected_decisions,
               "normalization": "uniform complete donor-month observed-support law; absent fuel is flat",
               "input_hashes": {str(p.relative_to(ROOT)): v139.sha(p) for p in
                                  (INPUT / "summary.json", INPUT / "profiles.npz", INPUT / "profile_index.csv", INPUT / "canonical_monthly_panel.csv", PROTOCOL)},
               "script_sha256": v139.sha(Path(__file__)), "response_sha256": v139.sha(OUT / "response_counts.npz"),
               "limitations": ["Donor-profile invariance under composition change is an assumption, not causal identification.",
                               "Later models must exclude target/calibration identities before accessing response donors.",
                               "These hypothetical source responses do not add independent calibration regions or target labels."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"identity checks: {len(panel) * 2}; cross-region checks: {len(checks)}; failures: 0")


if __name__ == "__main__":
    main()
