"""Matched V8 fuel-path diagnostics using a fitted, frozen V7 donor bank.

All routing and expert draws use released V7 forecasts.  The optional future
monthly mix is an explicitly illegal oracle mapping diagnostic: it never enters
donor weights, donor draws, or expert selection.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from transcif.research.dynamic_v2 import EF
from transcif.research.fuel_transport_v7 import recompose_paths


N_SAMPLES = 128
SEED = 17023


def _day_rng(origin, offset=0):
    # Identical to run_fuel_transport_v7.day_rng, including ns/us invariance.
    value = pd.Timestamp(origin).value // (86400 * 10**9)
    return np.random.default_rng(np.random.SeedSequence([SEED, int(value), offset]))


def _simplex_rows(value, n, name):
    rows = np.asarray(value, dtype=float)
    if (rows.shape != (n, len(EF)) or not np.isfinite(rows).all()
            or np.any(rows < 0) or not np.allclose(rows.sum(axis=1), 1., atol=1e-8, rtol=1e-8)):
        raise ValueError(f"{name} must be finite nonnegative eight-fuel simplex rows for every day")
    return rows


def _prediction(raw):
    clipped = np.clip(raw, 0., 1000.)
    return {"point": clipped.mean(axis=1), "ensemble": clipped}, float(np.mean(raw != clipped))


def _draw_hash(experts, donors):
    digest = hashlib.sha256()
    for value in (experts, donors):
        array = np.ascontiguousarray(value, dtype="<i8")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _oracle_recompose(bank, mix, donor_indices):
    n, samples = donor_indices.shape
    paths = recompose_paths(mix.reshape(n * samples, len(EF)),
                            bank.q_[donor_indices].reshape(n * samples, 24, len(EF)))
    return paths.reshape(n, samples, 24)


def generate_matched(bank, coarse, days, config, oracle_mix=None):
    """Generate common-donor direct, additive, and ratio-recomposed paths.

    ``config`` uses V7's ``forecast``, ``bandwidth``, and ``temperature`` keys.
    For ``adaptive_mixture``, each expert has its own V7 lawful donor kernel;
    the offset-997 expert draw chooses a whole path from that component.  The
    returned audit retains component weights as (day, expert, donor), final
    expert choices and final donor indices as (day, scenario).

    The additive path is raw donor CIF + EF dot (mapping mix - donor monthly
    mix), before final [0, 1000] clipping.  It anchors level while preserving
    every donor hourly deviation exactly before clipping.
    """
    origins = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
    n = len(origins)
    if not n or origins.hasnans:
        raise ValueError("days must contain nonmissing UTC origins")
    forecast = config["forecast"]
    bandwidth = float(config["bandwidth"])
    if forecast not in ("latest", "adaptive_mean", "adaptive_mixture"):
        raise ValueError(f"Unsupported V7 forecast {forecast!r}")
    experts = np.asarray(coarse["experts"], dtype=float)
    if experts.shape != (n, 3, len(EF)):
        raise ValueError("coarse experts must have shape (days, 3, 8)")
    for expert in range(3):
        _simplex_rows(experts[:, expert], n, "coarse experts")

    if forecast == "latest":
        component_mixes = experts[:, :1]
        probabilities = None
    else:
        probabilities = np.asarray(coarse["weights"][str(config["temperature"])], dtype=float)
        if (probabilities.shape != (n, 3) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-8, rtol=1e-8)):
            raise ValueError("V7 expert probabilities must be simplex rows")
        if forecast == "adaptive_mean":
            component_mixes = np.einsum("ne,nef->nf", probabilities, experts)[:, None, :]
        else:
            component_mixes = experts

    options = dict(mode="recompose_joint", bandwidth=bandwidth,
                   n_scenarios=N_SAMPLES, seed=SEED)
    components = [bank.predict(component_mixes[:, e], origins, **options)
                  for e in range(component_mixes.shape[1])]
    if forecast == "adaptive_mixture":
        expert_choices = np.stack([_day_rng(day, 997).choice(3, N_SAMPLES, p=probabilities[i])
                                   for i, day in enumerate(origins)])
    else:
        expert_choices = np.zeros((n, N_SAMPLES), dtype=int)
    rows = np.arange(n)[:, None]
    particles = np.arange(N_SAMPLES)[None, :]
    donor_components = np.stack([x["sampled_donor_indices"][:, :, 0] for x in components], axis=1)
    donor_indices = donor_components[rows, expert_choices, particles]
    legal_mix = component_mixes[rows, expert_choices]

    direct_raw = bank.raw_cif_[donor_indices]
    donor_mix = bank.donor_mix_[donor_indices]
    shifts = np.einsum("nsf,f->ns", legal_mix - donor_mix, EF)
    additive_raw = direct_raw + shifts[:, :, None]
    ratio_components = np.stack([x["ensemble"] for x in components], axis=1)
    ratio_raw = ratio_components[rows, expert_choices, particles]

    result = {}
    clip_fractions = {}
    for name, raw in (("direct_analog", direct_raw),
                      ("additive_anchor", additive_raw),
                      ("recompose_joint", ratio_raw)):
        result[name], clip_fractions[name] = _prediction(raw)
    shape_error = float(np.max(np.abs((additive_raw - direct_raw) - shifts[:, :, None])))

    if oracle_mix is not None:
        oracle_rows = _simplex_rows(oracle_mix, n, "oracle_mix")
        mapped_oracle = np.broadcast_to(oracle_rows[:, None, :], legal_mix.shape)
        oracle_shifts = np.einsum("nsf,f->ns", mapped_oracle - donor_mix, EF)
        oracle_additive = direct_raw + oracle_shifts[:, :, None]
        oracle_ratio = _oracle_recompose(bank, mapped_oracle, donor_indices)
        for name, raw in (("oracle_additive_anchor", oracle_additive),
                          ("oracle_recompose_joint", oracle_ratio)):
            result[name], clip_fractions[name] = _prediction(raw)

    result["audit"] = {
        "expert_choices": expert_choices,
        "sampled_donor_indices": donor_indices,
        "donor_weights": np.stack([x["donor_weights"] for x in components], axis=1),
        "draw_sha256": _draw_hash(expert_choices, donor_indices),
        "preclip_additive_shape_max_abs_error": shape_error,
        "clip_fraction": clip_fractions,
        "oracle_mix_used": oracle_mix is not None,
    }
    return result
