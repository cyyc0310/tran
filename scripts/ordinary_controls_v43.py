"""Pure fixed blends and expanded empirical-range controls; no fitting or scores."""
from __future__ import annotations

from typing import Any

import numpy as np

from scripts import capacity_routing_v36 as v36


PARENT_ARMS = tuple(f'{scope}-{family}-{model}' for scope in ('Legacy', 'Expanded')
                    for family in ('Size', 'SizeCap', 'DirectEF') for model in ('Ridge', 'ET'))
PAIRS = (
    ('Pool-SizeCapET', 'Legacy-SizeCap-ET', 'Expanded-SizeCap-ET'),
    ('Pool-DirectEFET', 'Legacy-DirectEF-ET', 'Expanded-DirectEF-ET'),
    ('Capacity-SizeCapET', 'Expanded-Size-ET', 'Expanded-SizeCap-ET'),
    ('Capacity-DirectEFET', 'Expanded-Size-ET', 'Expanded-DirectEF-ET'),
    ('StrongB0-SizeCapET', 'Legacy-Size-Ridge', 'Expanded-SizeCap-ET'),
    ('StrongB0-DirectEFET', 'Legacy-Size-Ridge', 'Expanded-DirectEF-ET'),
    ('Anchor-SizeCapET', 'Capacity-OP', 'Expanded-SizeCap-ET'),
    ('Anchor-DirectEFET', 'Capacity-OP', 'Expanded-DirectEF-ET'),
    ('Router-SizeCapET', 'CoalOilUpper', 'Expanded-SizeCap-ET'),
    ('Router-DirectEFET', 'CoalOilUpper', 'Expanded-DirectEF-ET'))
WEIGHTS = (.25, .5, .75)
ARMS = tuple(f'Mix-{name}-w{w}' for name, _, _ in PAIRS for w in WEIGHTS)+tuple(
    'ExpandedRange-'+recipe for recipe in v36.RECIPES[5:])
ATOL = 1e-9


def _real(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f'Real {name} required')
    try:
        array = np.asarray(raw, float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Real numeric {name} required') from exc
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f'Finite correctly shaped {name} required')
    return array


def _ef(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = _real(value, name, shape)
    if np.any((array < 400.-ATOL) | (array > 1000.+ATOL)):
        raise ValueError(f'{name} outside fixed EF bounds')
    return array


def _limits(q: np.ndarray) -> dict[str, Any]:
    return dict(training_rows=len(q), qmin=q.min(0), qmax=q.max(0),
                capEFmax=float((400+600*q[:, 0]+400*q[:, 2]).max()))


def build(ef12: np.ndarray, cif12: np.ndarray, mg: np.ndarray, anchorEF: np.ndarray,
          old_q: np.ndarray, new_q: np.ndarray, query_q: np.ndarray) -> dict[str, Any]:
    """Mix EF first; binary empirical-range arms preserve archived endpoints exactly.

    Caller binds source masks, feature names and dates. old_q must be a projection
    of new_q by that metadata; no floating row matching is used to infer provenance.
    """
    ef12 = _ef(ef12, 'parent EF', (12, 6, 128))
    cif12 = _real(cif12, 'parent CIF', (12, 6, 128))
    mg = _real(mg, 'MG particles', (6, 128, 5))
    if np.any(mg < 0) or not np.allclose(mg.sum(-1), 1., atol=v36.TOLERANCE, rtol=0):
        raise ValueError('Nonnegative simplex MG particles required')
    if not np.allclose(cif12, ef12*mg[None, ..., 0], atol=ATOL, rtol=0):
        raise ValueError('Parent EF/CIF must align with identical MG particles')
    anchor = _ef(anchorEF, 'Capacity-OP EF', (6, 128))
    if not np.array_equal(anchor, np.broadcast_to(anchor[:, :1], anchor.shape)):
        raise ValueError('Capacity anchor must be particle-constant within each query')
    old_q = _real(old_q, 'old training q'); new_q = _real(new_q, 'expanded training q')
    query_q = _real(query_q, 'query capacity shares', (6, 4))
    old_weights, expanded_weights = v36.weights(old_q, query_q), v36.weights(new_q, query_q)
    endpoints = PARENT_ARMS+('Capacity-OP', 'CoalOilUpper')
    endpoint_ef = {name: ef12[i].copy() for i, name in enumerate(PARENT_ARMS)}
    endpoint_cif = {name: cif12[i].copy() for i, name in enumerate(PARENT_ARMS)}
    endpoint_ef['Capacity-OP'] = anchor.copy()
    endpoint_cif['Capacity-OP'] = anchor*mg[..., 0]
    base = 'Legacy-Size-Ridge'
    endpoint_ef['CoalOilUpper'] = np.where(old_weights[8, :, None] == 1, anchor, endpoint_ef[base])
    endpoint_cif['CoalOilUpper'] = np.where(old_weights[8, :, None] == 1,
                                          endpoint_cif['Capacity-OP'], endpoint_cif[base])
    ef, cif, weights, recipes = [], [], [], []
    for name, left, right in PAIRS:
        for w in WEIGHTS:
            blended = (1-w)*endpoint_ef[left]+w*endpoint_ef[right]
            ef.append(blended); cif.append(blended*mg[..., 0]); weights.append(np.full(6, w))
            recipes.append(dict(arm=f'Mix-{name}-w{w}', left=left, right=right,
                                recipe='fixed_global_mix', right_weight=w))
    for index, recipe in enumerate(v36.RECIPES[5:], 5):
        w = expanded_weights[index]
        ef.append(np.where(w[:, None] == 1, anchor, endpoint_ef[base]))
        cif.append(np.where(w[:, None] == 1, endpoint_cif['Capacity-OP'], endpoint_cif[base]))
        weights.append(w.copy())
        recipes.append(dict(arm='ExpandedRange-'+recipe, left=base, right='Capacity-OP',
                            recipe=recipe, weight_index=index))
    return dict(EF_draws=np.stack(ef), CIF_draws=np.stack(cif), weights=np.stack(weights),
        endpoints=endpoints, endpoint_EF=endpoint_ef, endpoint_CIF=endpoint_cif,
        audit=dict(arms=ARMS, recipes=recipes, old_weights=old_weights, expanded_weights=expanded_weights,
                   old_limits=_limits(old_q), new_limits=_limits(new_q),
                   empirical_ranges_are_statistical_support=False, new_fit_count=0))
