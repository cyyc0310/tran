"""Pure contribution residuals and retrospective source-level persistence diagnostics."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timezone
from typing import Any

import numpy as np


SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
FUELS = ('coal', 'gas', 'oil', 'other')
EF = np.array([1000., 400., 800., 400.])
SPLIT = datetime(2023, 7, 1, tzinfo=timezone.utc)
IDENTITY_ATOL = 1e-9
SUMMARY_FIELDS = ('d', 'k', 'EF_error_true_minus_pred')


def _real(value: Any, name: str, shape: tuple[int, ...] | None = None,
          nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f'Real {name} required')
    try:
        array = np.asarray(raw, float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Real numeric {name} required') from exc
    if ((shape is not None and array.shape != shape) or not np.isfinite(array).all()
            or (nonnegative and np.any(array < 0))):
        raise ValueError(f'Finite correctly shaped {name} required')
    return array


def residuals(G: np.ndarray, C: np.ndarray, U: np.ndarray) -> dict[str, np.ndarray]:
    """G/C_POS contributions remain defined at C_f=G_f=0; no G/C_f is formed."""
    G = _real(G, 'generation', nonnegative=True)
    if G.ndim != 2 or G.shape[1] != 4 or not len(G):
        raise ValueError('Nonempty (N,4) generation required')
    C = _real(C, 'capacities', G.shape, nonnegative=True)
    U = _real(U, 'source profile rates', G.shape, nonnegative=True)
    conflicts = np.argwhere((C == 0) & (G > 0))
    if len(conflicts):
        row, fuel = conflicts[0]
        raise ValueError(f'Zero capacity positive generation: row={row} fuel={FUELS[fuel]}')
    total_capacity = _real(C.sum(1), 'total POS capacity')
    if np.any(total_capacity <= 0):
        raise ValueError('Strictly positive total POS capacity required')
    x = _real(G/total_capacity[:, None], 'true contributions')
    xhat = _real((C/total_capacity[:, None])*U, 'predicted contributions')
    true_mass = _real(x.sum(1), 'true POS contribution mass')
    pred_mass = _real(xhat.sum(1), 'predicted POS contribution mass')
    if np.any(true_mass <= 0) or np.any(pred_mass <= 0):
        raise ValueError('Strictly positive true and predicted POS contribution mass required')
    r = x-xhat
    ef_true = (x/true_mass[:, None])@EF
    ef_pred = (xhat/pred_mass[:, None])@EF
    error = ef_true-ef_pred
    identity = np.sum((EF-ef_pred[:, None])*r, axis=1)/true_mass
    result = dict(x=x, xhat=xhat, r=r, d=r.sum(1), k=600*r[:, 0]+400*r[:, 2],
                  EF_true=ef_true, EF_pred=ef_pred, EF_error_true_minus_pred=error, identity_rhs=identity)
    for name, value in result.items():
        _real(value, name)
    if not np.allclose(error, identity, atol=IDENTITY_ATOL, rtol=0):
        raise ValueError('Carbon contribution identity failed; no numerical fallback')
    return result


def _period(value: Any) -> datetime:
    try:
        if isinstance(value, str):
            value = (datetime.combine(date.fromisoformat(value), time(), timezone.utc) if len(value) == 10
                     else datetime.fromisoformat(value.replace('Z', '+00:00')))
        elif isinstance(value, date) and not isinstance(value, datetime):
            value = datetime.combine(value, time(), timezone.utc)
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('Explicit timezone required for timestamps')
        return value.astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise ValueError('ISO date or explicitly timezone-aware period_start required') from exc


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float | None, str, dict[str, Any]]:
    diagnostics: dict[str, Any] = dict(n=len(x))
    centered, norms = [], []
    for name, vector in (('x', x), ('y', y)):
        constant = bool(np.all(vector == vector[0]))
        scale = float(np.max(np.abs(vector)))
        # Scaling first avoids squaring large values; no epsilon or threshold is added.
        value = vector/scale if scale > 0 else np.zeros_like(vector)
        value = value-value.mean()
        norm = float(np.linalg.norm(value))
        raw_norm = 0. if constant else float(norm*scale)
        if not np.isfinite(raw_norm):
            raise ValueError('Finite centered Pearson L2 scales required')
        diagnostics[f'{name}_constant'] = constant
        diagnostics[f'{name}_centered_l2'] = raw_norm
        centered.append(value); norms.append(norm)
    if diagnostics['x_constant'] or diagnostics['y_constant']:
        return None, 'undefined_constant', diagnostics
    value = float(centered[0]@centered[1]/(norms[0]*norms[1]))
    if not np.isfinite(value):
        raise ValueError('Nonfinite nonconstant Pearson calculation')
    return float(np.clip(value, -1., 1.)), 'defined', diagnostics


def summarize_pairs(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """All 7x6 source pairs, fixed UTC split, unweighted monthly means within source.

    Records require outer/source/key/period_start/d/k/EF_error_true_minus_pred.
    Carrying early fine-label bias is a retrospective diagnostic, not deployment.
    """
    if not isinstance(records, Sequence) or not len(records):
        raise ValueError('Nonempty sequence of monthly residual mappings required')
    groups: dict[tuple[str, str], list[tuple[datetime, str, np.ndarray]]] = {}
    seen = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError('Monthly residual mapping required')
        try:
            outer, source, key = row['outer'], row['source'], row['key']
            at = _period(row['period_start'])
            values = np.array([float(_real(row[field], field, ())) for field in SUMMARY_FIELDS])
        except KeyError as exc:
            raise ValueError(f'Missing residual metadata/field: {exc}') from exc
        if (outer not in SOURCES or source not in SOURCES or source == outer
                or not isinstance(key, str) or not key):
            raise ValueError('Registered distinct outer/source and nonempty key required')
        identity = (outer, source, key)
        if identity in seen:
            raise ValueError(f'Duplicate residual key: {identity}')
        seen.add(identity)
        groups.setdefault((outer, source), []).append((at, key, values))
    expected = {(outer, source) for outer in SOURCES for source in SOURCES if source != outer}
    if set(groups) != expected:
        raise ValueError('Complete seven outers with six held sources each required')
    pairs = []
    for outer in SOURCES:
        for source in SOURCES:
            if source == outer:
                continue
            rows = sorted(groups[(outer, source)], key=lambda row: (row[0], row[1]))
            means, counts = {}, {}
            for label, early in (('early', True), ('late', False)):
                selected = [row[2] for row in rows if (row[0] < SPLIT) == early]
                if not selected:
                    raise ValueError(f'Missing {label} original source months for {outer}/{source}')
                counts[label] = len(selected)
                means[label] = _real(np.mean(selected, axis=0), 'source-period means')
            pair = dict(outer=outer, source=source, early_count=counts['early'], late_count=counts['late'])
            for label in ('early', 'late'):
                for index, name in enumerate(('d', 'k', 'EFbias')):
                    pair[f'{label}_{name}'] = float(means[label][index])
            pairs.append(pair)
    outers = []
    for outer in SOURCES:
        selected = [pair for pair in pairs if pair['outer'] == outer]
        early = np.array([pair['early_EFbias'] for pair in selected])
        late = np.array([pair['late_EFbias'] for pair in selected])
        d = np.array([pair['early_d'] for pair in selected])
        corr, status, fine_diagnostics = _pearson(early, late)
        coarse_corr, coarse_status, coarse_diagnostics = _pearson(d, late)
        zero = (early == 0) | (late == 0)
        record = dict(outer=outer, source_count=6, EFbias_pearson=corr, EFbias_pearson_status=status,
            same_sign_count=int(np.sum((np.sign(early) == np.sign(late)) & ~zero)),
            zero_pairs=int(zero.sum()), carry_EF_MAE=float(np.mean(np.abs(late-early))),
            zero_EF_MAE=float(np.mean(np.abs(late))), early_d_late_EFbias_pearson=coarse_corr,
            early_d_late_EFbias_pearson_status=coarse_status)
        for prefix, details in (('EFbias_pearson', fine_diagnostics),
                                ('early_d_late_EFbias_pearson', coarse_diagnostics)):
            record.update({f'{prefix}_{name}': value for name, value in details.items()})
        outers.append(record)
    return dict(pairs=pairs, outers=outers)
