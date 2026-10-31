"""Algebraic seasonal-level / analog-shape factorial composition."""
import hashlib
import json
from pathlib import Path
import numpy as np

ALPHAS = (0., .25, .5, 1.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compose(predictions, residuals, alpha):
    """Residual identity follows E=Y-P; no target observations are needed."""
    if alpha not in ALPHAS:
        raise ValueError('Alpha must belong to frozen grid')
    delta = predictions['analog_shape'] - predictions['latest']
    if not np.allclose(delta.mean(axis=-1), 0, atol=1e-9):
        raise ValueError('Shape displacement must have zero daily mean')
    return (predictions['seasonal'] + alpha * delta,
            residuals['seasonal'] + alpha * (residuals['analog_shape'] - residuals['latest']))


def select_source_alpha(residuals):
    """Source residual MAE only; same-bank reuse is exploratory optimism."""
    risks = {a: float(np.abs(residuals['seasonal'] + a * (
        residuals['analog_shape'] - residuals['latest'])).mean()) for a in ALPHAS}
    return min(ALPHAS, key=risks.get), risks


def load_component(folder, protocol):
    """Verify archived immutable fold before composing any arrays."""
    folder = Path(folder)
    seal = json.loads((folder/'prediction_seal.json').read_text())
    if seal.get('target_labels_read_before_seal') is not False:
        raise ValueError('Missing pre-label component seal')
    if sha(folder/'predictions.npz') != seal['prediction_sha256']:
        raise ValueError('Component hash mismatch')
    indices = seal.get('calibration_indices')
    if not indices or not seal.get('policy', {}).get('folds'):
        raise ValueError('Missing source-index alignment evidence')
    with np.load(folder/'predictions.npz', allow_pickle=False) as archive:
        values = {k: archive[k].copy() for k in archive.files}
    needed = ('seasonal','latest','analog_shape','analog_shape_half','source_policy')
    predictions = {k: values[k] for k in needed}
    residuals = {k: values['residual_'+k] for k in needed}
    if any(v.shape != residuals['latest'].shape for v in residuals.values()):
        raise ValueError('Unaligned source residual shapes')
    if len(indices) != len(residuals['latest']):
        raise ValueError('Unaligned source indices')
    if any(v.ndim != 2 or v.shape[1] != 24 or not np.isfinite(v).all() for v in (*predictions.values(), *residuals.values())):
        raise ValueError('Invalid component arrays')
    if any(v.shape != predictions['latest'].shape for v in predictions.values()) or len(values['origins']) != len(predictions['latest']):
        raise ValueError('Unaligned prediction origins')
    folds = seal['policy']['folds']
    if sum(f['days'] for f in folds) <= max(indices) or min(indices) < 0:
        raise ValueError('Source indices outside source-fold support')
    if any(f['held_source'] in f['fit_sources'] for f in folds):
        raise ValueError('Source inner fold leakage')
    # A single shared indices vector applies to every method in this archive.
    explicit = seal.get('calibration_indices_by_method', {})
    if any(list(v) != indices for v in explicit.values()):
        raise ValueError('Unaligned source indices by method')
    if not np.allclose(residuals['analog_shape_half'], .5*(residuals['latest']+residuals['analog_shape'])):
        raise ValueError('Residual alignment identity failed')
    return predictions, residuals, values['origins'], seal


def freeze_inputs(root, data, files, output):
    """Snapshot all dependencies before source fitting; never refresh hashes."""
    root, data, output = Path(root), Path(data), Path(output)
    output.mkdir(parents=True)
    manifest = {'hashes': {f: sha(root/f) for f in files},
                'data_hashes': {str(p.resolve()): sha(p) for p in sorted(data.glob('*.csv'))}}
    for f in files:
        destination = output/f
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root/f).read_bytes())
    (output/'prefit_freeze.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_frozen_inputs(root, manifest):
    """Fail closed before labels if fitting changed source or data artifacts."""
    paths = {str(Path(root)/f): h for f, h in manifest['hashes'].items()}
    paths.update(manifest['data_hashes'])
    for path, digest in paths.items():
        if not Path(path).exists() or sha(path) != digest:
            raise ValueError('Frozen input changed before target label access: '+path)
    return True
