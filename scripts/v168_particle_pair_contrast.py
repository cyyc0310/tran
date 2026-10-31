"""Pair the frozen particle stress queries without fitting or retuning models."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from v151_hourly_law import sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v168-masked-shape-value-r1'
OUT = ROOT / 'results/v168-particle-pair-contrast-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source_summary = json.loads((SOURCE / 'summary.json').read_text())
    if source_summary['assertion_failures']:
        raise ValueError('source diagnostic failed')
    files = sorted(SOURCE.glob('fold_*/particle_stress.csv'))
    if len(files) != 48:
        raise ValueError('stress block denominator')
    for path in files:
        if sha(path) != source_summary['output_hashes'][str(path.relative_to(SOURCE))]:
            raise ValueError('stress input changed')
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files + [SOURCE / 'summary.json', Path(__file__)]}
    stress = pd.concat([pd.read_csv(p, dtype={'mask_id': str}) for p in files], ignore_index=True)
    stress = stress.loc[stress.mask_id.str.contains('-')]
    keys = ['fold', 'profile_id', 'mask_id', 'raw_id']
    joint = stress.loc[stress.method.eq('conditional_joint')]
    independent = stress.loc[stress.method.eq('conditional_independent')]
    pairs = joint.merge(independent, on=keys, suffixes=('_joint', '_independent'), validate='one_to_one')
    if len(pairs) != 199:
        raise ValueError('contrast-pair stress denominator')
    for size in [1024, 4096]:
        pairs[f'delta_{size}'] = pairs[f'crps_{size}_joint'] - pairs[f'crps_{size}_independent']
    pairs['delta_change'] = pairs.delta_4096 - pairs.delta_1024
    if not np.isfinite(pairs[['delta_1024', 'delta_4096', 'delta_change']]).all().all():
        raise ValueError('nonfinite particle contrast')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('particle stress input changed')
    OUT.mkdir(parents=True)
    pairs.to_csv(OUT / 'paired_stress.csv', index=False)
    result = {'assertion_failures': 0, 'pair_queries': len(pairs),
              'mean_joint_minus_independent_1024': float(pairs.delta_1024.mean()),
              'mean_joint_minus_independent_4096': float(pairs.delta_4096.mean()),
              'mean_absolute_coupling_change': float(pairs.delta_change.abs().mean()),
              'max_absolute_coupling_change': float(pairs.delta_change.abs().max()),
              'direction_changed_queries': int((np.sign(pairs.delta_1024) != np.sign(pairs.delta_4096)).sum()),
              'input_hashes': hashes, 'output_hashes': {'paired_stress.csv': sha(OUT / 'paired_stress.csv')},
              'scope': 'Equal-case first-query numerical sensitivity, not the primary equal-region full-hour comparison. No refit or particle-count selection.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['input_hashes', 'output_hashes']}, indent=2))


if __name__ == '__main__':
    main()
