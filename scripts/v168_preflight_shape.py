"""Proper-source and synthetic identities for the conditional masking diagnostic."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hourly_fuel_data import HourlyFuelData
from masked_shape_completion import (CompletionForest, ShapePanel, independent_particles,
                                     joint_particles, proposals_for, score_completion)
from v151_hourly_law import EF, sha
from v168_masked_shape_value import hidden_intervention

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v168-shape-preflight-r2'
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    q = np.array([[.5, 1.5], [1.5, .5]])
    full = np.ones((2, 8))
    full[:, :2] = q
    independent = np.array([[.5, .5], [.5, 1.5], [1.5, .5], [1.5, 1.5]])
    S = np.array([.5, .5, 0, 0, 0, 0, 0, 0])
    actual = float((S * EF) @ full[0] / (S @ full[0]))
    joint_score = score_completion(S, full[0], (0, 1), q, full[0], actual)
    independent_score = score_completion(S, full[0], (0, 1), independent, full[0], actual)
    assert abs(joint_score['q_crps'] - independent_score['q_crps']) < 1e-12
    assert abs(joint_score['crps'] - independent_score['crps']) > 1
    values = np.array([[0., 1.], [1., 3.], [3., 0.]])
    draw = joint_particles(values, np.array([.2, .3, .5]))
    shuffled = independent_particles(draw, 'v168/preflight')
    for j in range(2):
        assert np.array_equal(np.sort(shuffled[:, j]), np.repeat(np.sort(draw[:, j]), 4))
    single = independent_particles(draw[:, :1], 'v168/single')
    assert np.array_equal(single, draw[:, :1])
    panel = ShapePanel(HourlyFuelData(ROOT), pd.read_csv(BASE / 'source_sampling.csv'))
    metadata = json.loads((BASE / 'fold_0_asof_30d/summary.json').read_text())
    query = pd.read_csv(BASE / 'fold_0_asof_30d/query_inputs.csv')
    q = int(query.loc[query.role.eq('test'), 'profile_id'].iloc[0])
    ids = panel.queries(q)[:2]
    with threadpool_limits(limits=2):
        model = CompletionForest(panel, metadata['source_regions'], (1,))
        hidden_intervention(panel, ids, (1,))
        distribution = model.distributions(ids)
        mean_error = 0.
        for position, item in enumerate(distribution):
            values, weights = item['conditional']
            mean = np.average(values, axis=0, weights=weights)
            predicted = np.asarray(model.models['conditional'].forest.predict(panel.features(ids[position:position + 1], (1,)))).reshape(-1)
            mean_error = max(mean_error, float(np.max(abs(mean - predicted))))
            proposals, _ = proposals_for(item, f'preflight/{position}')
            assert np.array_equal(proposals['conditional_joint'], proposals['conditional_independent'])
    if mean_error > 1e-9:
        raise ValueError('leaf mean and forest point prediction differ')
    OUT.mkdir(parents=True)
    result = {'assertion_failures': 0, 'synthetic_equal_q_crps': joint_score['q_crps'],
              'synthetic_joint_cif_crps': joint_score['crps'],
              'synthetic_independent_cif_crps': independent_score['crps'],
              'conditional_mean_identity_max_error': mean_error, 'hidden_feature_cases': len(ids),
              'script_sha256': sha(Path(__file__)), 'helper_sha256': sha(ROOT / 'scripts/masked_shape_completion.py'),
              'scope': 'Synthetic scores and proper-source model identities; no held-out hourly risk performance evaluated.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
