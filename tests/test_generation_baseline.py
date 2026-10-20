from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from transcif.models.baselines import PublicGenerationBoosting


def test_generation_baseline_transfers_capacity_scale_and_respects_external_efs(bundle):
    pytest.importorskip("sklearn")
    _, doc, grids = bundle
    grid = grids["source_coal"]
    examples = [grid.example(origin) for origin in doc["origins"][:4]]
    model = PublicGenerationBoosting().fit(examples, [grid.labels(e) for e in examples])
    example = grids["target_mixed"].example(doc["origins"][-1])
    scaled = replace(example, values=deepcopy(example.values))
    scaled.values["capacity"] *= 2
    scaled.values["demand"] *= 2
    generation = model.predict_generation([example])
    np.testing.assert_allclose(model.predict_generation([scaled]), 2 * generation, rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(model.predict([scaled]), model.predict([example]), rtol=1e-5)
    assert (generation >= 0).all()
    assert (generation <= example.values["capacity"] + 1e-6).all()
    expected = (generation * example.values["ef"]).sum(-1) / generation.sum(-1)
    np.testing.assert_allclose(model.predict([example]), expected, rtol=1e-5)
