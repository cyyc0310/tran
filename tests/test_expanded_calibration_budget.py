"""Legacy calibration selection is governed by source role and exact metadata."""
import importlib
from pathlib import Path

import pandas as pd
import pytest


def test_legacy_selection_ignores_target_labels_and_rejects_target_calibration(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('analyze_expanded_calibration_budget')
    rows = pd.DataFrame(dict(country=['GB', 'GB', 'US', 'GB'], region=['GB', 'GB', 'CISO', 'GB'],
        origin=['2023-01-15', '2023-01-25', '2023-01-15', '2023-07-15'],
        valid_at=['2023-01-15', '2023-01-25', '2023-01-15', '2023-07-15'], label_row=[1, 2, 3, 4],
        split=['calibration', 'calibration', 'calibration', 'development'], truth=[1., 2., 3., 4.]))
    old = rows.iloc[[0, 2, 3]].copy()
    assert module.legacy_indices(rows, old, ('GB',), 'US').tolist() == [0]
    changed = rows.copy(); changed['truth'] = -1e20
    assert module.legacy_indices(changed, old, ('GB',), 'US').tolist() == [0]
    with pytest.raises(ValueError, match='target'):
        module.legacy_indices(rows, old, ('US',), 'US')
    with pytest.raises(ValueError, match='not fully preserved'):
        module.legacy_indices(rows.iloc[1:], old, ('GB',), 'US')


def test_duplicate_legacy_identity_cannot_double_weight_a_source_hour(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('analyze_expanded_calibration_budget')
    rows = pd.DataFrame(dict(country=['GB'], region=['GB'], origin=['2023-01-15'],
        valid_at=['2023-01-15'], label_row=[1], split=['calibration']))
    with pytest.raises(ValueError, match='ambiguous'):
        module.legacy_indices(rows, pd.concat([rows, rows]), ('GB',), 'US')
