"""Execution and aggregation guards for zero-fit V35 diagnostics."""
import numpy as np
import pandas as pd
import pytest

from scripts import run_conditional_diagnostics_v35 as runner


def test_existing_attempt_is_preserved_before_any_input_access(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    monkeypatch.setattr(runner, 'contract', lambda: pytest.fail('Existing attempt must reject first'))
    with pytest.raises(FileExistsError):
        runner.run()


def test_failed_calculation_keeps_start_receipt_and_refuses_retry(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    monkeypatch.setattr(runner, 'OUT', out)
    monkeypatch.setattr(runner, 'contract', lambda: dict(input_hashes={}, code_hashes={}))
    monkeypatch.setattr(runner, 'compute', lambda _: (_ for _ in ()).throw(ValueError('synthetic failure')))
    with pytest.raises(ValueError, match='synthetic failure'):
        runner.run()
    receipt = (out / 'analysis_started.json').read_bytes()
    assert {p.name for p in out.iterdir()} == {'freeze.json', 'analysis_started.json'}
    with pytest.raises(FileExistsError):
        runner.run()
    assert (out / 'analysis_started.json').read_bytes() == receipt


def test_region_rmse_and_macro_are_not_pooled_across_months():
    case = pd.DataFrame(dict(region=['A', 'A', 'B'], arm=['x'] * 3,
                             cif_abs_error=[0., 4., 1.], cif_sq_error=[0., 16., 1.]))
    region = runner.aggregate(case, ['region', 'arm'], ('cif',))
    assert region.set_index('region').loc['A', 'cif_rmse'] == pytest.approx(np.sqrt(8.))
    assert region.cif_abs_error.mean() == 1.5
    assert region.cif_abs_error.mean() != pytest.approx(case.cif_abs_error.mean())
    assert region.cif_rmse.mean() == pytest.approx((np.sqrt(8.) + 1.) / 2.)
    assert region.scored_months.tolist() == [2, 1]


def test_bound_input_symlink_rejected_even_with_matching_bytes(tmp_path):
    source = tmp_path / 'file'
    source.write_text('fixed')
    link = tmp_path / 'link'
    link.symlink_to(source)
    with pytest.raises(ValueError, match='Regular non-symlink'):
        runner.verify({str(link): runner.sha(source)})


def test_archive_loader_has_explicit_member_whitelist(monkeypatch):
    accessed = []

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __getitem__(self, key):
            accessed.append(key)
            assert key in runner.MEMBERS
            return np.asarray([])

    monkeypatch.setattr(runner.np, 'load', lambda *args, **kwargs: Archive())
    with pytest.raises(ValueError, match='Exact V34 archive grid'):
        runner.load_archive('unopened', 'MISO')
    assert tuple(accessed) == runner.MEMBERS
    assert 'M8_draws' not in accessed
