"""Tests for source-region stack collector (Task 1.3).

Stubs each direction predictor with the *real* per-direction signature so
monkeypatching actually intercepts the call paths in ``collector.py``.
"""

import numpy as np
import pytest

from transcif.config import HORIZON, SEQ_LEN, TEST_STRIDE, TRAIN_FRACTION
from transcif.data.windows import build_windows
from transcif.models.zeroshot.collector import collect_source_stacks
from transcif.models.zeroshot.fusion import DIRECTION_ORDER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_stub_region_data(n_hours: int = 1000,
                           train_sentinel: float = -1.0,
                           test_sentinel: float = 1.0) -> dict:
    """Synthetic region with sentinel CIF values for leak detection."""
    split = int(n_hours * TRAIN_FRACTION)
    rs = np.random.rand(n_hours).astype(np.float32) * 0.5
    cif = np.ones(n_hours, dtype=np.float32) * train_sentinel
    cif[split:] = test_sentinel
    return {
        "rs": rs,
        "cif": cif,
        "mean_rs": float(rs.mean()),
        "ef_r": 0.0,
        "ef_nr": 500.0,
        "config": np.array([rs.mean(), 0.5], dtype=np.float32),
    }


def _count_test_windows(n_hours: int) -> int:
    """Replicate collector's TEST window count for stub shape."""
    split = int(n_hours * TRAIN_FRACTION)
    rs_template = np.zeros(n_hours, dtype=np.float32)
    cif_template = np.zeros(n_hours, dtype=np.float32)
    x_test, _, _ = build_windows(
        rs_template[split - SEQ_LEN:],
        cif_template[split - SEQ_LEN:],
        seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE,
    )
    return len(x_test)


def _stub_all_directions(monkeypatch, value_per_direction: dict | None = None) -> None:
    """Monkeypatch all 5 direction modules with correct-signature stubs.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        value_per_direction: optional ``{direction: constant_cif}`` map. If
            ``None``, all directions return 100.0.
    """
    if value_per_direction is None:
        value_per_direction = {d: 100.0 for d in DIRECTION_ORDER}

    def _resolve_value(direction: str, region_name: str,
                       all_regions: dict, n_windows: int) -> np.ndarray:
        val = value_per_direction[direction]
        return np.full((n_windows, HORIZON), val, dtype=np.float32)

    # RAG signature: train(all_regions, target_name, seed, device) -> (model, bank)
    #                predict(model, bank, x_rs, config, ef_r, ef_nr) -> (n, H)
    def rag_train(all_regions, target_name, seed=0, device=None, **kw):
        return ("rag_stub", {"bank": "data"})

    def rag_predict(model, bank, x_rs, config, ef_r, ef_nr, *, region=None,
                    all_regions=None):
        n = x_rs.shape[0]
        return _resolve_value("rag", region, all_regions or {}, n)

    # Phys-IRM signature: train(all_regions, target_name, seed, gamma_irm,
    #                           lambda_cif, device) -> (model, _)
    #                     predict(model, x_rs, config, ef_r, ef_nr) -> (n, H)
    def phys_train(all_regions, target_name, seed=0, gamma_irm=0.1,
                   lambda_cif=0.5, device=None, **kw):
        return ("phys_stub", {})

    def phys_predict(model, x_rs, config, ef_r, ef_nr, *, region=None,
                     all_regions=None):
        return _resolve_value("phys", region, all_regions or {}, x_rs.shape[0])

    # Causal signature: train(all_regions, target_name, seed, device) -> (model, _)
    #                   predict(model, x_rs, config, ef_r, ef_nr) -> (n, H)
    def causal_train(all_regions, target_name, seed=0, device=None, **kw):
        return ("causal_stub", {})

    def causal_predict(model, x_rs, config, ef_r, ef_nr, *, region=None,
                       all_regions=None):
        return _resolve_value("causal", region, all_regions or {}, x_rs.shape[0])

    # ICL signature: train(all_regions, target_name, seed, device) -> model
    #                predict(model, all_regions, region_name, x_rs, ef_r, ef_nr) -> (n, H)
    def icl_train(all_regions, target_name, seed=0, device=None, **kw):
        return "icl_stub"

    def icl_predict(model, all_regions, region_name, x_rs, ef_r, ef_nr):
        return _resolve_value("icl", region_name, all_regions, x_rs.shape[0])

    # Hier signature: train(all_regions, target_name, seed, device) -> model
    #                 predict(model, x_rs, config, ef_r, ef_nr) -> (n, H)
    def hier_train(all_regions, target_name, seed=0, device=None, **kw):
        return "hier_stub"

    def hier_predict(model, x_rs, config, ef_r, ef_nr, *, region=None,
                     all_regions=None):
        return _resolve_value("hier", region, all_regions or {}, x_rs.shape[0])

    from transcif.models.zeroshot import rag, phys_irm, causal, icl, hier

    monkeypatch.setattr(rag, "train_rag_zero_shot", rag_train)
    monkeypatch.setattr(rag, "predict_rag_zs", rag_predict)
    monkeypatch.setattr(phys_irm, "train_phys_irm", phys_train)
    monkeypatch.setattr(phys_irm, "predict_phys_irm", phys_predict)
    monkeypatch.setattr(causal, "train_causal_zero_shot", causal_train)
    monkeypatch.setattr(causal, "predict_causal_zs", causal_predict)
    monkeypatch.setattr(icl, "train_icl", icl_train)
    monkeypatch.setattr(icl, "predict_icl_zs", icl_predict)
    monkeypatch.setattr(hier, "train_hier", hier_train)
    monkeypatch.setattr(hier, "predict_hier_zs", hier_predict)


# ---------------------------------------------------------------------------
# Shape & membership tests
# ---------------------------------------------------------------------------

def test_collector_returns_correct_shapes_with_stubbed_predictors(monkeypatch):
    """Stacks have shape (n_i, 5, HORIZON); true CIF has (n_i, HORIZON)."""
    n_hours = 1000
    expected_n = _count_test_windows(n_hours)

    all_regions = {
        "SOURCE1": _make_stub_region_data(n_hours, -1.0, 1.0),
        "SOURCE2": _make_stub_region_data(n_hours, -2.0, 2.0),
        "TARGET": _make_stub_region_data(n_hours, -3.0, 3.0),
    }

    _stub_all_directions(monkeypatch)

    stacks, true_cifs, names = collect_source_stacks(
        all_regions=all_regions,
        target_name="TARGET",
        seed=0,
        device=None,
        source_names=None,
        progress=False,
    )

    assert len(stacks) == 2
    assert len(true_cifs) == 2
    assert set(names) == {"SOURCE1", "SOURCE2"}

    for i, name in enumerate(names):
        assert stacks[i].shape == (expected_n, len(DIRECTION_ORDER), HORIZON), (
            f"{name}: stack shape {stacks[i].shape}, expected "
            f"({expected_n}, 5, {HORIZON})"
        )
        assert true_cifs[i].shape == (expected_n, HORIZON)


def test_collector_excludes_target_region(monkeypatch):
    """Target region never appears in source names."""
    all_regions = {
        "SOURCE1": _make_stub_region_data(),
        "SOURCE2": _make_stub_region_data(),
        "TARGET": _make_stub_region_data(),
    }
    _stub_all_directions(monkeypatch)

    _, _, names = collect_source_stacks(
        all_regions=all_regions,
        target_name="TARGET",
        seed=0, device=None,
        source_names=None, progress=False,
    )

    assert "TARGET" not in names
    assert len(names) == 2


def test_collector_respects_explicit_source_list(monkeypatch):
    """Explicit source_names is honored; non-listed regions excluded."""
    all_regions = {
        "SOURCE1": _make_stub_region_data(),
        "SOURCE2": _make_stub_region_data(),
        "SOURCE3": _make_stub_region_data(),
        "TARGET": _make_stub_region_data(),
    }
    _stub_all_directions(monkeypatch)

    _, _, names = collect_source_stacks(
        all_regions=all_regions,
        target_name="TARGET",
        seed=0, device=None,
        source_names=["SOURCE1", "SOURCE3"],
        progress=False,
    )

    assert set(names) == {"SOURCE1", "SOURCE3"}
    assert "SOURCE2" not in names
    assert "TARGET" not in names


def test_collector_rejects_target_in_explicit_sources(monkeypatch):
    """Explicit source list containing target_name raises ValueError."""
    all_regions = {
        "SOURCE1": _make_stub_region_data(),
        "TARGET": _make_stub_region_data(),
    }
    _stub_all_directions(monkeypatch)

    with pytest.raises(ValueError, match="cannot be in source_names"):
        collect_source_stacks(
            all_regions=all_regions,
            target_name="TARGET",
            seed=0, device=None,
            source_names=["SOURCE1", "TARGET"],
            progress=False,
        )


def test_collector_handles_empty_source_list():
    """Empty source list returns empty triple."""
    all_regions = {"TARGET": _make_stub_region_data()}

    stacks, true_cifs, names = collect_source_stacks(
        all_regions=all_regions,
        target_name="TARGET",
        seed=0, device=None,
        source_names=[], progress=False,
    )

    assert stacks == [] and true_cifs == [] and names == []


def test_collector_matches_list_lengths(monkeypatch):
    """Returned lists have matching length."""
    all_regions = {
        f"SOURCE{i}": _make_stub_region_data() for i in range(1, 4)
    }
    all_regions["TARGET"] = _make_stub_region_data()
    _stub_all_directions(monkeypatch)

    stacks, true_cifs, names = collect_source_stacks(
        all_regions=all_regions,
        target_name="TARGET",
        seed=0, device=None,
        source_names=["SOURCE1", "SOURCE2", "SOURCE3"],
        progress=False,
    )

    assert len(stacks) == len(true_cifs) == len(names) == 3


# ---------------------------------------------------------------------------
# Train/test leak detection
# ---------------------------------------------------------------------------

def test_collector_no_train_test_leak(monkeypatch):
    """True CIF (computed from y_cif_test inside collector) contains only the
    TEST sentinel — never the TRAIN sentinel. This is the load-bearing leak
    assertion: if the collector's window slicing accidentally included train
    indices, the true CIF would contain the negative sentinel.
    """
    all_regions = {
        "SOURCE": _make_stub_region_data(1000, train_sentinel=-1.0,
                                         test_sentinel=1.0),
        "TARGET": _make_stub_region_data(1000, train_sentinel=-3.0,
                                         test_sentinel=3.0),
    }
    _stub_all_directions(monkeypatch)

    _, true_cifs, names = collect_source_stacks(
        all_regions=all_regions,
        target_name="TARGET",
        seed=0, device=None,
        source_names=["SOURCE"], progress=False,
    )

    assert names == ["SOURCE"]
    assert np.all(true_cifs[0] == 1.0), \
        "True CIF must be the TEST sentinel (+1.0), not TRAIN (-1.0)"
    assert (true_cifs[0] > 0).all(), \
        "True CIF must be strictly positive (test portion only)"


# ---------------------------------------------------------------------------
# Integration with real data slicing (no direction stubs needed)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_collector_window_slicing_matches_build_windows():
    """Verify collector's TEST slice matches build_windows output."""
    n_hours = 1000
    region = _make_stub_region_data(n_hours, train_sentinel=-1.0,
                                    test_sentinel=10.0)
    rs = region["rs"]
    cif = region["cif"]
    split = int(n_hours * TRAIN_FRACTION)

    _, _, y_cif_test = build_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:],
        seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE,
    )

    assert len(y_cif_test) > 0
    assert np.all(y_cif_test == 10.0), "All TEST targets must be +10.0 sentinel"
    assert cif[:split].mean() < 0 and cif[split:].mean() > 0
