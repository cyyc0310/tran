from dataclasses import replace

import numpy as np
import pytest
import torch

from transcif.models.dispatch import ProbabilisticDispatchNet
from transcif.models.ensemble import (mixture, predict_components, uncertainty_decomposition,
                                     independent_hour_ablation)
from transcif.data.dataset import stack_inputs
from transcif.training.transfer import independent_groups, fit_support, support_diagnostics
from transcif.evaluation.scheduling import choose_block, choose_robust_block, score_blocks, mechanism_decisions
from transcif.evaluation.metrics import forecast_metrics


def test_transitive_overlap_and_country_are_kept_together(bundle):
    _, _, grids = bundle
    base = grids["source_coal"]
    sources = {"a": replace(base, name="a", coverage=frozenset(["x"]), jurisdiction="one"),
               "b": replace(base, name="b", coverage=frozenset(["x", "y"]), jurisdiction="two"),
               "c": replace(base, name="c", coverage=frozenset(["y"]), jurisdiction="three"),
               "d": replace(base, name="d", coverage=frozenset(["z"]), jurisdiction="one")}
    assert independent_groups(sources) == [["a", "b", "c"], ["d"]]
    assert independent_groups(sources, "jurisdiction") == [["a", "b", "c", "d"]]


def test_support_uses_metadata_only_and_detects_unrepresented_configuration(bundle):
    _, doc, grids = bundle
    sources = [grids[name].example(doc["origins"][0]) for name in ("source_coal", "source_gas")]
    support = fit_support(sources, [["source_coal"], ["source_gas"]])
    np.testing.assert_array_equal(support_diagnostics(sources, support)["supported"], True)
    novel = grids["target_mixed"].example(doc["origins"][0])
    novel.values["capacity"][:] = [0, 0, 0, 0, 1000, 0]
    novel.values["mix"][:] = [0, 0, 0, 0, 1, 0]
    assert not support_diagnostics([novel], support)["supported"][0]
    for grid in grids.values():
        grid.labels_path.unlink()
    assert fit_support(sources, [["source_coal"], ["source_gas"]]) == support
    with pytest.raises(ValueError, match="two independent"):
        fit_support(sources, [["source_coal", "source_gas"]])


def test_variance_decomposition_is_exact_for_uniform_empirical_mixture():
    rng = np.random.default_rng(4)
    cif = rng.normal(size=(3, 4, 20, 8)) + np.arange(4)[None, :, None, None]
    stats = uncertainty_decomposition(cif)
    mixed = mixture({"cif": cif})["cif"]
    np.testing.assert_allclose(stats["total_variance"], mixed.var(1), atol=1e-12)
    np.testing.assert_allclose(stats["within_mechanism_variance"] +
                               stats["between_mechanism_variance"], stats["total_variance"])


def test_hour_permutation_preserves_marginals_and_breaks_shared_path_noise():
    rng = np.random.default_rng(8)
    samples = np.repeat(rng.normal(size=(2, 1000, 1)), 12, axis=-1)
    shuffled = independent_hour_ablation(samples, seed=7)
    np.testing.assert_array_equal(np.sort(samples, axis=1), np.sort(shuffled, axis=1))
    np.testing.assert_array_equal(shuffled, independent_hour_ablation(samples, seed=7))
    assert abs(np.corrcoef(shuffled[0, :, 0], shuffled[0, :, 1])[0, 1]) < .1
    truth = np.zeros((2, 12))
    a, b = forecast_metrics(truth, samples), forecast_metrics(truth, shuffled)
    assert a["crps"] == pytest.approx(b["crps"])
    assert a["mae"] == pytest.approx(b["mae"])
    assert a["energy_score"] != pytest.approx(b["energy_score"], abs=1e-3)


def test_worst_finite_mechanism_has_a_different_action_than_mixture():
    # Two mechanisms: mixture mean prefers hour 0 (6 vs 7), worst prefers 1.
    cif = np.array([[[[0., 7.], [0., 7.]], [[12., 7.], [12., 7.]]]])
    components = {"cif": cif, "demand": np.ones_like(cif), "unserved": np.zeros_like(cif)}
    decisions = mechanism_decisions(components, [True], duration=1)
    assert decisions["mixture_mean"][0] == 0
    assert decisions["worst_mechanism_mean"][0] == 1
    assert decisions["worst_mechanism_cvar"][0] == 1
    assert decisions["accepted"][0]
    components["unserved"][:, 0] = .02
    assert not mechanism_decisions(components, [True], duration=1)["accepted"][0]
    assert not mechanism_decisions(components, [False], duration=1)["accepted"][0]
    with pytest.raises(ValueError):
        choose_robust_block(np.ones((3, 4)))


def test_abstention_does_not_hide_bad_proposals():
    truth = np.array([[100., 0.], [80., 0.]])
    stats = score_blocks(truth, np.array([0, 0]), duration=1, accepted=[False, False])
    assert stats["recommendation_rate"] == stats["accepted_windows"] == 0
    assert stats["accepted_proxy_cost_g_per_kwh"] is None
    assert stats["proxy_cost_g_per_kwh"] == stats["oracle_regret_g_per_kwh"] == 90


def test_finite_mechanism_cvar_is_not_mislabeled_as_all_mixtures_dro():
    # P1: 1% cost100, P2: constant10; each CVaR90=10 at hour0.
    # Their equal mixture has CVaR90=14.5. Hour1 always costs12.
    samples = np.full((2, 100, 2), 12.)
    samples[0, :, 0] = 0
    samples[0, -1, 0] = 100
    samples[1, :, 0] = 10
    assert choose_robust_block(samples, duration=1, risk="cvar") == 0
    assert choose_block(samples.reshape(200, 2), duration=1, risk="cvar") == 1


def test_component_paths_remain_feasible_after_scale_change(bundle):
    _, doc, grids = bundle
    example = grids["source_coal"].example(doc["origins"][0])
    models = [ProbabilisticDispatchNet(), ProbabilisticDispatchNet()]
    components = predict_components(models, [example], scenarios=16, latent_scale=4., seed=5)
    np.testing.assert_allclose(components["generation"].sum(-1) + components["unserved"],
                               components["demand"], atol=1e-3, rtol=1e-5)
    assert (components["generation"] <= example.values["capacity"] + 1e-3).all()
    assert mixture(components)["cif"].shape == (1, 32, 24)
    inputs = stack_inputs([example])
    first = models[0](inputs, 4, deterministic=True)
    second = models[0](inputs, 4, deterministic=True)
    assert torch.equal(first.cif, second.cif)
    torch.testing.assert_close(first.cif[:, 0], first.cif[:, 3])
    with pytest.raises(ValueError, match="latent_scale"):
        models[0](inputs, 4, latent_scale=float("nan"))


def test_chunked_energy_score_matches_direct_pairwise_definition():
    rng = np.random.default_rng(5)
    samples, truth = rng.normal(size=(2, 65, 7)), rng.normal(size=(2, 7))
    pair = np.linalg.norm(samples[:, :, None] - samples[:, None, :], axis=-1).mean()
    expected = (np.linalg.norm(samples - truth[:, None], axis=-1).mean() - pair / 2) / 7**.5
    assert forecast_metrics(truth, samples)["energy_score"] == pytest.approx(expected)
