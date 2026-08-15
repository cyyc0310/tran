"""Integration test for Task 5.1 orchestrator (smoke test).

This test mocks the expensive predictor training and source collection
to verify the row schema and orchestration logic without running the
full evaluation pipeline.

Test strategy:
- Mock _build_predictors to return dummy predictors
- Mock collect_source_stacks to return dummy stacks
- Verify evaluate_target returns correct schema
- Verify summary statistics computation
"""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from scripts.experiments import run_fused_five_full


class TestFusedFiveFullSmoke:
    """Test Task 5.1 orchestrator schema and smoke output."""

    def test_row_schema_structure(self):
        """Test that evaluate_target returns correct row schema."""
        rng = np.random.default_rng(0)

        # Mock the expensive operations
        mock_predictors = {
            "rag": lambda x, cfg, ef_r, ef_nr: rng.random((len(x), 24)),
            "phys": lambda x, cfg, ef_r, ef_nr: rng.random((len(x), 24)),
            "causal": lambda x, cfg, ef_r, ef_nr: rng.random((len(x), 24)),
            "icl": lambda x, cfg, ef_r, ef_nr: rng.random((len(x), 24)),
            "hier": lambda x, cfg, ef_r, ef_nr: rng.random((len(x), 24)),
        }

        # Create dummy data
        n_windows = 10
        x_test = rng.random((n_windows, 336))
        config = np.array([0.0, 100.0])  # dummy config
        ef_r, ef_nr = 0.0, 800.0
        rs = rng.random(1000)
        cif = rng.random(1000)
        y_true = rng.random((n_windows, 24))

        # Mock all expensive imports and operations
        mock_source_stacks = [rng.random((5, 5, 24))]  # 5 directions, 24 horizon
        mock_source_true = [rng.random((5, 24))]
        mock_source_names = ["SOURCE1"]

        # Mock fusion method evaluation to return dummy metrics
        mock_fusion_metrics = (
            {"mae": 35.0, "rmse": 45.0, "smape": 8.5},  # base
            {"mae": 34.0, "rmse": 44.0, "smape": 8.0},  # plus
        )

        with patch('scripts.experiments.run_fused_five_full._build_predictors',
                   return_value=mock_predictors), \
             patch('scripts.experiments.run_fused_five_full._train_basismix'), \
             patch('transcif.models.zeroshot.collector.collect_source_stacks',
                   return_value=(mock_source_stacks, mock_source_true, mock_source_names)), \
             patch('scripts.experiments.run_fused_five_full._eval_fusion_method',
                   return_value=mock_fusion_metrics):

            # Create minimal mock region data
            all_regions = {
                "TEST_TARGET": {
                    "config": config,
                    "ef_r": ef_r,
                    "ef_nr": ef_nr,
                    "rs": rs,
                    "cif": cif,
                }
            }

            # Mock the window building to return our dummy data
            with patch('scripts.experiments.run_fused_five_full.build_windows',
                       return_value=(x_test, None, y_true)):
                # Mock zs_plus_predict
                with patch('scripts.experiments.run_fused_five_full.zs_plus_predict',
                           return_value=rng.random((n_windows, 24))):
                    row = run_fused_five_full.evaluate_target(
                        target="TEST_TARGET",
                        all_regions=all_regions,
                        seed=0,
                        src_limit=1,
                        output_json=Path("nonexistent.json"),
                        resume=False
                    )

        # Verify row structure
        assert row is not None, "evaluate_target should return a row"
        assert row["target"] == "TEST_TARGET"
        assert row["seed"] == 0

        # Verify all required methods are present
        required_methods = [
            "rag", "rag_plus",
            "phys", "phys_plus",
            "causal", "causal_plus",
            "icl", "icl_plus",
            "hier", "hier_plus",
            "equal", "equal_plus",
            "basismix", "basismix_plus",
            "persistence"
        ]

        for method in required_methods:
            assert method in row, f"Row missing method: {method}"
            assert "mae" in row[method], f"{method} missing mae metric"
            assert "rmse" in row[method], f"{method} missing rmse metric"
            assert "smape" in row[method], f"{method} missing smape metric"

    def test_summary_statistics_shape(self):
        """Test that compute_summary_statistics returns correct shape."""
        # Create dummy results
        results = [
            {
                "target": "R1",
                "seed": 0,
                "rag": {"mae": 40.0, "rmse": 50.0, "smape": 10.0},
                "rag_plus": {"mae": 38.0, "rmse": 48.0, "smape": 9.5},
                "basismix": {"mae": 35.0, "rmse": 45.0, "smape": 8.5},
                "basismix_plus": {"mae": 34.0, "rmse": 44.0, "smape": 8.0},
                "persistence": {"mae": 60.0, "rmse": 70.0, "smape": 15.0},
            },
            {
                "target": "R2",
                "seed": 0,
                "rag": {"mae": 42.0, "rmse": 52.0, "smape": 11.0},
                "rag_plus": {"mae": 39.0, "rmse": 49.0, "smape": 10.0},
                "basismix": {"mae": 36.0, "rmse": 46.0, "smape": 9.0},
                "basismix_plus": {"mae": 35.0, "rmse": 45.0, "smape": 8.5},
                "persistence": {"mae": 62.0, "rmse": 72.0, "smape": 16.0},
            },
        ]

        summary = run_fused_five_full.compute_summary_statistics(results)

        # Verify structure
        for method in ["rag", "rag_plus", "basismix", "basismix_plus", "persistence"]:
            assert method in summary, f"Summary missing method: {method}"
            assert "mae" in summary[method], f"{method} missing mae stats"
            assert "rmse" in summary[method], f"{method} missing rmse stats"
            assert "smape" in summary[method], f"{method} missing smape stats"

            # Verify stats structure
            for metric in ["mae", "rmse", "smape"]:
                stats = summary[method][metric]
                assert "median" in stats, f"{method}.{metric} missing median"
                assert "mean" in stats, f"{method}.{metric} missing mean"
                assert "std" in stats, f"{method}.{metric} missing std"
                assert isinstance(stats["median"], float)
                assert isinstance(stats["mean"], float)
                assert isinstance(stats["std"], float)

    def test_summary_statistics_values(self):
        """Test that summary statistics are computed correctly."""
        # Create controlled test data
        results = [
            {
                "target": "R1",
                "seed": 0,
                "rag": {"mae": 40.0, "rmse": 50.0, "smape": 10.0},
            },
            {
                "target": "R2",
                "seed": 0,
                "rag": {"mae": 42.0, "rmse": 52.0, "smape": 11.0},
            },
            {
                "target": "R3",
                "seed": 0,
                "rag": {"mae": 38.0, "rmse": 48.0, "smape": 9.0},
            },
        ]

        summary = run_fused_five_full.compute_summary_statistics(results)

        # Verify median computation (sorted: 38, 40, 42 → median = 40)
        assert summary["rag"]["mae"]["median"] == 40.0
        # Verify mean computation ((40 + 42 + 38) / 3 = 40)
        assert abs(summary["rag"]["mae"]["mean"] - 40.0) < 0.01
        # Verify std computation (std of [40, 42, 38] ≈ 1.63)
        assert abs(summary["rag"]["mae"]["std"] - 1.63) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
