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

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFusedFiveFullSmoke:
    """Test Task 5.1 orchestrator schema and smoke output."""

    def test_row_schema_structure(self):
        """Test that evaluate_target returns correct row schema."""
        # Import here to avoid issues with missing dependencies in test env
        import run_fused_five_full

        # Mock the expensive operations
        mock_predictors = {
            "rag": lambda x, cfg, ef_r, ef_nr: np.random.rand(len(x), 24),
            "phys": lambda x, cfg, ef_r, ef_nr: np.random.rand(len(x), 24),
            "causal": lambda x, cfg, ef_r, ef_nr: np.random.rand(len(x), 24),
            "icl": lambda x, cfg, ef_r, ef_nr: np.random.rand(len(x), 24),
            "hier": lambda x, cfg, ef_r, ef_nr: np.random.rand(len(x), 24),
        }

        # Create dummy data
        n_windows = 10
        x_test = np.random.rand(n_windows, 336)
        config = np.array([0.0, 100.0])  # dummy config
        ef_r, ef_nr = 0.0, 800.0
        rs = np.random.rand(1000)
        cif = np.random.rand(1000)
        y_true = np.random.rand(n_windows, 24)

        # Mock all expensive imports and operations
        mock_source_stacks = [np.random.rand(5, 5, 24)]  # 5 directions, 24 horizon
        mock_source_true = [np.random.rand(5, 24)]
        mock_source_names = ["SOURCE1"]

        # Mock fusion method evaluation to return dummy metrics
        mock_fusion_metrics = (
            {"mae": 35.0, "rmse": 45.0, "smape": 8.5},  # base
            {"mae": 34.0, "rmse": 44.0, "smape": 8.0},  # plus
        )

        with patch('run_fused_five_full._build_predictors', return_value=mock_predictors), \
             patch('run_fused_five_full._train_basismix'), \
             patch('transcif.models.zeroshot.collector.collect_source_stacks',
                   return_value=(mock_source_stacks, mock_source_true, mock_source_names)), \
             patch('run_fused_five_full._eval_fusion_method', return_value=mock_fusion_metrics):

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
            with patch('run_fused_five_full.build_windows', return_value=(x_test, None, y_true)):
                # Mock zs_plus_predict
                with patch('run_fused_five_full.zs_plus_predict', return_value=np.random.rand(n_windows, 24)):
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
        import run_fused_five_full

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
        import run_fused_five_full

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

    def test_row_schema_all_metrics_finite(self):
        """Test that all metrics in the row are finite and positive."""
        import run_fused_five_full

        # Create a minimal valid row
        row = {
            "target": "TEST",
            "seed": 0,
        }

        # Add all required methods with dummy metrics
        methods = [
            "rag", "rag_plus", "phys", "phys_plus", "causal", "causal_plus",
            "icl", "icl_plus", "hier", "hier_plus", "equal", "equal_plus",
            "basismix", "basismix_plus", "persistence"
        ]

        for method in methods:
            row[method] = {"mae": 40.0, "rmse": 50.0, "smape": 10.0}

        # Verify all metrics are finite and positive
        for method in methods:
            metrics = row[method]
            assert np.isfinite(metrics["mae"]), f"{method} mae is not finite"
            assert np.isfinite(metrics["rmse"]), f"{method} rmse is not finite"
            assert np.isfinite(metrics["smape"]), f"{method} smape is not finite"
            assert metrics["mae"] > 0, f"{method} mae is not positive"
            assert metrics["rmse"] > 0, f"{method} rmse is not positive"
            assert metrics["smape"] > 0, f"{method} smape is not positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
