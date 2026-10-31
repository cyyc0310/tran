"""Versioned W recovery entry point with canonical sealed-score selection.

The original V10 producer and selection algorithm are unchanged. The original
score phase writes and seals region_metrics.csv, but returns its in-memory
DataFrame to the source selector. Floating values can differ by one bit after
CSV serialization, so the original exact choice verifier may reject an
otherwise identical selection. This entry point passes the sealed CSV values
to the original selector, making production and verification use the same
serialized score table. It must run in a fresh output directory.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_information_value_v10 as runner

ORIGINAL_SCORE_PHASE = runner.score_phase
ORIGINAL_CODE_PATHS = runner._code_paths


def canonical_score_phase(output: Path, frozen: dict[str, Any], stage: str,
                          seal: dict[str, Any]) -> pd.DataFrame:
    """Return exactly the just-sealed region score bytes to the selector."""
    ORIGINAL_SCORE_PHASE(output, frozen, stage, seal)
    path = Path(output) / f"{stage}_scores" / "region_metrics.csv"
    return pd.read_csv(path)


def canonical_code_paths() -> list[Path]:
    """Include this versioned entry point in the new prefit code closure."""
    return [*ORIGINAL_CODE_PATHS(), Path(__file__).resolve()]


def main() -> None:
    runner.score_phase = canonical_score_phase
    runner._code_paths = canonical_code_paths
    runner.main()


if __name__ == "__main__":
    main()
