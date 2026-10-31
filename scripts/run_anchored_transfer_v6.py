"""Second-round anchored residual transfer using the unchanged frozen V6 runner.

Both this wrapper and the bridge module are explicitly frozen dependencies.
External reuse checks their original source-prefit hashes and the complete
name-to-mode bridge policy before allowing reuse of fitted source artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from coarse_bridge_v6 import AnchorTransport, MODES, bridge_policy

_SPEC = importlib.util.spec_from_file_location("anchored_v6_original_runner", ROOT / "scripts/run_conditional_transfer_v6.py")
R6 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(R6)
V4, V5 = R6.V4, R6.V5
VARIANT = "anchored_v6"
PROTOCOL = "docs/ANCHORED_TRANSFER_V6_PROTOCOL.md"
BRIDGE_FILES = ("scripts/coarse_bridge_v6.py", "scripts/run_anchored_transfer_v6.py")
MULTIPLIERS = {"x1": 1., "x1p1": 1.1, "x1p2": 1.2, "x1p25": 1.25, "x1p5": 1.5, "x2": 2.}
CONFIGS = {f"{mode}_{suffix}": {"mode": mode, "multiplier": value}
           for mode in MODES for suffix, value in MULTIPLIERS.items()}
CONFIGS.update({name: deepcopy(parameters) for name, parameters in R6.CONFIGS.items()
                if "old_calibration" in parameters})

# These globals belong to a private imported runner module, not the canonical
# module used by other tests/tasks. Its functions resolve this private grid.
R6.VARIANT, R6.PROTOCOL, R6.MODES, R6.CONFIGS = VARIANT, PROTOCOL, MODES, CONFIGS
R6.MULTIPLIERS = MULTIPLIERS
V4.PROTOCOL = PROTOCOL
_ORIGINAL_GRID = R6.grid_parameters
_ORIGINAL_VALIDATE = R6.validate_grid
_ORIGINAL_LOAD_SELECTION = R6.load_selection
_ORIGINAL_FIT = R6.fit_phase


def grid_parameters():
    return {**_ORIGINAL_GRID(), "multipliers": dict(MULTIPLIERS), "bridge_policy": bridge_policy()}


def validate_grid(config):
    # Full dictionary equality also protects noanchor's name-to-mode meaning.
    if config.get("bridge_policy") != bridge_policy():
        raise ValueError("Anchored bridge_policy differs from the frozen source policy")
    _ORIGINAL_VALIDATE(config)


def verify_bridge_source(source, config):
    validate_grid(config)
    manifest = json.loads((Path(source) / "prefit_freeze.json").read_text())
    for relative in BRIDGE_FILES:
        path = ROOT / relative
        if manifest.get("code_hashes", {}).get(str(path)) != V4.sha(path):
            raise ValueError("Source-selected bridge script changed or was not frozen: " + relative)
    return manifest


def load_selection(path):
    source, config, selected = _ORIGINAL_LOAD_SELECTION(path)
    verify_bridge_source(source, config)
    return source, config, selected


@contextmanager
def _fit_bridge_context():
    """Limit temporary core/freeze substitution to the delegated fit call."""
    import transcif.research.conditional_transfer_v6 as core
    original_class, original_freeze = core.ConditionalTransport, V4.freeze_inputs

    def freeze_with_bridge(root, data, output, extra_files=(), external_data=None):
        dependencies = [*extra_files, *(ROOT / relative for relative in BRIDGE_FILES)]
        return original_freeze(root, data, output, extra_files=dependencies, external_data=external_data)

    core.ConditionalTransport, V4.freeze_inputs = AnchorTransport, freeze_with_bridge
    try:
        yield
    finally:
        core.ConditionalTransport, V4.freeze_inputs = original_class, original_freeze


def fit_phase(data, baseline, historical, output, *, external_data=None, selection=None):
    if external_data is not None:
        if selection is None:
            raise ValueError("Confirmation requires a frozen anchored source selection")
        # Validate the complete source bridge closure before delegated copying.
        load_selection(selection)
    with _fit_bridge_context():
        return _ORIGINAL_FIT(data, baseline, historical, output, external_data=external_data, selection=selection)


R6.grid_parameters = grid_parameters
R6.validate_grid = validate_grid
R6.load_selection = load_selection
R6.fit_phase = fit_phase
predict_seal_phase = R6.predict_seal_phase
score_phase = R6.score_phase
SOURCES, HOLDOUTS, POINTS, DELAYS = R6.SOURCES, R6.HOLDOUTS, R6.POINTS, R6.DELAYS


if __name__ == "__main__":
    R6.main()
