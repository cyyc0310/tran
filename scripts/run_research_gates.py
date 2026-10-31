"""Run the frozen research pilot; does not modify upstream input data."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcif.research.eia_proxy import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/upstream-staging/eia-2022-2024")
    parser.add_argument("--output", type=Path, default=ROOT / "results/research")
    parser.add_argument("--stage", choices=["prepare", "gates12", "gate3", "all"], default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("prepare", "all"):
        prepare(args.source, args.output / "data")
    if args.stage in ("gates12", "all"):
        from transcif.research.gates12 import run
        run(args.output / "gates12")
    if args.stage in ("gate3", "all"):
        from transcif.research.gate3 import run
        run_dir = args.output / "gate3"
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit("Use a fresh output directory; refusing to overwrite a prior gate3 experiment")
        run_dir.mkdir(parents=True, exist_ok=True)
        protocol = ROOT / "docs/RESEARCH_GATES_PROTOCOL.md"
        shutil.copyfile(protocol, run_dir / "frozen_protocol.md")
        files = sorted((ROOT / "src/transcif/research").glob("*.py")) + [Path(__file__), protocol]
        import numpy, pandas, sklearn
        provenance = {"python": platform.python_version(), "numpy": numpy.__version__,
                      "pandas": pandas.__version__, "sklearn": sklearn.__version__,
                      "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                      "derived_input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sorted((args.output / "data").glob("*.csv"))},
                      "data_audit_sha256": hashlib.sha256((args.output / "data/audit.json").read_bytes()).hexdigest()}
        (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        run(args.output / "data", run_dir)


if __name__ == "__main__":
    main()
