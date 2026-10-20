import json
from pathlib import Path
import subprocess
import sys


def test_audit_identifies_untyped_labels_and_percent_units_without_changing_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    csv = source / "UK.csv"
    content = ("hour,cif_real_gco2_per_kwh,total_energy_so\n"
               "2023-01-01 00:00:00,100,100\n2023-01-01 01:00:00,110,99.9\n")
    csv.write_text(content)
    output = tmp_path / "audit.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_data.py"
    result = subprocess.run([sys.executable, str(script), "--directory", str(source),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text())["files"]["UK.csv"]
    assert not record["timezone_explicit"]
    assert record["energy_near_100_fraction"] == 1
    assert len(record["missing_six_fuel_supervision"]) == 6
    assert any("provenance absent" in finding for finding in record["findings"])
    assert csv.read_text() == content
