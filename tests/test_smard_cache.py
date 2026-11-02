import io
import json
from pathlib import Path
import runpy

import pytest


def cache_function():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/fetch_smard_generation.py'))['cache_json']


def test_bad_cached_response_is_quarantined_and_fetched(monkeypatch, tmp_path):
    path = tmp_path/'series.json'
    path.write_text('{"error":"temporary failure"}')
    calls = []
    def response(url, timeout):
        calls.append(url)
        return io.BytesIO(b'{"series":[[1,0.0]]}')
    monkeypatch.setattr('urllib.request.urlopen', response)
    result = cache_function()('https://example.test/series.json', path)
    assert result == {'series': [[1, 0.]]}
    assert len(calls) == 1
    assert len(list(tmp_path.glob('*.invalid-*.json'))) == 1
    assert json.loads(path.read_text()) == result


def test_error_json_is_not_written_as_valid_cache(monkeypatch, tmp_path):
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(b'{"error":"unavailable"}'))
    path = tmp_path/'index.json'
    with pytest.raises(ValueError):
        cache_function()('https://example.test/index_hour.json', path)
    assert not path.exists()
