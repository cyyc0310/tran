from pathlib import Path
import sys
from urllib.error import HTTPError, URLError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_weather_transfer_pipeline import invoke, retry_collection  # noqa: E402


def test_transient_failure_retries_and_preserves_success():
    calls, notices, pauses = [], [], []
    def work():
        calls.append(1)
        if len(calls) < 2:
            raise URLError('SSL EOF')
        return 'verified completed windows reused'
    assert retry_collection(work, lambda *args: notices.append(args), pause=pauses.append).startswith('verified')
    assert len(calls) == 2 and len(notices) == 1 and pauses == [10.]


@pytest.mark.parametrize('error', [ValueError('corrupt hash'), HTTPError('url', 404, 'missing', {}, None)])
def test_integrity_and_permanent_http_failure_are_never_retried(error):
    calls = []
    def work():
        calls.append(1)
        raise error
    with pytest.raises(type(error)):
        retry_collection(work, lambda *args: None, pause=lambda _: None)
    assert len(calls) == 1


def test_network_retries_are_bounded():
    calls = []
    def work():
        calls.append(1)
        raise TimeoutError('timeout')
    with pytest.raises(TimeoutError):
        retry_collection(work, lambda *args: None, pause=lambda _: None)
    assert len(calls) == 3


def test_argument_context_restored_even_on_failure():
    before = sys.argv
    def work():
        assert sys.argv[1:] == ['--output', 'fixed']
        raise ValueError('bad artifact')
    with pytest.raises(ValueError):
        invoke(work, ['--output', 'fixed'])
    assert sys.argv is before
