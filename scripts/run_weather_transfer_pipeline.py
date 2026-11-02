"""Sequential seasonal pilot with bounded network recovery and no PID watchers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError

import analyze_weather_transfer_pilot as analysis
import collect_weather_transfer_pilot as collection
import run_weather_transfer_pilot as modeling


def retry_collection(call, notify, *, attempts: int = 3, pause=time.sleep):
    """Retry transient network failures only; data/schema/hash errors fail closed."""
    if not isinstance(attempts, int) or isinstance(attempts, bool) or not 1 <= attempts <= 3:
        raise ValueError('attempts must be 1..3')
    for attempt in range(1, attempts+1):
        try:
            return call()
        except (URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code != 429 and error.code < 500:
                raise
            if attempt == attempts:
                raise
            notify(attempt, error)
            pause(10.*attempt)


def invoke(main, arguments: list[str]) -> None:
    original = sys.argv
    try:
        sys.argv = [main.__module__, *arguments]
        main()
    finally:
        sys.argv = original


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args(); cache = args.cache.resolve()
    state_path, journal = cache/'pipeline_state.json', cache/'pipeline_events.jsonl'
    if journal.exists():
        raise FileExistsError('pipeline journal already exists; inspect the previous execution before resuming')

    def update(stage: str, **values) -> None:
        state = {'stage': stage, 'updated_at': datetime.now(timezone.utc).isoformat(), **values}
        with journal.open('a') as handle:
            handle.write(json.dumps(state)+'\n')
        temporary = cache/'.pipeline_state.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(state_path)
        print(json.dumps(state), flush=True)

    if args.models.exists() or args.analysis.exists():
        raise FileExistsError('fresh model and analysis outputs required')
    model_receipt = json.loads((cache/'reviewed_execution_receipt.json').read_text())
    analysis_receipt = json.loads((cache/'pre_score_analysis_receipt.json').read_text())

    def verify_reviewed_code() -> None:
        for name, expected in model_receipt['files'].items():
            if collection.digest(collection.ROOT/name) != expected:
                raise ValueError('reviewed model code changed')
        if collection.digest(Path(analysis_receipt['analyzer'])) != analysis_receipt['analyzer_sha256']:
            raise ValueError('pre-score analysis code changed')

    verify_reviewed_code()
    receipt = {str(Path(__file__).resolve()): collection.digest(Path(__file__)),
        str(cache/'reviewed_execution_receipt.json'): collection.digest(cache/'reviewed_execution_receipt.json'),
        str(cache/'pre_score_analysis_receipt.json'): collection.digest(cache/'pre_score_analysis_receipt.json')}
    (cache/'pipeline_execution_seal.json').write_text(json.dumps(receipt, indent=2)+'\n')
    try:
        update('collecting', checkpoint_policy='completed compact windows verified on every attempt')
        retry_collection(lambda: invoke(collection.main, ['--output', str(cache)]),
            lambda attempt, error: update('retrying_collection', failed_attempt=attempt,
                                         error_type=type(error).__name__, error=str(error)))
        verify_reviewed_code()
        update('running_real_weather_models')
        invoke(modeling.main, ['--cache', str(cache), '--output', str(args.models.resolve())])
        update('models_complete')
        verify_reviewed_code()
        update('running_analysis')
        invoke(analysis.main, ['--experiment', str(args.models.resolve()), '--output', str(args.analysis.resolve())])
        for name, expected in receipt.items():
            if collection.digest(Path(name)) != expected:
                raise RuntimeError('pipeline code/receipt changed during execution')
        update('analysis_complete', output=str(args.analysis.resolve()))
    except BaseException as error:
        update('pipeline_failed', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
