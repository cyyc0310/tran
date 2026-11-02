"""Cache public SMARD generation series; preserve every original fuel category."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

FILTERS = {'1223': 'lignite', '1224': 'nuclear', '1225': 'wind_offshore',
           '1226': 'hydro', '1227': 'other_conventional', '1228': 'other_renewable',
           '4066': 'biomass', '4067': 'wind_onshore', '4068': 'solar',
           '4069': 'hard_coal', '4070': 'pumped_storage', '4071': 'gas'}
BASE = 'https://www.smard.de/app/chart_data'


def cache_json(url: str, path: Path) -> dict:
    expected = 'timestamps' if url.endswith('index_hour.json') else 'series'
    def validate(content: bytes) -> dict:
        data = json.loads(content)
        if not isinstance(data, dict) or not isinstance(data.get(expected), list):
            raise ValueError(f'invalid SMARD {expected} schema: {url}')
        if expected == 'timestamps' and not all(isinstance(t, int) for t in data[expected]):
            raise ValueError(f'invalid SMARD timestamp values: {url}')
        return data
    if path.exists():
        content = path.read_bytes()
        try:
            return validate(content)
        except ValueError:
            invalid = path.with_name(path.stem+'.invalid-'+hashlib.sha256(content).hexdigest()[:12]+'.json')
            path.replace(invalid)
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                content = response.read()
            break
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 1:
                raise
            time.sleep(1)
    data = validate(content)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--limit-weeks', type=int)
    parser.add_argument('--cache', type=Path, default=Path('data/smard-de/raw'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    collector = Path(__file__).read_bytes()
    args.cache.mkdir(parents=True, exist_ok=True)
    start = int(datetime(args.year, 1, 1, tzinfo=timezone.utc).timestamp()*1000)
    end = int(datetime(args.year+1, 1, 1, tzinfo=timezone.utc).timestamp()*1000)
    tasks = []
    errors = []
    for code in FILTERS:
        try:
            index = cache_json(f'{BASE}/{code}/DE/index_hour.json', args.cache/f'{code}_index_hour.json')
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({'stage': 'index', 'filter': code, 'error': str(exc)})
            continue
        weeks = [t for t in index['timestamps'] if start-8*86400_000 < t < end]
        if args.limit_weeks is not None:
            weeks = weeks[:args.limit_weeks]
        tasks.extend((code, t) for t in weeks)
    def fetch(task):
        code, stamp = task
        name = f'{code}_DE_hour_{stamp}.json'
        path = args.cache/name
        url = f'{BASE}/{code}/DE/{name}'
        data = cache_json(url, path)
        if 'series' not in data or not isinstance(data['series'], list):
            raise ValueError(f'bad series: {url}')
        return {'filter': code, 'category': FILTERS[code], 'url': url, 'path': str(path),
                'records': len(data['series']), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(task, pool.submit(fetch, task)) for task in tasks]
        for task, future in futures:
            try:
                records.append(future.result())
            except Exception as exc:
                errors.append({'task': task, 'error': str(exc)})
    manifest = {'status': 'RAW_GENERATION_CACHE_ONLY' if not errors else 'PARTIAL_DOWNLOAD',
                'year': args.year, 'limit_weeks': args.limit_weeks,
                'attribution': 'Bundesnetzagentur | SMARD.de',
                'licence_source': 'https://www.smard.de/en/datennutzung',
                'mapping_not_yet_harmonized': True, 'records': records, 'errors': errors}
    manifest['collector_sha256'] = hashlib.sha256(collector).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.collector.py').write_bytes(collector)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'files': len(records), 'errors': len(errors), 'output': str(args.output)}))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
