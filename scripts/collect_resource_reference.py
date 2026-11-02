"""Acquire fixed 2016–2020 public climate, then aggregate complete local months."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from collect_weather_transfer_pilot import ROOT, COORDINATES, ZONES
from transcif.data.weather_reference import YEARS, acquire, parameters, monthly_resources, climatology, digest


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    zones = {**ZONES, 'Shanghai_deployment_probe': 'Asia/Shanghai'}
    plan = dict(years=list(YEARS), points={k: list(v) for k, v in COORDINATES.items()}, zones=zones,
                earliest_forecast_origin='2022-05-15T00:00:00Z',
                budget='W2 fixed historical climate; not realized weather in 2022/2023',
                original_public_availability_certified=False)
    plan_path = out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('existing reference plan differs')
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    code = [Path(__file__), ROOT/'src/transcif/data/weather_reference.py',
            ROOT/'scripts/collect_weather_transfer_pilot.py']
    seals = {str(p): digest(p) for p in code}
    rows, raw_seals = [], {}
    for region, (lat, lon) in COORDINATES.items():
        for year in YEARS:
            directory = out/f'{region}/{year}'
            frame, _ = acquire(directory, parameters(lat, lon, year))
            if frame.instant_at.max() >= pd.Timestamp(plan['earliest_forecast_origin']):
                raise ValueError('reference reaches forecast period')
            annual = monthly_resources(frame, year, zones[region]); annual['region'] = region
            rows.append(annual)
            raw_seals.update({str(p.relative_to(out)): digest(p) for p in
                              (directory/'response.json', directory/'receipt.json')})
            print(region, year, 'verified', len(frame), 'UTC hours; 12 full local months', flush=True)
    annual = pd.concat(rows, ignore_index=True)
    annual.to_csv(out/'annual_resources.csv', index=False)
    climatology(annual).to_csv(out/'climatology.csv', index=False)
    for name, expected in seals.items():
        if digest(Path(name)) != expected:
            raise ValueError('collector code changed during collection')
    files = {**raw_seals, **{name: digest(out/name) for name in ('plan.json', 'annual_resources.csv', 'climatology.csv')}}
    (out/'manifest.json').write_text(json.dumps(dict(files=files, code=seals, complete=True,
        years=list(YEARS), points=len(COORDINATES), region_year_months=len(annual),
        eligible_for_issued_public=False, available_at=None,
        full_reference=True, cif_experiment_completed=False), indent=2)+'\n')
    print('COMPLETE', len(annual), 'region-year-months', flush=True)


if __name__ == '__main__':
    main()
