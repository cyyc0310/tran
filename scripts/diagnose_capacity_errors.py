"""POST-HOC ORACLE DIAGNOSTIC ONLY: attribute existing errors, never export a forecast."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from transcif.research import capacity_transfer, multicountry
from run_cross_country import ROOT, digest


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    parent = ROOT / 'results/three-country-capacity-r2'
    table_path = ROOT / 'results/public-capacity-input-r1/capacity.json'
    table = capacity_transfer.load_capacity(table_path)
    paths = [Path(__file__), table_path, parent / 'prediction_seal.json']
    paths += list(parent.glob('*.joblib')) + list(parent.glob('*.csv')) + list(parent.glob('*_metadata.json'))
    for item in json.loads((parent / 'prediction_seal.json').read_text()):
        if digest(parent / item['path']) != item['sha256']:
            raise ValueError('parent prediction seal changed')
    for name in json.loads((parent / 'provenance.json').read_text()):
        if name != 'scripts/run_three_country.py':
            paths.append(ROOT / name)
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    raw, _ = multicountry.load_countries(ROOT)
    rows, cohorts = [], []
    for path in sorted(parent.glob('*_metadata.json')):
        tag = path.name.removesuffix('_metadata.json')
        target = json.loads(path.read_text())['target_country']
        model = joblib.load(parent / f'{tag}.joblib')
        for region, frame in raw.items():
            if frame.country.iloc[0] != target:
                continue
            months = multicountry.public_months(frame)
            public = multicountry.public_features(months, frame.clock)
            public['raw_row'] = public.index
            query = public.dropna(subset=list(multicountry.HISTORY))
            query = query.loc[query.clock.dt.year.eq(2023) & query.clock.dt.month.gt(6)]
            predicted = pd.read_csv(parent / f'{tag}_{region}.csv')
            assert len(predicted) == len(query)
            np.testing.assert_array_equal(predicted.time.to_numpy(),
                                          frame.loc[query.raw_row, 'time'].astype(str).to_numpy())
            cap = table['regions'][region]
            cap_ef = 400. + 600. * cap['coal_mw'] / (cap['coal_mw'] + cap['gas_mw'])
            fine = frame.loc[query.raw_row, list(multicountry.FINE)].to_numpy()
            true_thermal = fine[:, :4].sum(axis=1)
            s = true_thermal / fine.sum(axis=1)
            y = fine @ multicountry.FACTORS / fine.sum(axis=1)
            q = query[list(multicountry.GROUPS)].to_numpy() * model['profile'][query.clock.dt.hour.to_numpy()]
            estimated_s = q[:, 0] / q.sum(axis=1)
            baseline = cap_ef * estimated_s
            np.testing.assert_allclose(baseline, predicted.capacity_anchor_shaped, atol=1e-10, rtol=0.)
            hourly_ef = np.divide(fine @ multicountry.FACTORS, true_thermal,
                                  out=np.full(len(fine), cap_ef), where=true_thermal > 0)
            np.testing.assert_allclose(s * hourly_ef, y, atol=1e-10, rtol=0.)
            periods = query.clock.dt.to_period('M')
            for p in periods.unique():
                mask = periods.eq(p).to_numpy()
                eligible = p in months.index
                cohorts.append({'configuration': tag, 'region': region, 'period': str(p),
                                'forecast_hours': int(mask.sum()), 'full_current_month_available': eligible})
                if not eligible:
                    continue
                current = months.loc[p].to_numpy()
                oracle_q = current * model['profile'][query.loc[mask].clock.dt.hour.to_numpy()]
                oracle_month_s = oracle_q[:, 0] / oracle_q.sum(axis=1)
                full_month = frame.loc[frame.clock.dt.to_period('M').eq(p), list(multicountry.FINE)].to_numpy()
                thermal_sum = full_month[:, :4].sum()
                if thermal_sum <= 0:
                    raise ValueError('monthly effective factor undefined')
                month_ef = float((full_month @ multicountry.FACTORS).sum() / thermal_sum)
                candidates = {
                    'baseline': baseline[mask],
                    'ORACLE_current_month_coarse': cap_ef * oracle_month_s,
                    'ORACLE_hourly_coarse': cap_ef * s[mask],
                    'ORACLE_current_month_factor': month_ef * estimated_s[mask],
                    'ORACLE_hourly_factor': hourly_ef[mask] * estimated_s[mask],
                    'ORACLE_current_month_both': month_ef * oracle_month_s,
                    'ORACLE_hourly_both': hourly_ef[mask] * s[mask],
                }
                for name, values in candidates.items():
                    rows.append({'configuration': tag, 'target': target, 'region': region,
                                 'month': p.month, 'mode': name, 'hours': int(mask.sum()),
                                 'mae': float(np.abs(values - y[mask]).mean()),
                                 'bias': float((values - y[mask]).mean()),
                                 'capacity_ef': cap_ef, 'oracle_monthly_ef': month_ef})
    x = pd.DataFrame(rows)
    x.to_csv(out / 'monthly_oracle_diagnostics.csv', index=False)
    pd.DataFrame(cohorts).to_csv(out / 'cohorts.csv', index=False)
    fields = ['mae', 'bias', 'capacity_ef', 'oracle_monthly_ef']
    regional = x.groupby(['configuration', 'target', 'region', 'mode'])[fields].mean().reset_index()
    country = regional.groupby(['configuration', 'target', 'mode'])[fields].mean().reset_index()
    country.to_csv(out / 'country_oracle_diagnostics.csv', index=False)
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f'diagnostic input changed: {name}')
    (out / 'audit.json').write_text(json.dumps({
        'status': 'POST_HOC_TARGET_ORACLES_NOT_DEPLOYABLE_NOT_MODEL_RESULTS',
        'input_hashes': hashes, 'known_future_information_used_intentionally': True,
        'new_model_fitted': False, 'changes_to_frozen_predictions': False,
        'all_fine_labels_previous_development_targets': True,
        'comparison_cohort': 'same rows with >=99% valid current-month coverage',
        'limits': 'Oracle error reductions are non-additive and not achievable forecast gains',
    }, indent=2))
    print(country.loc[country.configuration.eq('US_to_DE')].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
