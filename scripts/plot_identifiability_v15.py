"""Present frozen structural diagnostics; never select a prediction method."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/identifiability-v15'


def main() -> None:
    """Verify both completed phases and export a zero-origin comparison plot."""
    for stage in ('source', 'target'):
        receipt = json.loads((DATA/f'{stage}_completed.json').read_text())
        for name, expected in receipt['output_hashes'].items():
            with Path(name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('Changed phase artifact: '+name)
    macro = {s: pd.read_csv(DATA/f'{s}_macro.csv').query("ef == 'proxy'").set_index('budget')
             for s in ('source', 'target')}
    regional = pd.read_csv(DATA/'target_region.csv').query("ef == 'proxy'")
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ['#4878A8', '#DA8B32']
    coarse = ['month', 'day', '12h', '6h', '3h', '1h']
    labels = ['Month*', 'Day', '12 h', '6 h', '3 h', '1 h']
    for ax, metric, title in (
        (axes[0, 0], 'hour_width', 'A. Hourly ambiguity after exact fuel totals + hourly G'),
        (axes[0, 1], 'daily_mean_width', 'B. Ambiguity in arithmetic daily CIF mean'),
    ):
        x = np.arange(len(coarse))
        for i, stage in enumerate(('source', 'target')):
            ax.bar(x+(i-.5)*.36, macro[stage].loc[coarse, metric], .36,
                   label='Sources 2025 H1' if stage == 'source' else 'Viewed targets 2026 H1', color=colors[i])
        ax.set_xticks(x, labels)
        ax.set_ylim(bottom=0)
        ax.set_ylabel('Mean sharp width (g/kWh)')
        ax.set_title(title, loc='left', fontsize=11)
        ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(frameon=False)
    names = ['day', 'day_coal', 'day_gas', 'day_emitting', 'day_coal_gas', 'day_emitting_coal']
    short = ['None', 'Coal', 'Gas', 'Emitting', 'Coal + gas', 'Emitting + coal']
    ax = axes[1, 0]
    x = np.arange(len(names))
    for i, stage in enumerate(('source', 'target')):
        ax.bar(x+(i-.5)*.36, macro[stage].loc[names, 'hour_width'], .36, color=colors[i])
    ax.set_xticks(x, short, rotation=18, ha='right')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Mean hourly sharp width (g/kWh)')
    ax.set_title('C. Additional hourly channel, given exact daily fuels + G', loc='left', fontsize=11)
    ax.grid(axis='y', alpha=.2)
    ax = axes[1, 1]
    for i, name in enumerate(('day_emitting_coal', 'hour_emitting_coal_only')):
        values = []
        for region in ('PACE', 'LGEE'):
            row = regional.loc[regional.region.eq(region) & regional.budget.eq(name), 'hour_width']
            if len(row) != 1:
                raise ValueError('Unique regional diagnostic required')
            values.append(float(row.iloc[0]))
        values.append(float(macro['target'].loc[name, 'hour_width']))
        ax.bar(np.arange(3)+(i-.5)*.36, values, .36,
               label='With exact daily fuel totals' if i == 0 else 'Without any fuel totals', color=colors[i])
    ax.set_xticks(np.arange(3), ['PACE', 'LGEE', 'Target macro'])
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Mean hourly sharp width (g/kWh)')
    ax.set_title('D. Emitting + coal channels: dependence on daily fuel truth', loc='left', fontsize=11)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.2)
    fig.suptitle('Information sufficiency for the fixed-EF generation proxy', fontsize=15)
    fig.text(.05, .035, 'Retrospective oracle constraints, not predictive intervals or coverage gains. '
             '*Month uses retained complete valid days only.\n'
             'Equal-region macro means; no capacity/ramp assumptions; '
             'source choices frozen before target diagnosis. Zero-origin axes.', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .96), h_pad=3)
    for suffix in ('png', 'pdf'):
        fig.savefig(DATA/f'identifiability_v15.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
