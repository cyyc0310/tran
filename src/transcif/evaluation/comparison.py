"""Paired grid/group and time-block resampling for point-error comparisons."""

import numpy as np


def paired_block_comparison(reference_errors, candidate_errors, groups=None,
                            block_windows=7, draws=2000, seed=0):
    """Errors have axes grid, training seed, ordered forecast window, horizon.

    Resample independent grids or whole jurisdictions, and circular moving
    blocks of windows. All grids/methods share each temporal draw, preserving
    common time shocks. Training seeds are averaged, not treated as new grids.
    RMSE is computed per grid/seed before macro averaging, as in benchmark.py.
    Intervals describe this empirical resampling scheme, not arbitrary domains.
    """
    a, b = np.asarray(reference_errors, dtype=float), np.asarray(candidate_errors, dtype=float)
    if (a.shape != b.shape or a.ndim != 4 or min(a.shape) < 1 or
            not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("aligned finite errors with axes grid, seed, window, horizon required")
    if (not isinstance(block_windows, int) or not 1 <= block_windows <= a.shape[2] or
            not isinstance(draws, int) or draws < 2 or seed < 0):
        raise ValueError("valid block length, draws>=2 and nonnegative seed required")
    groups = list(range(len(a))) if groups is None else list(groups)
    if len(groups) != len(a):
        raise ValueError("one independent-group label per grid required")
    units = list(dict.fromkeys(groups))
    indices = [np.array([i for i, group in enumerate(groups) if group == unit]) for unit in units]
    windows = a.shape[2]
    # Store per-window errors; do not repeatedly expand the hourly axis.
    summaries = [(np.abs(e).mean(-1), np.square(e).mean(-1)) for e in (a, b)]

    def evaluate(window_indices):
        methods = []
        for absolute, squared in summaries:
            grid_mae = absolute[:, :, window_indices].mean(-1).mean(-1)
            grid_rmse = np.sqrt(squared[:, :, window_indices].mean(-1)).mean(-1)
            methods.append({"mae": np.array([grid_mae[index].mean() for index in indices]),
                            "rmse": np.array([grid_rmse[index].mean() for index in indices])})
        return methods

    point = evaluate(np.arange(windows))
    samples = {key: [] for key in ("mae", "rmse")}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        selected_units = rng.integers(len(units), size=len(units))
        starts = rng.integers(windows, size=(windows + block_windows - 1) // block_windows)
        selected_windows = ((starts[:, None] + np.arange(block_windows)) % windows).ravel()[:windows]
        left, right = evaluate(selected_windows)
        for key in samples:
            samples[key].append(float((right[key][selected_units] - left[key][selected_units]).mean()))
    metrics = {}
    for key, distribution in samples.items():
        metrics[key] = {"reference": float(point[0][key].mean()), "candidate": float(point[1][key].mean()),
            "candidate_minus_reference": float((point[1][key] - point[0][key]).mean()),
            "percentile_95_interval": np.quantile(distribution, [.025, .975]).tolist()}
    return {"metrics": metrics, "independent_groups": len(units), "grids": len(a),
        "training_seeds": a.shape[1], "windows": windows, "horizon": a.shape[3],
        "block_windows": block_windows, "draws": draws, "seed": seed,
        "aggregation": "mean within grid/seed, then fixed training seeds, grids within group, groups",
        "limitations": ["circular moving blocks assume stationary regular forecast-window sampling",
            "training seeds are fixed and averaged; they do not increase the number of independent grids",
            "empirical intervals do not certify new-domain generalization or adjust for development selection",
            "few independent groups or short test periods make interval estimates unstable"]}
