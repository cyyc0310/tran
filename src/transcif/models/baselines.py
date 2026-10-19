"""Same-public-input ridge and published-mix constant references."""

import numpy as np


def public_features(examples):
    result = []
    for example in examples:
        x = example.values
        h = x["calendar"].shape[0]
        cap = x["capacity"] / x["capacity"].sum()
        config = np.r_[cap, x["mix"], x["ef"] / 1000, x["cost"] / 100]
        weather = x["weather"].mean(0) / [30, 15, 1000]
        demand = x["demand"].mean(0) / x["capacity"].sum()
        result.append(np.c_[weather, demand, x["calendar"], np.tile(config, (h, 1))])
    return np.asarray(result, dtype=float)


class PublicRidge:
    def __init__(self, ridge=1.):
        self.ridge = ridge

    def fit(self, examples, labels):
        features = public_features(examples).reshape(-1, 31)
        truth = np.stack([y["cif"] for y in labels]).ravel()
        self.mean, self.std = features.mean(0), features.std(0).clip(1e-6)
        x = np.c_[np.ones(len(features)), (features - self.mean) / self.std]
        penalty = np.eye(x.shape[1]) * self.ridge
        penalty[0, 0] = 0
        self.weight = np.linalg.solve(x.T @ x + penalty, x.T @ truth)
        return self

    def predict(self, examples):
        features = public_features(examples)
        x = np.concatenate([np.ones((*features.shape[:2], 1)),
                            (features - self.mean) / self.std], axis=-1)
        upper = np.array([e.values["ef"].max() for e in examples])[:, None]
        return np.clip(x @ self.weight, 0, upper)


class PublicBoosting:
    """MAE-trained nonlinear reference with the same 31 public input summaries.

    Fixed training budget; no random validation split, target IDs or target fit.
    This is not an EnsembleCI or foundation-model reproduction.
    """

    def __init__(self, seed=0):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError as exc:
            raise ImportError("tree baseline requires: pip install -e '.[trees]'") from exc
        self.model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300,
            learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.,
            early_stopping=False, random_state=seed)

    def fit(self, examples, labels):
        features = public_features(examples)
        truth = np.stack([y["cif"] for y in labels])
        from threadpoolctl import threadpool_limits
        regions = [e.region for e in examples]
        counts = {name: regions.count(name) for name in set(regions)}
        weights = np.repeat([1 / counts[name] for name in regions], features.shape[1])
        with threadpool_limits(limits=1):
            self.model.fit(features.reshape(-1, 31), truth.ravel(), sample_weight=weights / weights.mean())
        return self

    def predict(self, examples):
        from threadpoolctl import threadpool_limits
        features = public_features(examples)
        with threadpool_limits(limits=1):
            point = self.model.predict(features.reshape(-1, 31)).reshape(features.shape[:2])
        upper = np.array([e.values["ef"].max() for e in examples])[:, None]
        return np.clip(point, 0, upper)


def mix_constant(examples):
    return np.stack([np.full(e.values["demand"].shape[-1],
                            e.values["mix"] @ e.values["ef"]) for e in examples])


def residual_scenarios(point, residuals, examples, size=64, seed=0):
    """Source-validation residual PATHS only; no target residual calibration."""
    if residuals.ndim != 2 or residuals.shape[1] != point.shape[-1] or len(residuals) == 0:
        raise ValueError("source residual paths must match the horizon")
    rng = np.random.default_rng(seed)
    draws = residuals[rng.integers(len(residuals), size=(len(point), size))]
    upper = np.array([e.values["ef"].max() for e in examples])[:, None, None]
    return np.clip(point[:, None] + draws, 0, upper)
