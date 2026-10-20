"""Same-public-input ridge and published-mix constant references."""

import numpy as np

from transcif.physics.astro import wind_capacity_factor


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


def public_dispatch_features(examples):
    """Same public information plus the neural context encoder's seven power features."""
    base, physical = public_features(examples), []
    for example in examples:
        x = example.values
        fractions = x["capacity"] / x["capacity"].sum()
        wind = wind_capacity_factor(x["weather"][..., 1]).mean(0)
        solar = np.clip(.85 * x["weather"][..., 2] / 1000, 0, 1).mean(0)
        load = x["demand"].mean(0) / x["capacity"].sum()
        wind_potential, solar_potential = fractions[4] * wind, fractions[5] * solar
        residual = load - fractions[2] - wind_potential - solar_potential
        thermal = max(fractions[[0, 1, 3]].sum(), 1e-4)
        ramp = np.r_[0., np.diff(load)]
        physical.append(np.stack([wind, solar, wind_potential, solar_potential,
                                  residual, np.clip(residual / thermal, -5, 5), ramp], -1))
    return np.concatenate([base, np.asarray(physical)], -1)


class PublicGenerationBoosting:
    """Capacity-normalized source fuel supervision, fixed public EFs at inference.

    Six MAE boosting models predict fuel capacity factors from public features.
    Bounded output is not claimed to satisfy demand balance or joint dispatch.
    This matches the neural model's source generation supervision information.
    """

    def __init__(self, seed=0):
        self.models = [PublicBoosting(seed + fuel).model for fuel in range(6)]

    def fit(self, examples, labels):
        from threadpoolctl import threadpool_limits
        features = public_dispatch_features(examples)
        capacity = np.stack([e.values["capacity"] for e in examples])
        generation = np.stack([label["generation"] for label in labels])
        regions = [e.region for e in examples]
        counts = {name: regions.count(name) for name in set(regions)}
        weights = np.repeat([1 / counts[name] for name in regions], features.shape[1])
        with threadpool_limits(limits=1):
            for fuel, model in enumerate(self.models):
                supported = np.repeat(capacity[:, fuel] > 0, features.shape[1])
                if not supported.any():
                    self.models[fuel] = None
                    continue
                factor = generation[..., fuel] / np.maximum(capacity[:, None, fuel], 1e-12)
                model.fit(features.reshape(-1, features.shape[-1])[supported], factor.ravel()[supported],
                          sample_weight=weights[supported] / weights[supported].mean())
        return self

    def predict_generation(self, examples):
        from threadpoolctl import threadpool_limits
        features = public_dispatch_features(examples)
        capacity = np.stack([e.values["capacity"] for e in examples])
        generation = np.zeros((*features.shape[:2], 6))
        with threadpool_limits(limits=1):
            for fuel, model in enumerate(self.models):
                if model is None:
                    if (capacity[:, fuel] > 0).any():
                        raise ValueError("target fuel has no source capacity-factor supervision")
                    continue
                factor = model.predict(features.reshape(-1, features.shape[-1])).reshape(features.shape[:2])
                generation[..., fuel] = np.clip(factor, 0, 1) * capacity[:, None, fuel]
        return generation

    def predict(self, examples):
        generation = self.predict_generation(examples)
        if (generation.sum(-1) <= 0).any():
            raise ValueError("positive predicted generation required for generation-boosting CIF")
        ef = np.stack([example.values["ef"] for example in examples])
        return (generation * ef[:, None]).sum(-1) / generation.sum(-1)


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
