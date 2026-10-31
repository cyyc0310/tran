"""Controlled ambiguity with a dense one-dimensional reference posterior.

The second standard-normal parameter is a common fuel-logit shift. It cancels
exactly from softmax and is integrated out analytically in the reference.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyMechanism:
    case: str
    amplitude: float = 1.6
    squared_observation_sd: float = 0.25
    sign_observation_sd: float = 0.25
    dimension: int = 2

    def __post_init__(self) -> None:
        if self.case not in {"harmful", "harmless", "resolved"}:
            raise ValueError(f"Unknown toy case: {self.case}")

    def historical_generation(self, z: np.ndarray) -> np.ndarray:
        """Explicit synthetic MW whose aggregate explains the observation.

        Total generation is constant within a day at 1 + z0**2. Opposite
        phases have identical fuel energies as well as identical total energy.
        Thus the magnitude information comes from total generation, NOT from
        fuel shares (which are exactly 1/2 over this canonical day).
        """
        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.dimension or not np.isfinite(z).all():
            raise ValueError("Finite toy inputs need shape (particles, 2).")
        pattern = np.sin(2 * np.pi * (np.arange(24) + .5) / 24)
        contrast = .65 * z[:, 0, None] * pattern
        logits = np.stack([contrast, -contrast], axis=-1)
        logits -= logits.max(axis=-1, keepdims=True)
        shares = np.exp(logits)
        shares /= shares.sum(axis=-1, keepdims=True)
        return (1 + z[:, 0, None, None] ** 2) * shares

    def historical_observations(self, z: np.ndarray) -> dict[str, np.ndarray]:
        generation = self.historical_generation(z)
        total_energy = generation.sum(axis=(1, 2))  # 24 one-hour intervals
        # The additional resolved-case statistic is an invertible historical
        # peak-hour fuel log-ratio, rather than a measurement of a hidden z.
        hour = 6
        pattern = np.sin(2 * np.pi * (hour + .5) / 24)
        peak_contrast = np.log(generation[:, hour, 0] / generation[:, hour, 1]) / (1.3 * pattern)
        return {"normalized_daily_total_minus_one": total_energy / 24 - 1,
                "daily_fuel_mwh": generation.sum(axis=1), "peak_contrast": peak_contrast}

    def evaluate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.dimension:
            raise ValueError("Toy inputs need shape (particles, 2).")
        likelihood = -0.5 * np.square((np.square(z[:, 0]) - self.amplitude**2) / self.squared_observation_sd)
        if self.case == "resolved":
            likelihood -= 0.5 * np.square((z[:, 0] - self.amplitude) / self.sign_observation_sd)
        parameter = np.square(z[:, 0]) / self.amplitude if self.case == "harmless" else z[:, 0]
        pattern = np.sin(2.0 * np.pi * (np.arange(24) + 0.5) / 24.0)
        contrast = 0.65 * parameter[:, None] * pattern[None, :]
        logits = np.stack((contrast, -contrast), axis=2) + 0.25 * z[:, 1, None, None]
        logits -= logits.max(axis=2, keepdims=True)
        shares = np.exp(logits)
        shares /= shares.sum(axis=2, keepdims=True)
        # A closed, nonnegative two-fuel mix with fixed synthetic emission factors.
        predictions = shares @ np.array([1000.0, 350.0])
        return likelihood, predictions

    def reference(self, points: int = 20001) -> dict[str, np.ndarray]:
        if points < 101 or points % 2 != 1:
            raise ValueError("Reference requires an odd grid with at least 101 points.")
        axis = np.linspace(-6.0, 6.0, points)
        z = np.column_stack((axis, np.zeros_like(axis)))
        likelihood, predictions = self.evaluate(z)
        log_weights = likelihood - 0.5 * np.square(axis)
        weights = np.exp(log_weights - log_weights.max())
        # Trapezoidal quadrature on a uniform grid; omitted prior tails < 2e-9.
        weights[[0, -1]] *= 0.5
        weights /= weights.sum()
        return {"z": z, "weights": weights, "predictions": predictions}

    def test_truths(self, count: int = 16, seed: int = 73021) -> np.ndarray:
        reference = self.reference()
        rng = np.random.default_rng(seed)
        indexes = rng.choice(len(reference["weights"]), size=count, p=reference["weights"])
        return reference["predictions"][indexes]
