"""Synthetic inverse problems with independently integrable active coordinates.

These are numerical inference tests, not calibrated simulators of real grids.
Samplers receive only ``evaluate`` and ambient dimension, never active axes or
quadrature. All coordinates have a standard normal prior. Nuisance coordinates
are integrated out exactly when constructing the two-dimensional reference.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class StressCase:
    name: str
    dimension: int = 2
    amplitude: float = 1.6
    noise: float = .25
    shift: float = 0.0
    curve: float = 0.0
    rotation_seed: int | None = None
    harmless: bool = False
    resolved: bool = False
    four_modes: bool = False
    joint_phase: bool = False
    age: float = 0.0
    transient: bool = False

    def __post_init__(self):
        if self.dimension < 2 or self.noise <= 0 or self.age < 0:
            raise ValueError("dimension >= 2, positive noise and nonnegative age required")

    @property
    def axes(self):
        if self.rotation_seed is None:
            return np.eye(self.dimension)[:, :2]
        matrix = np.random.default_rng(self.rotation_seed).normal(size=(self.dimension, self.dimension))
        return np.linalg.qr(matrix)[0][:, :2]

    def metadata(self):
        return asdict(self)

    def _outputs(self, v, w):
        if self.four_modes:
            # Historical normalized energy is 1 + a nonnegative quartic.
            residual = ((v * v - .8**2) * (v * v - 2.0**2)) / 2
        else:
            residual = v * v - self.amplitude**2
        likelihood = -.5 * (residual / self.noise) ** 2
        if self.resolved:
            # Additional synthetic historical phase-sensitive observation.
            likelihood -= .5 * ((v - self.amplitude) / .25) ** 2
        parameter = v * v / self.amplitude if self.harmless else v
        rho = np.exp(-self.age / 3)
        future = rho * parameter + np.sqrt(1 - rho**2) * w
        hour = np.arange(24) + .5
        sine = np.sin(2 * np.pi * hour / 24)
        contrast = .65 * future[:, None] * sine
        if self.joint_phase:
            contrast += .45 * w[:, None] * np.cos(4 * np.pi * hour / 24)
        if self.transient:
            pulse = np.exp(-.5 * ((hour - 17) / 1.3)**2)
            contrast += .55 * w[:, None] * pulse
        predictions = 350 + 650 * expit(2 * contrast)
        return likelihood, predictions

    def evaluate(self, z):
        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.dimension or not np.isfinite(z).all():
            raise ValueError("finite latent array with declared dimension required")
        active = z @ self.axes
        u, w = active.T
        v = u + self.shift + self.curve * (w * w - 1)
        return self._outputs(v, w)

    def reference(self, points=1201, secondary_points=161):
        """Trapezoid integration in transformed (v,w), unit Jacobian.

        Unlike prior-particle samplers this grid resolves the likelihood in v.
        Its cost is recorded separately and never counted as candidate cost.
        Reference refinement remains necessary; it is not exact arithmetic.
        """
        if points < 101 or secondary_points < 41 or points % 2 != 1 or secondary_points % 2 != 1:
            raise ValueError("odd grid sizes >=101 and >=41 required")
        v_axis = np.linspace(-8, 8, points)
        w_axis = np.linspace(-6, 6, secondary_points)
        v, w = np.meshgrid(v_axis, w_axis, indexing="ij")
        v, w = v.ravel(), w.ravel()
        u = v - self.shift - self.curve * (w * w - 1)
        log_likelihood, predictions = self._outputs(v, w)
        log_mass = log_likelihood - .5 * (u * u + w * w)
        mass = np.exp(log_mass - log_mass.max()).reshape(points, secondary_points)
        mass[[0, -1], :] *= .5
        mass[:, [0, -1]] *= .5
        weights = mass.ravel() / mass.sum()
        return {"predictions": predictions, "weights": weights, "v": v, "w": w,
                "quadrature_evaluations": points * secondary_points}


DEVELOPMENT_CASES = (
    StressCase("dev_harmful"),
    StressCase("dev_harmless", harmless=True),
    StressCase("dev_resolved", resolved=True),
    StressCase("dev_rotated6", dimension=6, rotation_seed=391),
    StressCase("dev_curved", curve=.4, joint_phase=True),
)

# All settings fixed before evaluating this suite. No selection or tuning on
# these cases is permitted within the same experiment/version.
STRESS_CASES = (
    StressCase("stress_unequal_modes", shift=.65),
    StressCase("stress_narrow_rotated", dimension=6, noise=.12, rotation_seed=192),
    StressCase("stress_four_modes", four_modes=True, noise=.2),
    StressCase("stress_curved_rotated12", dimension=12, curve=.55, rotation_seed=813, joint_phase=True),
    StressCase("stress_aged", dimension=6, rotation_seed=144, age=5, joint_phase=True),
    StressCase("stress_transient", dimension=6, rotation_seed=981, transient=True, shift=.3),
    StressCase("stress_harmless12", dimension=12, harmless=True, rotation_seed=512),
    StressCase("stress_resolved_curved", dimension=6, curve=.3, resolved=True, rotation_seed=265),
)
