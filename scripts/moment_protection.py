"""Certified finite-support KL projection onto prior linear moments."""
from __future__ import annotations

import numpy as np


TOLERANCE = 1e-10


class MomentGeometry:
    """Reuse a prior's centered moment geometry for different proposed weights."""

    def __init__(self, prior, features, normalizer):
        prior, features = np.asarray(prior, float), np.asarray(features, float)
        if (prior.ndim != 1 or features.ndim != 2 or len(features) != len(prior) or not len(prior)
                or not np.isfinite(prior).all() or (prior <= 0).any() or abs(prior.sum() - 1) > 1e-12
                or not np.isfinite(features).all() or not np.isfinite(normalizer) or normalizer <= 0):
            raise ValueError('invalid moment geometry')
        self.prior = prior / prior.sum()
        features = features.copy()
        self.features = features
        self.mean = self.prior @ features
        self.normalizer = float(normalizer)
        span = np.ptp(features, axis=0)
        if not np.isfinite(span).all():
            raise ValueError('nonfinite feature range')
        active = span > 0
        centered = (features[:, active] - self.mean[active]) / span[active]
        if active.any():
            _, singular, right = np.linalg.svd(centered, full_matrices=False)
            keep = singular > singular[0] * 1e-13
            coordinates = centered @ right[keep].T
            covariance = coordinates.T @ (self.prior[:, None] * coordinates)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            # This invertible preconditioner changes no moment constraint.
            floor = max(1e-12, float(eigenvalues[-1]) * 1e-12)
            self.coordinates = (coordinates @ eigenvectors) / np.sqrt(np.maximum(eigenvalues, floor))
            self.coordinates -= self.prior @ self.coordinates
        else:
            self.coordinates = np.empty((len(prior), 0))
        self.rank = self.coordinates.shape[1]

    def error(self, weights):
        return float(np.max(abs(weights @ self.features - self.mean), initial=0.) / self.normalizer)

    def project(self, proposed):
        proposed = np.asarray(proposed, float)
        if (proposed.shape != self.prior.shape or not np.isfinite(proposed).all()
                or (proposed <= 0).any() or abs(proposed.sum() - 1) > 1e-12):
            raise ValueError('invalid proposed probability')
        proposed = proposed / proposed.sum()
        if self.error(proposed) <= TOLERANCE:
            return proposed, {'status': 'already_feasible', 'iterations': 0, 'rank': self.rank,
                              'moment_error': self.error(proposed), 'stationarity_error': 0.,
                              'pythagorean_error': 0., 'used_prior_fallback': False}
        Z, logq = self.coordinates, np.log(proposed)
        theta = np.zeros(self.rank)

        def evaluate(value):
            logits = logq + Z @ value
            maximum = float(logits.max())
            mass = np.exp(logits - maximum)
            total = mass.sum()
            weights = mass / total
            objective = maximum + np.log(total)
            gradient = weights @ Z
            centered = Z - gradient
            hessian = centered.T @ (weights[:, None] * centered)
            return float(objective), gradient, hessian, weights

        def objective_only(value):
            logits = logq + Z @ value
            maximum = float(logits.max())
            return maximum + np.log(np.exp(logits - maximum).sum())

        status, iterations = 'iteration_limit', 0
        for iterations in range(101):
            objective, gradient, hessian, weights = evaluate(theta)
            if self.error(weights) <= TOLERANCE:
                # Only subnormal machine mass is restored after underflow.
                result = np.maximum(weights, np.nextafter(0., 1.))
                result /= result.sum()
                logratio = np.log(result) - logq
                stationarity = float(result @ abs(logratio - Z @ theta + objective))
                pythagorean = float(abs(self.prior @ (np.log(self.prior) - logq)
                                        - self.prior @ (np.log(self.prior) - np.log(result))
                                        - result @ logratio))
                if ((result > 0).all() and self.error(result) <= TOLERANCE
                        and max(stationarity, pythagorean) <= 1e-8):
                    return result, {'status': 'certified', 'iterations': iterations, 'rank': self.rank,
                                    'moment_error': self.error(result), 'stationarity_error': stationarity,
                                    'pythagorean_error': pythagorean, 'used_prior_fallback': False}
            if iterations == 100 or not self.rank:
                break
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            except np.linalg.LinAlgError:
                status = 'hessian_eigendecomposition_failed'
                break
            floor = max(1e-14, float(eigenvalues[-1]) * 1e-12)
            step = -eigenvectors @ ((eigenvectors.T @ gradient) / np.maximum(eigenvalues, floor))
            length = float(np.linalg.norm(step))
            if length > 50:
                step *= 50 / length
            slope = float(gradient @ step)
            if not np.isfinite(step).all() or slope >= 0:
                status = 'non_descent_step'
                break
            accepted = False
            fraction = 1.
            for _ in range(30):
                trial = objective_only(theta + fraction * step)
                if np.isfinite(trial) and trial <= objective + 1e-4 * fraction * slope + 1e-14 * max(1., abs(objective)):
                    theta += fraction * step
                    accepted = True
                    break
                fraction *= .5
            if not accepted:
                status = 'line_search_failed'
                break
        # This is an explicit, recorded return to the reference law. It is not
        # reported as a solved projection or silently accepted approximation.
        return self.prior.copy(), {'status': status, 'iterations': iterations, 'rank': self.rank,
                                   'moment_error': self.error(self.prior), 'stationarity_error': None,
                                   'pythagorean_error': None, 'used_prior_fallback': True}
