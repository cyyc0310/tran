"""Verify probability lifting properties without hourly evaluation labels."""
import numpy as np

from coarse_distribution_lifting import lift_probabilities


def main():
    rng = np.random.default_rng(165)
    cases = 0
    for _ in range(50):
        a = np.r_[rng.uniform(0, 1000, 20), [400., 400.]]
        p = rng.uniform(.01, 1, len(a))
        p /= p.sum()
        z = rng.uniform(-100, 1100, 100)
        w = rng.uniform(.01, 1, len(z))
        w /= w.sum()
        for nearest in [False, True]:
            q, info = lift_probabilities(a, p, z, w, nearest)
            assert np.isfinite(q).all() and (q > 0).all() and abs(q.sum() - 1) < 1e-12
            assert abs(q[-1] / q[-2] - p[-1] / p[-2]) < 1e-10
            if not nearest:
                assert abs(q @ a - w @ np.clip(z, a.min(), a.max())) < 1e-8
            cases += 1
    a, p = np.array([400., 400.]), np.array([.2, .8])
    q, _ = lift_probabilities(a, p, np.array([0., 1000.]), np.array([.4, .6]))
    assert np.max(abs(q - p)) < 1e-12
    q, _ = lift_probabilities(a + np.array([-1e-13, 1e-13]), p, np.array([0., 1000.]), np.array([.4, .6]))
    assert np.max(abs(q - p)) < 1e-12
    print({'assertion_failures': 0, 'random_projection_cases': cases, 'constant_anchor_case': 1})


if __name__ == '__main__':
    main()
