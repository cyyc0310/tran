"""Numerical checks independent of held-out hourly outcomes."""
import numpy as np

from scenario_reweighting import entropy_update


def main():
    rng = np.random.default_rng(164)
    cases = 0
    for _ in range(30):
        p = rng.uniform(.01, 1, 40)
        p /= p.sum()
        a = rng.uniform(0, 1000, 40)
        for target in [float(p @ a), float(np.mean(a)), -10., 1010.]:
            for cap in [None, .1]:
                q, info = entropy_update(p, a, target, cap)
                assert np.isfinite(q).all() and (q > 0).all()
                assert abs(q.sum() - 1) < 1e-12
                assert abs(info['achieved'] - q @ a) < 1e-10
                if cap is not None:
                    assert info['kl'] <= cap + 1e-10
                if cap is None and a.min() < target < a.max():
                    assert abs(q @ a - target) < 1e-7
                if target == float(p @ a):
                    assert np.max(abs(q - p)) < 1e-12
                cases += 1
    p = np.array([.2, .8])
    q, info = entropy_update(p, np.full(2, 400.), 500.)
    assert np.array_equal(p, q) and info['convex_hull_clipped']
    print({'assertion_failures': 0, 'random_tilt_cases': cases, 'constant_anchor_case': 1})


if __name__ == '__main__':
    main()
