# Pinned NFourSID source

This directory preserves the Python NFourSID implementation used for the Cascaded Tanks sparse-output state-space baseline.

- Upstream: <https://github.com/spmvg/nfoursid>
- Pinned source commit: `1e306879f9e697cd275f17d01dc96182b74ac97d`
- Upstream package release: `nfoursid 1.0.2`
- License: MIT; the upstream license is preserved as `LICENSE`.
- This implementation is an independent Python N4SID implementation. It is not MATLAB System Identification Toolbox and must not be described as an exact reproduction of the Forgione–Piga MATLAB result.

The upstream Python package source files are copied without modification under `src/nfoursid/`.

| Source file | SHA-256 |
|---|---|
| `src/nfoursid/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/nfoursid/kalman.py` | `f3ae6d2f597a682313159c202575b3b5f0e5341eff816f719d1948e70ff4bfec` |
| `src/nfoursid/nfoursid.py` | `a3e32e6d08c4ca7ea7d247111841b3e5bbd44f0b4daf05686615ab892f309565` |
| `src/nfoursid/state_space.py` | `75d2febc24b145dc566bdc0824b9660c85d794200256384677f7b7515aebfdb8` |
| `src/nfoursid/utils.py` | `bcc555ae51fd96ae522880d934cb3d475309a03c0ce78f6bb92158fc72918d25` |
| `LICENSE` | `c0615b31b9d61f4359b332390b5920b921e73ad5ea324a9df129ee28b9f89637` |

The NFourSID implementation estimates a stochastic state-space model and covariance matrix. The benchmark script uses its documented Kalman filter for sparse measurement updates, initialized with the least-squares source-record initial state estimate. See [the upstream algorithm documentation](https://nfoursid.readthedocs.io/en/latest/index.html) and [the pinned API source](https://github.com/spmvg/nfoursid/tree/1e306879f9e697cd275f17d01dc96182b74ac97d).
