# Forgione–Piga continuous-time neural system-identification source

This directory pins the upstream model source and its Cascaded Tanks training example for the author-model comparison. The source repository is licensed under Apache-2.0. This directory preserves the license and unmodified upstream files; project adaptations live in the project `scripts/` and protocol documents.

- Upstream repository: <https://github.com/forgi86/sysid-neural-continuous>
- Upstream commit: `956aa7ecd73061b0321c6f6098d74c9e4a7ce67a`
- Paper: Forgione, M. & Piga, D. (2021). “Continuous-time system identification with neural networks: Model structures and fitting criteria.” *European Journal of Control*, 59, 69–81. <https://doi.org/10.1016/j.ejcon.2021.01.008>
- Dataset: the upstream `examples/CTS/data/dataBenchmark.csv` is byte-for-byte equal to `CascadedTanksFiles/dataBenchmark.csv` in this project’s CC BY-SA 4.0 4TU archive. Both have 30,014 bytes and SHA-256 `ef2388ed822f3aef4aa80d6b0f2b466dd80b361786b3eafc7a2957c31ea323a7`.
- Preserved upstream files: `torchid/ssmodels_ct.py`, `examples/CTS/CTS_fit_truncated.py`, and `LICENSE.md`.

SHA-256 values:

| File | SHA-256 |
|---|---|
| `torchid/ssmodels_ct.py` | `cc15c6f5be2849b3d52a7488344cf8cb832a371fc97dcd5bdfae520d4ad920d7` |
| `examples/CTS/CTS_fit_truncated.py` | `02576aa7908d0f3e2cd6664dfa4c0a3314c67fa76f5d616e779ff3e5a7e1aca0` |
| `LICENSE.md` | `fc7e8ab1c7a5059c25186b88b1c8768c9b5d8ed088e45af07c6d5e1674f94a2d` |
