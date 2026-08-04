"""TransCIF: zero-shot cross-region carbon intensity forecasting.

A physics-informed framework for predicting carbon intensity (CIF) in
data-scarce regions using only two physical scalars (renewable share mean
and non-renewable emission factor) plus real-time renewable share.

Package layout:
    config      Global constants and region configuration loaders
    data        Data loading, windowing, dataset wrappers
    physics     Physics decomposition layer (Theorem 1/2)
    models      Base models and zero-shot research directions (zeroshot/)
    calibration Test-time calibration (ZS+ / conformal)
    evaluation  Metrics and LORO experiment orchestration
    training    Training utilities (schedulers, etc.)
"""

__version__ = "0.1.0"
