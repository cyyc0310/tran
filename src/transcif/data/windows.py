"""Sliding-window construction for TransCIF sequences.

Builds (x_rs, y_rs, y_cif) windows from a region's renewable-share and CIF
timeseries. Used by both training (stride = TRAIN_STRIDE) and evaluation
(stride = TEST_STRIDE).
"""

import numpy as np

from transcif.config import SEQ_LEN, HORIZON, TRAIN_STRIDE


def build_windows(rs, cif, seq_len=SEQ_LEN, horizon=HORIZON, stride=TRAIN_STRIDE):
    """Build (x_rs, y_rs, y_cif) sliding windows.

    Args:
        rs     : renewable-share array (float32)
        cif    : carbon-intensity array (float32)
        seq_len: input history length
        horizon: forecast horizon
        stride : window step

    Returns stacked arrays ``x_rs`` (n, seq_len), ``y_rs`` (n, horizon),
    ``y_cif`` (n, horizon). Empty arrays with correct trailing dims if the
    series is too short.
    """
    window = seq_len + horizon
    x_rs, y_rs, y_cif = [], [], []
    for start in range(0, len(rs) - window + 1, stride):
        x_rs.append(rs[start:start + seq_len])
        y_rs.append(rs[start + seq_len:start + window])
        y_cif.append(cif[start + seq_len:start + window])
    if not x_rs:
        return np.empty((0, seq_len)), np.empty((0, horizon)), np.empty((0, horizon))
    return np.stack(x_rs), np.stack(y_rs), np.stack(y_cif)
