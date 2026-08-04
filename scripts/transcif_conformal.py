"""Conformal prediction with adaptive online & state-conditioned calibration.

P2 items from IMPROVEMENT_PLAN.md:
  - Split-conformal (existing, preserved)
  - Per-horizon conformal
  - Adaptive Online Conformal (AOC) with forgetting factor
  - State-Conditioned Conformal: per-regime calibration

Usage:
    from transcif_conformal import (
        split_conformal_calibrate,
        split_conformal_calibrate_per_horizon,
        AdaptiveOnlineConformal,
        StateConditionedConformal,
        compute_crps, compute_coverage, compute_interval_width,
    )
"""

import numpy as np


# ---------------------------------------------------------------------------
# Split-conformal (existing – reference implementation)
# ---------------------------------------------------------------------------

def split_conformal_calibrate(y_true_cal, y_pred_cal, coverage=0.90):
    """Compute conformal prediction half-width from calibration residuals.

    Uses the standard split-conformal method with MEAN score across horizons.
    q = ceil((n+1) * coverage) / n quantile of mean(|y - y_hat|).
    """
    scores = np.abs(y_true_cal - y_pred_cal)
    if scores.ndim == 2:
        scores = scores.mean(axis=1)
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * coverage) / n)
    halfwidth = float(np.quantile(scores, q_level, method="higher"))
    return halfwidth, scores


def split_conformal_calibrate_per_horizon(y_true_cal, y_pred_cal, coverage=0.90):
    """Per-horizon conformal intervals (tighter than global)."""
    assert y_true_cal.ndim == 2 and y_pred_cal.ndim == 2
    n, h = y_true_cal.shape
    halfwidths = np.zeros(h)
    q_level = min(1.0, np.ceil((n + 1) * coverage) / n)
    for t in range(h):
        scores = np.abs(y_true_cal[:, t] - y_pred_cal[:, t])
        halfwidths[t] = float(np.quantile(scores, q_level, method="higher"))
    return halfwidths


# ---------------------------------------------------------------------------
# Adaptive Online Conformal (AOC)
# ---------------------------------------------------------------------------

class AdaptiveOnlineConformal:
    """Adaptive online conformal prediction with forgetting factor.

    Reference: Zaffran et al., "Adaptive Conformal Predictions for Time Series",
    ICML 2022.

    Instead of a fixed calibration quantile, this maintains a rolling window of
    recent residuals and updates the half-width at each time step so that the
    running empirical coverage tracks the nominal level.

    Key parameters:
        coverage    : target marginal coverage (e.g. 0.90)
        gamma       : step size for quantile updates (learning rate)
        window      : rolling window size for running coverage estimate
        forget      : exponential forgetting factor (0 < forget <= 1)

    The update rule (per horizon):
        hw_{t+1} = hw_t + gamma * (alpha - exceeded_t)
    where alpha = 1 - coverage and exceeded_t = 1 if observation falls outside
    the current interval, 0 otherwise.
    """

    def __init__(self, horizon=24, coverage=0.90, gamma=0.005,
                 window=100, forget=1.0, init_hw=None):
        self.horizon = horizon
        self.coverage = coverage
        self.alpha = 1.0 - coverage
        self.gamma = gamma
        self.window = window
        self.forget = forget
        # Per-horizon state
        self.hw = np.full(horizon, init_hw if init_hw is not None
                          else 0.01, dtype=np.float64)
        self.buffer = []       # rolling (pred, true) pairs
        self.t = 0

    def update(self, y_pred, y_true):
        """Update the conformal half-widths with one new observation.

        Args:
            y_pred : (horizon,) or (1, horizon) point forecast
            y_true : (horizon,) or (1, horizon) ground truth
        """
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(y_true).ravel()
        self.t += 1

        # Update per-horizon half-width via ACI rule
        for h in range(self.horizon):
            exceeded = 1.0 if abs(y_pred[h] - y_true[h]) > self.hw[h] else 0.0
            self.hw[h] += self.gamma * (self.alpha - exceeded)
            # Clamp to a reasonable positive range
            self.hw[h] = max(1e-6, min(self.hw[h], 1e3))

        # Maintain rolling buffer for auxiliary monitoring
        self.buffer.append((y_pred.copy(), y_true.copy()))
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

    def predict_interval(self, y_pred):
        """Return (lower, upper) bounds for a new point forecast."""
        y_pred = np.asarray(y_pred).ravel()
        return y_pred - self.hw, y_pred + self.hw

    def running_coverage(self):
        """Empirical coverage over the rolling window."""
        if not self.buffer:
            return 0.0
        covered = 0
        total = 0
        for pred, true in self.buffer[-self.window:]:
            lower = pred - self.hw
            upper = pred + self.hw
            covered += np.sum((true >= lower) & (true <= upper))
            total += len(true)
        return covered / max(total, 1)

    def get_halfwidths(self):
        return self.hw.copy()

    def to_dict(self):
        return {
            "horizon": self.horizon,
            "coverage": self.coverage,
            "gamma": self.gamma,
            "window": self.window,
            "forget": self.forget,
            "halfwidths": self.hw.tolist(),
            "t": self.t,
        }

    @classmethod
    def from_dict(cls, d):
        inst = cls(
            horizon=d["horizon"], coverage=d["coverage"],
            gamma=d["gamma"], window=d["window"], forget=d["forget"])
        inst.hw = np.array(d["halfwidths"])
        inst.t = d["t"]
        return inst


# ---------------------------------------------------------------------------
# State-Conditioned Conformal
# ---------------------------------------------------------------------------

class StateConditionedConformal:
    """Conformal prediction with separate calibration per state/regime.

    States are defined by a discrete classifier that maps each time step to a
    regime label.  A separate split-conformal quantile is computed for each
    state, yielding tighter intervals in homogeneous regimes.

    Typical states for CIF forecasting:
        - low / mid / high renewable share
        - low / high volatility (recent share stdev)
        - normal / extreme ramp
        - weekday / weekend
    """

    def __init__(self, horizon=24, coverage=0.90):
        self.horizon = horizon
        self.coverage = coverage
        self._calibrated = False
        self._hw_per_state = {}      # state_label → ndarray(h,)
        self._state_labels = []

    def calibrate(self, y_pred_cal, y_true_cal, state_labels_cal):
        """Calibrate from labelled calibration windows.

        Args:
            y_pred_cal : (N, horizon)
            y_true_cal : (N, horizon)
            state_labels_cal : (N,) discrete state ids
        """
        unique_states = np.unique(state_labels_cal)
        self._state_labels = list(unique_states)
        for s in unique_states:
            mask = state_labels_cal == s
            n_s = mask.sum()
            if n_s < 5:
                continue  # too few samples; skip this state
            hw = split_conformal_calibrate_per_horizon(
                y_pred_cal[mask], y_true_cal[mask], self.coverage)
            self._hw_per_state[s] = hw
        self._calibrated = True
        # Fallback: global per-horizon
        self._hw_global = split_conformal_calibrate_per_horizon(
            y_pred_cal, y_true_cal, self.coverage)

    def predict_interval(self, y_pred, state_label):
        """Return (lower, upper) given a point forecast and its state."""
        y_pred = np.asarray(y_pred).ravel()
        hw = self._hw_per_state.get(state_label, self._hw_global)
        return y_pred - hw, y_pred + hw

    def get_halfwidth(self, state_label=None):
        if state_label is not None:
            return self._hw_per_state.get(state_label, self._hw_global)
        return self._hw_global

    @property
    def is_calibrated(self):
        return self._calibrated

    def to_dict(self):
        return {
            "horizon": self.horizon,
            "coverage": self.coverage,
            "state_labels": [int(s) for s in self._state_labels],
            "hw_per_state": {int(k): v.tolist()
                             for k, v in self._hw_per_state.items()},
            "hw_global": self._hw_global.tolist() if self._calibrated else None,
        }


# ---------------------------------------------------------------------------
# Utility functions (preserved from conformal_prediction.py)
# ---------------------------------------------------------------------------

def compute_crps(y_true, y_pred, halfwidth):
    """CRPS for a uniform prediction interval [pred - hw, pred + hw]."""
    a = y_pred - halfwidth
    b = y_pred + halfwidth
    width = b - a
    crps_vals = np.where(
        y_true < a,
        (a - y_true) + width / 3,
        np.where(
            y_true > b,
            (y_true - b) + width / 3,
            ((y_true - a) ** 2 + (b - y_true) ** 2) / (2 * width) + width / 6,
        ),
    )
    return float(np.mean(crps_vals))


def compute_coverage(y_true, y_pred, halfwidth):
    """Empirical coverage: fraction of points within interval."""
    if isinstance(halfwidth, np.ndarray) and halfwidth.ndim == 1:
        lower = y_pred - halfwidth[np.newaxis, :]
        upper = y_pred + halfwidth[np.newaxis, :]
    else:
        lower = y_pred - halfwidth
        upper = y_pred + halfwidth
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def compute_interval_width(halfwidth):
    """Mean interval width."""
    if isinstance(halfwidth, np.ndarray):
        return float(2 * np.mean(halfwidth))
    return float(2 * halfwidth)


# ---------------------------------------------------------------------------
# State classifier for CIF forecasting
# ---------------------------------------------------------------------------

def classify_state(config, rs_window, ef_nr=None):
    """Classify a forecast window into a discrete state label for conformal.

    Returns:
        state_label : int  0=low-rs, 1=mid-rs, 2=high-rs
                    (with volatility sub-class: base + 3*vol_bucket)
    """
    mean_rs = float(np.mean(rs_window[-48:])) if len(rs_window) >= 48 else float(config[0])
    vol = float(np.std(rs_window[-48:])) if len(rs_window) >= 48 else 0.01

    # Renewable-share bucket
    if mean_rs < 0.15:
        rs_bucket = 0
    elif mean_rs < 0.35:
        rs_bucket = 1
    else:
        rs_bucket = 2

    # Volatility bucket
    if vol < 0.03:
        vol_bucket = 0
    elif vol < 0.08:
        vol_bucket = 1
    else:
        vol_bucket = 2

    return rs_bucket * 3 + vol_bucket  # 0-8, 9 states total
