"""Observable coarse-bias anchor plus a source-fitted fine-error remainder.

This uses the frozen V6 regression/transport algorithms unchanged. It anchors
the daily level to a released coarse hindcast bias before learning the remaining
fine error; it is a mechanism composition, not a new statistical theory.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from transcif.research.calibration_v4 import _daily
from transcif.research.conditional_transfer_v6 import ConditionalTransport as _BASE_TRANSPORT
from transcif.research.conditional_transfer_v6 import FEATURE_NAMES


MODES = ("global", "loglinear", "extratrees", "structured", "noanchor")
BRIDGE_POLICY = {
    "name": "observable_coarse_bias_identity_plus_source_remainder",
    "version": 1,
    "anchor_priority": [
        {"bias": "seasonal_hindcast_bias", "support": "seasonal_hindcast_valid_months"},
        {"bias": "harmonic_hindcast_bias", "support": "harmonic_hindcast_valid_months"},
        {"bias": "latest_hindcast_bias", "support": "hindcast_months"},
    ],
    "unsupported_anchor": 0.,
    "mode_map": {"global": "global", "loglinear": "loglinear", "extratrees": "extratrees",
                 "structured": "structured", "noanchor": "structured"},
    "anchored_modes": ["global", "loglinear", "extratrees", "structured"],
    "remainder_fit": "truth_minus_anchor; original_point_unchanged",
    "prediction_shift": ["corrected_point", "ensemble", "lower", "upper", "bias"],
    "feature_names": list(FEATURE_NAMES),
}


def bridge_policy():
    return deepcopy(BRIDGE_POLICY)


def coarse_anchor(features):
    """Choose a signed anchor from already-lawful, truth-free features only."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("features must be finite (N,F) in the frozen FEATURE_NAMES order.")
    anchor = np.zeros(len(x))
    selected = np.zeros(len(x), dtype=bool)
    for entry in BRIDGE_POLICY["anchor_priority"]:
        support = x[:, FEATURE_NAMES.index(entry["support"])]
        if np.any(support < 0):
            raise ValueError("Hindcast support counts must be nonnegative.")
        take = (support > 0) & ~selected
        anchor[take] = x[take, FEATURE_NAMES.index(entry["bias"])]
        selected |= take
    return anchor


class AnchorTransport(_BASE_TRANSPORT):
    """Fit the unchanged V6 model on fine-minus-observable-coarse errors.

    Four modes use the same anchor and observed inputs. ``noanchor`` is exactly
    the frozen structured estimator, with no subtraction or prediction shift.
    Labels/dates/regions retain the original proper-fit and bank validation.
    """

    def __init__(self, source_regions, mode="structured", alpha=.1, fit_year=2024):
        if mode not in MODES:
            raise ValueError(f"Unknown bridge mode {mode!r}; expected {MODES}.")
        self.bridge_mode = mode
        self.uses_anchor = mode != "noanchor"
        super().__init__(source_regions, mode=BRIDGE_POLICY["mode_map"][mode],
                         alpha=alpha, fit_year=fit_year)

    def fit(self, point, truth, features, regions, *, days, level_scales=None, shape_scales=None):
        point, _ = _daily(point, "point")
        truth, _ = _daily(truth, "truth")
        x = self._features(features, len(point))
        anchor = coarse_anchor(x) if self.uses_anchor else np.zeros(len(point))
        super().fit(point, truth - anchor[:, None], x, regions, days=days,
                    level_scales=level_scales, shape_scales=shape_scales)
        self.audit_["bridge_mode"] = self.bridge_mode
        self.audit_["bridge_policy"] = bridge_policy()
        self.audit_["raw_point_unchanged"] = True
        self.audit_["coarse_anchor"] = {"used": self.uses_anchor, "min": float(anchor.min()),
                                       "max": float(anchor.max()), "mean": float(anchor.mean())}
        return self

    def predict(self, point, features, *, level_scales=None, shape_scales=None,
                n_samples=128, seed=17023, multiplier=1.):
        result = super().predict(point, features, level_scales=level_scales, shape_scales=shape_scales,
                                 n_samples=n_samples, seed=seed, multiplier=multiplier)
        if self.uses_anchor:
            anchor = coarse_anchor(features)
            for key in ("corrected_point", "lower", "upper"):
                result[key] = result[key] + anchor[:, None]
            result["ensemble"] = result["ensemble"] + anchor[:, None, None]
            result["bias"] = result["bias"] + anchor
        return result
