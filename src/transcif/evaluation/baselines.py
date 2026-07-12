"""Naive-transfer baseline (the "lower bound" from the design doc's experiment plan):
apply the source-trained encoder directly to target-domain data, reconstruct CI with the
SOURCE region's emission factor table, with no residual correction and no dominant-
variable reweighting."""

import numpy as np
import torch

from transcif.models.encoder import DomainInvariantEncoder
from transcif.physics.cif import cif_from_shares, get_emission_factors


def naive_transfer_predict(
    encoder: DomainInvariantEncoder,
    x_target: torch.Tensor,
    source_region_code: str,
) -> np.ndarray:
    with torch.no_grad():
        renew_share_pred, _ = encoder(x_target)
    renew_factor, nonrenew_factor = get_emission_factors(source_region_code)
    return cif_from_shares(renew_share_pred.numpy(), renew_factor, nonrenew_factor)
