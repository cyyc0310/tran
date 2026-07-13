"""Stage 1: domain-invariant encoder that fuses LT-MWKC and CV-DWCC to predict the
future RenewShare trajectory (the reparameterized, transferable prediction target)."""

import torch
import torch.nn as nn

from transcif.models.cv_dwcc import CVDWCC
from transcif.models.lt_mwkc import LTMWKC


class DomainInvariantEncoder(nn.Module):
    def __init__(
        self,
        num_variables: int = 3,
        horizon: int = 24,
        lt_feature_dim: int = 32,
        cv_feature_dim: int = 16,
    ):
        super().__init__()
        self.lt_mwkc = LTMWKC(in_channels=num_variables, feature_dim=lt_feature_dim)
        self.cv_dwcc = CVDWCC(num_variables=num_variables, feature_dim=cv_feature_dim)

        fused_dim = lt_feature_dim + cv_feature_dim * num_variables
        self.predict = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lt_input = x.permute(0, 2, 1)
        lt_features = self.lt_mwkc(lt_input).mean(dim=-1)

        cv_features, dominant_idx = self.cv_dwcc(x)
        cv_pooled = cv_features.mean(dim=(2, 4))
        cv_pooled = cv_pooled.reshape(cv_pooled.shape[0], -1)

        fused = torch.cat([lt_features, cv_pooled], dim=-1)
        renew_share_pred = self.predict(fused)
        return renew_share_pred, dominant_idx
