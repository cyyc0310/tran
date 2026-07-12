import math

import torch
from transcif.models.encoder import DomainInvariantEncoder
from transcif.training.train_source import train_source_domain


def _make_synthetic_renew_share_dataset(num_samples: int, seq_len: int, horizon: int):
    """RenewShare follows a smooth diurnal sine pattern; LoadNorm/TempAnomaly are
    correlated noise. The target is the sine pattern shifted forward by `seq_len` steps,
    a learnable relationship a working encoder should pick up within a few epochs."""
    torch.manual_seed(7)
    t = torch.linspace(0, 4 * math.pi, seq_len + horizon)
    base = (torch.sin(t) + 1) / 2

    x_list, y_list = [], []
    for _ in range(num_samples):
        phase_shift = torch.empty(1).uniform_(0, 2 * math.pi).item()
        shifted = (torch.sin(t + phase_shift) + 1) / 2
        renew_share = shifted[:seq_len]
        target = shifted[seq_len : seq_len + horizon]

        load_norm = 0.5 + 0.1 * torch.randn(seq_len)
        temp_anomaly = 0.1 * torch.randn(seq_len)
        sample = torch.stack([renew_share, load_norm, temp_anomaly], dim=-1)
        x_list.append(sample)
        y_list.append(target)

    return torch.stack(x_list), torch.stack(y_list)


def test_train_source_domain_reduces_loss():
    x_train, y_train = _make_synthetic_renew_share_dataset(num_samples=16, seq_len=48, horizon=12)
    encoder = DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)

    losses = train_source_domain(encoder, x_train, y_train, epochs=30, lr=5e-3, consistency_weight=0.05)

    assert len(losses) == 30
    assert losses[-1] < losses[0] * 0.7
