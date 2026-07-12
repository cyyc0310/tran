import torch
from transcif.models.encoder import DomainInvariantEncoder
from transcif.calibration.dominant_reweight import (
    recompute_dominant_variable,
    reweight_lt_mwkc_alpha,
)


def test_recompute_dominant_variable_identifies_self_predictive_channel():
    """RenewShare is a smooth, slowly-varying signal; the other two channels are each a
    scaled copy of it plus independent noise, so both have a genuine partial dependence on
    RenewShare while being only indirectly related to each other (through that shared
    dependence, not a direct link). This gives CV-DWCC's local weighted regression a real
    signal to detect: RenewShare should locally out-predict the other's noisy copy of it,
    making RenewShare the dominant predictor for both other targets far more often than a
    coin flip (verified reliable across seeds 0-19; see task-11-report.md)."""
    torch.manual_seed(3)
    encoder = DomainInvariantEncoder(num_variables=3, horizon=6, lt_feature_dim=8, cv_feature_dim=4)

    dependence_strength = 3.0
    independent_noise_std = 0.3

    t = torch.linspace(0, 6 * torch.pi, 80)
    renew_share = ((torch.sin(t) + 1) / 2).unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1)
    independent_noise = torch.randn(3, 80, 2) * independent_noise_std
    dependent_others = dependence_strength * renew_share + independent_noise
    calibration_x = torch.cat([renew_share, dependent_others], dim=-1)

    dominant_idx = recompute_dominant_variable(encoder, calibration_x)
    assert dominant_idx == 0


def test_reweight_lt_mwkc_alpha_only_changes_targeted_branch():
    encoder = DomainInvariantEncoder(num_variables=3, horizon=6, lt_feature_dim=8, cv_feature_dim=4)
    before_first = encoder.lt_mwkc.branches[0].alpha.detach().clone()
    before_last = encoder.lt_mwkc.branches[-1].alpha.detach().clone()

    reweight_lt_mwkc_alpha(encoder, dominant_variable_idx=0, boost=2.0)

    after_first = encoder.lt_mwkc.branches[0].alpha.detach()
    after_last = encoder.lt_mwkc.branches[-1].alpha.detach()

    assert not torch.allclose(before_first, after_first)
    torch.testing.assert_close(before_last, after_last)
