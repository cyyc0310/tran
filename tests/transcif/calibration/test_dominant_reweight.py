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


def test_reweight_lt_mwkc_alpha_changes_model_output():
    """Regression guard for the final-review finding that reweighting used to change
    predictions by at most ~1.4e-4 on a [0, 1] scale — numerically present but
    scientifically inert. Checks both that (a) the branch-fusion mechanism itself shifts
    LTMWKC's own output by a substantial relative margin (not a diluted fraction of a
    percent), and (b) that shift measurably propagates through to the encoder's final
    prediction. Measuring LTMWKC's own output (rather than only the final, Sigmoid-diluted
    prediction) avoids flakiness: empirically, this relative change is robustly above 5%
    (measured 7.4%-10.2% across seeds 0, 1, 2, 3, 11, 42, 100), comfortably distinguishing
    the fix from the old near-inert behavior (~1.4e-4 absolute change in final output)
    without relying on the diluted final-output magnitude directly."""
    torch.manual_seed(11)
    encoder = DomainInvariantEncoder(num_variables=3, horizon=6, lt_feature_dim=8, cv_feature_dim=4)
    x = torch.randn(4, 48, 3)
    lt_input = x.permute(0, 2, 1)

    with torch.no_grad():
        lt_before = encoder.lt_mwkc(lt_input)
        before, _ = encoder(x)

    reweight_lt_mwkc_alpha(encoder, dominant_variable_idx=0, boost=3.0)

    with torch.no_grad():
        lt_after = encoder.lt_mwkc(lt_input)
        after, _ = encoder(x)

    lt_relative_change = (lt_after - lt_before).abs().mean() / lt_before.abs().mean()
    assert lt_relative_change.item() > 0.05
    assert not torch.allclose(before, after)
