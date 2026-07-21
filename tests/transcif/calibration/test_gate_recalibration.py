import pytest
import torch
from transcif.models.encoder import DomainInvariantEncoder, PersistenceSkipEncoder
from transcif.calibration.gate_recalibration import recalibrate_persistence_gate


def test_recalibrate_persistence_gate_raises_without_gate_logit():
    encoder = DomainInvariantEncoder(num_variables=2, horizon=6, lt_feature_dim=8, cv_feature_dim=4)
    x = torch.rand(4, 30, 2)
    y = torch.rand(4, 6)

    with pytest.raises(ValueError):
        recalibrate_persistence_gate(encoder, x, y)


def test_recalibrate_persistence_gate_only_updates_gate_logit():
    torch.manual_seed(5)
    base = DomainInvariantEncoder(num_variables=2, horizon=6, lt_feature_dim=8, cv_feature_dim=4)
    encoder = PersistenceSkipEncoder(base)
    before = {name: param.detach().clone() for name, param in encoder.named_parameters() if name != "gate_logit"}

    x = torch.rand(8, 30, 2)
    y = torch.rand(8, 6)
    recalibrate_persistence_gate(encoder, x, y, epochs=10)

    for name, param in encoder.named_parameters():
        if name == "gate_logit":
            continue
        torch.testing.assert_close(before[name], param.detach())


def test_recalibrate_persistence_gate_moves_toward_one_when_persistence_is_perfect():
    """When the calibration target exactly equals the last observed value (perfect
    persistence), recalibration should push the gate up from a neutral start."""
    torch.manual_seed(7)
    base = DomainInvariantEncoder(num_variables=2, horizon=6, lt_feature_dim=8, cv_feature_dim=4)
    encoder = PersistenceSkipEncoder(base)
    with torch.no_grad():
        encoder.gate_logit.fill_(0.0)  # neutral start, sigmoid(0) = 0.5

    x = torch.rand(16, 30, 2)
    last_observed = x[:, -1, 0:1].expand(-1, 6)
    y_perfect_persistence = last_observed.clone()

    gate_after = recalibrate_persistence_gate(encoder, x, y_perfect_persistence, epochs=150, lr=5e-2)

    assert gate_after > 0.5


def test_recalibrate_persistence_gate_moves_toward_zero_when_persistence_is_a_poor_predictor():
    """When the calibration target is unrelated to the last observed value, recalibration
    should push the gate down from a persistence-favoring start."""
    torch.manual_seed(7)
    base = DomainInvariantEncoder(num_variables=2, horizon=6, lt_feature_dim=8, cv_feature_dim=4)
    encoder = PersistenceSkipEncoder(base)
    with torch.no_grad():
        encoder.gate_logit.fill_(2.0)  # persistence-favoring start, sigmoid(2) ~= 0.88

    x = torch.rand(16, 30, 2) * 0.2  # last-observed values clustered near 0
    y_unrelated = torch.rand(16, 6) * 0.8 + 0.2  # targets clustered near 0.6, far from 0

    gate_after = recalibrate_persistence_gate(encoder, x, y_unrelated, epochs=150, lr=5e-2)

    assert gate_after < 0.88
