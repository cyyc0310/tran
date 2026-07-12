import torch
from transcif.physics.residual import ResidualCorrectionHead, fit_residual_head


def test_residual_head_output_shape():
    head = ResidualCorrectionHead(input_dim=4, hidden_dim=8)
    features = torch.randn(10, 4)
    output = head(features)
    assert output.shape == (10,)


def test_fit_residual_head_learns_known_linear_relationship():
    torch.manual_seed(0)
    true_weights = torch.tensor([2.0, -1.0, 0.5])
    train_features = torch.randn(200, 3)
    train_targets = train_features @ true_weights + 0.01 * torch.randn(200)

    head = ResidualCorrectionHead(input_dim=3, hidden_dim=16)
    fit_residual_head(head, train_features, train_targets, epochs=300, lr=1e-2)

    test_features = torch.randn(50, 3)
    test_targets = test_features @ true_weights
    with torch.no_grad():
        predictions = head(test_features)
    mse = torch.mean((predictions - test_targets) ** 2).item()
    assert mse < 0.5
