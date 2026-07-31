import torch

from transformer_disentanglement.epistasis import (
    PairEpistasisRegressor,
    SiteOnlyEpistasisControl,
    double_mutation_epistasis,
    robust_standardize,
    weighted_double_center,
    weighted_gauge_error,
)


def test_double_mutation_epistasis_removes_additive_effects() -> None:
    base = torch.tensor(2.0)
    left = torch.tensor([2.5, 1.5])
    right = torch.tensor([3.0, 1.0])
    interaction = torch.tensor([[0.2, -0.1], [0.3, 0.4]])
    double = left[:, None] + right[None, :] - base + interaction
    recovered = double_mutation_epistasis(base, left, right, double)
    assert torch.allclose(recovered, interaction)


def test_robust_standardize_is_shift_and_scale_invariant() -> None:
    values = torch.tensor([-3.0, -1.0, 0.0, 2.0, 20.0])
    standardized, _, _ = robust_standardize(values)
    transformed, _, _ = robust_standardize(7.0 + 4.0 * values)
    assert torch.allclose(standardized, transformed, atol=1e-6)
    assert torch.isclose(standardized.median(), torch.tensor(0.0))


def test_weighted_double_center_obeys_both_gauges() -> None:
    torch.manual_seed(3)
    blocks = torch.randn(5, 4, 4)
    left = torch.rand(5, 4)
    right = torch.rand(5, 4)
    left = left / left.sum(dim=-1, keepdim=True)
    right = right / right.sum(dim=-1, keepdim=True)
    centered = weighted_double_center(blocks, left, right)
    assert float(weighted_gauge_error(centered, left, right)) < 1e-6


def test_additive_site_field_is_removed_by_double_centering() -> None:
    torch.manual_seed(5)
    row = torch.randn(3, 4)
    column = torch.randn(3, 4)
    additive = row[:, :, None] + column[:, None, :]
    left = torch.softmax(torch.randn(3, 4), dim=-1)
    right = torch.softmax(torch.randn(3, 4), dim=-1)
    centered = weighted_double_center(additive, left, right)
    assert torch.max(torch.abs(centered)) < 1e-6


def test_projected_pair_regressor_has_exact_weighted_zero_marginals() -> None:
    torch.manual_seed(7)
    model = PairEpistasisRegressor(
        hidden_dim=12, states=5, rank=3, pair_dim=4, projected=True
    )
    left_hidden = torch.randn(6, 12)
    right_hidden = torch.randn(6, 12)
    left = torch.softmax(torch.randn(6, 5), dim=-1)
    right = torch.softmax(torch.randn(6, 5), dim=-1)
    blocks = model(left_hidden, right_hidden, left, right)
    assert blocks.shape == (6, 5, 5)
    assert float(weighted_gauge_error(blocks, left, right).detach()) < 2e-6


def test_site_only_control_has_no_multiplicative_parameters() -> None:
    model = SiteOnlyEpistasisControl(hidden_dim=8, states=3, residual_dim=10)
    left_hidden = torch.randn(4, 8)
    right_hidden = torch.randn(4, 8)
    probabilities = torch.full((4, 3), 1 / 3)
    blocks = model(
        left_hidden, right_hidden, probabilities, probabilities
    )
    mixed_difference = (
        blocks[:, 1:, 1:]
        - blocks[:, :-1, 1:]
        - blocks[:, 1:, :-1]
        + blocks[:, :-1, :-1]
    )
    assert torch.max(torch.abs(mixed_difference)) < 1e-6
