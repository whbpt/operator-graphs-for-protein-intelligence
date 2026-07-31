import torch

from transformer_disentanglement.gauge_set_aggregation import (
    MarginalOrthogonalSetAggregator,
)


def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(307)
    probabilities = torch.softmax(torch.randn(3, 7), dim=-1)
    messages = torch.randn(3, 5, 7)
    messages = messages - torch.sum(
        probabilities[:, None] * messages, dim=-1, keepdim=True
    )
    scores = torch.randn(3, 5)
    return messages, scores, probabilities


def test_zero_initialized_aggregator_matches_additive_message() -> None:
    messages, scores, probabilities = make_inputs()
    aggregator = MarginalOrthogonalSetAggregator(states=7, hidden_dim=11)
    output = aggregator(messages, scores, probabilities)
    weights = torch.softmax(scores, dim=-1)
    expected = torch.sum(weights[..., None] * messages, dim=-2)
    assert torch.allclose(output["interaction"], expected, atol=1e-6)
    assert torch.allclose(output["correction"], torch.zeros_like(expected))


def test_aggregated_message_obeys_target_marginal_gauge() -> None:
    messages, scores, probabilities = make_inputs()
    aggregator = MarginalOrthogonalSetAggregator(states=7, hidden_dim=11)
    with torch.no_grad():
        aggregator.residual_decoder.weight.normal_()
        aggregator.residual_decoder.bias.normal_()
    output = aggregator(messages, scores, probabilities)
    gauge = torch.sum(probabilities * output["interaction"], dim=-1)
    assert float(gauge.abs().max().detach()) < 2e-6


def test_aggregator_is_permutation_invariant_over_neighbors() -> None:
    messages, scores, probabilities = make_inputs()
    aggregator = MarginalOrthogonalSetAggregator(states=7, hidden_dim=11)
    with torch.no_grad():
        aggregator.residual_decoder.weight.normal_()
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original = aggregator(messages, scores, probabilities)["interaction"]
    permuted = aggregator(
        messages[:, permutation], scores[:, permutation], probabilities
    )["interaction"]
    assert torch.allclose(original, permuted, atol=1e-6)


def test_aggregator_residual_receives_gradients() -> None:
    messages, scores, probabilities = make_inputs()
    aggregator = MarginalOrthogonalSetAggregator(states=7, hidden_dim=11)
    target = torch.randn(3, 7)
    loss = (aggregator(messages, scores, probabilities)["interaction"] - target).square().mean()
    loss.backward()
    assert aggregator.residual_decoder.weight.grad is not None
    assert float(aggregator.residual_decoder.weight.grad.abs().sum()) > 0


def test_correction_rms_is_bounded_relative_to_additive_message() -> None:
    messages, scores, probabilities = make_inputs()
    aggregator = MarginalOrthogonalSetAggregator(
        states=7, hidden_dim=11, max_correction_ratio=0.4
    )
    with torch.no_grad():
        aggregator.residual_decoder.weight.normal_(std=100.0)
        aggregator.residual_decoder.bias.normal_(std=100.0)
    output = aggregator(messages, scores, probabilities)
    additive_rms = output["additive"].square().mean(dim=-1).sqrt()
    correction_rms = output["correction"].square().mean(dim=-1).sqrt()
    assert torch.all(correction_rms <= 0.4 * additive_rms + 1e-6)
