from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def weighted_target_center(
    values: torch.Tensor, probabilities: torch.Tensor
) -> torch.Tensor:
    """Remove the target-category marginal under site probabilities."""
    if values.shape != probabilities.shape:
        raise ValueError("values and probabilities must have the same shape")
    normalized = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        1e-8
    )
    return values - torch.sum(normalized * values, dim=-1, keepdim=True)


class MarginalOrthogonalSetAggregator(nn.Module):
    """Nonlinear permutation-invariant aggregation with an exact target gauge.

    The module starts as the ordinary softmax-weighted additive interaction. A
    zero-initialized residual decoder can learn dependencies among the selected
    pair messages, while the final projection prevents it from duplicating the
    target marginal branch.
    """

    def __init__(
        self,
        states: int = 20,
        hidden_dim: int = 32,
        routing_temperature: float = 1.0,
        max_correction_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if states < 2:
            raise ValueError("states must be at least two")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive")
        if max_correction_ratio <= 0:
            raise ValueError("max_correction_ratio must be positive")
        self.states = states
        self.hidden_dim = hidden_dim
        self.routing_temperature = routing_temperature
        self.max_correction_ratio = max_correction_ratio
        self.pair_norm = nn.LayerNorm(states + 2)
        self.pair_encoder = nn.Linear(states + 2, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim + states)
        self.context_projection = nn.Linear(hidden_dim + states, hidden_dim)
        self.residual_decoder = nn.Linear(hidden_dim, states)
        nn.init.zeros_(self.residual_decoder.weight)
        nn.init.zeros_(self.residual_decoder.bias)

    def forward(
        self,
        messages: torch.Tensor,
        scores: torch.Tensor,
        marginal_probabilities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if messages.ndim < 2 or messages.shape[-1] != self.states:
            raise ValueError("messages must have shape [..., neighbors, states]")
        if scores.shape != messages.shape[:-1]:
            raise ValueError("scores must match the message neighbor axes")
        if marginal_probabilities.shape != messages.shape[:-2] + (self.states,):
            raise ValueError("marginal probabilities have incompatible shape")

        weights = torch.softmax(scores / self.routing_temperature, dim=-1)
        additive = torch.sum(weights[..., None] * messages, dim=-2)

        score_mean = torch.sum(weights * scores, dim=-1, keepdim=True)
        score_variance = torch.sum(
            weights * (scores - score_mean).square(), dim=-1, keepdim=True
        )
        standardized_scores = (scores - score_mean) / score_variance.sqrt().clamp_min(
            1e-6
        )
        pair_features = torch.cat(
            [messages, weights[..., None], standardized_scores[..., None]], dim=-1
        )
        encoded = F.silu(self.pair_encoder(self.pair_norm(pair_features)))
        pooled = torch.sum(weights[..., None] * encoded, dim=-2)
        context = torch.cat([pooled, additive], dim=-1)
        hidden = F.silu(self.context_projection(self.context_norm(context)))
        raw_correction = weighted_target_center(
            self.residual_decoder(hidden), marginal_probabilities
        )
        additive_rms = torch.sqrt(
            additive.square().mean(dim=-1, keepdim=True) + 1e-12
        )
        correction_rms = torch.sqrt(
            raw_correction.square().mean(dim=-1, keepdim=True) + 1e-12
        )
        correction_cap = self.max_correction_ratio * additive_rms.clamp_min(1e-6)
        correction = raw_correction * (
            correction_cap
            / torch.sqrt(correction_rms.square() + correction_cap.square())
        )
        interaction = weighted_target_center(
            additive + correction, marginal_probabilities
        )
        return {
            "interaction": interaction,
            "additive": weighted_target_center(additive, marginal_probabilities),
            "correction": correction,
            "raw_correction": raw_correction,
            "weights": weights,
        }
