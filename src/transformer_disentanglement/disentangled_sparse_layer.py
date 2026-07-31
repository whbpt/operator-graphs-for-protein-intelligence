from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class DisentangledSparseInteractionLayer(nn.Module):
    """Single-sequence layer with marginal background and sparse typed fields."""

    def __init__(
        self,
        hidden_dim: int,
        states: int = 20,
        rank: int = 16,
        index_dim: int = 32,
        pair_dim: int = 32,
        pair_mlp_dim: int = 64,
        neighbors: int = 8,
        local_kernel: int = 5,
        local_exclusion: int = 2,
        routing_mode: str = "topk",
        soft_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if local_kernel % 2 == 0:
            raise ValueError("local_kernel must be odd")
        self.states = states
        self.rank = rank
        self.neighbors = neighbors
        self.local_exclusion = local_exclusion
        if routing_mode not in {"topk", "soft"}:
            raise ValueError("routing_mode must be 'topk' or 'soft'")
        self.routing_mode = routing_mode
        self.soft_temperature = soft_temperature

        self.background_norm = nn.LayerNorm(hidden_dim)
        self.factor_norm = nn.LayerNorm(hidden_dim)
        self.index_norm = nn.LayerNorm(hidden_dim)
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.local_norm = nn.LayerNorm(hidden_dim)

        self.background_projection = nn.Linear(hidden_dim, states)
        self.factor_projection = nn.Linear(hidden_dim, states * rank)
        self.index_left = nn.Linear(hidden_dim, index_dim, bias=False)
        self.index_right = nn.Linear(hidden_dim, index_dim, bias=False)
        self.index_bias = nn.Parameter(torch.tensor(0.0))
        self.index_dim = index_dim
        self.pair_projection = nn.Linear(hidden_dim, pair_dim)
        self.pair_decoder = nn.Sequential(
            nn.Linear(pair_dim * 3, pair_mlp_dim),
            nn.SiLU(),
            nn.Linear(pair_mlp_dim, rank + 1),
        )
        with torch.no_grad():
            self.pair_decoder[-1].bias[0] = -4.0

        self.local_depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=local_kernel,
            padding=local_kernel // 2,
            groups=hidden_dim,
        )
        self.local_pointwise = nn.Linear(hidden_dim, hidden_dim)
        self.interaction_to_hidden = nn.Linear(states, hidden_dim, bias=False)

    @staticmethod
    def symmetric_pair_features(
        left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [left + right, torch.abs(left - right), left * right], dim=-1
        )

    def index_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.index_norm(hidden)
        left = self.index_left(normalized)
        right = self.index_right(normalized)
        scores = torch.einsum("bid,bjd->bij", left, right)
        scores = 0.5 * (scores + scores.transpose(1, 2))
        return scores * self.index_dim**-0.5 + self.index_bias

    def select_neighbors(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = scores.shape[1]
        positions = torch.arange(length, device=scores.device)
        separation = torch.abs(positions[:, None] - positions[None, :])
        valid = separation > self.local_exclusion
        masked = scores.masked_fill(~valid[None, :, :], -torch.inf)
        count = min(self.neighbors, max(length - (2 * self.local_exclusion + 1), 1))
        values, indices = torch.topk(masked, k=count, dim=-1)
        return indices, values

    def set_routing_mode(self, routing_mode: str) -> None:
        if routing_mode not in {"topk", "soft"}:
            raise ValueError("routing_mode must be 'topk' or 'soft'")
        self.routing_mode = routing_mode

    def forward(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        gauge_probabilities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, length, hidden_dim]")
        if tokens.shape != hidden.shape[:2]:
            raise ValueError("tokens must have shape [batch, length]")

        local_input = self.local_norm(hidden).transpose(1, 2)
        local_update = self.local_depthwise(local_input).transpose(1, 2)
        local_update = self.local_pointwise(F.silu(local_update))
        background_hidden = hidden + local_update

        background_logits = self.background_projection(
            self.background_norm(background_hidden)
        )
        marginal_probabilities = background_logits.softmax(dim=-1)
        if gauge_probabilities is None:
            gauge_probabilities = marginal_probabilities.detach()
        gauge_probabilities = gauge_probabilities / gauge_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        factors = self.factor_projection(self.factor_norm(background_hidden)).reshape(
            hidden.shape[0], hidden.shape[1], self.states, self.rank
        )
        categorical_mean = torch.sum(
            gauge_probabilities[..., None] * factors,
            dim=2,
            keepdim=True,
        )
        factors = factors - categorical_mean
        factors = factors / factors.square().mean(
            dim=2, keepdim=True
        ).sqrt().clamp_min(1e-6)

        scores = self.index_scores(background_hidden)
        pair_state = self.pair_projection(self.pair_norm(background_hidden))
        if self.routing_mode == "soft":
            batch, length = tokens.shape
            positions = torch.arange(length, device=hidden.device)
            separation = torch.abs(positions[:, None] - positions[None, :])
            valid = separation > self.local_exclusion
            masked_scores = scores.masked_fill(~valid[None, :, :], -torch.inf)
            routing_weights = torch.softmax(
                masked_scores / self.soft_temperature, dim=-1
            )
            left_state = pair_state[:, :, None, :].expand(-1, -1, length, -1)
            right_state = pair_state[:, None, :, :].expand(-1, length, -1, -1)
            pair_features = self.symmetric_pair_features(left_state, right_state)
            decoded = self.pair_decoder(pair_features)
            amplitude = F.softplus(decoded[..., 0])
            mode = decoded[..., 1:]
            mode = mode / mode.square().mean(
                dim=-1, keepdim=True
            ).sqrt().clamp_min(1e-6)
            known = tokens < self.states
            clipped = tokens.clamp(max=self.states - 1)
            gather_index = clipped[..., None, None].expand(-1, -1, 1, self.rank)
            observed_right_factor = torch.gather(
                factors, dim=2, index=gather_index
            ).squeeze(2)
            observed_right_factor = observed_right_factor * known[..., None]
            interaction_logits = torch.einsum(
                "biar,bjr,bijr,bij,bij->bia",
                factors,
                observed_right_factor,
                mode,
                amplitude,
                routing_weights,
            )
            neighbor_indices, selected_scores = self.select_neighbors(scores)
        else:
            neighbor_indices, selected_scores = self.select_neighbors(scores)
            batch, length, neighbors = neighbor_indices.shape
            batch_index = torch.arange(batch, device=hidden.device)[:, None, None]
            left_state = pair_state[:, :, None, :].expand(-1, -1, neighbors, -1)
            right_state = pair_state[batch_index, neighbor_indices]
            pair_features = self.symmetric_pair_features(left_state, right_state)
            decoded = self.pair_decoder(pair_features)
            amplitude = F.softplus(decoded[..., 0])
            mode = decoded[..., 1:]
            mode = mode / mode.square().mean(
                dim=-1, keepdim=True
            ).sqrt().clamp_min(1e-6)
            left_factor = factors[:, :, None, :, :].expand(
                -1, -1, neighbors, -1, -1
            )
            right_factor = factors[batch_index, neighbor_indices]
            neighbor_tokens = tokens[batch_index, neighbor_indices]
            known = neighbor_tokens < self.states
            clipped = neighbor_tokens.clamp(max=self.states - 1)
            gather_index = clipped[..., None, None].expand(
                -1, -1, -1, 1, self.rank
            )
            observed_right_factor = torch.gather(
                right_factor, dim=3, index=gather_index
            ).squeeze(3)
            observed_right_factor = observed_right_factor * known[..., None]
            interaction_logits = torch.einsum(
                "bikar,bikr,bikr,bik->bia",
                left_factor,
                observed_right_factor,
                mode,
                amplitude,
            ) / math.sqrt(max(neighbors, 1))
            routing_weights = None

        updated_hidden = (
            background_hidden
            + self.interaction_to_hidden(interaction_logits)
        )
        return {
            "hidden": updated_hidden,
            "background_logits": background_logits,
            "marginal_probabilities": marginal_probabilities,
            "interaction_logits": interaction_logits,
            "logits": background_logits + interaction_logits,
            "index_scores": scores,
            "neighbor_indices": neighbor_indices,
            "selected_scores": selected_scores,
            "routing_weights": routing_weights,
            "amplitude": amplitude,
            "mode": mode,
            "factors": factors,
        }
