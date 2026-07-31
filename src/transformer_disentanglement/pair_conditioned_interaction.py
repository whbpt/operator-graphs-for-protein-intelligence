from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PairConditionedInteractionHead(nn.Module):
    """Marginal-free categorical interactions with pair-specific mode mixtures."""

    def __init__(
        self,
        hidden_dim: int,
        states: int = 20,
        rank: int = 32,
        pair_dim: int = 32,
        pair_mlp_dim: int = 64,
    ) -> None:
        super().__init__()
        self.states = states
        self.rank = rank
        self.marginal_norm = nn.LayerNorm(hidden_dim)
        self.factor_norm = nn.LayerNorm(hidden_dim)
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.marginal_projection = nn.Linear(hidden_dim, states)
        self.factor_projection = nn.Linear(hidden_dim, states * rank)
        self.pair_projection = nn.Linear(hidden_dim, pair_dim)
        self.pair_decoder = nn.Sequential(
            nn.Linear(pair_dim * 3, pair_mlp_dim),
            nn.SiLU(),
            nn.Linear(pair_mlp_dim, rank + 1),
        )
        with torch.no_grad():
            self.pair_decoder[-1].bias[0] = -1.0

    def forward(
        self,
        hidden: torch.Tensor,
        gauge_probabilities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        marginal_logits = self.marginal_projection(self.marginal_norm(hidden))
        marginal_probabilities = marginal_logits.softmax(dim=-1)
        if gauge_probabilities is None:
            gauge_probabilities = marginal_probabilities.detach()
        gauge_probabilities = gauge_probabilities / gauge_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        factors = self.factor_projection(self.factor_norm(hidden)).reshape(
            len(hidden), self.states, self.rank
        )
        categorical_mean = torch.sum(
            gauge_probabilities[..., None] * factors, dim=1, keepdim=True
        )
        factors = factors - categorical_mean
        factor_rms = factors.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        factors = factors / factor_rms
        pair_state = self.pair_projection(self.pair_norm(hidden))
        return {
            "marginal_logits": marginal_logits,
            "marginal_probabilities": marginal_probabilities,
            "factors": factors,
            "pair_state": pair_state,
        }

    @staticmethod
    def symmetric_pair_features(
        left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [left + right, torch.abs(left - right), left * right], dim=-1
        )

    def pair_parameters(
        self, pair_state: torch.Tensor, pairs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.symmetric_pair_features(
            pair_state[pairs[:, 0]], pair_state[pairs[:, 1]]
        )
        decoded = self.pair_decoder(features)
        amplitude = F.softplus(decoded[:, 0])
        mode = decoded[:, 1:]
        mode = mode / mode.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
            1e-6
        )
        return amplitude, mode

    @staticmethod
    def pair_blocks(
        factors: torch.Tensor,
        pairs: torch.Tensor,
        amplitude: torch.Tensor,
        mode: torch.Tensor,
    ) -> torch.Tensor:
        left = factors[pairs[:, 0]]
        right = factors[pairs[:, 1]]
        blocks = torch.einsum("par,pbr,pr->pab", left, right, mode)
        return blocks * amplitude[:, None, None]
