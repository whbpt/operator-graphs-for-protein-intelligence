from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def robust_standardize(
    values: torch.Tensor,
    minimum_scale: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Median/MAD standardization for a one-dimensional family statistic."""
    location = values.median()
    scale = 1.4826 * (values - location).abs().median()
    if float(scale) < minimum_scale:
        scale = values.std(unbiased=False).clamp_min(minimum_scale)
    return (values - location) / scale, location, scale


def double_mutation_epistasis(
    base_loss: torch.Tensor,
    left_single_loss: torch.Tensor,
    right_single_loss: torch.Tensor,
    double_loss: torch.Tensor,
) -> torch.Tensor:
    """Return the non-additive part of a double perturbation loss."""
    return (
        double_loss
        - left_single_loss[..., :, None]
        - right_single_loss[..., None, :]
        + base_loss[..., None, None]
    )


def weighted_double_center(
    blocks: torch.Tensor,
    left_probabilities: torch.Tensor,
    right_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Project categorical pair blocks out of both weighted marginal spaces."""
    left_probabilities = left_probabilities / left_probabilities.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    right_probabilities = right_probabilities / right_probabilities.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    left_effect = torch.einsum(
        "...a,...ab->...b", left_probabilities, blocks
    )
    right_effect = torch.einsum(
        "...ab,...b->...a", blocks, right_probabilities
    )
    grand_effect = torch.einsum(
        "...a,...ab,...b->...",
        left_probabilities,
        blocks,
        right_probabilities,
    )
    return (
        blocks
        - left_effect[..., None, :]
        - right_effect[..., :, None]
        + grand_effect[..., None, None]
    )


def weighted_gauge_error(
    blocks: torch.Tensor,
    left_probabilities: torch.Tensor,
    right_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Maximum absolute weighted row or column marginal."""
    left = torch.einsum("...a,...ab->...b", left_probabilities, blocks)
    right = torch.einsum("...ab,...b->...a", blocks, right_probabilities)
    return torch.maximum(left.abs().amax(), right.abs().amax())


class PairEpistasisRegressor(nn.Module):
    """Low-rank categorical pair field with optional two-sided projection."""

    def __init__(
        self,
        hidden_dim: int,
        states: int,
        rank: int = 8,
        pair_dim: int = 16,
        pair_mlp_dim: int = 64,
        projected: bool = True,
    ) -> None:
        super().__init__()
        self.states = states
        self.rank = rank
        self.projected = projected
        self.factor_norm = nn.LayerNorm(hidden_dim)
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.left_factor = nn.Linear(hidden_dim, states * rank)
        self.right_factor = nn.Linear(hidden_dim, states * rank)
        self.pair_projection = nn.Linear(hidden_dim, pair_dim)
        self.pair_decoder = nn.Sequential(
            nn.Linear(pair_dim * 3, pair_mlp_dim),
            nn.SiLU(),
            nn.Linear(pair_mlp_dim, rank + 1),
        )
        with torch.no_grad():
            self.pair_decoder[-1].bias[0] = -2.0

    @staticmethod
    def symmetric_pair_features(
        left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [left + right, torch.abs(left - right), left * right], dim=-1
        )

    def forward(
        self,
        left_hidden: torch.Tensor,
        right_hidden: torch.Tensor,
        left_probabilities: torch.Tensor,
        right_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        left = self.left_factor(self.factor_norm(left_hidden)).reshape(
            *left_hidden.shape[:-1], self.states, self.rank
        )
        right = self.right_factor(self.factor_norm(right_hidden)).reshape(
            *right_hidden.shape[:-1], self.states, self.rank
        )
        left_pair = self.pair_projection(self.pair_norm(left_hidden))
        right_pair = self.pair_projection(self.pair_norm(right_hidden))
        decoded = self.pair_decoder(
            self.symmetric_pair_features(left_pair, right_pair)
        )
        amplitude = F.softplus(decoded[..., 0])
        mode = decoded[..., 1:]
        mode = mode / mode.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
            1e-6
        )
        blocks = torch.einsum("...ar,...br,...r->...ab", left, right, mode)
        blocks = blocks * amplitude[..., None, None]
        if self.projected:
            blocks = weighted_double_center(
                blocks, left_probabilities, right_probabilities
            )
        return blocks


class SiteOnlyEpistasisControl(nn.Module):
    """Additive row-plus-column field with no categorical pair interaction."""

    def __init__(
        self,
        hidden_dim: int,
        states: int,
        residual_dim: int = 64,
    ) -> None:
        super().__init__()
        self.left = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, residual_dim),
            nn.SiLU(),
            nn.Linear(residual_dim, states),
        )
        self.right = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, residual_dim),
            nn.SiLU(),
            nn.Linear(residual_dim, states),
        )
        nn.init.zeros_(self.left[-1].weight)
        nn.init.zeros_(self.left[-1].bias)
        nn.init.zeros_(self.right[-1].weight)
        nn.init.zeros_(self.right[-1].bias)

    def forward(
        self,
        left_hidden: torch.Tensor,
        right_hidden: torch.Tensor,
        left_probabilities: torch.Tensor,
        right_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        del left_probabilities, right_probabilities
        left = self.left(left_hidden)
        right = self.right(right_hidden)
        return left[..., :, None] + right[..., None, :]
