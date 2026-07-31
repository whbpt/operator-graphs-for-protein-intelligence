from __future__ import annotations

import torch
from torch import nn


class SymmetricBilinearIndexer(nn.Module):
    """Narrow DSA-style scalar indexer with symmetric pair probabilities."""

    def __init__(self, hidden_dim: int, index_dim: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.left = nn.Linear(hidden_dim, index_dim, bias=False)
        self.right = nn.Linear(hidden_dim, index_dim, bias=False)
        self.bias = nn.Parameter(torch.tensor(-2.0))
        self.index_dim = index_dim

    def encode(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm(hidden)
        return self.left(normalized), self.right(normalized)

    def pair_logits(
        self, hidden: torch.Tensor, pairs: torch.Tensor
    ) -> torch.Tensor:
        left, right = self.encode(hidden)
        scale = self.index_dim**-0.5
        forward = torch.sum(left[pairs[:, 0]] * right[pairs[:, 1]], dim=-1)
        reverse = torch.sum(left[pairs[:, 1]] * right[pairs[:, 0]], dim=-1)
        return 0.5 * scale * (forward + reverse) + self.bias

    def pair_probabilities(
        self, hidden: torch.Tensor, pairs: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(self.pair_logits(hidden, pairs))

    def full_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        left, right = self.encode(hidden)
        logits = left @ right.T
        logits = 0.5 * self.index_dim**-0.5 * (logits + logits.T) + self.bias
        return logits * (
            1.0
            - torch.eye(
                len(hidden), device=hidden.device, dtype=logits.dtype
            )
        )

    def full_probabilities(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.full_logits(hidden)
        probabilities = torch.sigmoid(logits)
        return probabilities * (
            1.0
            - torch.eye(
                len(hidden), device=hidden.device, dtype=probabilities.dtype
            )
        )


class SymmetricPairMLPIndexer(nn.Module):
    """Symmetric nonlinear pair baseline operating on compressed site states."""

    def __init__(
        self,
        hidden_dim: int,
        pair_dim: int = 32,
        mlp_dim: int = 64,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.site_projection = nn.Linear(hidden_dim, pair_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim * 3, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, 1),
        )
        nn.init.constant_(self.pair_mlp[-1].bias, -2.0)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.site_projection(self.norm(hidden))

    @staticmethod
    def pair_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [left + right, torch.abs(left - right), left * right], dim=-1
        )

    def pair_logits(
        self, hidden: torch.Tensor, pairs: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.encode(hidden)
        features = self.pair_features(
            encoded[pairs[:, 0]], encoded[pairs[:, 1]]
        )
        return self.pair_mlp(features).squeeze(-1)

    def pair_probabilities(
        self, hidden: torch.Tensor, pairs: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(self.pair_logits(hidden, pairs))

    def full_logits(
        self, hidden: torch.Tensor, chunk_size: int = 64
    ) -> torch.Tensor:
        encoded = self.encode(hidden)
        length = len(hidden)
        rows = []
        for start in range(0, length, chunk_size):
            left = encoded[start : start + chunk_size, None, :]
            right = encoded[None, :, :]
            features = self.pair_features(
                left.expand(-1, length, -1),
                right.expand(len(left), -1, -1),
            )
            rows.append(self.pair_mlp(features).squeeze(-1))
        logits = torch.cat(rows, dim=0)
        logits = 0.5 * (logits + logits.T)
        return logits * (
            1.0
            - torch.eye(
                length, device=hidden.device, dtype=logits.dtype
            )
        )

    def full_probabilities(
        self, hidden: torch.Tensor, chunk_size: int = 64
    ) -> torch.Tensor:
        logits = self.full_logits(hidden, chunk_size=chunk_size)
        probabilities = torch.sigmoid(logits)
        return probabilities * (
            1.0
            - torch.eye(
                len(hidden), device=hidden.device, dtype=probabilities.dtype
            )
        )
