from __future__ import annotations

import torch
from torch import nn


class MarginalInteractionHead(nn.Module):
    """Factorized marginal and symmetric categorical interaction predictor.

    The head accepts one sequence representation [length, hidden_dim]. Optional
    MSA marginals only define the training-time gauge; inference can use the
    predicted marginal distribution instead.
    """

    def __init__(
        self,
        hidden_dim: int,
        states: int = 20,
        rank: int = 16,
        gate_dim: int = 32,
    ) -> None:
        super().__init__()
        self.states = states
        self.rank = rank
        self.marginal_norm = nn.LayerNorm(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.marginal_projection = nn.Linear(hidden_dim, states)
        self.factor_projection = nn.Linear(hidden_dim, states * rank)
        self.gate_left_projection = nn.Linear(hidden_dim, gate_dim, bias=False)
        self.gate_right_projection = nn.Linear(hidden_dim, gate_dim, bias=False)
        self.gate_dim = gate_dim
        self.gate_bias = nn.Parameter(torch.tensor(-1.5))
        initial_scale = torch.linspace(0.08, -0.08, rank)
        self.mode_scale = nn.Parameter(initial_scale)

    def forward(
        self,
        hidden: torch.Tensor,
        gauge_probabilities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim != 2:
            raise ValueError("hidden must have shape [length, hidden_dim]")
        normalized = self.norm(hidden)
        marginal_hidden = self.marginal_norm(hidden)
        gate_hidden = self.gate_norm(hidden)
        marginal_logits = self.marginal_projection(marginal_hidden)
        marginal_probabilities = marginal_logits.softmax(dim=-1)
        if gauge_probabilities is None:
            gauge_probabilities = marginal_probabilities.detach()
        if gauge_probabilities.shape != marginal_probabilities.shape:
            raise ValueError("gauge probabilities must match marginal probabilities")
        gauge_probabilities = gauge_probabilities / gauge_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        factors = self.factor_projection(normalized).reshape(
            hidden.shape[0], self.states, self.rank
        )
        categorical_mean = torch.sum(
            gauge_probabilities[..., None] * factors, dim=1, keepdim=True
        )
        factors = factors - categorical_mean
        # Pair strength is represented only by the sparse gate, not by a
        # site-specific factor norm that can trivially track entropy.
        factor_rms = factors.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        factors = factors / factor_rms
        return {
            "marginal_logits": marginal_logits,
            "marginal_probabilities": marginal_probabilities,
            "factors": factors,
            "mode_scale": self.mode_scale,
            "gate_left": self.gate_left_projection(gate_hidden),
            "gate_right": self.gate_right_projection(gate_hidden),
            "gate_bias": self.gate_bias,
        }

    @staticmethod
    def pair_gate_logits(
        gate_left: torch.Tensor,
        gate_right: torch.Tensor,
        gate_bias: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        scale = gate_left.shape[-1] ** -0.5
        forward = torch.sum(
            gate_left[pairs[:, 0]] * gate_right[pairs[:, 1]], dim=-1
        )
        reverse = torch.sum(
            gate_left[pairs[:, 1]] * gate_right[pairs[:, 0]], dim=-1
        )
        return 0.5 * scale * (forward + reverse) + gate_bias

    @staticmethod
    def pair_gates(
        gate_left: torch.Tensor,
        gate_right: torch.Tensor,
        gate_bias: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            MarginalInteractionHead.pair_gate_logits(
                gate_left, gate_right, gate_bias, pairs
            )
        )

    @staticmethod
    def full_gate_logits(
        gate_left: torch.Tensor,
        gate_right: torch.Tensor,
        gate_bias: torch.Tensor,
        zero_diagonal: bool = True,
    ) -> torch.Tensor:
        scale = gate_left.shape[-1] ** -0.5
        logits = gate_left @ gate_right.T
        logits = 0.5 * scale * (logits + logits.T) + gate_bias
        if zero_diagonal:
            logits = logits * (
                1.0
                - torch.eye(
                    logits.shape[0], device=logits.device, dtype=logits.dtype
                )
            )
        return logits

    @staticmethod
    def full_gates(
        gate_left: torch.Tensor,
        gate_right: torch.Tensor,
        gate_bias: torch.Tensor,
        zero_diagonal: bool = True,
    ) -> torch.Tensor:
        logits = MarginalInteractionHead.full_gate_logits(
            gate_left,
            gate_right,
            gate_bias,
            zero_diagonal=zero_diagonal,
        )
        gates = torch.sigmoid(logits)
        if zero_diagonal:
            gates = gates * (
                1.0
                - torch.eye(
                    gates.shape[0], device=gates.device, dtype=gates.dtype
                )
            )
        return gates

    @staticmethod
    def pair_blocks(
        factors: torch.Tensor,
        mode_scale: torch.Tensor,
        pairs: torch.Tensor,
        gates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pairs.ndim != 2 or pairs.shape[-1] != 2:
            raise ValueError("pairs must have shape [pair_count, 2]")
        left = factors[pairs[:, 0]]
        right = factors[pairs[:, 1]]
        blocks = torch.einsum("par,pbr,r->pab", left, right, mode_scale)
        if gates is not None:
            blocks = blocks * gates[:, None, None]
        return blocks

    @staticmethod
    def full_blocks(
        factors: torch.Tensor,
        mode_scale: torch.Tensor,
        gates: torch.Tensor | None = None,
        zero_diagonal: bool = True,
    ) -> torch.Tensor:
        blocks = torch.einsum("iar,jbr,r->ijab", factors, factors, mode_scale)
        if gates is not None:
            blocks = blocks * gates[:, :, None, None]
        if zero_diagonal:
            mask = 1.0 - torch.eye(
                blocks.shape[0], device=blocks.device, dtype=blocks.dtype
            )
            blocks = blocks * mask[:, :, None, None]
        return blocks

    @staticmethod
    def full_score_map(
        factors: torch.Tensor,
        mode_scale: torch.Tensor,
        gates: torch.Tensor | None = None,
        zero_diagonal: bool = True,
    ) -> torch.Tensor:
        """Compute block Frobenius norms without materializing [L, L, q, q]."""
        categorical_gram = torch.einsum("iar,ias->irs", factors, factors)
        squared = torch.einsum(
            "irs,jrs,r,s->ij",
            categorical_gram,
            categorical_gram,
            mode_scale,
            mode_scale,
        ).clamp_min(0.0)
        scores = torch.sqrt(squared + 1e-12)
        if gates is not None:
            scores = scores * gates
        if zero_diagonal:
            scores = scores * (
                1.0
                - torch.eye(
                    scores.shape[0], device=scores.device, dtype=scores.dtype
                )
            )
        return scores
