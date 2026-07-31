from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DualStreamOrthogonalInteractionLayer(nn.Module):
    """Dynamic routing keys with stable marginal-orthogonal categorical values."""

    def __init__(
        self,
        stable_dim: int,
        task_dim: int,
        states: int = 20,
        rank: int = 8,
        index_dim: int = 16,
        pair_dim: int = 16,
        pair_mlp_dim: int = 64,
        neighbors: int = 8,
        local_exclusion: int = 2,
        routing_mode: str = "topk",
        routing_temperature: float = 1.0,
        rank_mode: str = "fixed",
        gate_temperature: float = 1.0,
        value_mode: str = "site_shared",
        adapter_count: int = 8,
        adapter_topk: int = 2,
        adapter_bias_update_speed: float = 0.0,
    ) -> None:
        super().__init__()
        if routing_mode not in {"soft", "topk"}:
            raise ValueError("routing_mode must be 'soft' or 'topk'")
        if rank_mode not in {"fixed", "adaptive"}:
            raise ValueError("rank_mode must be 'fixed' or 'adaptive'")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        if value_mode not in {"site_shared", "pair_residual"}:
            raise ValueError("value_mode must be 'site_shared' or 'pair_residual'")
        if adapter_count < 1:
            raise ValueError("adapter_count must be positive")
        if not 1 <= adapter_topk <= adapter_count:
            raise ValueError("adapter_topk must be between 1 and adapter_count")
        if adapter_bias_update_speed < 0:
            raise ValueError("adapter_bias_update_speed must be non-negative")
        self.states = states
        self.rank = rank
        self.index_dim = index_dim
        self.neighbors = neighbors
        self.local_exclusion = local_exclusion
        self.routing_mode = routing_mode
        self.routing_temperature = routing_temperature
        self.rank_mode = rank_mode
        self.gate_temperature = gate_temperature
        self.value_mode = value_mode
        self.adapter_count = adapter_count
        self.adapter_topk = adapter_topk
        self.adapter_bias_update_speed = adapter_bias_update_speed

        self.background_norm = nn.LayerNorm(stable_dim)
        self.factor_norm = nn.LayerNorm(stable_dim)
        self.value_norm = nn.LayerNorm(stable_dim)
        self.task_index_norm = nn.LayerNorm(task_dim)
        self.background_projection = nn.Linear(stable_dim, states)
        self.factor_projection = nn.Linear(stable_dim, states * rank)
        self.value_projection = nn.Linear(stable_dim, pair_dim)
        self.mode_decoder = nn.Sequential(
            nn.Linear(pair_dim * 3, pair_mlp_dim),
            nn.SiLU(),
            nn.Linear(pair_mlp_dim, rank),
        )
        self.index_left = nn.Linear(task_dim, index_dim, bias=False)
        self.index_right = nn.Linear(task_dim, index_dim, bias=False)
        self.index_bias = nn.Parameter(torch.tensor(0.0))
        self.interaction_scale = nn.Parameter(torch.tensor(0.01))
        self.interaction_to_stable = nn.Linear(states, stable_dim, bias=False)
        self.interaction_to_task = nn.Linear(states, task_dim, bias=False)
        self.mode_gate_projection = (
            nn.Linear(task_dim, rank, bias=False)
            if rank_mode == "adaptive"
            else None
        )
        if value_mode == "pair_residual":
            self.adapter_decoder = nn.Sequential(
                nn.Linear(pair_dim * 3, pair_mlp_dim),
                nn.SiLU(),
                nn.Linear(pair_mlp_dim, adapter_count),
            )
            self.adapter_bank = nn.Parameter(
                torch.empty(adapter_count, states, rank)
            )
            nn.init.normal_(self.adapter_bank, std=0.2)
            self.adapter_scale_logit = nn.Parameter(torch.tensor(-1.0))
            self.register_buffer(
                "adapter_routing_bias", torch.zeros(adapter_count)
            )
        else:
            self.adapter_decoder = None
            self.register_parameter("adapter_bank", None)
            self.register_parameter("adapter_scale_logit", None)

    @staticmethod
    def symmetric_pair_features(
        left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [left + right, torch.abs(left - right), left * right], dim=-1
        )

    @staticmethod
    def directional_pair_features(
        left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([left, right, left * right], dim=-1)

    @staticmethod
    def project_pair_blocks(
        blocks: torch.Tensor,
        left_probabilities: torch.Tensor,
        right_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the exact two-sided weighted gauge to assembled pair fields."""
        left_mean = torch.einsum(
            "...a,...ac->...c", left_probabilities, blocks
        )
        projected = blocks - left_mean[..., None, :]
        right_mean = torch.einsum(
            "...ac,...c->...a", projected, right_probabilities
        )
        return projected - right_mean[..., :, None]

    def sparse_adapter_weights(self, logits: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(logits, dim=-1)
        if self.adapter_topk == self.adapter_count:
            return weights
        if not hasattr(self, "adapter_routing_bias"):
            raise RuntimeError("adapter routing bias is not initialized")
        routing_logits = logits + self.adapter_routing_bias
        selected = torch.topk(
            routing_logits, k=self.adapter_topk, dim=-1
        ).indices
        mask = torch.zeros_like(weights).scatter_(-1, selected, 1.0)
        if self.training and self.adapter_bias_update_speed > 0:
            with torch.no_grad():
                load = mask.reshape(-1, self.adapter_count).sum(dim=0)
                target = load.mean()
                direction = torch.sign(load - target)
                self.adapter_routing_bias.sub_(
                    self.adapter_bias_update_speed * direction
                )
                self.adapter_routing_bias.sub_(self.adapter_routing_bias.mean())
        weights = weights * mask
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def pair_factor_residuals(
        self, left_state: torch.Tensor, right_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.value_mode != "pair_residual":
            shape = left_state.shape[:-1] + (self.states, self.rank)
            zeros = torch.zeros(
                shape, device=left_state.device, dtype=left_state.dtype
            )
            return zeros, zeros
        if (
            self.adapter_decoder is None
            or self.adapter_bank is None
            or self.adapter_scale_logit is None
        ):
            raise RuntimeError("pair residual adapters are not initialized")
        left_weights = self.sparse_adapter_weights(
            self.adapter_decoder(
                self.directional_pair_features(left_state, right_state)
            )
        )
        right_weights = self.sparse_adapter_weights(
            self.adapter_decoder(
                self.directional_pair_features(right_state, left_state)
            )
        )
        scale = torch.sigmoid(self.adapter_scale_logit)
        left_residual = scale * torch.einsum(
            "...m,mar->...ar", left_weights, self.adapter_bank
        )
        right_residual = scale * torch.einsum(
            "...m,mar->...ar", right_weights, self.adapter_bank
        )
        return left_residual, right_residual

    def assemble_pair_blocks(
        self,
        left_factor: torch.Tensor,
        right_factor: torch.Tensor,
        left_state: torch.Tensor,
        right_state: torch.Tensor,
        mode: torch.Tensor,
        left_probabilities: torch.Tensor,
        right_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        left_residual, right_residual = self.pair_factor_residuals(
            left_state, right_state
        )
        blocks = torch.einsum(
            "...ar,...cr,...r->...ac",
            left_factor + left_residual,
            right_factor + right_residual,
            mode,
        )
        return self.project_pair_blocks(
            blocks, left_probabilities, right_probabilities
        )

    def set_routing_mode(self, routing_mode: str) -> None:
        if routing_mode not in {"soft", "topk"}:
            raise ValueError("routing_mode must be 'soft' or 'topk'")
        self.routing_mode = routing_mode

    def index_scores(self, task_hidden: torch.Tensor) -> torch.Tensor:
        query, key = self.index_features(task_hidden)
        scores = torch.einsum("bid,bjd->bij", query, key)
        return scores * self.index_dim**-0.5 + self.index_bias

    def index_features(
        self, task_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return MIPS features exactly equivalent to the symmetric index score."""
        normalized = self.task_index_norm(task_hidden)
        left = self.index_left(normalized)
        right = self.index_right(normalized)
        scale = 2**-0.5
        query = scale * torch.cat([right, left], dim=-1)
        key = scale * torch.cat([left, right], dim=-1)
        return query, key

    def valid_pair_mask(self, length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(length, device=device)
        separation = torch.abs(positions[:, None] - positions[None, :])
        return separation > self.local_exclusion

    def stable_values(
        self,
        stable_hidden: torch.Tensor,
        gauge_probabilities: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        background_logits = self.background_projection(
            self.background_norm(stable_hidden)
        )
        marginal_probabilities = background_logits.softmax(dim=-1)
        if gauge_probabilities is None:
            gauge_probabilities = marginal_probabilities.detach()
        gauge_probabilities = gauge_probabilities / gauge_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        factors = self.factor_projection(self.factor_norm(stable_hidden)).reshape(
            *stable_hidden.shape[:2], self.states, self.rank
        )
        categorical_mean = torch.sum(
            gauge_probabilities[..., None] * factors, dim=2, keepdim=True
        )
        factors = factors - categorical_mean
        factors = factors / factors.square().mean(
            dim=2, keepdim=True
        ).sqrt().clamp_min(1e-6)
        value_state = self.value_projection(self.value_norm(stable_hidden))
        return (
            background_logits,
            marginal_probabilities,
            factors,
            value_state,
        )

    def decode_modes(
        self, left_state: torch.Tensor, right_state: torch.Tensor
    ) -> torch.Tensor:
        mode = self.mode_decoder(
            self.symmetric_pair_features(left_state, right_state)
        )
        return mode / mode.square().mean(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-6)

    def mode_task_state(self, task_hidden: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(task_hidden, (task_hidden.shape[-1],))

    def decode_mode_gates(
        self, left_state: torch.Tensor, right_state: torch.Tensor
    ) -> torch.Tensor:
        shape = left_state.shape[:-1] + (self.rank,)
        if self.rank_mode == "fixed":
            return torch.ones(shape, device=left_state.device, dtype=left_state.dtype)
        if self.mode_gate_projection is None:
            raise RuntimeError("adaptive rank requires a mode gate projection")
        left = self.mode_gate_projection(left_state)
        right = self.mode_gate_projection(right_state)
        logits = (left + right) * 2**-0.5 + left * right
        return torch.sigmoid(logits / self.gate_temperature)

    @staticmethod
    def gate_effective_rank(gates: torch.Tensor) -> torch.Tensor:
        squared = gates.square()
        return squared.sum(dim=-1).square() / squared.square().sum(
            dim=-1
        ).clamp_min(1e-8)

    def pair_modes(
        self,
        left_value: torch.Tensor,
        right_value: torch.Tensor,
        left_task: torch.Tensor,
        right_task: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signed_modes = self.decode_modes(left_value, right_value)
        gates = self.decode_mode_gates(left_task, right_task)
        modes = signed_modes * gates
        modes = modes / modes.square().mean(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-6)
        return modes, gates, self.gate_effective_rank(gates)

    def sampled_index_scores(
        self, task_hidden: torch.Tensor, pairs: torch.Tensor
    ) -> torch.Tensor:
        """Score sampled pairs without materializing the full quadratic matrix."""
        query, key = self.index_features(task_hidden)
        batch = torch.arange(len(task_hidden), device=task_hidden.device)[:, None]
        left_index = pairs[..., 0]
        right_index = pairs[..., 1]
        score = torch.sum(
            query[batch, left_index] * key[batch, right_index], dim=-1
        )
        return score * self.index_dim**-0.5 + self.index_bias

    def sampled_pair_outputs(
        self,
        factors: torch.Tensor,
        value_state: torch.Tensor,
        task_hidden: torch.Tensor,
        pairs: torch.Tensor,
        marginal_probabilities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode sampled value blocks and their dynamic mode budgets."""
        batch = torch.arange(len(factors), device=factors.device)[:, None]
        left_index = pairs[..., 0]
        right_index = pairs[..., 1]
        left_factor = factors[batch, left_index]
        right_factor = factors[batch, right_index]
        task_state = self.mode_task_state(task_hidden)
        mode, gates, effective_rank = self.pair_modes(
            value_state[batch, left_index],
            value_state[batch, right_index],
            task_state[batch, left_index],
            task_state[batch, right_index],
        )
        if marginal_probabilities is None:
            if self.value_mode == "pair_residual":
                raise ValueError(
                    "marginal_probabilities are required for pair_residual values"
                )
            blocks = torch.einsum(
                "npar,npcr,npr->npac", left_factor, right_factor, mode
            )
        else:
            blocks = self.assemble_pair_blocks(
                left_factor,
                right_factor,
                value_state[batch, left_index],
                value_state[batch, right_index],
                mode,
                marginal_probabilities[batch, left_index],
                marginal_probabilities[batch, right_index],
            )
        return blocks, gates, effective_rank

    def sampled_pair_blocks(
        self,
        factors: torch.Tensor,
        value_state: torch.Tensor,
        task_hidden: torch.Tensor,
        pairs: torch.Tensor,
        marginal_probabilities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode sampled pair blocks for dense teacher supervision."""
        return self.sampled_pair_outputs(
            factors,
            value_state,
            task_hidden,
            pairs,
            marginal_probabilities,
        )[0]

    def forward(
        self,
        stable_hidden: torch.Tensor,
        task_hidden: torch.Tensor,
        tokens: torch.Tensor,
        gauge_probabilities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if stable_hidden.ndim != 3 or task_hidden.ndim != 3:
            raise ValueError("hidden streams must have shape [batch, length, dim]")
        if stable_hidden.shape[:2] != task_hidden.shape[:2]:
            raise ValueError("stable and task streams must share batch and length")
        if tokens.shape != stable_hidden.shape[:2]:
            raise ValueError("tokens must have shape [batch, length]")

        (
            background_logits,
            marginal_probabilities,
            factors,
            value_state,
        ) = self.stable_values(stable_hidden, gauge_probabilities)
        scores = self.index_scores(task_hidden)
        mode_task_state = self.mode_task_state(task_hidden)
        batch, length = tokens.shape
        valid = self.valid_pair_mask(length, tokens.device)
        masked_scores = scores.masked_fill(~valid[None], -torch.inf)
        known = tokens < self.states
        clipped_tokens = tokens.clamp(max=self.states - 1)
        gather_index = clipped_tokens[..., None, None].expand(
            -1, -1, 1, self.rank
        )
        observed_factors = torch.gather(
            factors, dim=2, index=gather_index
        ).squeeze(2)
        observed_factors = observed_factors * known[..., None]

        if self.routing_mode == "soft":
            routing_weights = torch.softmax(
                masked_scores / self.routing_temperature, dim=-1
            )
            left_state = value_state[:, :, None, :].expand(-1, -1, length, -1)
            right_state = value_state[:, None, :, :].expand(-1, length, -1, -1)
            left_task_state = mode_task_state[:, :, None, :].expand(
                -1, -1, length, -1
            )
            right_task_state = mode_task_state[:, None, :, :].expand(
                -1, length, -1, -1
            )
            mode, mode_gates, effective_rank = self.pair_modes(
                left_state, right_state, left_task_state, right_task_state
            )
            if self.value_mode == "pair_residual":
                left_factor = factors[:, :, None, :, :].expand(
                    -1, -1, length, -1, -1
                )
                right_factor = factors[:, None, :, :, :].expand(
                    -1, length, -1, -1, -1
                )
                left_probabilities = marginal_probabilities[:, :, None, :].expand(
                    -1, -1, length, -1
                )
                right_probabilities = marginal_probabilities[:, None, :, :].expand(
                    -1, length, -1, -1
                )
                blocks = self.assemble_pair_blocks(
                    left_factor,
                    right_factor,
                    left_state,
                    right_state,
                    mode,
                    left_probabilities,
                    right_probabilities,
                )
                context_tokens = clipped_tokens[:, None, :].expand(
                    -1, length, -1
                )
                columns = context_tokens[..., None, None].expand(
                    -1, -1, -1, self.states, 1
                )
                messages = torch.gather(blocks, dim=-1, index=columns).squeeze(-1)
                messages = messages * known[:, None, :, None]
                interaction_logits = torch.sum(
                    routing_weights[..., None] * messages, dim=2
                )
            else:
                interaction_logits = torch.einsum(
                    "biar,bjr,bijr,bij->bia",
                    factors,
                    observed_factors,
                    mode,
                    routing_weights,
                )
            neighbor_indices = torch.topk(
                masked_scores,
                k=min(self.neighbors, int(valid.sum(dim=-1).min())),
                dim=-1,
            ).indices
            selected_scores = torch.gather(scores, 2, neighbor_indices)
            selected_weights = torch.gather(
                routing_weights, 2, neighbor_indices
            )
        else:
            available = int(valid.sum(dim=-1).min())
            count = min(self.neighbors, max(available, 1))
            selected_scores, neighbor_indices = torch.topk(
                masked_scores, k=count, dim=-1
            )
            selected_weights = torch.softmax(
                selected_scores / self.routing_temperature, dim=-1
            )
            batch_index = torch.arange(batch, device=tokens.device)[:, None, None]
            left_state = value_state[:, :, None, :].expand(-1, -1, count, -1)
            right_state = value_state[batch_index, neighbor_indices]
            left_task_state = mode_task_state[:, :, None, :].expand(
                -1, -1, count, -1
            )
            right_task_state = mode_task_state[batch_index, neighbor_indices]
            mode, mode_gates, effective_rank = self.pair_modes(
                left_state, right_state, left_task_state, right_task_state
            )
            left_factor = factors[:, :, None, :, :].expand(
                -1, -1, count, -1, -1
            )
            if self.value_mode == "pair_residual":
                right_factor = factors[batch_index, neighbor_indices]
                left_probabilities = marginal_probabilities[:, :, None, :].expand(
                    -1, -1, count, -1
                )
                right_probabilities = marginal_probabilities[
                    batch_index, neighbor_indices
                ]
                blocks = self.assemble_pair_blocks(
                    left_factor,
                    right_factor,
                    left_state,
                    right_state,
                    mode,
                    left_probabilities,
                    right_probabilities,
                )
                context_tokens = clipped_tokens[batch_index, neighbor_indices]
                columns = context_tokens[..., None, None].expand(
                    -1, -1, -1, self.states, 1
                )
                messages = torch.gather(blocks, dim=-1, index=columns).squeeze(-1)
                messages = messages * known[batch_index, neighbor_indices][..., None]
                interaction_logits = torch.sum(
                    selected_weights[..., None] * messages, dim=2
                )
            else:
                right_observed = observed_factors[batch_index, neighbor_indices]
                interaction_logits = torch.einsum(
                    "bikar,bikr,bikr,bik->bia",
                    left_factor,
                    right_observed,
                    mode,
                    selected_weights,
                )
            routing_weights = None

        interaction_logits = self.interaction_scale * interaction_logits
        stable_update = self.interaction_to_stable(interaction_logits)
        task_update = self.interaction_to_task(interaction_logits)
        return {
            "stable_hidden": stable_hidden + stable_update,
            "task_hidden": task_hidden + task_update,
            "background_logits": background_logits,
            "marginal_probabilities": marginal_probabilities,
            "interaction_logits": interaction_logits,
            "logits": background_logits + interaction_logits,
            "index_scores": scores,
            "neighbor_indices": neighbor_indices,
            "selected_scores": selected_scores,
            "selected_weights": selected_weights,
            "routing_weights": routing_weights,
            "factors": factors,
            "value_state": value_state,
            "mode": mode,
            "mode_gates": mode_gates,
            "effective_rank": effective_rank,
            "active_modes": (mode_gates >= 0.5).sum(dim=-1),
        }
