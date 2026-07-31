from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from transformer_disentanglement.content_tile_routing import ContentTileRouter
from transformer_disentanglement.disentangled_sparse_layer import (
    DisentangledSparseInteractionLayer,
)
from transformer_disentanglement.dual_stream_layer import (
    DualStreamOrthogonalInteractionLayer,
)


class DisentangledProteinLM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        states: int = 20,
        rank: int = 8,
        index_dim: int = 16,
        pair_dim: int = 16,
        neighbors: int = 8,
        layers: int = 1,
        max_length: int = 320,
        routing_mode: str = "topk",
        soft_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(states + 2, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.layers = nn.ModuleList(
            [
                DisentangledSparseInteractionLayer(
                    hidden_dim=hidden_dim,
                    states=states,
                    rank=rank,
                    index_dim=index_dim,
                    pair_dim=pair_dim,
                    pair_mlp_dim=hidden_dim,
                    neighbors=neighbors,
                    local_kernel=5,
                    local_exclusion=2,
                    routing_mode=routing_mode,
                    soft_temperature=soft_temperature,
                )
                for _ in range(layers)
            ]
        )

    def set_routing_mode(self, routing_mode: str) -> None:
        for layer in self.layers:
            layer.set_routing_mode(routing_mode)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor | list]:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        outputs = []
        for layer in self.layers:
            output = layer(hidden, tokens)
            hidden = output["hidden"]
            outputs.append(output)
        return {
            "logits": outputs[-1]["logits"],
            "background_logits": outputs[-1]["background_logits"],
            "hidden": hidden,
            "layers": outputs,
        }


class LocalProteinLM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        states: int = 20,
        layers: int = 1,
        max_length: int = 320,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(states + 2, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.depthwise = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=5,
                    padding=2,
                    groups=hidden_dim,
                )
                for _ in range(layers)
            ]
        )
        self.pointwise = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, states)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor | list]:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        for norm, depthwise, pointwise in zip(
            self.norms, self.depthwise, self.pointwise
        ):
            local = depthwise(norm(hidden).transpose(1, 2)).transpose(1, 2)
            hidden = hidden + pointwise(F.silu(local))
        logits = self.output(self.output_norm(hidden))
        return {
            "logits": logits,
            "background_logits": logits,
            "hidden": hidden,
            "layers": [],
        }


class TransformerProteinLM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        states: int = 20,
        layers: int = 1,
        heads: int = 4,
        max_length: int = 320,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(states + 2, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, states)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor | list]:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        hidden = self.encoder(hidden)
        logits = self.output(self.output_norm(hidden))
        return {
            "logits": logits,
            "background_logits": logits,
            "hidden": hidden,
            "layers": [],
        }


class MarginalOrthogonalResidualLM(nn.Module):
    """Local background plus a per-site marginal-orthogonal residual control."""

    def __init__(
        self,
        hidden_dim: int = 64,
        states: int = 20,
        residual_dim: int = 64,
        max_length: int = 320,
    ) -> None:
        super().__init__()
        self.background = LocalProteinLM(
            hidden_dim=hidden_dim,
            states=states,
            layers=1,
            max_length=max_length,
        )
        self.residual_norm = nn.LayerNorm(hidden_dim)
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim, residual_dim),
            nn.SiLU(),
            nn.Linear(residual_dim, states),
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor | list]:
        background_output = self.background(tokens)
        background_logits = background_output["logits"]
        probabilities = background_logits.softmax(dim=-1).detach()
        raw_residual = self.residual_mlp(
            self.residual_norm(background_output["hidden"])
        )
        marginal_mean = torch.sum(
            probabilities * raw_residual, dim=-1, keepdim=True
        )
        residual_logits = raw_residual - marginal_mean
        return {
            "logits": background_logits + residual_logits,
            "background_logits": background_logits,
            "residual_logits": residual_logits,
            "hidden": background_output["hidden"],
            "layers": [],
        }


class DualStreamProteinLM(nn.Module):
    """Linear-mixer task stream plus stable orthogonal sparse interaction values."""

    def __init__(
        self,
        stable_dim: int = 64,
        task_dim: int = 64,
        states: int = 20,
        rank: int = 8,
        index_dim: int = 16,
        pair_dim: int = 16,
        pair_mlp_dim: int | None = None,
        neighbors: int = 8,
        max_length: int = 320,
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
        if task_dim % 2:
            raise ValueError("task_dim must be even for the bidirectional task mixer")
        self.stable_embedding = nn.Embedding(states + 2, stable_dim)
        self.position_embedding = nn.Embedding(max_length, stable_dim)
        self.stable_norm = nn.LayerNorm(stable_dim)
        self.stable_depthwise = nn.Conv1d(
            stable_dim,
            stable_dim,
            kernel_size=5,
            padding=2,
            groups=stable_dim,
        )
        self.stable_pointwise = nn.Linear(stable_dim, stable_dim)
        self.task_input_norm = nn.LayerNorm(stable_dim)
        self.task_mixer = nn.GRU(
            input_size=stable_dim,
            hidden_size=task_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        resolved_pair_mlp_dim = stable_dim if pair_mlp_dim is None else pair_mlp_dim
        if rank_mode == "adaptive" and pair_mlp_dim is None:
            mode_parameter_slope = pair_dim * 3 + 1 + rank
            resolved_pair_mlp_dim = max(
                8,
                round(
                    stable_dim
                    - task_dim * rank / mode_parameter_slope
                ),
            )
        self.interaction = DualStreamOrthogonalInteractionLayer(
            stable_dim=stable_dim,
            task_dim=task_dim,
            states=states,
            rank=rank,
            index_dim=index_dim,
            pair_dim=pair_dim,
            pair_mlp_dim=resolved_pair_mlp_dim,
            neighbors=neighbors,
            local_exclusion=2,
            routing_mode=routing_mode,
            routing_temperature=routing_temperature,
            rank_mode=rank_mode,
            gate_temperature=gate_temperature,
            value_mode=value_mode,
            adapter_count=adapter_count,
            adapter_topk=adapter_topk,
            adapter_bias_update_speed=adapter_bias_update_speed,
        )

    def set_routing_mode(self, routing_mode: str) -> None:
        self.interaction.set_routing_mode(routing_mode)

    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        stable = self.stable_embedding(tokens) + self.position_embedding(positions)[None]
        local = self.stable_depthwise(
            self.stable_norm(stable).transpose(1, 2)
        ).transpose(1, 2)
        stable = stable + self.stable_pointwise(F.silu(local))
        task, _ = self.task_mixer(self.task_input_norm(stable))
        return stable, task

    def forward(
        self, tokens: torch.Tensor, use_interaction: bool = True
    ) -> dict[str, torch.Tensor | None]:
        stable, task = self.encode(tokens)
        if use_interaction:
            output = self.interaction(stable, task, tokens)
            output["encoded_stable"] = stable
            output["encoded_task"] = task
            return output
        background, probabilities, factors, value_state = (
            self.interaction.stable_values(stable, None)
        )
        return {
            "stable_hidden": stable,
            "task_hidden": task,
            "encoded_stable": stable,
            "encoded_task": task,
            "background_logits": background,
            "marginal_probabilities": probabilities,
            "interaction_logits": torch.zeros_like(background),
            "logits": background,
            "index_scores": None,
            "neighbor_indices": None,
            "selected_scores": None,
            "selected_weights": None,
            "routing_weights": None,
            "factors": factors,
            "value_state": value_state,
            "mode": None,
            "mode_gates": None,
            "effective_rank": None,
            "active_modes": None,
        }


class ContentTileProteinLM(nn.Module):
    """Executable single-sequence model with content-tile candidate routing."""

    def __init__(
        self,
        stable_dim: int = 64,
        task_dim: int = 64,
        states: int = 20,
        rank: int = 8,
        index_dim: int = 16,
        pair_dim: int = 16,
        neighbors: int = 8,
        max_length: int = 320,
        tile_dim: int = 16,
        tiles: int = 12,
        selected_tiles: int = 5,
        candidate_budget: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = DualStreamProteinLM(
            stable_dim=stable_dim,
            task_dim=task_dim,
            states=states,
            rank=rank,
            index_dim=index_dim,
            pair_dim=pair_dim,
            neighbors=neighbors,
            max_length=max_length,
        )
        self.router = ContentTileRouter(
            stable_dim=stable_dim,
            task_dim=task_dim,
            tile_dim=tile_dim,
            tiles=tiles,
            selected_tiles=selected_tiles,
            candidate_budget=candidate_budget,
            neighbors=neighbors,
        )

    def forward(
        self, tokens: torch.Tensor, use_interaction: bool = True
    ) -> dict[str, torch.Tensor | None]:
        if len(tokens) != 1:
            raise ValueError("prototype content-tile model currently requires batch size 1")
        output = self.backbone(tokens, use_interaction=False)
        if not use_interaction:
            return output
        length = tokens.shape[1]
        task = output["encoded_task"]
        stable = output["encoded_stable"]
        exact_query, exact_key = self.backbone.interaction.index_features(task)
        valid = self.backbone.interaction.valid_pair_mask(length, tokens.device)
        query_positions = torch.arange(length, device=tokens.device)
        routed = self.router(
            stable,  # type: ignore[arg-type]
            task,  # type: ignore[arg-type]
            query_positions,
            exact_query,
            exact_key,
            valid,
            score_scale=self.backbone.interaction.index_dim**-0.5,
            score_bias=self.backbone.interaction.index_bias,
        )
        neighbors = routed.neighbor_indices
        count = neighbors.shape[-1]
        left = query_positions[:, None].expand(length, count)
        pairs = torch.stack([left, neighbors[0]], dim=-1).reshape(1, -1, 2)
        blocks = self.backbone.interaction.sampled_pair_blocks(
            output["factors"],  # type: ignore[arg-type]
            output["value_state"],  # type: ignore[arg-type]
            task,  # type: ignore[arg-type]
            pairs,
            output["marginal_probabilities"],  # type: ignore[arg-type]
        ).reshape(1, length, count, self.backbone.interaction.states, -1)
        context_tokens = tokens[0, neighbors[0]]
        valid_context = context_tokens < self.backbone.interaction.states
        safe_context = context_tokens.clamp_max(self.backbone.interaction.states - 1)
        columns = safe_context[..., None, None].expand(
            length, count, self.backbone.interaction.states, 1
        )
        messages = torch.gather(blocks[0], dim=-1, index=columns).squeeze(-1)
        messages = messages * valid_context[..., None]
        weights = torch.softmax(
            routed.neighbor_scores[0]
            / self.backbone.interaction.routing_temperature,
            dim=-1,
        )
        interaction_logits = self.backbone.interaction.interaction_scale * torch.sum(
            weights[..., None] * messages, dim=1
        )
        output["interaction_logits"] = interaction_logits[None]
        output["logits"] = output["background_logits"] + interaction_logits[None]
        output["neighbor_indices"] = neighbors
        output["selected_scores"] = routed.neighbor_scores
        output["selected_weights"] = weights[None]
        output["candidate_indices"] = routed.candidate_indices
        output["tile_assignments"] = routed.hard_assignments
        output["selected_tiles"] = routed.selected_tiles
        return output
