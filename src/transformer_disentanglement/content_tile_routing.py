from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ContentTileRoutingOutput:
    candidate_indices: torch.Tensor
    candidate_scores: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_scores: torch.Tensor
    selected_tiles: torch.Tensor
    hard_assignments: torch.Tensor
    evaluated_pairs: torch.Tensor
    evaluated_tiles: torch.Tensor


class ContentTileRouter(nn.Module):
    """Capacity-balanced content tiles followed by bounded exact pair scoring."""

    def __init__(
        self,
        stable_dim: int,
        task_dim: int,
        tile_dim: int = 16,
        tiles: int = 12,
        selected_tiles: int = 4,
        candidate_budget: int = 64,
        neighbors: int = 8,
        sinkhorn_steps: int = 20,
        assignment_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if min(
            tile_dim,
            tiles,
            selected_tiles,
            candidate_budget,
            neighbors,
            sinkhorn_steps,
        ) <= 0:
            raise ValueError("router dimensions and budgets must be positive")
        if selected_tiles > tiles:
            raise ValueError("selected_tiles cannot exceed tiles")
        self.stable_dim = stable_dim
        self.task_dim = task_dim
        self.tile_dim = tile_dim
        self.tiles = tiles
        self.selected_tiles_count = selected_tiles
        self.candidate_budget = candidate_budget
        self.neighbors = neighbors
        self.sinkhorn_steps = sinkhorn_steps
        self.assignment_temperature = assignment_temperature
        self.stable_projection = nn.Linear(stable_dim, tile_dim, bias=False)
        self.query_projection = nn.Linear(task_dim, tile_dim, bias=False)
        self.tile_prototypes = nn.Parameter(torch.empty(tiles, tile_dim))
        nn.init.normal_(self.tile_prototypes, std=tile_dim**-0.5)

    def projected_states(
        self,
        stable_hidden: torch.Tensor,
        task_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stable = F.layer_norm(stable_hidden, (stable_hidden.shape[-1],))
        task = F.layer_norm(task_hidden, (task_hidden.shape[-1],))
        keys = F.normalize(self.stable_projection(stable), dim=-1)
        queries = F.normalize(self.query_projection(task), dim=-1)
        prototypes = F.normalize(self.tile_prototypes, dim=-1)
        return keys, queries, prototypes

    def assignment_logits(
        self,
        stable_hidden: torch.Tensor,
        task_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, queries, prototypes = self.projected_states(stable_hidden, task_hidden)
        assignments = torch.einsum("bld,md->blm", keys, prototypes)
        tile_scores = torch.einsum("bld,md->blm", queries, prototypes)
        return assignments, tile_scores

    def soft_balanced_assignments(self, logits: torch.Tensor) -> torch.Tensor:
        """Return row-stochastic assignments with approximately equal tile mass."""
        scaled = logits / self.assignment_temperature
        probabilities = torch.exp(scaled - scaled.amax(dim=-1, keepdim=True))
        target_column_mass = logits.shape[1] / self.tiles
        for _ in range(self.sinkhorn_steps):
            probabilities = probabilities / probabilities.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            probabilities = probabilities * (
                target_column_mass
                / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
            )
        return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )

    @staticmethod
    def balanced_hard_assignments(logits: torch.Tensor) -> torch.Tensor:
        """Greedily assign every token to a bounded-capacity tile."""
        if logits.ndim != 2:
            raise ValueError("logits must have shape [length, tiles]")
        length, tiles = logits.shape
        capacity = ceil(length / tiles)
        flat_order = torch.argsort(logits.detach().cpu().flatten(), descending=True)
        assignments = torch.full((length,), -1, dtype=torch.long)
        counts = torch.zeros(tiles, dtype=torch.long)
        for flat_index in flat_order.tolist():
            token = flat_index // tiles
            tile = flat_index % tiles
            if assignments[token] < 0 and counts[tile] < capacity:
                assignments[token] = tile
                counts[tile] += 1
            if bool(torch.all(assignments >= 0)):
                break
        if bool(torch.any(assignments < 0)):
            raise RuntimeError("capacity-balanced assignment did not cover every token")
        return assignments.to(logits.device)

    def routing_kl_loss(
        self,
        stable_hidden: torch.Tensor,
        task_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        target_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Fit a tile mixture to a dense target distribution over partner positions."""
        assignment_logits, all_tile_scores = self.assignment_logits(
            stable_hidden, task_hidden
        )
        assignments = self.soft_balanced_assignments(assignment_logits)
        membership = assignments / assignments.sum(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        batch = torch.arange(len(task_hidden), device=task_hidden.device)[:, None]
        tile_probabilities = torch.softmax(
            all_tile_scores[batch, query_positions], dim=-1
        )
        predicted = torch.einsum("bqm,blm->bql", tile_probabilities, membership)
        valid = target_probabilities > 0
        predicted = predicted.masked_fill(~valid, 0.0)
        predicted = predicted / predicted.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target = target_probabilities / target_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return F.kl_div(
            predicted.clamp_min(1e-8).log(),
            target,
            reduction="batchmean",
        )

    def forward(
        self,
        stable_hidden: torch.Tensor,
        task_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        exact_query: torch.Tensor,
        exact_key: torch.Tensor,
        valid_pair_mask: torch.Tensor,
        score_scale: float,
        score_bias: torch.Tensor | float = 0.0,
    ) -> ContentTileRoutingOutput:
        if len(stable_hidden) != 1:
            raise ValueError("prototype content-tile routing currently requires batch size 1")
        if query_positions.ndim != 1:
            raise ValueError("query_positions must have shape [queries]")
        assignment_logits, all_tile_scores = self.assignment_logits(
            stable_hidden, task_hidden
        )
        hard = self.balanced_hard_assignments(assignment_logits[0])
        candidate_rows = []
        candidate_score_rows = []
        neighbor_rows = []
        neighbor_score_rows = []
        tile_rows = []
        pair_counts = []
        tile_counts = []
        for row, position in enumerate(query_positions.tolist()):
            valid = valid_pair_mask[row]
            tile_order = torch.argsort(all_tile_scores[0, position], descending=True)
            candidates: list[int] = []
            chosen_tiles: list[int] = []
            for tile_tensor in tile_order:
                tile = int(tile_tensor)
                positions = torch.nonzero(
                    (hard == tile) & valid, as_tuple=False
                ).squeeze(-1)
                if not len(positions):
                    continue
                chosen_tiles.append(tile)
                remaining = self.candidate_budget - len(candidates)
                if len(positions) > remaining:
                    affinity = assignment_logits[0, positions, tile]
                    positions = positions[torch.topk(affinity, k=remaining).indices]
                candidates.extend(positions.tolist())
                if (
                    len(chosen_tiles) >= self.selected_tiles_count
                    and len(candidates) >= min(self.candidate_budget, int(valid.sum()))
                ):
                    break
                if len(candidates) >= self.candidate_budget:
                    break
            candidate_tensor = torch.tensor(
                candidates, device=stable_hidden.device, dtype=torch.long
            )
            scores = torch.sum(
                exact_query[0, row][None] * exact_key[0, candidate_tensor], dim=-1
            )
            scores = scores * score_scale + score_bias
            neighbor_count = min(self.neighbors, len(candidate_tensor))
            selected = torch.topk(scores, k=neighbor_count)
            padded_tiles = chosen_tiles[: self.tiles]
            padded_tiles.extend([-1] * (self.tiles - len(padded_tiles)))
            candidate_rows.append(candidate_tensor)
            candidate_score_rows.append(scores)
            neighbor_rows.append(candidate_tensor[selected.indices])
            neighbor_score_rows.append(selected.values)
            tile_rows.append(
                torch.tensor(padded_tiles, device=stable_hidden.device)
            )
            pair_counts.append(len(candidate_tensor))
            tile_counts.append(len(chosen_tiles))
        return ContentTileRoutingOutput(
            candidate_indices=torch.stack(candidate_rows)[None],
            candidate_scores=torch.stack(candidate_score_rows)[None],
            neighbor_indices=torch.stack(neighbor_rows)[None],
            neighbor_scores=torch.stack(neighbor_score_rows)[None],
            selected_tiles=torch.stack(tile_rows)[None],
            hard_assignments=hard[None],
            evaluated_pairs=torch.tensor(pair_counts, device=stable_hidden.device)[None],
            evaluated_tiles=torch.tensor(tile_counts, device=stable_hidden.device)[None],
        )
