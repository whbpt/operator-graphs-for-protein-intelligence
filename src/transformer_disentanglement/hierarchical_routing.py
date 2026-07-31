from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class HierarchicalRoutingOutput:
    candidate_indices: torch.Tensor
    candidate_scores: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_scores: torch.Tensor
    evaluated_pairs: torch.Tensor
    evaluated_nodes: torch.Tensor


class HierarchicalSegmentRouter(nn.Module):
    """Learned coarse segment routing followed by exact candidate scoring."""

    def __init__(
        self,
        task_dim: int,
        node_dim: int = 16,
        branching: int = 4,
        leaf_size: int = 4,
        beam_size: int = 8,
        candidate_budget: int = 32,
        neighbors: int = 8,
    ) -> None:
        super().__init__()
        if min(node_dim, branching, leaf_size, beam_size, candidate_budget, neighbors) <= 0:
            raise ValueError("router dimensions and budgets must be positive")
        self.task_dim = task_dim
        self.node_dim = node_dim
        self.branching = branching
        self.leaf_size = leaf_size
        self.beam_size = beam_size
        self.candidate_budget = candidate_budget
        self.neighbors = neighbors
        self.query_projection = nn.Linear(task_dim, node_dim, bias=False)
        self.key_projection = nn.Linear(task_dim, node_dim, bias=False)

    def split_range(self, segment: tuple[int, int]) -> list[tuple[int, int]]:
        start, end = segment
        length = end - start
        if length <= self.leaf_size:
            return [segment]
        parts = min(self.branching, ceil(length / self.leaf_size))
        base, remainder = divmod(length, parts)
        children = []
        cursor = start
        for index in range(parts):
            width = base + (1 if index < remainder else 0)
            children.append((cursor, cursor + width))
            cursor += width
        return children

    def levels(self, length: int) -> list[list[tuple[int, int]]]:
        levels = [[(0, length)]]
        while any(end - start > self.leaf_size for start, end in levels[-1]):
            next_level = []
            for segment in levels[-1]:
                next_level.extend(self.split_range(segment))
            levels.append(next_level)
        return levels

    @staticmethod
    def segment_means(
        key_state: torch.Tensor, segments: list[tuple[int, int]]
    ) -> torch.Tensor:
        prefix = torch.cat(
            [torch.zeros_like(key_state[:, :1]), key_state.cumsum(dim=1)], dim=1
        )
        starts = torch.tensor(
            [start for start, _ in segments], device=key_state.device
        )
        ends = torch.tensor([end for _, end in segments], device=key_state.device)
        sums = prefix[:, ends] - prefix[:, starts]
        lengths = (ends - starts).to(key_state.dtype)
        return sums / lengths[None, :, None]

    def projected_states(
        self, task_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = F.layer_norm(task_hidden, (task_hidden.shape[-1],))
        query = self.query_projection(normalized)
        key = self.key_projection(normalized)
        return query, key

    def level_scores(
        self,
        task_hidden: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[list[tuple[int, int]]]]:
        query_state, key_state = self.projected_states(task_hidden)
        batch = torch.arange(len(task_hidden), device=task_hidden.device)[:, None]
        selected_query = query_state[batch, query_positions]
        hierarchy = self.levels(task_hidden.shape[1])
        scores = []
        for segments in hierarchy[1:]:
            summaries = self.segment_means(key_state, segments)
            scores.append(
                torch.einsum("bqd,bnd->bqn", selected_query, summaries)
                * self.node_dim**-0.5
            )
        return scores, hierarchy[1:]

    def hierarchical_kl_loss(
        self,
        task_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        target_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        scores, levels = self.level_scores(task_hidden, query_positions)
        losses = []
        for level_scores, segments in zip(scores, levels):
            masses = torch.stack(
                [
                    target_probabilities[..., start:end].sum(dim=-1)
                    for start, end in segments
                ],
                dim=-1,
            )
            masses = masses / masses.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            losses.append(
                F.kl_div(
                    level_scores.log_softmax(dim=-1),
                    masses,
                    reduction="batchmean",
                )
            )
        return torch.stack(losses).mean() if losses else task_hidden.sum() * 0.0

    def forward(
        self,
        task_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        exact_query: torch.Tensor,
        exact_key: torch.Tensor,
        valid_pair_mask: torch.Tensor,
        score_scale: float,
        score_bias: torch.Tensor | float = 0.0,
    ) -> HierarchicalRoutingOutput:
        if len(task_hidden) != 1:
            raise ValueError("prototype hierarchical routing currently requires batch size 1")
        if query_positions.ndim != 1:
            raise ValueError("query_positions must have shape [queries]")
        query_state, key_state = self.projected_states(task_hidden)
        candidate_rows = []
        candidate_score_rows = []
        neighbor_rows = []
        neighbor_score_rows = []
        evaluated_pairs = []
        evaluated_nodes = []
        bias = score_bias
        for row, position in enumerate(query_positions.tolist()):
            beam = [(0, task_hidden.shape[1])]
            node_evaluations = 0
            while any(end - start > self.leaf_size for start, end in beam):
                expanded = []
                for segment in beam:
                    expanded.extend(self.split_range(segment))
                summaries = self.segment_means(key_state, expanded)[0]
                node_scores = torch.mv(
                    summaries, query_state[0, position]
                ) * self.node_dim**-0.5
                node_evaluations += len(expanded)
                count = min(self.beam_size, len(expanded))
                selected = torch.topk(node_scores, k=count).indices.tolist()
                beam = [expanded[index] for index in selected]
            candidates = []
            for start, end in beam:
                candidates.extend(range(start, end))
            valid = valid_pair_mask[row]
            candidates = sorted({index for index in candidates if bool(valid[index])})
            if len(candidates) < min(self.candidate_budget, int(valid.sum())):
                for index in torch.nonzero(valid, as_tuple=False).squeeze(-1).tolist():
                    if index not in candidates:
                        candidates.append(index)
                    if len(candidates) >= self.candidate_budget:
                        break
            candidate_tensor = torch.tensor(
                candidates, device=task_hidden.device, dtype=torch.long
            )
            scores = torch.sum(
                exact_query[0, row][None] * exact_key[0, candidate_tensor], dim=-1
            )
            scores = scores * score_scale + bias
            if len(candidate_tensor) > self.candidate_budget:
                retained = torch.topk(scores, k=self.candidate_budget).indices
                candidate_tensor = candidate_tensor[retained]
                scores = scores[retained]
            neighbor_count = min(self.neighbors, len(candidate_tensor))
            selected = torch.topk(scores, k=neighbor_count)
            candidate_rows.append(candidate_tensor)
            candidate_score_rows.append(scores)
            neighbor_rows.append(candidate_tensor[selected.indices])
            neighbor_score_rows.append(selected.values)
            evaluated_pairs.append(len(candidates))
            evaluated_nodes.append(node_evaluations)
        return HierarchicalRoutingOutput(
            candidate_indices=torch.stack(candidate_rows)[None],
            candidate_scores=torch.stack(candidate_score_rows)[None],
            neighbor_indices=torch.stack(neighbor_rows)[None],
            neighbor_scores=torch.stack(neighbor_score_rows)[None],
            evaluated_pairs=torch.tensor(evaluated_pairs)[None],
            evaluated_nodes=torch.tensor(evaluated_nodes)[None],
        )
