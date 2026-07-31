from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CandidateRoutingOutput:
    candidate_indices: torch.Tensor
    candidate_scores: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_scores: torch.Tensor
    bucket_candidates: torch.Tensor
    evaluated_pairs: torch.Tensor
    fallback_candidates: torch.Tensor


@dataclass
class TreeRoutingOutput:
    neighbor_indices: torch.Tensor
    neighbor_scores: torch.Tensor
    evaluated_pairs: torch.Tensor
    visited_nodes: torch.Tensor


@dataclass
class _KDNode:
    minimum: np.ndarray
    maximum: np.ndarray
    size: int
    indices: np.ndarray | None = None
    left: _KDNode | None = None
    right: _KDNode | None = None


class BoundingBoxMIPSTreeRouter(nn.Module):
    """Exact top-k MIPS with k-d boxes and best-first branch-and-bound."""

    def __init__(self, neighbors: int = 8, leaf_size: int = 8) -> None:
        super().__init__()
        if neighbors <= 0 or leaf_size <= 0:
            raise ValueError("neighbors and leaf_size must be positive")
        self.neighbors = neighbors
        self.leaf_size = leaf_size

    def _build(self, keys: np.ndarray, indices: np.ndarray) -> _KDNode:
        values = keys[indices]
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        if len(indices) <= self.leaf_size:
            return _KDNode(minimum, maximum, len(indices), indices=indices)
        split_dimension = int(np.argmax(maximum - minimum))
        order = np.argsort(values[:, split_dimension], kind="stable")
        midpoint = len(indices) // 2
        left_indices = indices[order[:midpoint]]
        right_indices = indices[order[midpoint:]]
        return _KDNode(
            minimum,
            maximum,
            len(indices),
            left=self._build(keys, left_indices),
            right=self._build(keys, right_indices),
        )

    @staticmethod
    def _upper_bound(query: np.ndarray, node: _KDNode) -> float:
        extreme = np.where(query >= 0, node.maximum, node.minimum)
        return float(np.dot(query, extreme))

    def _search(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        root: _KDNode,
        valid: np.ndarray,
        count: int,
    ) -> tuple[list[int], list[float], int, int]:
        frontier: list[tuple[float, int, _KDNode]] = []
        serial = 0
        heapq.heappush(frontier, (-self._upper_bound(query, root), serial, root))
        best: list[tuple[float, int]] = []
        evaluated = 0
        visited = 0
        while frontier:
            negative_bound, _, node = heapq.heappop(frontier)
            bound = -negative_bound
            if len(best) == count and bound <= best[0][0]:
                break
            visited += 1
            if node.indices is not None:
                indices = node.indices[valid[node.indices]]
                if not len(indices):
                    continue
                scores = keys[indices] @ query
                evaluated += len(indices)
                for index, score in zip(indices.tolist(), scores.tolist()):
                    item = (float(score), int(index))
                    if len(best) < count:
                        heapq.heappush(best, item)
                    elif item[0] > best[0][0]:
                        heapq.heapreplace(best, item)
                continue
            for child in (node.left, node.right):
                if child is None:
                    continue
                child_bound = self._upper_bound(query, child)
                if len(best) < count or child_bound > best[0][0]:
                    serial += 1
                    heapq.heappush(frontier, (-child_bound, serial, child))
        ordered = sorted(best, reverse=True)
        return (
            [index for _, index in ordered],
            [score for score, _ in ordered],
            evaluated,
            visited,
        )

    def forward(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        valid_pair_mask: torch.Tensor,
        score_scale: float,
        score_bias: torch.Tensor | float = 0.0,
    ) -> TreeRoutingOutput:
        if score_scale <= 0:
            raise ValueError("score_scale must be positive")
        if query_features.ndim != 3 or key_features.ndim != 3:
            raise ValueError("features must have shape [batch, length, dim]")
        batch, query_length, feature_dim = query_features.shape
        key_batch, key_length, key_feature_dim = key_features.shape
        if batch != key_batch or feature_dim != key_feature_dim:
            raise ValueError("query and key batch/feature dimensions must match")
        if valid_pair_mask.shape != (query_length, key_length):
            raise ValueError("valid_pair_mask must have shape [queries, keys]")
        available = int(valid_pair_mask.sum(dim=-1).min())
        count = min(self.neighbors, max(available, 1))
        queries = query_features.detach().cpu().numpy()
        keys = key_features.detach().cpu().numpy()
        valid = valid_pair_mask.detach().cpu().numpy().astype(bool)
        bias = (
            float(score_bias.detach().cpu())
            if isinstance(score_bias, torch.Tensor)
            else float(score_bias)
        )
        neighbor_indices = torch.empty(
            batch,
            query_length,
            count,
            dtype=torch.long,
            device=query_features.device,
        )
        neighbor_scores = torch.empty(
            batch,
            query_length,
            count,
            dtype=query_features.dtype,
            device=query_features.device,
        )
        evaluated_pairs = torch.empty(batch, query_length, dtype=torch.long)
        visited_nodes = torch.empty(batch, query_length, dtype=torch.long)
        all_indices = np.arange(key_length, dtype=np.int64)
        for batch_index in range(batch):
            root = self._build(keys[batch_index], all_indices)
            for query_index in range(query_length):
                indices, scores, evaluated, visited = self._search(
                    queries[batch_index, query_index],
                    keys[batch_index],
                    root,
                    valid[query_index],
                    count,
                )
                neighbor_indices[batch_index, query_index] = torch.tensor(
                    indices, device=query_features.device
                )
                neighbor_scores[batch_index, query_index] = (
                    torch.tensor(
                        scores,
                        device=query_features.device,
                        dtype=query_features.dtype,
                    )
                    * score_scale
                    + bias
                )
                evaluated_pairs[batch_index, query_index] = evaluated
                visited_nodes[batch_index, query_index] = visited
        return TreeRoutingOutput(
            neighbor_indices=neighbor_indices,
            neighbor_scores=neighbor_scores,
            evaluated_pairs=evaluated_pairs,
            visited_nodes=visited_nodes,
        )


class HashableTaskAdapter(nn.Module):
    """Learn a shared task embedding that is compatible with fixed LSH planes."""

    def __init__(self, task_dim: int, hash_dim: int = 32) -> None:
        super().__init__()
        self.task_dim = task_dim
        self.hash_dim = hash_dim
        self.projection = nn.Linear(task_dim, hash_dim, bias=False)

    def forward(self, task_hidden: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(task_hidden, (task_hidden.shape[-1],))
        return F.normalize(self.projection(normalized), dim=-1)

    @staticmethod
    def relaxed_codes(
        hash_features: torch.Tensor,
        hyperplanes: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        projections = torch.einsum(
            "bld,tfd->bltf", hash_features, hyperplanes
        )
        return torch.tanh(projections / temperature)

    @staticmethod
    def relaxed_scores(
        codes: torch.Tensor,
        query_positions: torch.Tensor,
        table_temperature: float = 0.1,
    ) -> torch.Tensor:
        if table_temperature <= 0:
            raise ValueError("table_temperature must be positive")
        batch = torch.arange(len(codes), device=codes.device)[:, None]
        query_codes = codes[batch, query_positions]
        affinity = torch.einsum(
            "bqtf,bltf->bqlt", query_codes, codes
        ) / codes.shape[-1]
        return table_temperature * (
            torch.logsumexp(affinity / table_temperature, dim=-1)
            - torch.log(
                torch.tensor(
                    codes.shape[-2],
                    device=codes.device,
                    dtype=codes.dtype,
                )
            )
        )

    @staticmethod
    def regularization(
        codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantization = (1.0 - codes.abs()).square().mean()
        balance = codes.mean(dim=1).square().mean()
        centered = codes - codes.mean(dim=1, keepdim=True)
        covariance = torch.einsum(
            "bltf,bltg->btfg", centered, centered
        ) / max(codes.shape[1], 1)
        identity = torch.eye(
            codes.shape[-1], device=codes.device, dtype=codes.dtype
        )
        off_diagonal = covariance * (1.0 - identity)
        decorrelation = off_diagonal.square().mean()
        return quantization, balance, decorrelation


class MultiTableLSHCandidateRouter(nn.Module):
    """Non-differentiable multi-table LSH followed by exact candidate scoring."""

    def __init__(
        self,
        feature_dim: int,
        tables: int = 4,
        bits: int = 6,
        candidate_budget: int = 32,
        neighbors: int = 8,
        hamming_radius: int = 0,
        seed: int = 20260712,
    ) -> None:
        super().__init__()
        if tables <= 0 or bits <= 0:
            raise ValueError("tables and bits must be positive")
        if bits > 20:
            raise ValueError("bits above 20 are not supported")
        if candidate_budget <= 0 or neighbors <= 0:
            raise ValueError("candidate_budget and neighbors must be positive")
        if hamming_radius not in {0, 1}:
            raise ValueError("hamming_radius must be 0 or 1")
        self.feature_dim = feature_dim
        self.tables = tables
        self.bits = bits
        self.candidate_budget = candidate_budget
        self.neighbors = neighbors
        self.hamming_radius = hamming_radius
        self.seed = seed
        generator = torch.Generator(device="cpu").manual_seed(seed)
        hyperplanes = torch.randn(
            tables, bits, feature_dim, generator=generator
        )
        hyperplanes = F.normalize(hyperplanes, dim=-1)
        bit_weights = 2 ** torch.arange(bits, dtype=torch.long)
        self.register_buffer("hyperplanes", hyperplanes, persistent=False)
        self.register_buffer("bit_weights", bit_weights, persistent=False)

    def hash_codes(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError("feature dimension does not match router")
        normalized = F.normalize(features, dim=-1)
        projections = torch.einsum(
            "bld,tfd->bltf", normalized, self.hyperplanes
        )
        return torch.sum(
            (projections >= 0).long() * self.bit_weights, dim=-1
        )

    def probe_codes(self, code: int) -> list[int]:
        probes = [code]
        if self.hamming_radius == 1:
            probes.extend(code ^ (1 << bit) for bit in range(self.bits))
        return probes

    @staticmethod
    def _bucket_maps(codes: torch.Tensor) -> list[list[dict[int, list[int]]]]:
        code_values = codes.detach().cpu().tolist()
        maps: list[list[dict[int, list[int]]]] = []
        for batch_codes in code_values:
            table_maps: list[dict[int, list[int]]] = []
            table_count = len(batch_codes[0])
            for table in range(table_count):
                buckets: dict[int, list[int]] = {}
                for position, position_codes in enumerate(batch_codes):
                    buckets.setdefault(position_codes[table], []).append(position)
                table_maps.append(buckets)
            maps.append(table_maps)
        return maps

    def forward(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        valid_pair_mask: torch.Tensor,
        score_scale: float,
        score_bias: torch.Tensor | float = 0.0,
        query_positions: torch.Tensor | None = None,
        hash_query_features: torch.Tensor | None = None,
        hash_key_features: torch.Tensor | None = None,
    ) -> CandidateRoutingOutput:
        if query_features.ndim != 3 or key_features.ndim != 3:
            raise ValueError("features must have shape [batch, length, dim]")
        batch, query_length, feature_dim = query_features.shape
        key_batch, key_length, key_feature_dim = key_features.shape
        if batch != key_batch or feature_dim != key_feature_dim:
            raise ValueError("query and key batch/feature dimensions must match")
        if valid_pair_mask.shape != (query_length, key_length):
            raise ValueError("valid_pair_mask must have shape [queries, keys]")
        if query_positions is None:
            query_positions = torch.arange(query_length)
        if query_positions.shape != (query_length,):
            raise ValueError("query_positions must have shape [queries]")
        if hash_query_features is None:
            hash_query_features = query_features
        if hash_key_features is None:
            hash_key_features = key_features
        if hash_query_features.shape[:2] != (batch, query_length):
            raise ValueError("hash query features must match query batch/length")
        if hash_key_features.shape[:2] != (batch, key_length):
            raise ValueError("hash key features must match key batch/length")
        query_position_values = query_positions.detach().cpu().tolist()
        available = int(valid_pair_mask.sum(dim=-1).min())
        candidate_count = min(self.candidate_budget, max(available, 1))
        neighbor_count = min(self.neighbors, candidate_count)
        query_codes = self.hash_codes(hash_query_features)
        key_codes = self.hash_codes(hash_key_features)
        bucket_maps = self._bucket_maps(key_codes)
        query_code_values = query_codes.detach().cpu().tolist()
        valid_lists = [
            torch.nonzero(valid_pair_mask[position], as_tuple=False)
            .squeeze(-1)
            .cpu()
            .tolist()
            for position in range(query_length)
        ]

        candidate_indices = torch.empty(
            batch,
            query_length,
            candidate_count,
            dtype=torch.long,
            device=query_features.device,
        )
        candidate_scores = torch.empty(
            batch,
            query_length,
            candidate_count,
            dtype=query_features.dtype,
            device=query_features.device,
        )
        neighbor_indices = torch.empty(
            batch,
            query_length,
            neighbor_count,
            dtype=torch.long,
            device=query_features.device,
        )
        neighbor_scores = torch.empty(
            batch,
            query_length,
            neighbor_count,
            dtype=query_features.dtype,
            device=query_features.device,
        )
        bucket_candidates = torch.empty(batch, query_length, dtype=torch.long)
        evaluated_pairs = torch.empty(batch, query_length, dtype=torch.long)
        fallback_candidates = torch.empty(batch, query_length, dtype=torch.long)

        for batch_index in range(batch):
            for query_index in range(query_length):
                candidates: set[int] = set()
                for table in range(self.tables):
                    code = query_code_values[batch_index][query_index][table]
                    for probe in self.probe_codes(code):
                        candidates.update(
                            bucket_maps[batch_index][table].get(probe, [])
                        )
                valid_set = set(valid_lists[query_index])
                candidates.intersection_update(valid_set)
                raw_bucket_count = len(candidates)
                if len(candidates) < candidate_count:
                    valid = valid_lists[query_index]
                    start = (
                        query_position_values[query_index] * 2654435761 + self.seed
                    ) % max(len(valid), 1)
                    ordered = valid[start:] + valid[:start]
                    for position in ordered:
                        candidates.add(position)
                        if len(candidates) >= candidate_count:
                            break
                ordered_candidates = sorted(candidates)
                candidate_tensor = torch.tensor(
                    ordered_candidates,
                    dtype=torch.long,
                    device=query_features.device,
                )
                scores = torch.sum(
                    query_features[batch_index, query_index][None]
                    * key_features[batch_index, candidate_tensor],
                    dim=-1,
                )
                scores = scores * score_scale + score_bias
                evaluated_pairs[batch_index, query_index] = len(candidate_tensor)
                if len(candidate_tensor) > candidate_count:
                    retained = torch.topk(scores, k=candidate_count).indices
                    candidate_tensor = candidate_tensor[retained]
                    scores = scores[retained]
                candidate_indices[batch_index, query_index] = candidate_tensor
                candidate_scores[batch_index, query_index] = scores
                selected = torch.topk(scores, k=neighbor_count)
                neighbor_indices[batch_index, query_index] = candidate_tensor[
                    selected.indices
                ]
                neighbor_scores[batch_index, query_index] = selected.values
                bucket_candidates[batch_index, query_index] = raw_bucket_count
                fallback_candidates[batch_index, query_index] = max(
                    candidate_count - raw_bucket_count, 0
                )

        return CandidateRoutingOutput(
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            neighbor_indices=neighbor_indices,
            neighbor_scores=neighbor_scores,
            bucket_candidates=bucket_candidates,
            evaluated_pairs=evaluated_pairs,
            fallback_candidates=fallback_candidates,
        )
