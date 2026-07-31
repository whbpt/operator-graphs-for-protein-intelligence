from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Iterable

import numpy as np
import torch
from scipy.linalg import expm
from scipy.optimize import minimize
from torch import nn


@dataclass(frozen=True)
class TreeEdge:
    child: "TreeNode"
    length: float

    def __post_init__(self) -> None:
        if not isfinite(self.length) or self.length < 0:
            raise ValueError("branch lengths must be finite and nonnegative")


@dataclass(frozen=True)
class TreeNode:
    label: str | None = None
    children: tuple[TreeEdge, ...] = ()

    def __post_init__(self) -> None:
        if self.children and self.label is not None:
            raise ValueError("internal nodes cannot carry observed leaf labels")
        if not self.children and self.label is None:
            raise ValueError("leaf nodes require a label")
        if self.children and len(self.children) < 2:
            raise ValueError("internal nodes require at least two children")

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass(frozen=True)
class SubstitutionModel:
    generator: np.ndarray
    stationary: np.ndarray

    def __post_init__(self) -> None:
        generator = np.asarray(self.generator, dtype=np.float64)
        stationary = np.asarray(self.stationary, dtype=np.float64)
        if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
            raise ValueError("generator must be a square matrix")
        if stationary.shape != (generator.shape[0],):
            raise ValueError("stationary distribution has the wrong shape")
        if np.max(np.abs(generator.sum(axis=1))) > 1e-10:
            raise ValueError("generator rows must sum to zero")
        off_diagonal = generator.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        if np.min(off_diagonal) < -1e-12:
            raise ValueError("generator off-diagonal entries must be nonnegative")
        if np.any(stationary <= 0) or not np.isclose(stationary.sum(), 1.0):
            raise ValueError("stationary distribution must be positive and normalized")
        if np.max(np.abs(stationary @ generator)) > 1e-9:
            raise ValueError("stationary distribution is not stationary for the generator")
        object.__setattr__(self, "generator", generator.copy())
        object.__setattr__(self, "stationary", stationary.copy())

    @property
    def states(self) -> int:
        return len(self.stationary)

    @classmethod
    def jukes_cantor(cls, states: int = 4, rate: float = 1.0) -> "SubstitutionModel":
        if states < 2 or rate <= 0:
            raise ValueError("states must be at least two and rate must be positive")
        generator = np.full((states, states), rate / (states - 1), dtype=np.float64)
        np.fill_diagonal(generator, -rate)
        return cls(generator, np.full(states, 1.0 / states))

    def transition(self, length: float) -> np.ndarray:
        if length < 0:
            raise ValueError("branch length must be nonnegative")
        matrix = expm(length * self.generator)
        matrix = np.maximum(matrix, 0.0)
        return matrix / matrix.sum(axis=1, keepdims=True)


def leaf_labels(tree: TreeNode) -> tuple[str, ...]:
    if tree.is_leaf:
        assert tree.label is not None
        return (tree.label,)
    labels = tuple(label for edge in tree.children for label in leaf_labels(edge.child))
    if len(set(labels)) != len(labels):
        raise ValueError("leaf labels must be unique")
    return labels


def canonical_topology(tree: TreeNode) -> str:
    if tree.is_leaf:
        assert tree.label is not None
        return tree.label
    children = sorted(canonical_topology(edge.child) for edge in tree.children)
    return "(" + ",".join(children) + ")"


def canonical_newick(tree: TreeNode, precision: int = 6) -> str:
    def render(node: TreeNode) -> str:
        if node.is_leaf:
            assert node.label is not None
            return node.label
        children = sorted(
            (
                f"{render(edge.child)}:{edge.length:.{precision}g}"
                for edge in node.children
            )
        )
        return "(" + ",".join(children) + ")"

    return render(tree) + ";"


def nontrivial_splits(tree: TreeNode) -> frozenset[frozenset[str]]:
    all_leaves = frozenset(leaf_labels(tree))
    splits: set[frozenset[str]] = set()

    def canonical_side(descendants: frozenset[str]) -> frozenset[str]:
        complement = all_leaves - descendants
        if len(descendants) < len(complement):
            return descendants
        if len(complement) < len(descendants):
            return complement
        return min(descendants, complement, key=lambda side: tuple(sorted(side)))

    def visit(node: TreeNode) -> frozenset[str]:
        if node.is_leaf:
            assert node.label is not None
            return frozenset((node.label,))
        descendants = frozenset().union(*(visit(edge.child) for edge in node.children))
        if 1 < len(descendants) < len(all_leaves) - 1:
            splits.add(canonical_side(descendants))
        return descendants

    visit(tree)
    return frozenset(splits)


def leaf_cherries(tree: TreeNode) -> frozenset[frozenset[str]]:
    adjacency: dict[int, list[int]] = {}
    leaf_vertex: dict[str, int] = {}
    next_vertex = 0

    def visit(node: TreeNode) -> int:
        nonlocal next_vertex
        vertex = next_vertex
        next_vertex += 1
        adjacency[vertex] = []
        if node.is_leaf:
            assert node.label is not None
            leaf_vertex[node.label] = vertex
        for edge in node.children:
            child = visit(edge.child)
            adjacency[vertex].append(child)
            adjacency[child].append(vertex)
        return vertex

    visit(tree)
    if len(leaf_vertex) < 2:
        return frozenset()
    if len(leaf_vertex) == 2:
        return frozenset((frozenset(leaf_vertex),))
    branching_vertex: dict[int, list[str]] = {}
    for label, leaf in leaf_vertex.items():
        previous = leaf
        current = adjacency[leaf][0]
        while len(adjacency[current]) == 2:
            following = next(
                neighbor for neighbor in adjacency[current] if neighbor != previous
            )
            previous, current = current, following
        branching_vertex.setdefault(current, []).append(label)
    cherries = {
        frozenset((left, right))
        for labels in branching_vertex.values()
        for left, right in combinations(labels, 2)
    }
    return frozenset(cherries)


def robinson_foulds_distance(left: TreeNode, right: TreeNode) -> int:
    if set(leaf_labels(left)) != set(leaf_labels(right)):
        raise ValueError("trees must have the same leaves")
    left_splits = nontrivial_splits(left)
    right_splits = nontrivial_splits(right)
    return len(left_splits - right_splits) + len(right_splits - left_splits)


def scale_tree(tree: TreeNode, scale: float) -> TreeNode:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if tree.is_leaf:
        return tree
    return TreeNode(
        children=tuple(
            TreeEdge(scale_tree(edge.child, scale), edge.length * scale)
            for edge in tree.children
        )
    )


def patristic_distances(
    tree: TreeNode, labels: Iterable[str] | None = None
) -> tuple[np.ndarray, tuple[str, ...]]:
    requested = tuple(sorted(leaf_labels(tree)) if labels is None else labels)
    if set(requested) != set(leaf_labels(tree)) or len(set(requested)) != len(requested):
        raise ValueError("labels must list every tree leaf exactly once")
    adjacency: dict[int, list[tuple[int, float]]] = {}
    leaf_vertex: dict[str, int] = {}
    next_vertex = 0

    def visit(node: TreeNode) -> int:
        nonlocal next_vertex
        vertex = next_vertex
        next_vertex += 1
        adjacency[vertex] = []
        if node.is_leaf:
            assert node.label is not None
            leaf_vertex[node.label] = vertex
        for edge in node.children:
            child = visit(edge.child)
            adjacency[vertex].append((child, edge.length))
            adjacency[child].append((vertex, edge.length))
        return vertex

    visit(tree)
    distances = np.zeros((len(requested), len(requested)), dtype=np.float64)
    for row, label in enumerate(requested):
        start = leaf_vertex[label]
        stack = [(start, -1, 0.0)]
        distance_by_vertex: dict[int, float] = {}
        while stack:
            vertex, parent, distance = stack.pop()
            distance_by_vertex[vertex] = distance
            for neighbor, length in adjacency[vertex]:
                if neighbor != parent:
                    stack.append((neighbor, vertex, distance + length))
        for column, other in enumerate(requested):
            distances[row, column] = distance_by_vertex[leaf_vertex[other]]
    return distances, requested


def _validate_alignment(
    alignment: np.ndarray, labels: Iterable[str], states: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    alignment = np.asarray(alignment, dtype=np.int64)
    labels = tuple(labels)
    if alignment.ndim != 2 or alignment.shape[0] != len(labels):
        raise ValueError("alignment must have shape [leaves, sites]")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    if np.any((alignment < -1) | (alignment >= states)):
        raise ValueError("alignment states must lie in [0, states) or equal -1")
    return alignment, labels


def phylogenetic_log_likelihood(
    tree: TreeNode,
    alignment: np.ndarray,
    labels: Iterable[str],
    model: SubstitutionModel,
) -> float:
    alignment, labels = _validate_alignment(alignment, labels, model.states)
    row_by_label = {label: row for row, label in enumerate(labels)}
    if set(row_by_label) != set(leaf_labels(tree)):
        raise ValueError("tree and alignment must contain the same leaf labels")
    sites = alignment.shape[1]

    def partial(node: TreeNode) -> tuple[np.ndarray, np.ndarray]:
        if node.is_leaf:
            assert node.label is not None
            observed = alignment[row_by_label[node.label]]
            values = np.ones((sites, model.states), dtype=np.float64)
            known = observed >= 0
            values[known] = 0.0
            values[np.nonzero(known)[0], observed[known]] = 1.0
            return values, np.zeros(sites, dtype=np.float64)

        values = np.ones((sites, model.states), dtype=np.float64)
        log_scale = np.zeros(sites, dtype=np.float64)
        for edge in node.children:
            child_values, child_scale = partial(edge.child)
            transported = child_values @ model.transition(edge.length).T
            values *= transported
            log_scale += child_scale
        normalizer = values.sum(axis=1)
        if np.any(normalizer <= 0):
            return values, np.full(sites, -np.inf)
        values /= normalizer[:, None]
        log_scale += np.log(normalizer)
        return values, log_scale

    root_values, log_scale = partial(tree)
    site_probability = root_values @ model.stationary
    if np.any(site_probability <= 0):
        return -np.inf
    return float(np.sum(np.log(site_probability) + log_scale))


def simulate_alignment(
    tree: TreeNode,
    sites: int,
    model: SubstitutionModel,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if sites <= 0:
        raise ValueError("sites must be positive")
    labels = tuple(sorted(leaf_labels(tree)))
    sequences: dict[str, np.ndarray] = {}
    root_states = rng.choice(model.states, size=sites, p=model.stationary)

    def sample_child(parent_states: np.ndarray, edge: TreeEdge) -> None:
        transition = model.transition(edge.length)
        uniforms = rng.random(sites)
        cumulative = np.cumsum(transition[parent_states], axis=1)
        cumulative[:, -1] = 1.0
        child_states = (uniforms[:, None] > cumulative).sum(axis=1)
        node = edge.child
        if node.is_leaf:
            assert node.label is not None
            sequences[node.label] = child_states
            return
        for child_edge in node.children:
            sample_child(child_states, child_edge)

    if tree.is_leaf:
        assert tree.label is not None
        sequences[tree.label] = root_states
    else:
        for edge in tree.children:
            sample_child(root_states, edge)
    return np.stack([sequences[label] for label in labels]), labels


def jukes_cantor_distances(alignment: np.ndarray, states: int = 4) -> np.ndarray:
    alignment = np.asarray(alignment, dtype=np.int64)
    if alignment.ndim != 2:
        raise ValueError("alignment must have shape [leaves, sites]")
    if states < 2:
        raise ValueError("states must be at least two")
    leaves = len(alignment)
    distances = np.zeros((leaves, leaves), dtype=np.float64)
    saturation = (states - 1) / states
    for left, right in combinations(range(leaves), 2):
        valid = (alignment[left] >= 0) & (alignment[right] >= 0)
        if not np.any(valid):
            raise ValueError("every sequence pair must share at least one observed site")
        mismatch = float(np.mean(alignment[left, valid] != alignment[right, valid]))
        argument = max(1.0 - mismatch / saturation, 1e-8)
        distance = -saturation * np.log(argument)
        distances[left, right] = distances[right, left] = distance
    return distances


class SymmetricMergeScorer(nn.Module):
    """Trainable residual on top of a neighbor-joining proposal score."""

    def __init__(self, feature_dim: int = 6, hidden_dim: int = 24) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self, features: torch.Tensor, base_scores: torch.Tensor
    ) -> torch.Tensor:
        if features.shape[:-1] != base_scores.shape:
            raise ValueError("features and base scores have incompatible shapes")
        return base_scores + self.residual(features).squeeze(-1)


@dataclass(frozen=True)
class ReconstructionResult:
    tree: TreeNode
    log_likelihood: float | None
    proposal_score: float
    candidate_trees: int


@dataclass(frozen=True)
class _Cluster:
    leaves: frozenset[str]
    node: TreeNode

    @property
    def size(self) -> int:
        return len(self.leaves)


@dataclass(frozen=True)
class _SearchState:
    clusters: tuple[_Cluster, ...]
    distances: np.ndarray
    proposal_score: float


def _candidate_features(
    state: _SearchState,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    count = len(state.clusters)
    row_sum = state.distances.sum(axis=1)
    positive = state.distances[state.distances > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    scale = max(scale, 1e-8)
    pairs: list[tuple[int, int]] = []
    features = []
    base_scores = []
    for left, right in combinations(range(count), 2):
        distance = state.distances[left, right]
        q_value = (count - 2) * distance - row_sum[left] - row_sum[right]
        left_size = state.clusters[left].size
        right_size = state.clusters[right].size
        features.append(
            [
                distance / scale,
                q_value / (count * scale),
                abs(row_sum[left] - row_sum[right]) / (count * scale),
                abs(left_size - right_size) / (left_size + right_size),
                np.log1p(left_size + right_size),
                (row_sum[left] + row_sum[right]) / (count * scale),
            ]
        )
        base_scores.append(-q_value / (count * scale))
        pairs.append((left, right))
    return pairs, np.asarray(features), np.asarray(base_scores)


def initial_merge_supervision(
    distances: np.ndarray,
    labels: Iterable[str],
    target_tree: TreeNode,
) -> tuple[tuple[tuple[str, str], ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = tuple(labels)
    if set(labels) != set(leaf_labels(target_tree)):
        raise ValueError("target tree and distance labels must agree")
    distances = np.asarray(distances, dtype=np.float64)
    if distances.shape != (len(labels), len(labels)):
        raise ValueError("distance matrix shape does not match labels")
    order = np.argsort(np.asarray(labels))
    ordered_labels = tuple(labels[index] for index in order)
    state = _SearchState(
        tuple(
            _Cluster(frozenset((label,)), TreeNode(label=label))
            for label in ordered_labels
        ),
        distances[np.ix_(order, order)],
        0.0,
    )
    pairs, features, base_scores = _candidate_features(state)
    named_pairs = tuple(
        (ordered_labels[left], ordered_labels[right]) for left, right in pairs
    )
    cherries = leaf_cherries(target_tree)
    target = torch.tensor(
        [
            float(frozenset((left, right)) in cherries)
            for left, right in named_pairs
        ],
        dtype=torch.float32,
    )
    if target.sum() == 0:
        raise ValueError("target tree does not expose any leaf cherry")
    target /= target.sum()
    return (
        named_pairs,
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(base_scores, dtype=torch.float32),
        target,
    )


def _merge_state(
    state: _SearchState,
    left: int,
    right: int,
    log_probability: float,
    minimum_branch: float,
) -> _SearchState:
    if left > right:
        left, right = right, left
    clusters = state.clusters
    count = len(clusters)
    distance = state.distances[left, right]
    row_sum = state.distances.sum(axis=1)
    if count > 2:
        correction = (row_sum[left] - row_sum[right]) / (2 * (count - 2))
    else:
        correction = 0.0
    left_length = max(0.5 * distance + correction, minimum_branch)
    right_length = max(distance - left_length, minimum_branch)
    parent = TreeNode(
        children=(
            TreeEdge(clusters[left].node, left_length),
            TreeEdge(clusters[right].node, right_length),
        )
    )
    merged = _Cluster(clusters[left].leaves | clusters[right].leaves, parent)
    retained = [index for index in range(count) if index not in (left, right)]
    new_clusters = [clusters[index] for index in retained] + [merged]
    order = sorted(
        range(len(new_clusters)),
        key=lambda index: tuple(sorted(new_clusters[index].leaves)),
    )
    new_clusters = [new_clusters[index] for index in order]

    raw_distances = np.zeros((len(retained) + 1, len(retained) + 1))
    for new_i, old_i in enumerate(retained):
        for new_j, old_j in enumerate(retained):
            raw_distances[new_i, new_j] = state.distances[old_i, old_j]
        raw_distances[new_i, -1] = raw_distances[-1, new_i] = max(
            0.5
            * (
                state.distances[old_i, left]
                + state.distances[old_i, right]
                - distance
            ),
            0.0,
        )
    new_distances = raw_distances[np.ix_(order, order)]
    return _SearchState(
        tuple(new_clusters),
        new_distances,
        state.proposal_score + log_probability,
    )


def _finish_tree(state: _SearchState, minimum_branch: float) -> TreeNode:
    clusters = state.clusters
    if len(clusters) == 1:
        return clusters[0].node
    if len(clusters) == 2:
        distance = max(state.distances[0, 1], 2 * minimum_branch)
        return TreeNode(
            children=(
                TreeEdge(clusters[0].node, distance / 2),
                TreeEdge(clusters[1].node, distance / 2),
            )
        )
    if len(clusters) != 3:
        raise ValueError("a neighbor-joining state must finish with two or three clusters")
    d01 = state.distances[0, 1]
    d02 = state.distances[0, 2]
    d12 = state.distances[1, 2]
    lengths = (
        max(0.5 * (d01 + d02 - d12), minimum_branch),
        max(0.5 * (d01 + d12 - d02), minimum_branch),
        max(0.5 * (d02 + d12 - d01), minimum_branch),
    )
    return TreeNode(
        children=tuple(
            TreeEdge(cluster.node, length)
            for cluster, length in zip(clusters, lengths)
        )
    )


def _forest_key(state: _SearchState) -> tuple[str, ...]:
    return tuple(sorted(canonical_topology(cluster.node) for cluster in state.clusters))


def reconstruct_from_distances(
    distances: np.ndarray,
    labels: Iterable[str],
    scorer: SymmetricMergeScorer | None = None,
    beam_size: int = 8,
    expansions_per_state: int = 4,
    minimum_branch: float = 1e-6,
) -> list[tuple[TreeNode, float]]:
    labels = tuple(labels)
    distances = np.asarray(distances, dtype=np.float64)
    if distances.shape != (len(labels), len(labels)):
        raise ValueError("distance matrix shape does not match labels")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("at least two unique labels are required")
    if not np.allclose(distances, distances.T, atol=1e-10):
        raise ValueError("distance matrix must be symmetric")
    if np.max(np.abs(np.diag(distances))) > 1e-10 or np.min(distances) < 0:
        raise ValueError("distance matrix must be nonnegative with a zero diagonal")
    if min(beam_size, expansions_per_state) <= 0:
        raise ValueError("beam sizes must be positive")

    order = np.argsort(np.asarray(labels))
    ordered_labels = tuple(labels[index] for index in order)
    ordered_distances = distances[np.ix_(order, order)]
    initial = _SearchState(
        tuple(_Cluster(frozenset((label,)), TreeNode(label=label)) for label in ordered_labels),
        ordered_distances,
        0.0,
    )
    beam = [initial]
    while len(beam[0].clusters) > 3:
        expanded: list[_SearchState] = []
        for state in beam:
            pairs, features, base_scores = _candidate_features(state)
            feature_tensor = torch.as_tensor(features, dtype=torch.float32)
            base_tensor = torch.as_tensor(base_scores, dtype=torch.float32)
            if scorer is None:
                scores = base_tensor
            else:
                device = next(scorer.parameters()).device
                scores = scorer(
                    feature_tensor.to(device),
                    base_tensor.to(device),
                )
            log_probabilities = torch.log_softmax(scores, dim=0)
            keep = min(expansions_per_state, len(pairs))
            selected = torch.topk(scores, k=keep).indices.tolist()
            for candidate in selected:
                left, right = pairs[candidate]
                expanded.append(
                    _merge_state(
                        state,
                        left,
                        right,
                        float(log_probabilities[candidate].detach()),
                        minimum_branch,
                    )
                )
        unique: dict[tuple[str, ...], _SearchState] = {}
        for state in expanded:
            key = _forest_key(state)
            if key not in unique or state.proposal_score > unique[key].proposal_score:
                unique[key] = state
        beam = sorted(
            unique.values(), key=lambda state: state.proposal_score, reverse=True
        )[:beam_size]
    candidates: dict[frozenset[frozenset[str]], tuple[TreeNode, float]] = {}
    for state in beam:
        tree = _finish_tree(state, minimum_branch)
        key = nontrivial_splits(tree)
        if key not in candidates or state.proposal_score > candidates[key][1]:
            candidates[key] = (tree, state.proposal_score)
    return sorted(candidates.values(), key=lambda item: item[1], reverse=True)


def _branch_lengths(tree: TreeNode) -> np.ndarray:
    lengths: list[float] = []

    def visit(node: TreeNode) -> None:
        for edge in node.children:
            lengths.append(edge.length)
            visit(edge.child)

    visit(tree)
    return np.asarray(lengths, dtype=np.float64)


def _replace_branch_lengths(tree: TreeNode, lengths: np.ndarray) -> TreeNode:
    cursor = 0

    def visit(node: TreeNode) -> TreeNode:
        nonlocal cursor
        if node.is_leaf:
            return node
        children = []
        for edge in node.children:
            length = float(lengths[cursor])
            cursor += 1
            children.append(TreeEdge(visit(edge.child), length))
        return TreeNode(children=tuple(children))

    replaced = visit(tree)
    if cursor != len(lengths):
        raise ValueError("branch-length vector has the wrong size")
    return replaced


def _optimize_branch_lengths(
    tree: TreeNode,
    alignment: np.ndarray,
    labels: tuple[str, ...],
    model: SubstitutionModel,
) -> tuple[TreeNode, float]:
    initial = np.maximum(_branch_lengths(tree), 1e-5)

    def objective(log_lengths: np.ndarray) -> float:
        candidate = _replace_branch_lengths(tree, np.exp(log_lengths))
        return -phylogenetic_log_likelihood(candidate, alignment, labels, model)

    optimum = minimize(
        objective,
        np.log(initial),
        method="L-BFGS-B",
        bounds=[(-11.5, 2.5)] * len(initial),
        options={"maxiter": 80, "ftol": 1e-9},
    )
    optimized = _replace_branch_lengths(tree, np.exp(optimum.x))
    return optimized, -float(optimum.fun)


def reconstruct_alignment(
    alignment: np.ndarray,
    labels: Iterable[str],
    model: SubstitutionModel | None = None,
    scorer: SymmetricMergeScorer | None = None,
    beam_size: int = 8,
    expansions_per_state: int = 4,
    optimize_branches: bool = True,
) -> ReconstructionResult:
    labels = tuple(labels)
    states = model.states if model is not None else 4
    alignment, labels = _validate_alignment(alignment, labels, states)
    distances = jukes_cantor_distances(alignment, states=states)
    candidates = reconstruct_from_distances(
        distances,
        labels,
        scorer=scorer,
        beam_size=beam_size,
        expansions_per_state=expansions_per_state,
    )
    if model is None:
        tree, proposal_score = candidates[0]
        return ReconstructionResult(tree, None, proposal_score, len(candidates))

    evaluated = []
    for tree, proposal_score in candidates:
        if optimize_branches:
            tree, log_likelihood = _optimize_branch_lengths(
                tree, alignment, labels, model
            )
        else:
            log_likelihood = phylogenetic_log_likelihood(
                tree, alignment, labels, model
            )
        evaluated.append((log_likelihood, proposal_score, tree))
    log_likelihood, proposal_score, tree = max(
        evaluated, key=lambda item: (item[0], item[1])
    )
    return ReconstructionResult(
        tree,
        log_likelihood,
        proposal_score,
        len(candidates),
    )
