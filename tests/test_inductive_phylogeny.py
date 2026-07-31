from itertools import product

import numpy as np
import torch

from transformer_disentanglement.inductive_phylogeny import (
    SubstitutionModel,
    SymmetricMergeScorer,
    TreeEdge,
    TreeNode,
    canonical_topology,
    initial_merge_supervision,
    leaf_cherries,
    nontrivial_splits,
    patristic_distances,
    phylogenetic_log_likelihood,
    reconstruct_alignment,
    reconstruct_from_distances,
    robinson_foulds_distance,
    simulate_alignment,
)


def quartet_tree() -> TreeNode:
    left = TreeNode(
        children=(
            TreeEdge(TreeNode(label="A"), 0.08),
            TreeEdge(TreeNode(label="B"), 0.12),
        )
    )
    right = TreeNode(
        children=(
            TreeEdge(TreeNode(label="C"), 0.10),
            TreeEdge(TreeNode(label="D"), 0.09),
        )
    )
    return TreeNode(children=(TreeEdge(left, 0.18), TreeEdge(right, 0.22)))


def test_transition_is_stochastic_and_obeys_semigroup() -> None:
    model = SubstitutionModel.jukes_cantor()
    first = model.transition(0.17)
    second = model.transition(0.31)
    combined = model.transition(0.48)
    np.testing.assert_allclose(first.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(first @ second, combined, atol=1e-12)
    np.testing.assert_allclose(
        model.stationary[:, None] * combined,
        model.stationary[None, :] * combined.T,
        atol=1e-12,
    )


def test_pruning_matches_explicit_ancestral_enumeration() -> None:
    model = SubstitutionModel.jukes_cantor()
    internal = TreeNode(
        children=(
            TreeEdge(TreeNode(label="B"), 0.13),
            TreeEdge(TreeNode(label="C"), 0.19),
        )
    )
    tree = TreeNode(
        children=(
            TreeEdge(TreeNode(label="A"), 0.07),
            TreeEdge(internal, 0.11),
        )
    )
    labels = ("A", "B", "C")
    alignment = np.asarray([[0], [1], [2]])
    transition_a = model.transition(0.07)
    transition_internal = model.transition(0.11)
    transition_b = model.transition(0.13)
    transition_c = model.transition(0.19)
    probability = 0.0
    for root_state, internal_state in product(range(4), repeat=2):
        probability += (
            model.stationary[root_state]
            * transition_a[root_state, 0]
            * transition_internal[root_state, internal_state]
            * transition_b[internal_state, 1]
            * transition_c[internal_state, 2]
        )
    observed = phylogenetic_log_likelihood(tree, alignment, labels, model)
    np.testing.assert_allclose(observed, np.log(probability), atol=1e-12)


def test_reversible_likelihood_is_invariant_to_computational_root() -> None:
    model = SubstitutionModel.jukes_cantor()
    center_rooted = TreeNode(
        children=(
            TreeEdge(TreeNode(label="A"), 0.20),
            TreeEdge(TreeNode(label="B"), 0.30),
            TreeEdge(TreeNode(label="C"), 0.40),
        )
    )
    old_center = TreeNode(
        children=(
            TreeEdge(TreeNode(label="B"), 0.30),
            TreeEdge(TreeNode(label="C"), 0.40),
        )
    )
    edge_rooted = TreeNode(
        children=(
            TreeEdge(TreeNode(label="A"), 0.08),
            TreeEdge(old_center, 0.12),
        )
    )
    alignment = np.asarray(
        [
            [0, 1, 2, 3, 0],
            [0, 1, 1, 3, 2],
            [2, 1, 2, 0, 0],
        ]
    )
    labels = ("A", "B", "C")
    center_score = phylogenetic_log_likelihood(
        center_rooted, alignment, labels, model
    )
    edge_score = phylogenetic_log_likelihood(
        edge_rooted, alignment, labels, model
    )
    np.testing.assert_allclose(center_score, edge_score, atol=1e-12)
    assert leaf_cherries(center_rooted) == leaf_cherries(edge_rooted)


def test_patristic_distances_obey_four_point_condition() -> None:
    distances, labels = patristic_distances(quartet_tree())
    index = {label: position for position, label in enumerate(labels)}
    sums = sorted(
        [
            distances[index["A"], index["B"]] + distances[index["C"], index["D"]],
            distances[index["A"], index["C"]] + distances[index["B"], index["D"]],
            distances[index["A"], index["D"]] + distances[index["B"], index["C"]],
        ]
    )
    np.testing.assert_allclose(sums[-1], sums[-2], atol=1e-12)


def test_exact_additive_distances_recover_quartet_split() -> None:
    expected = quartet_tree()
    distances, labels = patristic_distances(expected)
    candidates = reconstruct_from_distances(
        distances,
        labels,
        beam_size=4,
        expansions_per_state=3,
    )
    assert robinson_foulds_distance(expected, candidates[0][0]) == 0
    assert nontrivial_splits(candidates[0][0]) == frozenset(
        (frozenset(("A", "B")),)
    )


def test_reconstruction_is_equivariant_to_alignment_row_order() -> None:
    model = SubstitutionModel.jukes_cantor()
    alignment, labels = simulate_alignment(
        quartet_tree(), 1200, model, np.random.default_rng(17)
    )
    expected = reconstruct_alignment(
        alignment,
        labels,
        model=model,
        beam_size=4,
        expansions_per_state=3,
    )
    permutation = np.asarray([2, 0, 3, 1])
    permuted = reconstruct_alignment(
        alignment[permutation],
        tuple(labels[index] for index in permutation),
        model=model,
        beam_size=4,
        expansions_per_state=3,
    )
    assert nontrivial_splits(expected.tree) == nontrivial_splits(permuted.tree)
    np.testing.assert_allclose(
        expected.log_likelihood, permuted.log_likelihood, atol=1e-8
    )


def test_symmetric_residual_starts_at_neighbor_joining_score() -> None:
    torch.manual_seed(19)
    scorer = SymmetricMergeScorer(feature_dim=6, hidden_dim=8)
    features = torch.randn(7, 6)
    base_scores = torch.randn(7)
    observed = scorer(features, base_scores)
    assert torch.equal(observed, base_scores)
    observed.sum().backward()
    assert scorer.residual[0].weight.grad is not None
    assert scorer.residual[-1].weight.grad is not None


def test_merge_supervision_targets_unrooted_cherries_and_trains() -> None:
    tree = quartet_tree()
    distances, labels = patristic_distances(tree)
    pairs, features, base_scores, target = initial_merge_supervision(
        distances, labels, tree
    )
    positive = {
        frozenset(pair)
        for pair, probability in zip(pairs, target)
        if probability > 0
    }
    assert positive == leaf_cherries(tree)
    scorer = SymmetricMergeScorer(feature_dim=6, hidden_dim=8)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=0.05)
    initial_loss = None
    for _ in range(20):
        logits = scorer(features, base_scores)
        loss = -(target * logits.log_softmax(dim=0)).sum()
        if initial_loss is None:
            initial_loss = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert initial_loss is not None
    assert float(loss.detach()) < initial_loss


def test_simulated_quartet_is_recovered_end_to_end() -> None:
    model = SubstitutionModel.jukes_cantor()
    expected = quartet_tree()
    alignment, labels = simulate_alignment(
        expected, 2000, model, np.random.default_rng(23)
    )
    result = reconstruct_alignment(
        alignment,
        labels,
        model=model,
        beam_size=4,
        expansions_per_state=3,
    )
    assert result.log_likelihood is not None
    assert result.candidate_trees >= 1
    assert robinson_foulds_distance(expected, result.tree) == 0
    assert canonical_topology(result.tree)
