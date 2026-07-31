import numpy as np
import torch

from transformer_disentanglement.categorical_interactions import (
    block_transpose_symmetrize,
    blocks_to_matrix,
    categorical_double_center,
    matrix_to_blocks,
    weighted_categorical_center,
)
from transformer_disentanglement.interaction_model import MarginalInteractionHead
from transformer_disentanglement.data import (
    encode_alignment,
    sample_pssm_null,
    sample_global_composition_null,
    shuffle_columns_null,
    site_frequencies,
)
from transformer_disentanglement.metrics import (
    average_product_correction,
    binary_average_precision,
    binary_contact_precision,
    binary_roc_auc,
    leading_mode_fraction,
)
from transformer_disentanglement.seqmodels_benchmark import (
    EntropyNullModel,
    entropy_pair_scale,
    fit_entropy_null_model,
    residualize_entropy_background,
)
from transformer_disentanglement.relation_baselines import (
    SymmetricBilinearIndexer,
    SymmetricPairMLPIndexer,
)
from transformer_disentanglement.pair_conditioned_interaction import (
    PairConditionedInteractionHead,
)
from transformer_disentanglement.phylogeny import (
    residualize_phylogeny_modes,
)
from transformer_disentanglement.disentangled_sparse_layer import (
    DisentangledSparseInteractionLayer,
)
from transformer_disentanglement.demo_language_models import (
    DisentangledProteinLM,
    LocalProteinLM,
    MarginalOrthogonalResidualLM,
    TransformerProteinLM,
)
from transformer_disentanglement.task_gradient import (
    marginal_orthogonal_task_gradient,
)


def test_pssm_null_preserves_marginals() -> None:
    sequences = ["AAAA", "ACAC", "CACA", "CCCC"]
    rng = np.random.default_rng(3)
    null = sample_pssm_null(sequences, depth=20_000, rng=rng, keep_query=False)
    expected = site_frequencies(encode_alignment(sequences))
    observed = site_frequencies(encode_alignment(null))
    assert np.max(np.abs(expected - observed)) < 0.02


def test_apc_removes_rank_one_background() -> None:
    vector = np.asarray([1.0, 2.0, 3.0, 4.0])
    matrix = np.outer(vector, vector)
    corrected = average_product_correction(matrix)
    assert leading_mode_fraction(matrix) > 0.999
    assert np.max(np.abs(corrected)) < 1e-12


def test_column_shuffle_preserves_finite_sample_marginals() -> None:
    sequences = ["AAAA", "ACAC", "CACA", "CCCC"]
    rng = np.random.default_rng(5)
    shuffled = shuffle_columns_null(sequences, rng)
    expected = site_frequencies(encode_alignment(sequences))
    observed = site_frequencies(encode_alignment(shuffled))
    assert np.array_equal(expected, observed)


def test_global_null_preserves_overall_composition() -> None:
    sequences = ["AAAA", "ACAC", "CACA", "CCCC"]
    rng = np.random.default_rng(9)
    sampled = sample_global_composition_null(sequences, 20_000, rng)
    expected = np.bincount(encode_alignment(sequences).ravel(), minlength=22)
    observed = np.bincount(encode_alignment(sampled).ravel(), minlength=22)
    expected = expected / expected.sum()
    observed = observed / observed.sum()
    assert np.max(np.abs(expected - observed)) < 0.01


def test_categorical_projection_and_block_matrix_round_trip() -> None:
    rng = np.random.default_rng(11)
    blocks = rng.normal(size=(3, 3, 4, 4))
    centered = categorical_double_center(blocks)
    symmetric = block_transpose_symmetrize(centered)
    assert np.max(np.abs(symmetric.sum(axis=-1))) < 1e-12
    assert np.max(np.abs(symmetric.sum(axis=-2))) < 1e-12
    assert np.allclose(symmetric, symmetric.transpose(1, 0, 3, 2))
    matrix = blocks_to_matrix(symmetric)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(matrix_to_blocks(matrix, 3, 4), symmetric)


def test_weighted_categorical_projection_removes_marginals() -> None:
    rng = np.random.default_rng(13)
    blocks = rng.normal(size=(3, 3, 4, 4))
    probabilities = rng.dirichlet(np.ones(4), size=3)
    centered = weighted_categorical_center(blocks, probabilities)
    left = np.einsum("ia,ijab->ijb", probabilities, centered)
    right = np.einsum("ijab,jb->ija", centered, probabilities)
    assert np.max(np.abs(left)) < 1e-12
    assert np.max(np.abs(right)) < 1e-12


def test_factor_head_enforces_weighted_gauge_and_block_symmetry() -> None:
    torch.manual_seed(17)
    hidden = torch.randn(5, 7)
    probabilities = torch.softmax(torch.randn(5, 4), dim=-1)
    head = MarginalInteractionHead(hidden_dim=7, states=4, rank=3)
    output = head(hidden, gauge_probabilities=probabilities)
    gates = head.full_gates(
        output["gate_left"], output["gate_right"], output["gate_bias"]
    )
    blocks = head.full_blocks(output["factors"], output["mode_scale"], gates=gates)
    left = torch.einsum("ia,ijab->ijb", probabilities, blocks)
    right = torch.einsum("ijab,jb->ija", blocks, probabilities)
    assert torch.max(torch.abs(left)).item() < 1e-5
    assert torch.max(torch.abs(right)).item() < 1e-5
    assert torch.allclose(blocks, blocks.permute(1, 0, 3, 2), atol=1e-6)
    assert torch.allclose(gates, gates.T, atol=1e-6)


def test_binary_contact_precision_respects_mask() -> None:
    scores = np.zeros((5, 5))
    contacts = np.zeros((5, 5))
    mask = np.ones((5, 5), dtype=bool)
    scores[0, 4] = scores[4, 0] = 10.0
    contacts[0, 4] = contacts[4, 0] = 1.0
    assert binary_contact_precision(
        scores, contacts, mask, min_separation=1, top_fraction=0.2
    ) == 1.0


def test_entropy_background_residual_keeps_only_excess_strength() -> None:
    pssm = np.asarray([[0.5, 0.5], [0.8, 0.2], [0.5, 0.5]])
    pairs = np.asarray([[0, 1], [0, 2]])
    blocks = np.asarray(
        [
            [[0.1, -0.1], [-0.1, 0.1]],
            [[0.4, -0.4], [-0.4, 0.4]],
        ],
        dtype=np.float32,
    )
    scale = entropy_pair_scale(pssm, pairs)
    assert np.all(scale > 0)
    null = EntropyNullModel(location=0.3, scale=0.0)
    residual, reliability = residualize_entropy_background(
        blocks, pairs, pssm, null, z_threshold=0.0
    )
    assert reliability[0] == 0.0
    assert 0.0 < reliability[1] < 1.0
    np.testing.assert_allclose(residual[1], blocks[1] * reliability[1])


def test_fit_entropy_null_model_is_robust_to_one_outlier() -> None:
    pssm = np.full((4, 2), 0.5)
    pairs = np.asarray([[0, 1], [0, 2], [1, 2], [2, 3]])
    blocks = np.ones((4, 2, 2), dtype=np.float32) * 0.1
    blocks[-1] *= 100.0
    model = fit_entropy_null_model(blocks, pairs, pssm)
    assert model.location < 1.0


def test_binary_ranking_metrics_reward_perfect_order() -> None:
    scores = np.asarray([0.9, 0.8, 0.2, 0.1])
    labels = np.asarray([1, 1, 0, 0])
    assert binary_average_precision(scores, labels) == 1.0
    assert binary_roc_auc(scores, labels) == 1.0


def test_relation_indexers_are_symmetric_and_match_sampled_pairs() -> None:
    torch.manual_seed(23)
    hidden = torch.randn(6, 8)
    pairs = torch.tensor([[0, 4], [2, 5], [1, 3]])
    for model in [
        SymmetricBilinearIndexer(8, index_dim=4),
        SymmetricPairMLPIndexer(8, pair_dim=4, mlp_dim=7),
    ]:
        full = model.full_probabilities(hidden)
        sampled = model.pair_probabilities(hidden, pairs)
        assert torch.allclose(full, full.T, atol=1e-6)
        assert torch.allclose(sampled, full[pairs[:, 0], pairs[:, 1]], atol=1e-6)


def test_pair_conditioned_head_obeys_gauge_and_block_symmetry() -> None:
    torch.manual_seed(29)
    hidden = torch.randn(5, 8)
    probabilities = torch.softmax(torch.randn(5, 4), dim=-1)
    head = PairConditionedInteractionHead(
        hidden_dim=8, states=4, rank=3, pair_dim=5, pair_mlp_dim=7
    )
    output = head(hidden, gauge_probabilities=probabilities)
    forward_pairs = torch.tensor([[0, 3], [1, 4]])
    reverse_pairs = forward_pairs.flip(-1)
    amplitude, mode = head.pair_parameters(output["pair_state"], forward_pairs)
    reverse_amplitude, reverse_mode = head.pair_parameters(
        output["pair_state"], reverse_pairs
    )
    blocks = head.pair_blocks(
        output["factors"], forward_pairs, amplitude, mode
    )
    reverse_blocks = head.pair_blocks(
        output["factors"], reverse_pairs, reverse_amplitude, reverse_mode
    )
    left = torch.einsum("pa,pab->pb", probabilities[forward_pairs[:, 0]], blocks)
    right = torch.einsum("pab,pb->pa", blocks, probabilities[forward_pairs[:, 1]])
    assert torch.max(torch.abs(left)).item() < 1e-5
    assert torch.max(torch.abs(right)).item() < 1e-5
    assert torch.allclose(blocks, reverse_blocks.transpose(-1, -2), atol=1e-6)
    assert torch.allclose(amplitude, reverse_amplitude, atol=1e-6)
    assert torch.allclose(mode, reverse_mode, atol=1e-6)


def test_phylogeny_residual_projection_preserves_weighted_gauge() -> None:
    rng = np.random.default_rng(31)
    msa = rng.integers(0, 4, size=(20, 5), dtype=np.uint8)
    weights = rng.uniform(0.2, 1.0, size=20).astype(np.float32)
    pssm = np.stack(
        [
            np.bincount(msa[:, site], weights=weights, minlength=4)
            for site in range(msa.shape[1])
        ]
    )
    pssm /= pssm.sum(axis=-1, keepdims=True)
    pairs = np.asarray([[0, 3], [1, 4], [0, 2]])
    blocks = rng.normal(size=(3, 4, 4)).astype(np.float32)
    loadings = rng.normal(size=(5, 4, 2)).astype(np.float32)
    residual = residualize_phylogeny_modes(
        blocks, pairs, loadings, msa, weights, pssm
    )
    left = np.einsum("pa,pab->pb", pssm[pairs[:, 0]], residual)
    right = np.einsum("pab,pb->pa", residual, pssm[pairs[:, 1]])
    assert np.max(np.abs(left)) < 1e-5
    assert np.max(np.abs(right)) < 1e-5


def test_sparse_interaction_layer_has_zero_marginal_interaction_mean() -> None:
    torch.manual_seed(37)
    layer = DisentangledSparseInteractionLayer(
        hidden_dim=12,
        states=4,
        rank=3,
        index_dim=5,
        pair_dim=5,
        pair_mlp_dim=7,
        neighbors=2,
        local_exclusion=0,
    )
    hidden = torch.randn(2, 6, 12)
    tokens = torch.randint(0, 4, (2, 6))
    probabilities = torch.softmax(torch.randn(2, 6, 4), dim=-1)
    output = layer(hidden, tokens, gauge_probabilities=probabilities)
    mean = torch.sum(probabilities * output["interaction_logits"], dim=-1)
    assert output["hidden"].shape == hidden.shape
    assert output["logits"].shape == (2, 6, 4)
    assert output["neighbor_indices"].shape == (2, 6, 2)
    assert torch.max(torch.abs(mean)).item() < 1e-5


def test_masked_neighbor_has_no_categorical_interaction_message() -> None:
    torch.manual_seed(41)
    layer = DisentangledSparseInteractionLayer(
        hidden_dim=8,
        states=3,
        rank=2,
        index_dim=4,
        pair_dim=4,
        pair_mlp_dim=6,
        neighbors=3,
        local_exclusion=0,
    )
    hidden = torch.randn(1, 4, 8)
    tokens = torch.full((1, 4), 3)
    output = layer(hidden, tokens)
    assert torch.max(torch.abs(output["interaction_logits"])).item() == 0.0


def test_demo_language_models_produce_comparable_logits() -> None:
    tokens = torch.randint(0, 22, (2, 12))
    models = [
        DisentangledProteinLM(
            hidden_dim=16,
            rank=3,
            index_dim=4,
            pair_dim=4,
            neighbors=2,
        ),
        LocalProteinLM(hidden_dim=16),
        TransformerProteinLM(hidden_dim=16, heads=4),
    ]
    for model in models:
        output = model(tokens)
        assert output["logits"].shape == (2, 12, 20)
        assert output["background_logits"].shape == (2, 12, 20)


def test_soft_routing_preserves_zero_marginal_interaction_mean() -> None:
    torch.manual_seed(43)
    layer = DisentangledSparseInteractionLayer(
        hidden_dim=10,
        states=4,
        rank=3,
        index_dim=5,
        pair_dim=5,
        pair_mlp_dim=8,
        neighbors=2,
        local_exclusion=0,
        routing_mode="soft",
    )
    hidden = torch.randn(1, 6, 10)
    tokens = torch.randint(0, 4, (1, 6))
    probabilities = torch.softmax(torch.randn(1, 6, 4), dim=-1)
    output = layer(hidden, tokens, gauge_probabilities=probabilities)
    mean = torch.sum(probabilities * output["interaction_logits"], dim=-1)
    assert torch.max(torch.abs(mean)).item() < 1e-5
    assert output["routing_weights"].shape == (1, 6, 6)
    np.testing.assert_allclose(
        output["routing_weights"].sum(dim=-1).detach().numpy(), 1.0, atol=1e-6
    )


def test_task_gradient_target_is_marginal_orthogonal() -> None:
    torch.manual_seed(47)
    logits = torch.randn(7, 5)
    targets = torch.randint(0, 5, (7,))
    residual = marginal_orthogonal_task_gradient(logits, targets)
    probabilities = logits.softmax(dim=-1)
    mean = torch.sum(probabilities * residual, dim=-1)
    assert torch.max(torch.abs(mean)).item() < 1e-6


def test_site_residual_control_is_marginal_orthogonal() -> None:
    torch.manual_seed(53)
    model = MarginalOrthogonalResidualLM(hidden_dim=16, states=5, residual_dim=8)
    tokens = torch.randint(0, 7, (2, 10))
    output = model(tokens)
    probabilities = output["background_logits"].softmax(dim=-1)
    mean = torch.sum(probabilities * output["residual_logits"], dim=-1)
    assert torch.max(torch.abs(mean)).item() < 1e-6
