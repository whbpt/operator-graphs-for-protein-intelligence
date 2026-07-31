import torch

from transformer_disentanglement.candidate_routing import (
    BoundingBoxMIPSTreeRouter,
    HashableTaskAdapter,
    MultiTableLSHCandidateRouter,
)
from transformer_disentanglement.dual_stream_layer import (
    DualStreamOrthogonalInteractionLayer,
)


def make_layer_and_task() -> tuple[DualStreamOrthogonalInteractionLayer, torch.Tensor]:
    torch.manual_seed(101)
    layer = DualStreamOrthogonalInteractionLayer(
        stable_dim=12,
        task_dim=10,
        states=5,
        rank=3,
        index_dim=4,
        pair_dim=4,
        neighbors=3,
        local_exclusion=1,
    )
    return layer, torch.randn(2, 24, 10)


def test_index_features_reproduce_dense_scores() -> None:
    layer, task = make_layer_and_task()
    query, key = layer.index_features(task)
    reconstructed = (
        torch.einsum("bid,bjd->bij", query, key) * layer.index_dim**-0.5
        + layer.index_bias
    )
    assert torch.allclose(reconstructed, layer.index_scores(task), atol=1e-6)


def test_lsh_candidates_are_valid_and_exactly_scored() -> None:
    layer, task = make_layer_and_task()
    query, key = layer.index_features(task)
    valid = layer.valid_pair_mask(task.shape[1], task.device)
    router = MultiTableLSHCandidateRouter(
        feature_dim=query.shape[-1],
        tables=3,
        bits=5,
        candidate_budget=8,
        neighbors=3,
        seed=103,
    )
    output = router(
        query,
        key,
        valid,
        score_scale=layer.index_dim**-0.5,
        score_bias=layer.index_bias,
    )
    assert output.candidate_indices.shape == (2, 24, 8)
    assert output.neighbor_indices.shape == (2, 24, 3)
    positions = torch.arange(24)[None, :, None]
    assert torch.all(torch.abs(output.candidate_indices.cpu() - positions) > 1)
    dense = layer.index_scores(task)
    batch = torch.arange(2)[:, None, None]
    rows = torch.arange(24)[None, :, None]
    expected = dense[batch, rows, output.candidate_indices]
    assert torch.allclose(output.candidate_scores, expected, atol=1e-6)


def test_full_candidate_budget_recovers_dense_topk() -> None:
    layer, task = make_layer_and_task()
    query, key = layer.index_features(task)
    positions = torch.arange(task.shape[1])
    valid = positions[:, None] != positions[None, :]
    router = MultiTableLSHCandidateRouter(
        feature_dim=query.shape[-1],
        tables=1,
        bits=8,
        candidate_budget=24,
        neighbors=3,
        seed=107,
    )
    output = router(
        query,
        key,
        valid,
        score_scale=layer.index_dim**-0.5,
        score_bias=layer.index_bias,
    )
    dense = layer.index_scores(task).masked_fill(~valid[None], -torch.inf)
    expected = torch.topk(dense, k=3, dim=-1).indices
    assert torch.equal(
        output.neighbor_indices.sort(dim=-1).values,
        expected.sort(dim=-1).values,
    )


def test_hash_retrieval_evaluates_fewer_than_all_pairs() -> None:
    torch.manual_seed(109)
    query = torch.randn(1, 128, 16)
    key = torch.randn(1, 128, 16)
    positions = torch.arange(128)
    valid = torch.abs(positions[:, None] - positions[None, :]) > 2
    router = MultiTableLSHCandidateRouter(
        feature_dim=16,
        tables=3,
        bits=7,
        candidate_budget=24,
        neighbors=8,
        seed=113,
    )
    output = router(query, key, valid, score_scale=1.0)
    assert float(output.evaluated_pairs.float().mean()) < 40


def test_hash_adapter_relaxed_scores_train_and_hash_features_only_select() -> None:
    torch.manual_seed(127)
    adapter = HashableTaskAdapter(task_dim=10, hash_dim=8)
    router = MultiTableLSHCandidateRouter(
        feature_dim=8,
        tables=3,
        bits=4,
        candidate_budget=8,
        neighbors=3,
        seed=131,
    )
    task = torch.randn(2, 24, 10)
    hash_features = adapter(task)
    codes = adapter.relaxed_codes(hash_features, router.hyperplanes, 0.5)
    positions = torch.tensor([[3, 7], [4, 8]])
    relaxed = adapter.relaxed_scores(codes, positions)
    quantization, balance, decorrelation = adapter.regularization(codes)
    loss = relaxed.square().mean() + quantization + balance + decorrelation
    loss.backward()
    assert relaxed.shape == (2, 2, 24)
    assert adapter.projection.weight.grad is not None

    score_query = torch.randn(2, 2, 6)
    score_key = torch.randn(2, 24, 6)
    valid = torch.ones(2, 24, dtype=torch.bool)
    output = router(
        score_query,
        score_key,
        valid,
        score_scale=1.0,
        query_positions=torch.tensor([3, 7]),
        hash_query_features=hash_features[:, [3, 7]],
        hash_key_features=hash_features,
    )
    batch = torch.arange(2)[:, None, None]
    rows = torch.arange(2)[None, :, None]
    expected = torch.sum(
        score_query[:, :, None]
        * score_key[batch, output.candidate_indices],
        dim=-1,
    )
    assert torch.allclose(output.candidate_scores, expected)


def test_bounding_box_tree_exactly_recovers_dense_topk() -> None:
    torch.manual_seed(137)
    query = torch.randn(2, 5, 12)
    key = torch.randn(2, 64, 12)
    valid = torch.ones(5, 64, dtype=torch.bool)
    valid[:, :3] = False
    router = BoundingBoxMIPSTreeRouter(neighbors=6, leaf_size=4)
    output = router(query, key, valid, score_scale=0.25, score_bias=0.3)
    dense = torch.einsum("bqd,bld->bql", query, key) * 0.25 + 0.3
    dense = dense.masked_fill(~valid[None], -torch.inf)
    expected = torch.topk(dense, k=6, dim=-1)
    assert torch.equal(output.neighbor_indices, expected.indices)
    assert torch.allclose(output.neighbor_scores, expected.values, atol=1e-6)
    assert torch.all(output.evaluated_pairs <= valid.sum(dim=-1)[None])


def test_bounding_box_tree_prunes_low_dimensional_keys() -> None:
    torch.manual_seed(139)
    angles = torch.linspace(0, 2 * torch.pi, 256)
    key = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)[None]
    query = key[:, ::32]
    valid = torch.ones(len(query[0]), len(key[0]), dtype=torch.bool)
    router = BoundingBoxMIPSTreeRouter(neighbors=4, leaf_size=4)
    output = router(query, key, valid, score_scale=1.0)
    assert float(output.evaluated_pairs.float().mean()) < 64
