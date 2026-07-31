from math import ceil

import torch

from transformer_disentanglement.content_tile_routing import ContentTileRouter


def make_router() -> ContentTileRouter:
    torch.manual_seed(211)
    return ContentTileRouter(
        stable_dim=12,
        task_dim=10,
        tile_dim=6,
        tiles=4,
        selected_tiles=2,
        candidate_budget=12,
        neighbors=4,
        sinkhorn_steps=20,
    )


def test_soft_assignments_are_balanced_and_row_stochastic() -> None:
    router = make_router()
    logits = torch.randn(2, 20, 4)
    assignments = router.soft_balanced_assignments(logits)
    assert torch.allclose(
        assignments.sum(dim=-1), torch.ones(2, 20), atol=1e-5
    )
    expected = torch.full((2, 4), 5.0)
    assert torch.allclose(assignments.sum(dim=1), expected, atol=2e-3)


def test_hard_assignments_cover_tokens_with_bounded_capacity() -> None:
    router = make_router()
    logits = torch.randn(23, 4)
    assignments = router.balanced_hard_assignments(logits)
    counts = torch.bincount(assignments, minlength=4)
    assert len(assignments) == 23
    assert torch.all(assignments >= 0)
    assert int(counts.max()) <= ceil(23 / 4)


def test_tile_loss_backpropagates_to_both_stream_projections() -> None:
    router = make_router()
    stable = torch.randn(2, 20, 12)
    task = torch.randn(2, 20, 10)
    queries = torch.tensor([[3, 11], [5, 17]])
    target = torch.rand(2, 2, 20)
    target = target / target.sum(dim=-1, keepdim=True)
    loss = router.routing_kl_loss(stable, task, queries, target)
    loss.backward()
    assert router.stable_projection.weight.grad is not None
    assert router.query_projection.weight.grad is not None
    assert router.tile_prototypes.grad is not None


def test_forward_returns_valid_exactly_scored_candidates() -> None:
    router = make_router()
    stable = torch.randn(1, 20, 12)
    task = torch.randn(1, 20, 10)
    query_positions = torch.tensor([4, 15])
    exact_query = torch.randn(1, 2, 8)
    exact_key = torch.randn(1, 20, 8)
    positions = torch.arange(20)
    valid = torch.stack(
        [torch.abs(positions - position) > 1 for position in query_positions]
    )
    output = router(
        stable,
        task,
        query_positions,
        exact_query,
        exact_key,
        valid,
        score_scale=0.5,
        score_bias=0.2,
    )
    assert output.candidate_indices.shape == (1, 2, 12)
    assert output.neighbor_indices.shape == (1, 2, 4)
    counts = torch.bincount(output.hard_assignments[0], minlength=router.tiles)
    assert int(counts.max()) <= ceil(20 / router.tiles)
    for row in range(2):
        candidates = output.candidate_indices[0, row]
        assert torch.all(valid[row, candidates])
        expected = (
            torch.sum(exact_query[0, row][None] * exact_key[0, candidates], dim=-1)
            * 0.5
            + 0.2
        )
        assert torch.allclose(output.candidate_scores[0, row], expected)
