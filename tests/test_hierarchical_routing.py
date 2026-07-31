import torch

from transformer_disentanglement.hierarchical_routing import (
    HierarchicalSegmentRouter,
)


def make_router() -> HierarchicalSegmentRouter:
    torch.manual_seed(151)
    return HierarchicalSegmentRouter(
        task_dim=12,
        node_dim=6,
        branching=4,
        leaf_size=4,
        beam_size=4,
        candidate_budget=16,
        neighbors=4,
    )


def test_hierarchy_leaves_cover_sequence_once() -> None:
    router = make_router()
    levels = router.levels(37)
    leaves = levels[-1]
    positions = [position for start, end in leaves for position in range(start, end)]
    assert positions == list(range(37))
    assert all(end - start <= router.leaf_size for start, end in leaves)


def test_hierarchical_loss_backpropagates() -> None:
    router = make_router()
    task = torch.randn(2, 37, 12)
    queries = torch.tensor([[3, 11], [5, 17]])
    target = torch.rand(2, 2, 37)
    target = target / target.sum(dim=-1, keepdim=True)
    loss = router.hierarchical_kl_loss(task, queries, target)
    loss.backward()
    assert router.query_projection.weight.grad is not None
    assert router.key_projection.weight.grad is not None


def test_hierarchical_candidates_are_valid_and_exactly_scored() -> None:
    router = make_router()
    task = torch.randn(1, 37, 12)
    query_positions = torch.tensor([7, 21])
    exact_query = torch.randn(1, 2, 10)
    exact_key = torch.randn(1, 37, 10)
    positions = torch.arange(37)
    valid = torch.stack(
        [torch.abs(positions - position) > 2 for position in query_positions]
    )
    output = router(
        task,
        query_positions,
        exact_query,
        exact_key,
        valid,
        score_scale=0.5,
        score_bias=0.2,
    )
    assert output.candidate_indices.shape == (1, 2, 16)
    assert output.neighbor_indices.shape == (1, 2, 4)
    for row in range(2):
        assert torch.all(valid[row, output.candidate_indices[0, row]])
        expected = (
            torch.sum(
                exact_query[0, row][None]
                * exact_key[0, output.candidate_indices[0, row]],
                dim=-1,
            )
            * 0.5
            + 0.2
        )
        assert torch.allclose(output.candidate_scores[0, row], expected)
