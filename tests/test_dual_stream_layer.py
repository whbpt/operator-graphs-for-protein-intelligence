import torch

from transformer_disentanglement.demo_language_models import (
    ContentTileProteinLM,
    DualStreamProteinLM,
)
from transformer_disentanglement.dual_stream_layer import (
    DualStreamOrthogonalInteractionLayer,
)


def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(41)
    stable = torch.randn(2, 9, 12)
    task = torch.randn(2, 9, 10)
    tokens = torch.randint(0, 5, (2, 9))
    return stable, task, tokens


def make_layer(
    routing_mode: str = "topk",
    rank_mode: str = "fixed",
    value_mode: str = "site_shared",
) -> DualStreamOrthogonalInteractionLayer:
    torch.manual_seed(43)
    return DualStreamOrthogonalInteractionLayer(
        stable_dim=12,
        task_dim=10,
        states=5,
        rank=3,
        index_dim=4,
        pair_dim=4,
        pair_mlp_dim=12,
        neighbors=2,
        local_exclusion=1,
        routing_mode=routing_mode,
        rank_mode=rank_mode,
        value_mode=value_mode,
        adapter_count=5,
        adapter_topk=2,
    )


def test_task_stream_changes_routing_but_not_values() -> None:
    layer = make_layer()
    stable, task, tokens = make_inputs()
    baseline = layer(stable, task, tokens)
    changed = layer(stable, task + torch.randn_like(task), tokens)
    assert torch.allclose(
        baseline["background_logits"], changed["background_logits"]
    )
    assert torch.allclose(baseline["factors"], changed["factors"])
    assert not torch.allclose(baseline["index_scores"], changed["index_scores"])


def test_stable_stream_changes_values_but_not_routing() -> None:
    layer = make_layer()
    stable, task, tokens = make_inputs()
    baseline = layer(stable, task, tokens)
    changed = layer(stable + torch.randn_like(stable), task, tokens)
    assert torch.allclose(baseline["index_scores"], changed["index_scores"])
    assert not torch.allclose(
        baseline["background_logits"], changed["background_logits"]
    )
    assert not torch.allclose(baseline["factors"], changed["factors"])


def test_selected_pair_blocks_obey_both_weighted_gauges() -> None:
    layer = make_layer()
    stable, task, tokens = make_inputs()
    output = layer(stable, task, tokens)
    factors = output["factors"]
    probabilities = output["marginal_probabilities"]
    indices = output["neighbor_indices"]
    mode = output["mode"]
    batch = torch.arange(len(stable))[:, None, None]
    count = indices.shape[-1]
    left = factors[:, :, None].expand(-1, -1, count, -1, -1)
    right = factors[batch, indices]
    blocks = torch.einsum("nikar,nikcr,nikr->nikac", left, right, mode)
    left_probabilities = probabilities[:, :, None].expand(-1, -1, count, -1)
    right_probabilities = probabilities[batch, indices]
    left_gauge = torch.einsum(
        "nika,nikac->nikc", left_probabilities, blocks
    )
    right_gauge = torch.einsum(
        "nikac,nikc->nika", blocks, right_probabilities
    )
    assert float(left_gauge.abs().max().detach()) < 2e-6
    assert float(right_gauge.abs().max().detach()) < 2e-6


def test_all_masked_context_has_zero_interaction_message() -> None:
    layer = make_layer()
    stable, task, tokens = make_inputs()
    tokens.fill_(6)
    output = layer(stable, task, tokens)
    assert torch.count_nonzero(output["interaction_logits"]) == 0


def test_topk_routing_is_normalized_and_excludes_local_pairs() -> None:
    layer = make_layer()
    stable, task, tokens = make_inputs()
    output = layer(stable, task, tokens)
    weights = output["selected_weights"]
    indices = output["neighbor_indices"]
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    positions = torch.arange(tokens.shape[1])[None, :, None]
    assert torch.all(torch.abs(indices.cpu() - positions) > 1)


def test_soft_routing_is_normalized_over_valid_pairs() -> None:
    layer = make_layer("soft")
    stable, task, tokens = make_inputs()
    output = layer(stable, task, tokens)
    weights = output["routing_weights"]
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    valid = layer.valid_pair_mask(tokens.shape[1], tokens.device)
    assert torch.count_nonzero(weights.masked_select(~valid[None])) == 0


def test_sampled_scores_match_dense_scores() -> None:
    layer = make_layer()
    _, task, _ = make_inputs()
    pairs = torch.tensor([[[0, 3], [2, 7]], [[1, 5], [4, 8]]])
    sampled = layer.sampled_index_scores(task, pairs)
    dense = layer.index_scores(task)
    batch = torch.arange(2)[:, None]
    expected = dense[batch, pairs[..., 0], pairs[..., 1]]
    assert torch.allclose(sampled, expected, atol=1e-6)


def test_fixed_rank_reports_all_modes_active() -> None:
    layer = make_layer()
    stable, task, tokens = make_inputs()
    output = layer(stable, task, tokens)
    assert torch.all(output["mode_gates"] == 1)
    assert torch.allclose(
        output["effective_rank"],
        torch.full_like(output["effective_rank"], layer.rank),
    )
    assert torch.all(output["active_modes"] == layer.rank)


def test_adaptive_mode_gates_depend_on_task_not_stable_state() -> None:
    layer = make_layer(rank_mode="adaptive")
    stable, task, tokens = make_inputs()
    baseline = layer(stable, task, tokens)
    changed_task = layer(stable, task + torch.randn_like(task), tokens)
    changed_stable = layer(stable + torch.randn_like(stable), task, tokens)
    assert not torch.allclose(baseline["mode_gates"], changed_task["mode_gates"])
    assert torch.allclose(baseline["mode_gates"], changed_stable["mode_gates"])
    assert torch.all(baseline["effective_rank"] >= 1.0)
    assert torch.all(baseline["effective_rank"] <= layer.rank + 1e-5)


def test_adaptive_sampled_blocks_obey_both_weighted_gauges() -> None:
    layer = make_layer(rank_mode="adaptive")
    stable, task, _ = make_inputs()
    background, probabilities, factors, value_state = layer.stable_values(stable, None)
    del background
    pairs = torch.tensor([[[0, 3], [2, 7]], [[1, 5], [4, 8]]])
    blocks, gates, effective_rank = layer.sampled_pair_outputs(
        factors, value_state, task, pairs
    )
    batch = torch.arange(2)[:, None]
    left_probabilities = probabilities[batch, pairs[..., 0]]
    right_probabilities = probabilities[batch, pairs[..., 1]]
    left_gauge = torch.einsum("npa,npac->npc", left_probabilities, blocks)
    right_gauge = torch.einsum("npac,npc->npa", blocks, right_probabilities)
    assert float(left_gauge.abs().max().detach()) < 2e-6
    assert float(right_gauge.abs().max().detach()) < 2e-6
    assert gates.shape == (2, 2, layer.rank)
    assert effective_rank.shape == (2, 2)


def test_pair_residual_changes_basis_with_partner_state() -> None:
    layer = make_layer(value_mode="pair_residual")
    _, _, _, value_state = layer.stable_values(make_inputs()[0], None)
    left = value_state[:, 0]
    first_left, _ = layer.pair_factor_residuals(left, value_state[:, 3])
    second_left, _ = layer.pair_factor_residuals(left, value_state[:, 7])
    assert not torch.allclose(first_left, second_left)


def test_pair_residual_sampled_blocks_obey_both_weighted_gauges() -> None:
    layer = make_layer(value_mode="pair_residual")
    stable, task, _ = make_inputs()
    _, probabilities, factors, value_state = layer.stable_values(stable, None)
    pairs = torch.tensor([[[0, 3], [2, 7]], [[1, 5], [4, 8]]])
    blocks, _, _ = layer.sampled_pair_outputs(
        factors, value_state, task, pairs, probabilities
    )
    batch = torch.arange(2)[:, None]
    left_probabilities = probabilities[batch, pairs[..., 0]]
    right_probabilities = probabilities[batch, pairs[..., 1]]
    left_gauge = torch.einsum("npa,npac->npc", left_probabilities, blocks)
    right_gauge = torch.einsum("npac,npc->npa", blocks, right_probabilities)
    assert float(left_gauge.abs().max().detach()) < 2e-6
    assert float(right_gauge.abs().max().detach()) < 2e-6


def test_pair_residual_forward_executes_sparse_single_sequence() -> None:
    model = DualStreamProteinLM(
        stable_dim=16,
        task_dim=12,
        states=5,
        rank=3,
        index_dim=4,
        pair_dim=4,
        neighbors=2,
        max_length=16,
        value_mode="pair_residual",
        adapter_count=4,
        adapter_topk=2,
    )
    tokens = torch.randint(0, 5, (1, 9))
    output = model(tokens)
    assert output["logits"].shape == (1, 9, 5)
    assert output["neighbor_indices"].shape == (1, 9, 2)
    assert torch.isfinite(output["interaction_logits"]).all()


def test_adapter_load_bias_changes_selection_not_semantic_order() -> None:
    layer = make_layer(value_mode="pair_residual")
    layer.adapter_topk = 2
    layer.adapter_routing_bias.copy_(torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0]))
    logits = torch.tensor([[4.0, 0.0, -2.0, -3.0, -4.0]])
    weights = layer.sparse_adapter_weights(logits)
    selected = set(torch.nonzero(weights[0]).flatten().tolist())
    assert selected == {0, 1}
    assert weights[0, 0] > weights[0, 1]


def test_adapter_load_bias_moves_against_overloaded_adapter() -> None:
    layer = make_layer(value_mode="pair_residual")
    layer.adapter_topk = 1
    layer.adapter_bias_update_speed = 0.1
    layer.train()
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]]).repeat(8, 1)
    layer.sparse_adapter_weights(logits)
    assert layer.adapter_routing_bias[0] < 0
    assert torch.all(layer.adapter_routing_bias[1:] > 0)


def test_dual_stream_language_model_executes_without_teacher_inputs() -> None:
    model = DualStreamProteinLM(
        stable_dim=16,
        task_dim=12,
        states=5,
        rank=3,
        index_dim=4,
        pair_dim=4,
        neighbors=2,
        max_length=16,
    )
    tokens = torch.randint(0, 5, (2, 9))
    output = model(tokens)
    background = model(tokens, use_interaction=False)
    assert output["logits"].shape == (2, 9, 5)
    assert output["encoded_task"].shape == (2, 9, 12)
    assert background["logits"].shape == (2, 9, 5)


def test_content_tile_language_model_executes_from_one_sequence() -> None:
    model = ContentTileProteinLM(
        stable_dim=16,
        task_dim=12,
        states=5,
        rank=3,
        index_dim=4,
        pair_dim=4,
        neighbors=2,
        max_length=24,
        tile_dim=6,
        tiles=4,
        selected_tiles=2,
        candidate_budget=8,
    )
    tokens = torch.randint(0, 5, (1, 20))
    output = model(tokens)
    assert output["logits"].shape == (1, 20, 5)
    assert output["neighbor_indices"].shape == (1, 20, 2)
    assert output["candidate_indices"].shape == (1, 20, 8)
    assert output["tile_assignments"].shape == (1, 20)
    assert output["index_scores"] is None


def test_content_tile_model_masked_context_has_zero_interaction() -> None:
    model = ContentTileProteinLM(
        stable_dim=16,
        task_dim=12,
        states=5,
        rank=3,
        index_dim=4,
        pair_dim=4,
        neighbors=2,
        max_length=24,
        tile_dim=6,
        tiles=4,
        selected_tiles=2,
        candidate_budget=8,
    )
    tokens = torch.full((1, 20), 6)
    output = model(tokens)
    assert torch.count_nonzero(output["interaction_logits"]) == 0
