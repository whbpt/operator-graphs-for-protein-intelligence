import numpy as np

from transformer_disentanglement.block_structure import (
    block_concentration_metrics,
    oracle_partition_block_mask,
    oracle_sliding_block_mask,
    oracle_token_mask,
    top_position_fragmentation,
)


def test_oracle_token_mask_respects_validity_and_budget() -> None:
    signal = np.array([1.0, 5.0, 4.0, 3.0, 2.0])
    valid = np.array([True, False, True, True, True])
    selected = oracle_token_mask(signal, valid, budget=2)
    assert selected.tolist() == [False, False, True, True, False]


def test_oracle_blocks_prefer_concentrated_signal() -> None:
    signal = np.array([5.0, 4.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0])
    valid = np.ones(8, dtype=bool)
    selected = oracle_partition_block_mask(signal, valid, block_size=2, budget=4)
    assert selected.tolist() == [True, True, False, False, False, False, True, True]
    metrics = block_concentration_metrics(signal, valid, block_size=2, budget=4)
    assert metrics["block_to_token_mass"] == 1.0
    assert metrics["top_token_recall"] == 1.0


def test_fragmentation_distinguishes_clustered_and_scattered_positions() -> None:
    valid = np.ones(10, dtype=bool)
    clustered = np.array([0, 0, 5, 4, 3, 0, 0, 0, 0, 0], dtype=float)
    scattered = np.array([5, 0, 0, 4, 0, 0, 3, 0, 0, 0], dtype=float)
    clustered_metrics = top_position_fragmentation(clustered, valid, budget=3)
    scattered_metrics = top_position_fragmentation(scattered, valid, budget=3)
    assert clustered_metrics["components"] == 1
    assert scattered_metrics["components"] == 3
    assert clustered_metrics["span_ratio"] < scattered_metrics["span_ratio"]


def test_sliding_blocks_remove_partition_boundary_penalty() -> None:
    signal = np.array([0.0, 5.0, 5.0, 0.0, 0.0, 0.0])
    valid = np.ones(6, dtype=bool)
    partition = oracle_partition_block_mask(signal, valid, block_size=2, budget=2)
    sliding = oracle_sliding_block_mask(signal, valid, block_size=2, budget=2)
    assert signal[partition].sum() == 5.0
    assert signal[sliding].sum() == 10.0


def test_sliding_blocks_are_non_overlapping_under_multiple_block_budget() -> None:
    signal = np.array([5.0, 5.0, 0.0, 4.0, 4.0, 0.0])
    valid = np.ones(6, dtype=bool)
    selected = oracle_sliding_block_mask(signal, valid, block_size=2, budget=4)
    assert selected.tolist() == [True, True, False, True, True, False]
