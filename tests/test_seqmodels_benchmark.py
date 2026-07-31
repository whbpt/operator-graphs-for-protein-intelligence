import numpy as np

from transformer_disentanglement.seqmodels_benchmark import (
    connected_pair_blocks,
    weighted_log_odds_blocks,
    weighted_pssm,
)


def test_weighted_pssm_ignores_gap_state() -> None:
    msa = np.asarray([[0, 1], [0, 20], [1, 1]], dtype=np.uint8)
    weights = np.asarray([1.0, 1.0, 3.0], dtype=np.float32)
    pssm = weighted_pssm(msa, weights, pseudocount=1e-9)
    assert np.allclose(pssm.sum(axis=-1), 1.0)
    assert pssm[0, 1] > pssm[0, 0]
    assert pssm[1, 1] > 0.999


def test_connected_blocks_obey_weighted_gauge() -> None:
    msa = np.asarray(
        [[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=np.uint8
    )
    weights = np.ones(4, dtype=np.float32)
    pssm = weighted_pssm(msa, weights, states=2, pseudocount=1e-9)
    pairs = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    blocks = connected_pair_blocks(msa, weights, pairs, pssm=pssm, states=2)
    left = np.einsum("pa,pab->pb", pssm[pairs[:, 0]], blocks)
    right = np.einsum("pab,pb->pa", blocks, pssm[pairs[:, 1]])
    assert np.max(np.abs(left)) < 1e-6
    assert np.max(np.abs(right)) < 1e-6


def test_weighted_log_odds_blocks_obey_both_gauges() -> None:
    msa = np.asarray(
        [[0, 0], [0, 0], [0, 1], [1, 1], [1, 1]], dtype=np.uint8
    )
    weights = np.ones(5, dtype=np.float32)
    pssm = weighted_pssm(msa, weights, states=2, pseudocount=1e-4)
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    blocks = weighted_log_odds_blocks(
        msa, weights, pairs, pssm=pssm, states=2, prior_weight=1.0
    )
    left = np.einsum("pa,pab->pb", pssm[pairs[:, 0]], blocks)
    right = np.einsum("pab,pb->pa", blocks, pssm[pairs[:, 1]])
    assert np.max(np.abs(left)) < 1e-6
    assert np.max(np.abs(right)) < 1e-6
    assert float(np.sqrt(np.mean(blocks**2))) > 0


def test_leave_query_out_log_odds_ignore_query_residue() -> None:
    msa = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    weights = np.ones(4, dtype=np.float32)
    weights[0] = 0.0
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    pssm = weighted_pssm(msa, weights, states=2, pseudocount=1e-4)
    first = weighted_log_odds_blocks(msa, weights, pairs, pssm=pssm, states=2)
    changed = msa.copy()
    changed[0] = np.asarray([1, 1])
    second = weighted_log_odds_blocks(changed, weights, pairs, pssm=pssm, states=2)
    assert np.allclose(first, second)
