from __future__ import annotations

import numpy as np


def categorical_double_center(blocks: np.ndarray) -> np.ndarray:
    """Project the last two categorical axes into the zero-sum gauge."""
    if blocks.ndim < 2:
        raise ValueError("blocks must have at least two categorical axes")
    row_mean = blocks.mean(axis=-1, keepdims=True)
    column_mean = blocks.mean(axis=-2, keepdims=True)
    grand_mean = blocks.mean(axis=(-2, -1), keepdims=True)
    return blocks - row_mean - column_mean + grand_mean


def weighted_categorical_center(
    blocks: np.ndarray,
    row_probabilities: np.ndarray,
    column_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    """Remove one-body categorical effects under site-specific marginals.

    The returned block G[i, j] satisfies p_i^T G[i, j] = 0 and
    G[i, j] p_j = 0. Uniform probabilities recover ordinary double centering.
    """
    if blocks.ndim != 4:
        raise ValueError("blocks must have shape [length, length, states, states]")
    length_i, length_j, states_a, states_b = blocks.shape
    if row_probabilities.shape != (length_i, states_a):
        raise ValueError("row_probabilities has incompatible shape")
    if column_probabilities is None:
        column_probabilities = row_probabilities
    if column_probabilities.shape != (length_j, states_b):
        raise ValueError("column_probabilities has incompatible shape")
    if not np.allclose(row_probabilities.sum(axis=-1), 1.0):
        raise ValueError("row probabilities must sum to one")
    if not np.allclose(column_probabilities.sum(axis=-1), 1.0):
        raise ValueError("column probabilities must sum to one")

    row_effect = np.einsum("ia,ijab->ijb", row_probabilities, blocks)
    column_effect = np.einsum("ijab,jb->ija", blocks, column_probabilities)
    grand_effect = np.einsum(
        "ia,ijab,jb->ij", row_probabilities, blocks, column_probabilities
    )
    return (
        blocks
        - row_effect[:, :, None, :]
        - column_effect[:, :, :, None]
        + grand_effect[:, :, None, None]
    )


def block_transpose_symmetrize(blocks: np.ndarray) -> np.ndarray:
    """Enforce G[i, j, a, b] = G[j, i, b, a]."""
    if blocks.ndim != 4 or blocks.shape[0] != blocks.shape[1]:
        raise ValueError("blocks must have shape [length, length, states, states]")
    return 0.5 * (blocks + blocks.transpose(1, 0, 3, 2))


def zero_position_diagonal(blocks: np.ndarray) -> np.ndarray:
    result = blocks.copy()
    index = np.arange(result.shape[0])
    result[index, index] = 0.0
    return result


def blocks_to_matrix(blocks: np.ndarray) -> np.ndarray:
    """Flatten [i, j, a, b] blocks into [(i, a), (j, b)] matrix form."""
    if blocks.ndim != 4:
        raise ValueError("blocks must have shape [length, length, states, states]")
    length_i, length_j, states_a, states_b = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(
        length_i * states_a, length_j * states_b
    )


def matrix_to_blocks(
    matrix: np.ndarray, length: int, states: int
) -> np.ndarray:
    expected = length * states
    if matrix.shape != (expected, expected):
        raise ValueError(f"matrix must have shape {(expected, expected)}")
    return matrix.reshape(length, states, length, states).transpose(0, 2, 1, 3)


def block_frobenius_scores(blocks: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(blocks), axis=(-2, -1)))


def reconstruct_symmetric_modes(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    rank: int,
) -> np.ndarray:
    if rank < 1 or rank > len(eigenvalues):
        raise ValueError("rank must be between 1 and the number of supplied modes")
    order = np.argsort(np.abs(eigenvalues))[::-1][:rank]
    vectors = eigenvectors[:, order]
    return (vectors * eigenvalues[order]) @ vectors.T
