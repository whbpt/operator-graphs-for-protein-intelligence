from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


def normalized_entropy(frequencies: np.ndarray) -> np.ndarray:
    entropy = -np.sum(frequencies * np.log(frequencies + 1e-12), axis=-1)
    return entropy / np.log(frequencies.shape[-1])


def mutual_information(encoded: np.ndarray, states: int) -> np.ndarray:
    depth, length = encoded.shape
    frequencies = np.zeros((length, states), dtype=np.float64)
    for state in range(states):
        frequencies[:, state] = np.mean(encoded == state, axis=0)

    matrix = np.zeros((length, length), dtype=np.float64)
    for i in range(length):
        for j in range(i + 1, length):
            joint_index = encoded[:, i] * states + encoded[:, j]
            joint = np.bincount(joint_index, minlength=states * states).reshape(states, states)
            joint = joint.astype(np.float64) / depth
            independent = frequencies[i, :, None] * frequencies[j, None, :]
            valid = joint > 0
            value = np.sum(joint[valid] * np.log(joint[valid] / independent[valid]))
            matrix[i, j] = matrix[j, i] = value
    return matrix


def average_product_correction(matrix: np.ndarray) -> np.ndarray:
    row_sum = matrix.sum(axis=-1)
    total = row_sum.sum()
    if total <= 0:
        return matrix.copy()
    corrected = matrix - np.outer(row_sum, row_sum) / total
    np.fill_diagonal(corrected, 0.0)
    return corrected


def leading_mode_fraction(matrix: np.ndarray) -> float:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)
    denominator = np.sum(eigenvalues**2)
    return float(eigenvalues[-1] ** 2 / denominator) if denominator > 0 else 0.0


def leading_vector(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    _, eigenvectors = np.linalg.eigh(symmetric)
    vector = eigenvectors[:, -1]
    return vector if vector.sum() >= 0 else -vector


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    result = spearmanr(x, y)
    return float(result.statistic) if np.isfinite(result.statistic) else 0.0


def binary_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(np.asarray(scores))[::-1]
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def binary_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores), method="average")
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def pairwise_diagnostics(matrix: np.ndarray, entropy: np.ndarray) -> dict[str, float]:
    vector = leading_vector(matrix)
    return {
        "leading_mode_fraction": leading_mode_fraction(matrix),
        "row_sum_entropy_spearman": safe_spearman(matrix.sum(axis=-1), entropy),
        "leading_vector_entropy_spearman": safe_spearman(vector, entropy),
    }


def contact_precision(
    scores: np.ndarray,
    distances: np.ndarray,
    top_fraction: float = 1.0,
    min_separation: int = 6,
    contact_threshold: float = 8.0,
) -> float:
    length = scores.shape[0]
    i, j = np.triu_indices(length, k=min_separation)
    valid = np.isfinite(distances[i, j])
    i, j = i[valid], j[valid]
    if len(i) == 0:
        return float("nan")
    count = max(1, min(len(i), int(round(length * top_fraction))))
    ranking = np.argsort(scores[i, j])[::-1][:count]
    return float(np.mean(distances[i[ranking], j[ranking]] < contact_threshold))


def contact_prevalence(
    distances: np.ndarray,
    min_separation: int = 6,
    contact_threshold: float = 8.0,
) -> float:
    i, j = np.triu_indices(distances.shape[0], k=min_separation)
    valid = np.isfinite(distances[i, j])
    return float(np.mean(distances[i[valid], j[valid]] < contact_threshold))


def binary_contact_precision(
    scores: np.ndarray,
    contacts: np.ndarray,
    mask: np.ndarray,
    top_fraction: float = 1.0,
    min_separation: int = 6,
    contact_threshold: float = 0.01,
) -> float:
    length = scores.shape[0]
    i, j = np.triu_indices(length, k=min_separation)
    valid = mask[i, j].astype(bool)
    i, j = i[valid], j[valid]
    if len(i) == 0:
        return float("nan")
    count = max(1, min(len(i), int(round(length * top_fraction))))
    ranking = np.argsort(scores[i, j])[::-1][:count]
    return float(np.mean(contacts[i[ranking], j[ranking]] > contact_threshold))


def binary_contact_prevalence(
    contacts: np.ndarray,
    mask: np.ndarray,
    min_separation: int = 6,
    contact_threshold: float = 0.01,
) -> float:
    i, j = np.triu_indices(contacts.shape[0], k=min_separation)
    valid = mask[i, j].astype(bool)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(contacts[i[valid], j[valid]] > contact_threshold))


def separation_baseline(length: int) -> np.ndarray:
    positions = np.arange(length)
    return -np.abs(positions[:, None] - positions[None, :]).astype(np.float64)
