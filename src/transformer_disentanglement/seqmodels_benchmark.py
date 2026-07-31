from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SOURCE_AA_ORDER = "ARNDCQEGHILKMFPSTWYV-"
MODEL_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
SOURCE_TO_MODEL = np.asarray(
    [MODEL_AA_ORDER.index(amino_acid) if amino_acid != "-" else 20 for amino_acid in SOURCE_AA_ORDER],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class SeqmodelsFamily:
    identifier: str
    query: str
    query_positions: np.ndarray
    msa: np.ndarray
    weights: np.ndarray
    contacts: np.ndarray
    contact_mask: np.ndarray


@dataclass(frozen=True)
class EntropyNullModel:
    """Robust family-level background for categorical covariance strength."""

    location: float
    scale: float


def load_seqmodels_family(
    benchmark: str | Path,
    filename: str,
    identifier: str,
) -> SeqmodelsFamily:
    path = Path(benchmark) / "families" / filename
    with np.load(path, allow_pickle=False) as data:
        source_msa = data["x"]
        weights = data["x_w"].astype(np.float32)
        contacts = data["x_true"].astype(np.float32)
        contact_mask = data["x_mask"].astype(bool)
    query_positions = np.flatnonzero(source_msa[0] != 20)
    msa = SOURCE_TO_MODEL[source_msa[:, query_positions]]
    contacts = contacts[np.ix_(query_positions, query_positions)]
    contact_mask = contact_mask[np.ix_(query_positions, query_positions)]
    query = "".join(MODEL_AA_ORDER[state] for state in msa[0])
    return SeqmodelsFamily(
        identifier=identifier,
        query=query,
        query_positions=query_positions.astype(np.int16),
        msa=msa,
        weights=weights,
        contacts=contacts,
        contact_mask=contact_mask,
    )


def weighted_pssm(
    msa: np.ndarray,
    weights: np.ndarray,
    states: int = len(MODEL_AA_ORDER),
    pseudocount: float = 1e-3,
) -> np.ndarray:
    if msa.ndim != 2 or weights.shape != (msa.shape[0],):
        raise ValueError("MSA and weights have incompatible shapes")
    counts = np.zeros((msa.shape[1], states), dtype=np.float64)
    for state in range(states):
        counts[:, state] = np.sum(weights[:, None] * (msa == state), axis=0)
    counts += pseudocount
    return counts / counts.sum(axis=-1, keepdims=True)


def connected_pair_blocks(
    msa: np.ndarray,
    weights: np.ndarray,
    pairs: np.ndarray,
    pssm: np.ndarray | None = None,
    states: int = len(MODEL_AA_ORDER),
) -> np.ndarray:
    """Compute sampled real-minus-independent categorical pair blocks."""
    if pairs.ndim != 2 or pairs.shape[-1] != 2:
        raise ValueError("pairs must have shape [pair_count, 2]")
    if pssm is None:
        pssm = weighted_pssm(msa, weights, states=states)
    blocks = np.zeros((len(pairs), states, states), dtype=np.float64)
    for pair_index, (left, right) in enumerate(pairs):
        left_state = msa[:, left]
        right_state = msa[:, right]
        valid = (left_state < states) & (right_state < states)
        valid_weights = weights[valid].astype(np.float64)
        if valid_weights.sum() <= 0:
            continue
        joint_index = left_state[valid].astype(np.int64) * states + right_state[valid]
        joint = np.bincount(
            joint_index,
            weights=valid_weights,
            minlength=states * states,
        ).reshape(states, states)
        joint /= valid_weights.sum()
        blocks[pair_index] = joint - np.outer(pssm[left], pssm[right])

    row_probabilities = pssm[pairs[:, 0]]
    column_probabilities = pssm[pairs[:, 1]]
    row_effect = np.einsum("pa,pab->pb", row_probabilities, blocks)
    column_effect = np.einsum("pab,pb->pa", blocks, column_probabilities)
    grand_effect = np.einsum(
        "pa,pab,pb->p", row_probabilities, blocks, column_probabilities
    )
    centered = (
        blocks
        - row_effect[:, None, :]
        - column_effect[:, :, None]
        + grand_effect[:, None, None]
    )
    return centered.astype(np.float32)


def weighted_log_odds_blocks(
    msa: np.ndarray,
    weights: np.ndarray,
    pairs: np.ndarray,
    pssm: np.ndarray | None = None,
    states: int = len(MODEL_AA_ORDER),
    prior_weight: float = 1.0,
) -> np.ndarray:
    """Estimate shrunk categorical log odds and remove both marginals.

    The prior is the independent distribution implied by the supplied PSSM, so
    unobserved cells have zero interaction before the weighted gauge projection.
    Setting the query-row weight to zero produces a leave-query-out teacher.
    """
    if pairs.ndim != 2 or pairs.shape[-1] != 2:
        raise ValueError("pairs must have shape [pair_count, 2]")
    if prior_weight <= 0:
        raise ValueError("prior_weight must be positive")
    if pssm is None:
        pssm = weighted_pssm(msa, weights, states=states)
    blocks = np.zeros((len(pairs), states, states), dtype=np.float64)
    for pair_index, (left, right) in enumerate(pairs):
        left_state = msa[:, left]
        right_state = msa[:, right]
        valid = (left_state < states) & (right_state < states)
        valid_weights = weights[valid].astype(np.float64)
        independent = np.outer(pssm[left], pssm[right]).clip(min=1e-12)
        if valid_weights.sum() <= 0:
            joint = independent
        else:
            joint_index = left_state[valid].astype(np.int64) * states + right_state[valid]
            counts = np.bincount(
                joint_index,
                weights=valid_weights,
                minlength=states * states,
            ).reshape(states, states)
            joint = (counts + prior_weight * independent) / (
                valid_weights.sum() + prior_weight
            )
        blocks[pair_index] = np.log(joint.clip(min=1e-12)) - np.log(independent)

    row_probabilities = pssm[pairs[:, 0]]
    column_probabilities = pssm[pairs[:, 1]]
    row_effect = np.einsum("pa,pab->pb", row_probabilities, blocks)
    column_effect = np.einsum("pab,pb->pa", blocks, column_probabilities)
    grand_effect = np.einsum(
        "pa,pab,pb->p", row_probabilities, blocks, column_probabilities
    )
    centered = (
        blocks
        - row_effect[:, None, :]
        - column_effect[:, :, None]
        + grand_effect[:, None, None]
    )
    return centered.astype(np.float32)


def entropy_pair_scale(pssm: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Maximum covariance scale implied by the two categorical marginals."""
    diversity = 1.0 - np.sum(np.square(pssm), axis=-1)
    diversity = np.clip(diversity, 0.0, None)
    return np.sqrt(diversity[pairs[:, 0]] * diversity[pairs[:, 1]])


def fit_entropy_null_model(
    blocks: np.ndarray,
    pairs: np.ndarray,
    pssm: np.ndarray,
) -> EntropyNullModel:
    """Fit a robust null to covariance strength after marginal scaling."""
    strength = np.sqrt(np.mean(np.square(blocks), axis=(-2, -1)))
    marginal_scale = entropy_pair_scale(pssm, pairs).clip(min=1e-8)
    normalized = strength / marginal_scale
    location = float(np.median(normalized))
    scale = float(1.4826 * np.median(np.abs(normalized - location)))
    return EntropyNullModel(location=location, scale=max(scale, 1e-8))


def residualize_entropy_background(
    blocks: np.ndarray,
    pairs: np.ndarray,
    pssm: np.ndarray,
    null_model: EntropyNullModel,
    z_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the entropy-scaled background while preserving block signs."""
    strength = np.sqrt(np.mean(np.square(blocks), axis=(-2, -1)))
    marginal_scale = entropy_pair_scale(pssm, pairs)
    threshold = marginal_scale * (
        null_model.location + z_threshold * null_model.scale
    )
    residual_strength = np.maximum(strength - threshold, 0.0)
    reliability = residual_strength / np.maximum(strength, 1e-12)
    residual = blocks * reliability[:, None, None]
    return residual.astype(np.float32), reliability.astype(np.float32)
