from __future__ import annotations

import numpy as np
import torch


def weighted_centered_features(
    msa: np.ndarray,
    weights: np.ndarray,
    pssm: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Return sqrt-weighted, gap-masked centered one-hot MSA features."""
    states = pssm.shape[-1]
    msa_tensor = torch.from_numpy(msa.astype(np.int64)).to(device)
    valid = msa_tensor < states
    clipped = msa_tensor.clamp(max=states - 1)
    one_hot = torch.nn.functional.one_hot(clipped, states).float()
    probabilities = torch.from_numpy(pssm).to(device)
    centered = (one_hot - probabilities[None, :, :]) * valid[:, :, None]
    normalized_weight = weights.astype(np.float32) / max(float(weights.sum()), 1e-8)
    sqrt_weight = torch.from_numpy(np.sqrt(normalized_weight)).to(device)
    return centered * sqrt_weight[:, None, None]


def randomized_phylogeny_loadings(
    weighted_features: torch.Tensor,
    rank: int,
    seed: int,
    power_iterations: int = 2,
) -> np.ndarray:
    """Fit leading sequence modes and return their site/category loadings."""
    flat = weighted_features.reshape(len(weighted_features), -1)
    torch.manual_seed(seed)
    omega = torch.randn(
        flat.shape[1], rank, device=flat.device, dtype=flat.dtype
    )
    basis, _ = torch.linalg.qr(flat @ omega, mode="reduced")
    for _ in range(power_iterations):
        basis, _ = torch.linalg.qr(
            flat @ (flat.T @ basis), mode="reduced"
        )
    small = (basis.T @ flat).cpu().numpy()
    left_small, _, _ = np.linalg.svd(small, full_matrices=False)
    sequence_modes = basis @ torch.from_numpy(
        left_small.astype(np.float32)
    ).to(flat.device)
    loadings = flat.T @ sequence_modes[:, :rank]
    return (
        loadings.reshape(
            weighted_features.shape[1], weighted_features.shape[2], rank
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def weighted_project_pair_blocks(
    blocks: np.ndarray,
    pairs: np.ndarray,
    pssm: np.ndarray,
) -> np.ndarray:
    row_probability = pssm[pairs[:, 0]]
    column_probability = pssm[pairs[:, 1]]
    row_effect = np.einsum("pa,pab->pb", row_probability, blocks)
    column_effect = np.einsum("pab,pb->pa", blocks, column_probability)
    grand = np.einsum(
        "pa,pab,pb->p", row_probability, blocks, column_probability
    )
    return (
        blocks
        - row_effect[:, None, :]
        - column_effect[:, :, None]
        + grand[:, None, None]
    ).astype(np.float32)


def residualize_phylogeny_modes(
    blocks: np.ndarray,
    pairs: np.ndarray,
    loadings: np.ndarray,
    msa: np.ndarray,
    weights: np.ndarray,
    pssm: np.ndarray,
) -> np.ndarray:
    """Subtract low-rank sequence-mode covariance from sampled pair blocks."""
    correction = np.einsum(
        "pak,pbk->pab", loadings[pairs[:, 0]], loadings[pairs[:, 1]]
    )
    valid = (msa[:, pairs[:, 0]] < pssm.shape[-1]) & (
        msa[:, pairs[:, 1]] < pssm.shape[-1]
    )
    valid_fraction = np.sum(weights[:, None] * valid, axis=0) / max(
        float(weights.sum()), 1e-8
    )
    correction /= np.maximum(valid_fraction[:, None, None], 1e-8)
    return weighted_project_pair_blocks(blocks - correction, pairs, pssm)
