from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.train_seqmodels_factor_head import load_representation
from transformer_disentanglement.metrics import (
    binary_average_precision,
    binary_roc_auc,
)
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import (
    connected_pair_blocks,
    entropy_pair_scale,
    load_seqmodels_family,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", default="test")
    parser.add_argument("--max-families", type=int, default=12)
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def sample_long_pairs(
    length: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    left, right = np.triu_indices(length, k=24)
    replace = len(left) < count
    selected = rng.choice(len(left), size=count, replace=replace)
    return np.stack([left[selected], right[selected]], axis=-1).astype(np.int64)


def weighted_centered_one_hot(
    msa: np.ndarray,
    weights: np.ndarray,
    pssm: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
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


def randomized_left_modes(
    matrix: torch.Tensor,
    rank: int,
    seed: int,
    power_iterations: int = 2,
) -> torch.Tensor:
    torch.manual_seed(seed)
    omega = torch.randn(
        matrix.shape[1], rank, device=matrix.device, dtype=matrix.dtype
    )
    q, _ = torch.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(power_iterations):
        q, _ = torch.linalg.qr(matrix @ (matrix.T @ q), mode="reduced")
    small = (q.T @ matrix).cpu().numpy()
    left_small, _, _ = np.linalg.svd(small, full_matrices=False)
    return q @ torch.from_numpy(left_small.astype(np.float32)).to(matrix.device)


def projected_pair_blocks(
    weighted_features: torch.Tensor,
    msa: np.ndarray,
    weights: np.ndarray,
    pairs: np.ndarray,
    pssm: np.ndarray,
    chunk_size: int = 64,
) -> np.ndarray:
    blocks = []
    total_weight = float(weights.sum())
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start : start + chunk_size]
        left = weighted_features[:, chunk[:, 0], :]
        right = weighted_features[:, chunk[:, 1], :]
        block = torch.einsum("npa,npb->pab", left, right).cpu().numpy()
        valid = (msa[:, chunk[:, 0]] < 20) & (msa[:, chunk[:, 1]] < 20)
        valid_fraction = np.sum(weights[:, None] * valid, axis=0) / max(
            total_weight, 1e-8
        )
        block /= np.maximum(valid_fraction[:, None, None], 1e-8)
        blocks.append(block)
    blocks_array = np.concatenate(blocks, axis=0)
    row_probability = pssm[pairs[:, 0]]
    column_probability = pssm[pairs[:, 1]]
    row_effect = np.einsum("pa,pab->pb", row_probability, blocks_array)
    column_effect = np.einsum("pab,pb->pa", blocks_array, column_probability)
    grand = np.einsum(
        "pa,pab,pb->p", row_probability, blocks_array, column_probability
    )
    return (
        blocks_array
        - row_effect[:, None, :]
        - column_effect[:, :, None]
        + grand[:, None, None]
    ).astype(np.float32)


def score_metrics(
    blocks: np.ndarray,
    pairs: np.ndarray,
    pssm: np.ndarray,
    contacts: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    strength = np.sqrt(np.mean(np.square(blocks), axis=(-2, -1)))
    normalized = strength / entropy_pair_scale(pssm, pairs).clip(min=1e-8)
    valid = mask[pairs[:, 0], pairs[:, 1]]
    labels = contacts[pairs[:, 0], pairs[:, 1]] > 0.01
    count = max(1, int(round(0.1 * np.sum(valid))))
    ranking = np.argsort(normalized[valid])[::-1][:count]
    return {
        "contact_average_precision": binary_average_precision(
            normalized[valid], labels[valid]
        ),
        "contact_roc_auc": binary_roc_auc(normalized[valid], labels[valid]),
        "top10_precision": float(np.mean(labels[valid][ranking])),
        "score_mean": float(np.mean(normalized)),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    device = choose_device(args.device)
    frame = pd.read_csv(args.representations / "families.csv")
    frame = frame[frame["role"] == args.role].head(args.max_families)
    rows = []
    max_rank = max(args.ranks)
    for family_number, row in enumerate(frame.itertuples(index=False)):
        rng = np.random.default_rng(args.seed + int(row.index) * 104729)
        family = load_seqmodels_family(args.benchmark, row.file, row.x_id)
        _, pssm = load_representation(args.representations, row)
        pairs = sample_long_pairs(len(family.query), args.pairs, rng)
        raw_blocks = connected_pair_blocks(
            family.msa, family.weights, pairs, pssm=pssm
        )
        raw_metrics = score_metrics(
            raw_blocks,
            pairs,
            pssm,
            family.contacts,
            family.contact_mask,
        )
        rows.append(
            {"x_id": row.x_id, "rank_removed": 0, **raw_metrics}
        )

        weighted = weighted_centered_one_hot(
            family.msa, family.weights, pssm, device
        )
        flat = weighted.reshape(len(family.msa), -1)
        modes = randomized_left_modes(
            flat, max_rank, args.seed + family_number
        )
        for rank in args.ranks:
            basis = modes[:, :rank]
            residual_flat = flat - basis @ (basis.T @ flat)
            residual = residual_flat.reshape_as(weighted)
            blocks = projected_pair_blocks(
                residual,
                family.msa,
                family.weights,
                pairs,
                pssm,
            )
            metrics = score_metrics(
                blocks,
                pairs,
                pssm,
                family.contacts,
                family.contact_mask,
            )
            rows.append(
                {"x_id": row.x_id, "rank_removed": rank, **metrics}
            )

    result = pd.DataFrame(rows)
    result.to_csv(args.output / "per_family_metrics.csv", index=False)
    summary = result.groupby("rank_removed").mean(numeric_only=True)
    summary.to_csv(args.output / "summary.csv")
    (args.output / "run.json").write_text(
        json.dumps(vars(args) | {"device": str(device)}, default=str, indent=2)
    )
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
