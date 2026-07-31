from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import eigsh

from transformer_disentanglement.categorical_interactions import (
    block_frobenius_scores,
    block_transpose_symmetrize,
    blocks_to_matrix,
    categorical_double_center,
    matrix_to_blocks,
    reconstruct_symmetric_modes,
    zero_position_diagonal,
)
from transformer_disentanglement.contacts import load_3cnba_distances
from transformer_disentanglement.data import load_aln
from transformer_disentanglement.metrics import (
    average_product_correction,
    contact_precision,
    leading_mode_fraction,
    safe_spearman,
)
from transformer_disentanglement.protein_transformer import choose_device, load_model


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["esm2_8m", "esm2_35m"], default="esm2_8m")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reuse-jacobian", action="store_true")
    return parser.parse_args()


def extract_finite_mutation_jacobian(
    model,
    alphabet,
    sequence: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    batch_converter = alphabet.get_batch_converter()
    amino_acid_tokens = torch.tensor(
        [alphabet.get_idx(amino_acid) for amino_acid in AMINO_ACIDS],
        device=device,
    )

    _, _, baseline_tokens = batch_converter([("query", sequence)])
    baseline_tokens = baseline_tokens.to(device)
    with torch.no_grad():
        baseline = model(baseline_tokens)["logits"][0, 1:-1]
        baseline = baseline.index_select(-1, amino_acid_tokens).float().cpu()

    length = len(sequence)
    states = len(AMINO_ACIDS)
    mutations = [(position, state) for position in range(length) for state in range(states)]
    jacobian = np.empty((length, states, length, states), dtype=np.float32)

    for start in range(0, len(mutations), batch_size):
        batch_mutations = mutations[start : start + batch_size]
        sequences = []
        for position, state in batch_mutations:
            mutant = sequence[:position] + AMINO_ACIDS[state] + sequence[position + 1 :]
            sequences.append((f"mutant_{position}_{state}", mutant))
        _, _, tokens = batch_converter(sequences)
        tokens = tokens.to(device)
        with torch.no_grad():
            logits = model(tokens)["logits"][:, 1:-1]
            logits = logits.index_select(-1, amino_acid_tokens).float().cpu()
        differences = logits - baseline.unsqueeze(0)
        for batch_index, (position, state) in enumerate(batch_mutations):
            jacobian[:, :, position, state] = differences[batch_index].numpy()
    return jacobian.transpose(0, 2, 1, 3)


def score_row(
    name: str,
    blocks: np.ndarray,
    full_blocks: np.ndarray,
    distances: np.ndarray,
) -> tuple[dict[str, float | str], np.ndarray]:
    scores = block_frobenius_scores(blocks)
    np.fill_diagonal(scores, 0.0)
    corrected = average_product_correction(scores)
    full_matrix = blocks_to_matrix(full_blocks)
    matrix = blocks_to_matrix(blocks)
    denominator = np.linalg.norm(full_matrix)
    relative_error = np.linalg.norm(matrix - full_matrix) / denominator
    return (
        {
            "condition": name,
            "relative_frobenius_error": float(relative_error),
            "score_spearman_vs_full": safe_spearman(
                scores[np.triu_indices(scores.shape[0], k=1)],
                block_frobenius_scores(full_blocks)[np.triu_indices(scores.shape[0], k=1)],
            ),
            "score_leading_mode_fraction": leading_mode_fraction(scores),
            "raw_p_at_l": contact_precision(scores, distances),
            "apc_p_at_l": contact_precision(corrected, distances),
            "raw_long_p_at_l": contact_precision(scores, distances, min_separation=24),
            "apc_long_p_at_l": contact_precision(
                corrected, distances, min_separation=24
            ),
        },
        scores,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    sequence = load_aln(args.alignment)[0]
    distances = load_3cnba_distances(args.distances, args.reference)
    if len(sequence) != distances.shape[0]:
        raise ValueError("Query sequence and distance map have different lengths")

    cache_path = args.output / "categorical_jacobian.npz"
    if args.reuse_jacobian and cache_path.exists():
        raw_blocks = np.load(cache_path)["raw_blocks"]
        device = choose_device(args.device)
    else:
        device = choose_device(args.device)
        model, alphabet, architecture = load_model(args.model, device)
        if architecture != "single_sequence":
            raise ValueError("This experiment requires a single-sequence model")
        raw_blocks = extract_finite_mutation_jacobian(
            model, alphabet, sequence, device, args.batch_size
        )
        np.savez_compressed(cache_path, raw_blocks=raw_blocks)

    centered = categorical_double_center(raw_blocks)
    symmetric = block_transpose_symmetrize(centered)
    antisymmetric = 0.5 * (centered - centered.transpose(1, 0, 3, 2))
    full_blocks = zero_position_diagonal(symmetric)
    full_matrix = blocks_to_matrix(full_blocks).astype(np.float64)

    component_rows = []
    centered_norm = np.linalg.norm(zero_position_diagonal(centered))
    for name, component in (
        ("raw", raw_blocks),
        ("categorical_centered", centered),
        ("symmetric", symmetric),
        ("antisymmetric", antisymmetric),
    ):
        component = zero_position_diagonal(component)
        row, _ = score_row(name, component, full_blocks, distances)
        component_norm = np.linalg.norm(component)
        row["tensor_norm"] = float(component_norm)
        row["squared_norm_fraction_of_centered"] = float(
            component_norm**2 / centered_norm**2
        )
        component_rows.append(row)
    pd.DataFrame(component_rows).to_csv(
        args.output / "component_metrics.csv", index=False
    )

    max_rank = min(max(args.ranks), full_matrix.shape[0] - 2)
    eigenvalues, eigenvectors = eigsh(full_matrix, k=max_rank, which="LM")
    np.savez_compressed(
        args.output / "categorical_modes.npz",
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors.astype(np.float32),
    )

    rows = []
    maps: dict[str, np.ndarray] = {}
    full_row, full_scores = score_row("full", full_blocks, full_blocks, distances)
    rows.append(full_row)
    maps["full"] = full_scores
    for rank in sorted(set(args.ranks)):
        if rank > max_rank:
            continue
        reconstructed = reconstruct_symmetric_modes(
            eigenvalues, eigenvectors, rank
        )
        blocks = matrix_to_blocks(reconstructed, len(sequence), len(AMINO_ACIDS))
        row, scores = score_row(f"rank_{rank}", blocks, full_blocks, distances)
        row["rank"] = rank
        rows.append(row)
        maps[f"rank_{rank}"] = scores

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "compression_metrics.csv", index=False)
    (args.output / "run.json").write_text(
        json.dumps(vars(args) | {"device": str(device), "length": len(sequence)}, default=str, indent=2)
    )

    selected = ["full"]
    selected.extend(f"rank_{rank}" for rank in args.ranks if f"rank_{rank}" in maps)
    columns = min(4, len(selected))
    rows_count = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(4 * columns, 4 * rows_count))
    axes = np.asarray(axes).reshape(-1)
    for axis, name in zip(axes, selected):
        image = axis.imshow(average_product_correction(maps[name]), cmap="viridis")
        axis.set_title(name)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes[len(selected) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output / "compression_maps.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
