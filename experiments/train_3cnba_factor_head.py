from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from transformer_disentanglement.categorical_interactions import (
    block_frobenius_scores,
    block_transpose_symmetrize,
    weighted_categorical_center,
    zero_position_diagonal,
)
from transformer_disentanglement.contacts import load_3cnba_distances
from transformer_disentanglement.data import ALPHABET, encode_alignment, load_aln
from transformer_disentanglement.interaction_model import MarginalInteractionHead
from transformer_disentanglement.metrics import (
    average_product_correction,
    contact_precision,
    leading_mode_fraction,
    safe_spearman,
)
from transformer_disentanglement.protein_transformer import (
    choose_device,
    extract_single_sequence_representation,
    load_model,
)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--jacobian", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["esm2_8m", "esm2_35m"], default="esm2_8m")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--sampled-pairs", type=int, default=512)
    parser.add_argument("--sparse-weight", type=float, default=2e-4)
    parser.add_argument("--strength-weight", type=float, default=0.25)
    parser.add_argument(
        "--entropy-leak-weight",
        type=float,
        default=0.0,
        help=(
            "Ablation only. Statistical decorrelation can remove genuine interactions; "
            "weighted zero-sum gauge is the default separation criterion."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def msa_amino_acid_frequencies(sequences: list[str], pseudocount: float = 1e-3) -> np.ndarray:
    encoded = encode_alignment(sequences)
    state_indices = np.asarray([ALPHABET.index(amino_acid) for amino_acid in AMINO_ACIDS])
    counts = np.stack([np.mean(encoded == index, axis=0) for index in state_indices], axis=-1)
    counts += pseudocount
    return counts / counts.sum(axis=-1, keepdims=True)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(probabilities + 1e-12), axis=-1)


def energy_support_fraction(scores: np.ndarray, fraction: float = 0.9) -> float:
    values = scores[np.triu_indices(scores.shape[0], k=1)] ** 2
    if values.sum() <= 0:
        return 0.0
    ordered = np.sort(values)[::-1]
    count = int(np.searchsorted(np.cumsum(ordered), fraction * ordered.sum()) + 1)
    return float(count / len(values))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    sequences = load_aln(args.alignment)
    sequence = sequences[0]
    pssm = msa_amino_acid_frequencies(sequences)
    distances = load_3cnba_distances(args.distances, args.reference)
    raw_blocks = np.load(args.jacobian)["raw_blocks"]
    if raw_blocks.shape[:2] != (len(sequence), len(sequence)):
        raise ValueError("Jacobian and sequence lengths differ")

    target_blocks = weighted_categorical_center(raw_blocks, pssm)
    target_blocks = zero_position_diagonal(
        block_transpose_symmetrize(target_blocks)
    ).astype(np.float32)
    target_rms = float(np.sqrt(np.mean(target_blocks**2)))
    normalized_target = target_blocks / max(target_rms, 1e-8)
    target_scores = block_frobenius_scores(target_blocks)
    np.fill_diagonal(target_scores, 0.0)

    device = choose_device(args.device)
    model, alphabet, architecture = load_model(args.model, device)
    if architecture != "single_sequence":
        raise ValueError("Factor-head demo requires a single-sequence model")
    hidden = extract_single_sequence_representation(
        model, alphabet, sequence, device
    )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    hidden_tensor = torch.from_numpy(hidden).to(device)
    pssm_tensor = torch.from_numpy(pssm.astype(np.float32)).to(device)
    target_tensor = torch.from_numpy(normalized_target).to(device)
    head = MarginalInteractionHead(
        hidden_dim=hidden.shape[-1], states=len(AMINO_ACIDS), rank=args.rank
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    pair_i, pair_j = np.triu_indices(len(sequence), k=1)
    pair_weights = target_scores[pair_i, pair_j]
    pair_weights = pair_weights + 0.05 * np.mean(pair_weights)
    pair_weights = pair_weights / pair_weights.sum()
    history = []
    for step in range(1, args.steps + 1):
        importance_count = args.sampled_pairs // 2
        uniform_count = args.sampled_pairs - importance_count
        importance = rng.choice(
            len(pair_i), size=importance_count, replace=True, p=pair_weights
        )
        uniform = rng.integers(0, len(pair_i), size=uniform_count)
        selected = np.concatenate([importance, uniform])
        pairs = torch.from_numpy(
            np.stack([pair_i[selected], pair_j[selected]], axis=-1).astype(np.int64)
        ).to(device)

        output = head(hidden_tensor, gauge_probabilities=pssm_tensor)
        gates = head.pair_gates(
            output["gate_left"], output["gate_right"], output["gate_bias"], pairs
        )
        predicted = head.pair_blocks(
            output["factors"], output["mode_scale"], pairs, gates=gates
        )
        expected = target_tensor[pairs[:, 0], pairs[:, 1]]
        interaction_loss = torch.mean((predicted - expected) ** 2)
        predicted_strength = torch.sqrt(predicted.square().mean(dim=(-2, -1)) + 1e-8)
        expected_strength = torch.sqrt(expected.square().mean(dim=(-2, -1)) + 1e-8)
        strength_loss = torch.mean((predicted_strength - expected_strength) ** 2)
        log_probabilities = torch.log_softmax(output["marginal_logits"], dim=-1)
        marginal_loss = -torch.mean(torch.sum(pssm_tensor * log_probabilities, dim=-1))
        predicted_entropy = -torch.sum(
            output["marginal_probabilities"]
            * torch.log(output["marginal_probabilities"].clamp_min(1e-8)),
            dim=-1,
        )
        target_entropy = -torch.sum(pssm_tensor * torch.log(pssm_tensor), dim=-1)
        entropy_loss = torch.mean((predicted_entropy - target_entropy) ** 2)
        group_sparsity = gates.mean()
        full_gates = head.full_gates(
            output["gate_left"], output["gate_right"], output["gate_bias"]
        )
        gate_row_strength = full_gates.mean(dim=-1)
        centered_gate = gate_row_strength - gate_row_strength.mean()
        centered_entropy = target_entropy - target_entropy.mean()
        entropy_correlation = torch.sum(centered_gate * centered_entropy) / (
            torch.linalg.vector_norm(centered_gate)
            * torch.linalg.vector_norm(centered_entropy)
            + 1e-8
        )
        loss = (
            interaction_loss
            + args.strength_weight * strength_loss
            + 0.05 * marginal_loss
            + 0.05 * entropy_loss
            + args.sparse_weight * group_sparsity
            + args.entropy_leak_weight * entropy_correlation.square()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "interaction_loss": float(interaction_loss.detach().cpu()),
                    "strength_loss": float(strength_loss.detach().cpu()),
                    "marginal_loss": float(marginal_loss.detach().cpu()),
                    "entropy_loss": float(entropy_loss.detach().cpu()),
                    "group_sparsity": float(group_sparsity.detach().cpu()),
                    "gate_entropy_correlation": float(
                        entropy_correlation.detach().cpu()
                    ),
                }
            )

    head.eval()
    with torch.no_grad():
        output = head(hidden_tensor, gauge_probabilities=pssm_tensor)
        predicted_gates = head.full_gates(
            output["gate_left"], output["gate_right"], output["gate_bias"]
        )
        predicted_blocks = head.full_blocks(
            output["factors"], output["mode_scale"], gates=predicted_gates
        ).cpu().numpy() * target_rms
        predicted_gates = predicted_gates.cpu().numpy()
        predicted_pssm = output["marginal_probabilities"].cpu().numpy()
        mode_scale = output["mode_scale"].cpu().numpy()

    predicted_scores = block_frobenius_scores(predicted_blocks)
    np.fill_diagonal(predicted_scores, 0.0)
    target_vector = target_scores[np.triu_indices(len(sequence), k=1)]
    predicted_vector = predicted_scores[np.triu_indices(len(sequence), k=1)]
    target_matrix_norm = np.linalg.norm(target_blocks)
    weighted_left = np.einsum("ia,ijab->ijb", pssm, predicted_blocks)
    weighted_right = np.einsum("ijab,jb->ija", predicted_blocks, pssm)
    metrics = {
        "model": args.model,
        "rank": args.rank,
        "steps": args.steps,
        "device": str(device),
        "target_rms": target_rms,
        "marginal_kl": float(
            np.mean(
                np.sum(
                    pssm * (np.log(pssm + 1e-12) - np.log(predicted_pssm + 1e-12)),
                    axis=-1,
                )
            )
        ),
        "entropy_spearman": safe_spearman(entropy(pssm), entropy(predicted_pssm)),
        "interaction_relative_error": float(
            np.linalg.norm(predicted_blocks - target_blocks) / target_matrix_norm
        ),
        "score_spearman": safe_spearman(target_vector, predicted_vector),
        "target_long_p_at_l": contact_precision(
            target_scores, distances, min_separation=24
        ),
        "predicted_long_p_at_l": contact_precision(
            predicted_scores, distances, min_separation=24
        ),
        "predicted_apc_long_p_at_l": contact_precision(
            average_product_correction(predicted_scores), distances, min_separation=24
        ),
        "target_leading_mode_fraction": leading_mode_fraction(target_scores),
        "predicted_leading_mode_fraction": leading_mode_fraction(predicted_scores),
        "target_energy_support_90": energy_support_fraction(target_scores),
        "predicted_energy_support_90": energy_support_fraction(predicted_scores),
        "gate_mean": float(predicted_gates.mean()),
        "gate_active_fraction_0_5": float(np.mean(predicted_gates > 0.5)),
        "gate_entropy_spearman": safe_spearman(
            predicted_gates.mean(axis=-1), entropy(pssm)
        ),
        "weighted_gauge_max_abs": float(
            max(np.max(np.abs(weighted_left)), np.max(np.abs(weighted_right)))
        ),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    np.savez_compressed(
        args.output / "predictions.npz",
        predicted_pssm=predicted_pssm,
        target_pssm=pssm,
        predicted_blocks=predicted_blocks,
        target_blocks=target_blocks,
        predicted_scores=predicted_scores,
        target_scores=target_scores,
        mode_scale=mode_scale,
        predicted_gates=predicted_gates,
    )
    torch.save(head.state_dict(), args.output / "factor_head.pt")
    (args.output / "run.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = (
        ("Teacher interaction", target_scores),
        ("Predicted interaction", predicted_scores),
        ("Predicted + APC", average_product_correction(predicted_scores)),
        ("Absolute error", np.abs(predicted_scores - target_scores)),
    )
    for axis, (title, matrix) in zip(axes, panels):
        image = axis.imshow(matrix, cmap="viridis")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(args.output / "factor_head_maps.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
