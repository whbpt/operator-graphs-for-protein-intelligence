from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transformer_disentanglement.contacts import load_3cnba_distances
from transformer_disentanglement.data import (
    ALPHABET,
    encode_alignment,
    load_aln,
    sample_global_composition_null,
    sample_pssm_null,
    shuffle_columns_null,
    site_frequencies,
    subsample_alignment,
)
from transformer_disentanglement.metrics import (
    average_product_correction,
    contact_prevalence,
    contact_precision,
    mutual_information,
    normalized_entropy,
    pairwise_diagnostics,
    safe_spearman,
    separation_baseline,
)
from transformer_disentanglement.protein_transformer import (
    choose_device,
    extract_attention,
    load_model,
    symmetrized_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--null-replicates", type=int, default=2)
    parser.add_argument("--column-null-replicates", type=int, default=2)
    parser.add_argument("--global-null-replicates", type=int, default=1)
    parser.add_argument("--include-query-repeat", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model",
        choices=["esm2_8m", "esm2_35m", "msa_transformer"],
        default="esm2_8m",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-transformer", action="store_true")
    return parser.parse_args()


def classical_metrics(
    name: str, sequences: list[str], distances: np.ndarray
) -> tuple[dict[str, float | str], np.ndarray, np.ndarray, np.ndarray]:
    encoded = encode_alignment(sequences)
    frequencies = site_frequencies(encoded)
    entropy = normalized_entropy(frequencies)
    mi = mutual_information(encoded, states=len(ALPHABET))
    mi_apc = average_product_correction(mi)
    metrics: dict[str, float | str] = {
        "condition": name,
        "mean_entropy": float(entropy.mean()),
        "mi_contact_p_at_l_raw": contact_precision(mi, distances),
        "mi_contact_p_at_l_apc": contact_precision(mi_apc, distances),
        "mi_long_contact_p_at_l_raw": contact_precision(mi, distances, min_separation=24),
        "mi_long_contact_p_at_l_apc": contact_precision(
            mi_apc, distances, min_separation=24
        ),
    }
    metrics.update({f"mi_{key}": value for key, value in pairwise_diagnostics(mi, entropy).items()})
    return metrics, entropy, mi, mi_apc


def transformer_metrics(
    metrics: dict[str, float | str],
    attentions: np.ndarray,
    contacts: np.ndarray,
    entropy: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    symmetric = symmetrized_attention(attentions)
    head_rows = []
    for layer in range(symmetric.shape[0]):
        for head in range(symmetric.shape[1]):
            matrix = symmetric[layer, head]
            corrected = average_product_correction(matrix)
            diagnostics = pairwise_diagnostics(matrix, entropy)
            head_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    **diagnostics,
                    "raw_p_at_l": contact_precision(matrix, distances),
                    "apc_p_at_l": contact_precision(corrected, distances),
                    "raw_long_p_at_l": contact_precision(
                        matrix, distances, min_separation=24
                    ),
                    "apc_long_p_at_l": contact_precision(
                        corrected, distances, min_separation=24
                    ),
                }
            )
    heads = pd.DataFrame(head_rows)
    metrics.update(
        {
            "transformer_contact_p_at_l": contact_precision(contacts, distances),
            "transformer_long_contact_p_at_l": contact_precision(
                contacts, distances, min_separation=24
            ),
            "attention_median_leading_fraction": float(heads.leading_mode_fraction.median()),
            "attention_max_leading_fraction": float(heads.leading_mode_fraction.max()),
            "attention_median_abs_entropy_corr": float(
                heads.leading_vector_entropy_spearman.abs().median()
            ),
            "attention_max_abs_entropy_corr": float(
                heads.leading_vector_entropy_spearman.abs().max()
            ),
            "attention_best_raw_p_at_l": float(heads.raw_p_at_l.max()),
            "attention_best_apc_p_at_l": float(heads.apc_p_at_l.max()),
            "attention_best_raw_long_p_at_l": float(heads.raw_long_p_at_l.max()),
            "attention_best_apc_long_p_at_l": float(heads.apc_long_p_at_l.max()),
            "attention_heads_improved_by_apc": float(
                np.mean(heads.apc_p_at_l > heads.raw_p_at_l)
            ),
        }
    )
    metrics["attention_head_entropy_row_sum_corr"] = safe_spearman(
        heads.row_sum_entropy_spearman.to_numpy(),
        heads.leading_vector_entropy_spearman.to_numpy(),
    )
    return symmetric


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_sequences = load_aln(args.alignment)
    real_sequences = subsample_alignment(all_sequences, args.depth, rng)
    distances = load_3cnba_distances(args.distances, args.reference)
    if len(real_sequences[0]) != distances.shape[0]:
        raise ValueError(
            f"Alignment length {len(real_sequences[0])} != distance map {distances.shape[0]}"
        )

    conditions: list[tuple[str, list[str]]] = [("real", real_sequences)]
    for replicate in range(args.null_replicates):
        conditions.append(
            (
                f"pssm_null_{replicate}",
                sample_pssm_null(real_sequences, args.depth, rng, keep_query=True),
            )
        )
    for replicate in range(args.column_null_replicates):
        conditions.append(
            (
                f"column_shuffle_{replicate}",
                shuffle_columns_null(real_sequences, rng),
            )
        )
    for replicate in range(args.global_null_replicates):
        conditions.append(
            (
                f"global_composition_{replicate}",
                sample_global_composition_null(real_sequences, args.depth, rng),
            )
        )
    if args.include_query_repeat:
        conditions.append(("query_repeat", [real_sequences[0]] * args.depth))

    device = choose_device(args.device)
    model = alphabet = architecture = None
    if not args.skip_transformer:
        model, alphabet, architecture = load_model(args.model, device)

    rows = []
    maps: dict[
        str,
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ],
    ] = {}
    real_frequencies = site_frequencies(encode_alignment(real_sequences))
    separation_scores = separation_baseline(distances.shape[0])
    for name, sequences in conditions:
        metrics, entropy, mi, mi_apc = classical_metrics(name, sequences, distances)
        metrics["depth"] = args.depth
        metrics["seed"] = args.seed
        metrics["device"] = str(device)
        metrics["contact_prevalence_sep6"] = contact_prevalence(distances)
        metrics["contact_prevalence_sep24"] = contact_prevalence(
            distances, min_separation=24
        )
        metrics["separation_baseline_p_at_l"] = contact_precision(
            separation_scores, distances
        )
        metrics["separation_baseline_long_p_at_l"] = contact_precision(
            separation_scores, distances, min_separation=24
        )
        frequencies = site_frequencies(encode_alignment(sequences))
        metrics["pssm_mae_vs_real"] = float(np.mean(np.abs(frequencies - real_frequencies)))
        attention_mean = None
        attention_heads = None
        contacts = None
        if model is not None and alphabet is not None and architecture is not None:
            attentions, contacts = extract_attention(
                model,
                alphabet,
                architecture,
                sequences,
                device,
                batch_size=args.batch_size,
            )
            attention_heads = transformer_metrics(
                metrics, attentions, contacts, entropy, distances
            )
            attention_mean = attention_heads.mean(axis=(0, 1))
            metrics["transformer_model"] = args.model
        rows.append(metrics)
        maps[name] = (mi, mi_apc, attention_mean, attention_heads, contacts)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "metrics.csv", index=False)
    (args.output / "run.json").write_text(
        json.dumps(vars(args) | {"device": str(device)}, default=str, indent=2)
    )

    columns = 4 if model is not None else 2
    figure, axes = plt.subplots(len(conditions), columns, figsize=(4 * columns, 4 * len(conditions)))
    axes = np.asarray(axes).reshape(len(conditions), columns)
    for row_index, (name, _) in enumerate(conditions):
        mi, mi_apc, attention_mean, _, contacts = maps[name]
        panels = [("MI", mi), ("MI + APC", mi_apc)]
        if attention_mean is not None:
            panels.append(("Mean sym. attention", attention_mean))
        if contacts is not None:
            panels.append(("Contact head", contacts))
        for column_index, (title, matrix) in enumerate(panels):
            axes[row_index, column_index].imshow(matrix, cmap="viridis")
            axes[row_index, column_index].set_title(f"{name}: {title}")
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
    figure.tight_layout()
    figure.savefig(args.output / "pairwise_maps.png", dpi=180)
    plt.close(figure)
    arrays = {}
    for name, (mi, mi_apc, attention_mean, attention_heads, contacts) in maps.items():
        arrays[f"{name}__mi"] = mi
        arrays[f"{name}__mi_apc"] = mi_apc
        if attention_mean is not None:
            arrays[f"{name}__attention_mean"] = attention_mean
        if attention_heads is not None:
            arrays[f"{name}__attention_heads"] = attention_heads.astype(np.float32)
        if contacts is not None:
            arrays[f"{name}__contacts"] = contacts
    np.savez_compressed(args.output / "pairwise_arrays.npz", **arrays)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
