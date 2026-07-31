from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transformer_disentanglement.contacts import load_3cnba_distances
from transformer_disentanglement.metrics import (
    average_product_correction,
    contact_precision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def condition_names(arrays, suffix: str) -> list[str]:
    return sorted(
        key.removesuffix(suffix)
        for key in arrays.files
        if key.endswith(suffix) and not key.startswith("real__")
    )


def group_names(names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name in names:
        if name.startswith("pssm_null_"):
            group = "pssm_null"
        elif name.startswith("column_shuffle_"):
            group = "column_shuffle"
        elif name.startswith("global_composition_"):
            group = "global_composition"
        else:
            group = name
        groups.setdefault(group, []).append(name)
    return groups


def mean_condition(arrays, names: list[str], suffix: str) -> np.ndarray:
    return np.mean([arrays[f"{name}{suffix}"] for name in names], axis=0)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.arrays)
    distances = load_3cnba_distances(args.distances, args.reference)
    groups = group_names(condition_names(arrays, "__contacts"))

    real_contacts = arrays["real__contacts"]
    real_heads = arrays["real__attention_heads"]
    real_mi = arrays["real__mi"]
    real_mi_apc = arrays["real__mi_apc"]
    summary_rows = []
    head_rows = []

    for group, names in groups.items():
        null_contacts = mean_condition(arrays, names, "__contacts")
        null_heads = mean_condition(arrays, names, "__attention_heads")
        null_mi = mean_condition(arrays, names, "__mi")
        null_mi_apc = mean_condition(arrays, names, "__mi_apc")
        contact_residual = real_contacts - null_contacts
        mi_residual = real_mi - null_mi
        mi_apc_residual = real_mi_apc - null_mi_apc
        head_residual = real_heads - null_heads

        for layer in range(head_residual.shape[0]):
            for head in range(head_residual.shape[1]):
                residual = head_residual[layer, head]
                head_rows.append(
                    {
                        "null_group": group,
                        "layer": layer,
                        "head": head,
                        "residual_long_p_at_l": contact_precision(
                            residual, distances, min_separation=24
                        ),
                        "absolute_residual_long_p_at_l": contact_precision(
                            np.abs(residual), distances, min_separation=24
                        ),
                        "apc_residual_long_p_at_l": contact_precision(
                            average_product_correction(residual),
                            distances,
                            min_separation=24,
                        ),
                    }
                )

        group_heads = [row for row in head_rows if row["null_group"] == group]
        best_signed = max(group_heads, key=lambda row: row["residual_long_p_at_l"])
        best_absolute = max(
            group_heads, key=lambda row: row["absolute_residual_long_p_at_l"]
        )
        summary_rows.append(
            {
                "null_group": group,
                "replicates": len(names),
                "contact_residual_p_at_l": contact_precision(
                    contact_residual, distances
                ),
                "contact_residual_long_p_at_l": contact_precision(
                    contact_residual, distances, min_separation=24
                ),
                "absolute_contact_residual_long_p_at_l": contact_precision(
                    np.abs(contact_residual), distances, min_separation=24
                ),
                "mi_residual_long_p_at_l": contact_precision(
                    mi_residual, distances, min_separation=24
                ),
                "mi_apc_residual_long_p_at_l": contact_precision(
                    mi_apc_residual, distances, min_separation=24
                ),
                "best_attention_residual_long_p_at_l": best_signed[
                    "residual_long_p_at_l"
                ],
                "best_attention_residual_layer": best_signed["layer"],
                "best_attention_residual_head": best_signed["head"],
                "best_absolute_attention_residual_long_p_at_l": best_absolute[
                    "absolute_residual_long_p_at_l"
                ],
                "best_absolute_attention_residual_layer": best_absolute["layer"],
                "best_absolute_attention_residual_head": best_absolute["head"],
            }
        )

        ground_truth = np.isfinite(distances) & (distances < 8.0)
        figure, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        for axis, title, matrix in zip(
            axes,
            [
                "Real contact head",
                f"{group} mean",
                "Absolute residual",
                "Structural contacts",
            ],
            [real_contacts, null_contacts, np.abs(contact_residual), ground_truth],
        ):
            axis.imshow(matrix, cmap="viridis")
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.tight_layout()
        figure.savefig(args.output / f"residual_{group}.png", dpi=180)
        plt.close(figure)

    summary = pd.DataFrame(summary_rows)
    heads = pd.DataFrame(head_rows)
    summary.to_csv(args.output / "residual_summary.csv", index=False)
    heads.to_csv(args.output / "residual_heads.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
