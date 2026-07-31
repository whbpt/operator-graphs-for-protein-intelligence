from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    specifications = [
        (
            "20260712",
            "end_to_end_soft_then_sparse_v1",
            "end_to_end_local_1600",
            "end_to_end_transformer_1600",
        ),
        (
            "20260713",
            "end_to_end_soft_then_sparse_seed13",
            "end_to_end_local_seed13_1600",
            "end_to_end_transformer_seed13_1600",
        ),
    ]
    stage_rows = []
    final_rows = []
    for seed, sparse_dir, local_dir, transformer_dir in specifications:
        sparse = read(args.results / sparse_dir)
        local = read(args.results / local_dir)
        transformer = read(args.results / transformer_dir)
        stages = {
            "Initial soft": sparse["initial_soft"],
            "Trained soft": sparse["trained_soft"],
            "Converted top-k": sparse["converted_topk"],
            "Adapted top-k": sparse["adapted_topk"],
        }
        for stage, values in stages.items():
            stage_rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "cross_entropy": values["masked_cross_entropy"],
                    "background_cross_entropy": values[
                        "background_cross_entropy"
                    ],
                    "interaction_contribution": values["masked_cross_entropy"]
                    - values["background_cross_entropy"],
                }
            )
        for model, values in [
            ("Local", local["evaluation"]),
            ("Sparse", sparse["adapted_topk"]),
            ("Transformer", transformer["evaluation"]),
        ]:
            final_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "cross_entropy": values["masked_cross_entropy"],
                    "accuracy": values["masked_accuracy"],
                }
            )
    stages = pd.DataFrame(stage_rows)
    final = pd.DataFrame(final_rows)
    stages.to_csv(args.output / "soft_sparse_stages.csv", index=False)
    final.to_csv(args.output / "replicated_model_results.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    stage_order = ["Initial soft", "Trained soft", "Converted top-k", "Adapted top-k"]
    for seed, group in stages.groupby("seed"):
        ordered = group.set_index("stage").loc[stage_order]
        axes[0].plot(
            stage_order,
            ordered["cross_entropy"],
            marker="o",
            label=seed,
        )
    axes[0].set_title("Dense-to-sparse conversion")
    axes[0].set_ylabel("Held-out cross entropy")
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].legend(frameon=False, fontsize=8)

    model_order = ["Local", "Sparse", "Transformer"]
    positions = np.arange(len(model_order))
    width = 0.35
    colors = ["#6b7280", "#d1495b"]
    for index, (seed, group) in enumerate(final.groupby("seed")):
        ordered = group.set_index("model").loc[model_order]
        axes[1].bar(
            positions + (index - 0.5) * width,
            ordered["cross_entropy"],
            width=width,
            color=colors[index],
            label=seed,
        )
    axes[1].set_xticks(positions, model_order)
    axes[1].set_ylim(2.85, 2.95)
    axes[1].set_title("Two-seed final comparison")
    axes[1].set_ylabel("Held-out cross entropy")
    axes[1].legend(frameon=False, fontsize=8)

    adapted = stages[stages["stage"] == "Adapted top-k"]
    axes[2].bar(
        adapted["seed"],
        adapted["interaction_contribution"],
        color=colors,
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_title("Typed interaction contribution")
    axes[2].set_ylabel("Total CE - background CE")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output / "soft_sparse_replication.png", dpi=200)
    plt.close(figure)

    print(stages.to_string(index=False))
    print()
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
