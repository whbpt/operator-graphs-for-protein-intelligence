from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.train_end_to_end_demo import mask_sequence
from transformer_disentanglement.demo_language_models import DisentangledProteinLM
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import load_seqmodels_family
from transformer_disentanglement.task_gradient import (
    normalized_task_gradient_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--seed12", type=Path, required=True)
    parser.add_argument("--seed13", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=20_000)
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device) -> DisentangledProteinLM:
    model = DisentangledProteinLM(
        hidden_dim=64,
        rank=8,
        index_dim=16,
        pair_dim=16,
        neighbors=8,
        layers=1,
        routing_mode="topk",
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def bootstrap_difference(
    values: np.ndarray,
    seed: int,
    iterations: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    frame = pd.read_csv(args.representations / "families.csv")
    frame = frame[frame["role"] == "validation"]
    specifications = [
        (20260712, args.seed12),
        (20260713, args.seed13),
    ]
    rows = []
    with torch.no_grad():
        for seed, checkpoint in specifications:
            model = load_model(checkpoint, device)
            for row in frame.itertuples(index=False):
                family = load_seqmodels_family(args.benchmark, row.file, row.x_id)
                rng = np.random.default_rng(seed + int(row.index) * 104729)
                masked, selected = mask_sequence(family.msa[0], 0.15, rng)
                tokens = torch.from_numpy(masked)[None].to(device)
                targets = torch.from_numpy(
                    family.msa[0, selected].astype(np.int64)
                ).to(device)
                output = model(tokens)
                logits = output["logits"][0, selected]
                background = output["background_logits"][0, selected]
                interaction = output["layers"][-1]["interaction_logits"][
                    0, selected
                ]
                target_field = normalized_task_gradient_target(
                    background, targets, target_rms=0.1
                )
                cosine = F.cosine_similarity(
                    interaction, target_field, dim=-1, eps=1e-8
                )
                total_loss = float(F.cross_entropy(logits, targets).cpu())
                background_loss = float(
                    F.cross_entropy(background, targets).cpu()
                )
                rows.append(
                    {
                        "seed": str(seed),
                        "x_id": row.x_id,
                        "total_cross_entropy": total_loss,
                        "background_cross_entropy": background_loss,
                        "interaction_contribution": total_loss - background_loss,
                        "task_gradient_cosine": float(cosine.mean().cpu()),
                        "interaction_rms": float(
                            torch.sqrt(interaction.square().mean()).cpu()
                        ),
                    }
                )
    per_family = pd.DataFrame(rows)
    per_family.to_csv(args.output / "per_family_metrics.csv", index=False)
    summaries = []
    for seed, group in per_family.groupby("seed"):
        mean, low, high = bootstrap_difference(
            group["interaction_contribution"].to_numpy(),
            int(seed),
            args.bootstrap,
        )
        summaries.append(
            {
                "seed": seed,
                "families": len(group),
                "interaction_contribution": mean,
                "ci_2_5": low,
                "ci_97_5": high,
                "task_gradient_cosine": float(
                    group["task_gradient_cosine"].mean()
                ),
                "interaction_rms": float(group["interaction_rms"].mean()),
            }
        )
    family_mean = per_family.groupby("x_id").mean(numeric_only=True)
    mean, low, high = bootstrap_difference(
        family_mean["interaction_contribution"].to_numpy(),
        20260714,
        args.bootstrap,
    )
    summaries.append(
        {
            "seed": "family-mean",
            "families": len(family_mean),
            "interaction_contribution": mean,
            "ci_2_5": low,
            "ci_97_5": high,
            "task_gradient_cosine": float(
                family_mean["task_gradient_cosine"].mean()
            ),
            "interaction_rms": float(family_mean["interaction_rms"].mean()),
        }
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "summary.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    colors = ["#6b7280", "#d1495b", "#2a9d8f"]
    error_low = summary["interaction_contribution"] - summary["ci_2_5"]
    error_high = summary["ci_97_5"] - summary["interaction_contribution"]
    axes[0].bar(
        summary["seed"],
        summary["interaction_contribution"],
        yerr=np.stack([error_low, error_high]),
        color=colors,
        capsize=4,
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("Interaction contribution")
    axes[0].set_ylabel("Total CE - background CE")
    axes[1].bar(
        summary["seed"], summary["task_gradient_cosine"], color=colors
    )
    axes[1].set_title("Task-gradient alignment")
    axes[1].set_ylabel("Mean cosine")
    axes[2].bar(summary["seed"], summary["interaction_rms"], color=colors)
    axes[2].set_title("Interaction magnitude")
    axes[2].set_ylabel("RMS logits")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output / "task_gradient_replication.png", dpi=200)
    plt.close(figure)

    metadata = {
        "bootstrap": args.bootstrap,
        "checkpoints": [str(args.seed12), str(args.seed13)],
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
