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
from transformer_disentanglement.demo_language_models import (
    DisentangledProteinLM,
    LocalProteinLM,
    TransformerProteinLM,
)
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import load_seqmodels_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_models(args: argparse.Namespace, device: torch.device):
    models = {
        "Local": LocalProteinLM(hidden_dim=64, layers=1).to(device),
        "Transformer": TransformerProteinLM(hidden_dim=64, layers=1).to(device),
        "Sparse": DisentangledProteinLM(
            hidden_dim=64,
            rank=8,
            index_dim=16,
            pair_dim=16,
            neighbors=8,
            layers=1,
        ).to(device),
    }
    checkpoints = {
        "Local": args.local,
        "Transformer": args.transformer,
        "Sparse": args.sparse,
    }
    for name, model in models.items():
        model.load_state_dict(
            torch.load(checkpoints[name], map_location=device, weights_only=True)
        )
        model.eval()
    return models


def paired_bootstrap(
    difference: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float, float]:
    indices = rng.integers(0, len(difference), size=(iterations, len(difference)))
    means = difference[indices].mean(axis=1)
    return (
        float(difference.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    models = load_models(args, device)
    frame = pd.read_csv(args.representations / "families.csv")
    frame = frame[frame["role"] == "validation"]
    rows = []
    with torch.no_grad():
        for row in frame.itertuples(index=False):
            family = load_seqmodels_family(args.benchmark, row.file, row.x_id)
            rng = np.random.default_rng(args.seed + int(row.index) * 104729)
            masked, selected = mask_sequence(family.msa[0], 0.15, rng)
            tokens = torch.from_numpy(masked)[None].to(device)
            targets = torch.from_numpy(
                family.msa[0, selected].astype(np.int64)
            ).to(device)
            for name, model in models.items():
                output = model(tokens)
                logits = output["logits"][0, selected]
                background = output["background_logits"][0, selected]
                record = {
                    "x_id": row.x_id,
                    "model": name,
                    "cross_entropy": float(F.cross_entropy(logits, targets).cpu()),
                    "background_cross_entropy": float(
                        F.cross_entropy(background, targets).cpu()
                    ),
                    "accuracy": float(
                        (logits.argmax(dim=-1) == targets).float().mean().cpu()
                    ),
                }
                if name == "Sparse":
                    record["interaction_rms"] = float(
                        torch.sqrt(
                            output["layers"][-1]["interaction_logits"][0, selected]
                            .square()
                            .mean()
                        ).cpu()
                    )
                else:
                    record["interaction_rms"] = 0.0
                rows.append(record)

    per_family = pd.DataFrame(rows)
    per_family.to_csv(args.output / "per_family_metrics.csv", index=False)
    pivot = per_family.pivot(index="x_id", columns="model", values="cross_entropy")
    sparse_background = (
        per_family[per_family["model"] == "Sparse"]
        .set_index("x_id")["background_cross_entropy"]
        .loc[pivot.index]
    )
    rng = np.random.default_rng(args.seed)
    comparisons = []
    for label, difference in [
        ("Sparse - Local", (pivot["Sparse"] - pivot["Local"]).to_numpy()),
        (
            "Sparse background - Local",
            (sparse_background - pivot["Local"]).to_numpy(),
        ),
        (
            "Sparse interaction contribution",
            (pivot["Sparse"] - sparse_background).to_numpy(),
        ),
        (
            "Transformer - Local",
            (pivot["Transformer"] - pivot["Local"]).to_numpy(),
        ),
    ]:
        mean, low, high = paired_bootstrap(
            difference, rng, args.bootstrap
        )
        comparisons.append(
            {"comparison": label, "mean": mean, "ci_2_5": low, "ci_97_5": high}
        )
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(args.output / "paired_bootstrap.csv", index=False)

    summaries = []
    result_directories = {
        "Local": args.local.parent,
        "Transformer": args.transformer.parent,
        "Sparse": args.sparse.parent,
    }
    for name in ["Local", "Sparse", "Transformer"]:
        external = json.loads(
            (result_directories[name] / "summary.json").read_text()
        )
        external_evaluation = external.get("evaluation")
        if external_evaluation is None:
            external_evaluation = external.get("post_adaptation")
        if external_evaluation is None:
            external_evaluation = external["adapted_topk"]
        group = per_family[per_family["model"] == name]
        summaries.append(
            {
                "model": name,
                "parameters": external["parameters"],
                "cross_entropy": float(group["cross_entropy"].mean()),
                "accuracy": float(group["accuracy"].mean()),
                "forward_tokens_per_second": (
                    external_evaluation["forward_tokens_per_second"]
                ),
                "peak_memory_mb": external["peak_device_memory_bytes"] / 1e6,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "model_summary.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    colors = ["#6b7280", "#d1495b", "#2a9d8f"]
    axes[0].bar(summary["model"], summary["cross_entropy"], color=colors)
    axes[0].set_title("Held-out masked-token loss")
    axes[0].set_ylabel("Cross entropy")
    axes[0].set_ylim(2.88, 2.95)
    axes[1].bar(
        summary["model"], summary["forward_tokens_per_second"], color=colors
    )
    axes[1].set_title("Forward throughput")
    axes[1].set_ylabel("Tokens / second")
    axes[2].bar(summary["model"], summary["parameters"], color=colors)
    axes[2].set_title("Parameter count")
    axes[2].set_ylabel("Parameters")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output / "end_to_end_demo_comparison.png", dpi=200)
    plt.close(figure)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print()
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.5f}"))


if __name__ == "__main__":
    main()
