from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from experiments.train_evolutionary_conditional_demo import (
    build_evolutionary_examples,
)
from transformer_disentanglement.demo_language_models import (
    DualStreamProteinLM,
    LocalProteinLM,
)
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def cosine_to_template(weights: np.ndarray) -> np.ndarray:
    template = weights.mean(axis=0)
    template /= max(np.linalg.norm(template), 1e-12)
    norms = np.linalg.norm(weights, axis=1).clip(min=1e-12)
    return weights @ template / norms


@torch.no_grad()
def main() -> None:
    cli = parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((cli.run / "run.json").read_text())
    args = SimpleNamespace(**config)
    device = choose_device(cli.device)

    background = LocalProteinLM(hidden_dim=args.stable_dim, layers=1).to(device)
    background.load_state_dict(
        torch.load(
            args.background_checkpoint, map_location=device, weights_only=True
        )
    )
    background.eval()
    families = pd.read_csv(Path(args.representations) / "families.csv")
    validation = families[families.role == "validation"].reset_index(drop=True)
    examples = build_evolutionary_examples(
        validation,
        args.eval_families,
        args,
        background,
        np.random.default_rng(args.seed + 3000),
        device,
    )
    del background

    model = DualStreamProteinLM(
        stable_dim=args.stable_dim,
        task_dim=args.task_dim,
        rank=args.rank,
        index_dim=args.index_dim,
        pair_dim=args.pair_dim,
        pair_mlp_dim=args.pair_mlp_dim,
        neighbors=args.neighbors,
        routing_mode="topk",
        value_mode=args.value_mode,
        adapter_count=args.adapter_count,
        adapter_topk=args.adapter_topk,
        adapter_bias_update_speed=getattr(args, "adapter_bias_update_speed", 0.0),
    ).to(device)
    model.load_state_dict(
        torch.load(cli.run / "model.pt", map_location=device, weights_only=True),
        strict=False,
    )
    model.eval()
    layer = model.interaction
    if layer.adapter_decoder is None:
        raise ValueError("run does not use pair_residual values")

    rows: list[dict[str, float | int | str]] = []
    all_weights = []
    for example in examples:
        tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
        pairs = example["pairs"].to(device)  # type: ignore[union-attr]
        output = model(tokens, use_interaction=False)
        value_state = output["value_state"][0]  # type: ignore[index]
        factors = output["factors"][0]  # type: ignore[index]
        left_state = value_state[pairs[:, 0]]
        right_state = value_state[pairs[:, 1]]
        left_weights = layer.sparse_adapter_weights(
            layer.adapter_decoder(
                layer.directional_pair_features(left_state, right_state)
            )
        )
        right_weights = layer.sparse_adapter_weights(
            layer.adapter_decoder(
                layer.directional_pair_features(right_state, left_state)
            )
        )
        left_residual, right_residual = layer.pair_factor_residuals(
            left_state, right_state
        )
        left_base = factors[pairs[:, 0]]
        right_base = factors[pairs[:, 1]]
        for index in range(len(pairs)):
            left_np = left_weights[index].cpu().numpy()
            right_np = right_weights[index].cpu().numpy()
            all_weights.extend([left_np, right_np])
            rows.append(
                {
                    "example": str(example["identifier"]),
                    "left_position": int(pairs[index, 0]),
                    "right_position": int(pairs[index, 1]),
                    "left_top_adapter": int(left_np.argmax()),
                    "right_top_adapter": int(right_np.argmax()),
                    "left_weight_entropy": float(
                        -(left_np * np.log(left_np + 1e-12)).sum()
                    ),
                    "right_weight_entropy": float(
                        -(right_np * np.log(right_np + 1e-12)).sum()
                    ),
                    "left_residual_ratio": float(
                        left_residual[index].square().mean().sqrt()
                        / left_base[index].square().mean().sqrt().clamp_min(1e-8)
                    ),
                    "right_residual_ratio": float(
                        right_residual[index].square().mean().sqrt()
                        / right_base[index].square().mean().sqrt().clamp_min(1e-8)
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    weights = np.asarray(all_weights)
    cosine = cosine_to_template(weights)
    top_counts = np.bincount(
        weights.argmax(axis=1), minlength=layer.adapter_count
    ).astype(np.float64)
    top_distribution = top_counts / top_counts.sum()
    summary = {
        "examples": len(examples),
        "directed_pair_sides": len(weights),
        "adapter_scale": float(torch.sigmoid(layer.adapter_scale_logit).cpu()),
        "mean_cosine_to_global_template": float(cosine.mean()),
        "p10_cosine_to_global_template": float(np.quantile(cosine, 0.1)),
        "top_adapter_distribution": top_distribution.tolist(),
        "top_adapter_effective_count": float(
            np.exp(-(top_distribution * np.log(top_distribution + 1e-12)).sum())
        ),
        "mean_weight_entropy": float(
            frame[["left_weight_entropy", "right_weight_entropy"]]
            .to_numpy()
            .mean()
        ),
        "mean_residual_ratio": float(
            frame[["left_residual_ratio", "right_residual_ratio"]]
            .to_numpy()
            .mean()
        ),
    }
    frame.to_csv(cli.output / "per_pair.csv", index=False)
    (cli.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
