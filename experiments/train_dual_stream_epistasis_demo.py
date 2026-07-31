from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.train_epistasis_identifiability import (
    build_examples,
    load_frozen_models,
)
from transformer_disentanglement.demo_language_models import DualStreamProteinLM
from transformer_disentanglement.epistasis import robust_standardize
from transformer_disentanglement.metrics import (
    binary_average_precision,
    safe_spearman,
)
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--background-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stable-dim", type=int, default=64)
    parser.add_argument("--task-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--index-dim", type=int, default=16)
    parser.add_argument("--pair-dim", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--train-families", type=int, default=64)
    parser.add_argument("--eval-families", type=int, default=24)
    parser.add_argument("--pairs-per-family", type=int, default=16)
    parser.add_argument("--mutation-states", default="0,1,2,3,4")
    parser.add_argument("--probe-fraction", type=float, default=0.15)
    parser.add_argument("--min-separation", type=int, default=6)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--sparse-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--index-weight", type=float, default=0.2)
    parser.add_argument("--shape-weight", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_local_initialization(
    model: DualStreamProteinLM, checkpoint: Path, device: torch.device
) -> None:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    mapping = {
        "stable_embedding.weight": "token_embedding.weight",
        "position_embedding.weight": "position_embedding.weight",
        "stable_norm.weight": "norms.0.weight",
        "stable_norm.bias": "norms.0.bias",
        "stable_depthwise.weight": "depthwise.0.weight",
        "stable_depthwise.bias": "depthwise.0.bias",
        "stable_pointwise.weight": "pointwise.0.weight",
        "stable_pointwise.bias": "pointwise.0.bias",
        "interaction.background_norm.weight": "output_norm.weight",
        "interaction.background_norm.bias": "output_norm.bias",
        "interaction.background_projection.weight": "output.weight",
        "interaction.background_projection.bias": "output.bias",
    }
    model_state = model.state_dict()
    for target, source in mapping.items():
        if model_state[target].shape != state[source].shape:
            raise ValueError(f"Checkpoint shape mismatch for {target}")
        model_state[target] = state[source]
    model.load_state_dict(model_state)


def strength_target(example: dict[str, torch.Tensor | str | float]) -> torch.Tensor:
    physical = example["physical_target"]  # type: ignore[assignment]
    strength = physical.square().mean(dim=(-2, -1)).sqrt().clamp_min(1e-10)
    standardized, _, _ = robust_standardize(torch.log(strength))
    return standardized


def top_metrics(scores: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    count = max(1, int(round(0.1 * len(target))))
    true_top = np.argsort(target)[-count:]
    predicted_top = np.argsort(scores)[-count:]
    labels = np.zeros(len(target), dtype=bool)
    labels[true_top] = True
    return (
        binary_average_precision(scores, labels),
        len(np.intersect1d(true_top, predicted_top)) / count,
    )


def pair_outputs(
    model: DualStreamProteinLM,
    output: dict[str, torch.Tensor | None],
    pairs: torch.Tensor,
    mutation_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_batch = pairs[None]
    scores = model.interaction.sampled_index_scores(
        output["encoded_task"], pair_batch  # type: ignore[arg-type]
    )[0]
    blocks = model.interaction.sampled_pair_blocks(
        output["factors"],  # type: ignore[arg-type]
        output["value_state"],  # type: ignore[arg-type]
        pair_batch,
    )[0]
    blocks = blocks[:, mutation_states][:, :, mutation_states]
    normalized_scores = (scores - scores.mean()) / scores.std(
        unbiased=False
    ).clamp_min(1e-6)
    normalized_blocks = blocks / blocks.square().mean().sqrt().clamp_min(1e-6)
    return normalized_scores, normalized_blocks


def example_loss(
    model: DualStreamProteinLM,
    example: dict[str, torch.Tensor | str | float],
    device: torch.device,
    mutation_states: torch.Tensor,
    use_interaction: bool,
    index_weight: float,
    shape_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
    probes = example["probe_positions"].to(device)  # type: ignore[union-attr]
    targets = example["probe_targets"].to(device)  # type: ignore[union-attr]
    pairs = example["pairs"].to(device)  # type: ignore[union-attr]
    shape_target = example["target"].to(device)  # type: ignore[union-attr]
    index_target = strength_target(example).to(device)
    output = model(tokens, use_interaction=use_interaction)
    logits = output["logits"][0, probes]  # type: ignore[index]
    mlm_loss = F.cross_entropy(logits, targets)
    predicted_index, predicted_shape = pair_outputs(
        model, output, pairs, mutation_states
    )
    index_loss = F.smooth_l1_loss(predicted_index, index_target)
    shape_loss = F.smooth_l1_loss(predicted_shape, shape_target)
    loss = mlm_loss + index_weight * index_loss + shape_weight * shape_loss
    return loss, {
        "mlm_loss": float(mlm_loss.detach().cpu()),
        "index_loss": float(index_loss.detach().cpu()),
        "shape_loss": float(shape_loss.detach().cpu()),
    }


def train_phase(
    model: DualStreamProteinLM,
    examples: list[dict[str, torch.Tensor | str | float]],
    steps: int,
    args: argparse.Namespace,
    device: torch.device,
    mutation_states: torch.Tensor,
    rng: np.random.Generator,
    use_interaction: bool,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    history = []
    model.train()
    for step in range(1, steps + 1):
        example = examples[int(rng.integers(len(examples)))]
        loss, metrics = example_loss(
            model,
            example,
            device,
            mutation_states,
            use_interaction,
            args.index_weight,
            args.shape_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == steps:
            history.append(
                {"step": step, "loss": float(loss.detach().cpu())} | metrics
            )
    return history


@torch.no_grad()
def evaluate(
    model: DualStreamProteinLM,
    examples: list[dict[str, torch.Tensor | str | float]],
    device: torch.device,
    mutation_states: torch.Tensor,
    use_interaction: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows = []
    for example in examples:
        tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
        probes = example["probe_positions"].to(device)  # type: ignore[union-attr]
        targets = example["probe_targets"].to(device)  # type: ignore[union-attr]
        pairs = example["pairs"].to(device)  # type: ignore[union-attr]
        shape_target = example["target"].to(device)  # type: ignore[union-attr]
        index_target = strength_target(example).to(device)
        output = model(tokens, use_interaction=use_interaction)
        logits = output["logits"][0, probes]  # type: ignore[index]
        background = output["background_logits"][0, probes]  # type: ignore[index]
        predicted_index, predicted_shape = pair_outputs(
            model, output, pairs, mutation_states
        )
        index_np = predicted_index.cpu().numpy()
        target_np = index_target.cpu().numpy()
        average_precision, recall = top_metrics(index_np, target_np)
        rows.append(
            {
                "family": example["identifier"],
                "cross_entropy": float(F.cross_entropy(logits, targets).cpu()),
                "background_cross_entropy": float(
                    F.cross_entropy(background, targets).cpu()
                ),
                "index_spearman": safe_spearman(index_np, target_np),
                "index_top_ap": average_precision,
                "index_top_recall": recall,
                "shape_mse": float(F.mse_loss(predicted_shape, shape_target).cpu()),
                "shape_correlation": safe_spearman(
                    predicted_shape.cpu().numpy().ravel(),
                    shape_target.cpu().numpy().ravel(),
                ),
                "interaction_rms": float(
                    output["interaction_logits"].square().mean().sqrt().cpu()  # type: ignore[union-attr]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        column: float(frame[column].mean())
        for column in frame.columns
        if column != "family"
    }, frame


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    args.hidden_dim = args.stable_dim
    args.features = "local"
    args.target_normalization = "family"
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    teacher, background = load_frozen_models(args, device)
    frame = pd.read_csv(args.representations / "families.csv")
    train_frame = frame[frame["role"] == "train"].reset_index(drop=True)
    eval_frame = frame[frame["role"] == "validation"].reset_index(drop=True)
    mutation_states = torch.tensor(
        [int(value) for value in args.mutation_states.split(",")], device=device
    )
    start = time.perf_counter()
    train_examples = build_examples(
        train_frame,
        args.train_families,
        args,
        teacher,
        background,
        mutation_states,
        np.random.default_rng(args.seed + 1000),
        device,
    )
    eval_examples = build_examples(
        eval_frame,
        args.eval_families,
        args,
        teacher,
        background,
        mutation_states,
        np.random.default_rng(args.seed + 3000),
        device,
    )
    data_seconds = time.perf_counter() - start
    del teacher, background

    model = DualStreamProteinLM(
        stable_dim=args.stable_dim,
        task_dim=args.task_dim,
        rank=args.rank,
        index_dim=args.index_dim,
        pair_dim=args.pair_dim,
        neighbors=args.neighbors,
        routing_mode="topk",
    ).to(device)
    load_local_initialization(model, args.background_checkpoint, device)
    initial, _ = evaluate(
        model, eval_examples, device, mutation_states, use_interaction=False
    )
    rng = np.random.default_rng(args.seed + 4000)
    warmup_history = train_phase(
        model,
        train_examples,
        args.warmup_steps,
        args,
        device,
        mutation_states,
        rng,
        use_interaction=False,
    )
    warmup_background, _ = evaluate(
        model, eval_examples, device, mutation_states, use_interaction=False
    )
    converted_topk, converted_frame = evaluate(
        model, eval_examples, device, mutation_states, use_interaction=True
    )
    sparse_history = train_phase(
        model,
        train_examples,
        args.sparse_steps,
        args,
        device,
        mutation_states,
        rng,
        use_interaction=True,
    )
    adapted_topk, adapted_frame = evaluate(
        model, eval_examples, device, mutation_states, use_interaction=True
    )
    summary = {
        "seed": args.seed,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "data_generation_seconds": data_seconds,
        "train_families": len(train_examples),
        "validation_families": len(eval_examples),
        "initial_background": initial,
        "trained_background": warmup_background,
        "converted_topk": converted_topk,
        "adapted_topk": adapted_topk,
    }
    pd.DataFrame(warmup_history).to_csv(
        args.output / "warmup_history.csv", index=False
    )
    pd.DataFrame(sparse_history).to_csv(
        args.output / "sparse_history.csv", index=False
    )
    converted_frame.to_csv(
        args.output / "converted_topk_per_family.csv", index=False
    )
    adapted_frame.to_csv(args.output / "adapted_topk_per_family.csv", index=False)
    torch.save(model.state_dict(), args.output / "model.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
