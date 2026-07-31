from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.train_end_to_end_demo import evaluate, mask_sequence
from experiments.train_warmstarted_sparse_demo import load_local_background
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
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--index-dim", type=int, default=16)
    parser.add_argument("--pair-dim", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--soft-steps", type=int, default=300)
    parser.add_argument("--sparse-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--soft-temperature", type=float, default=0.5)
    parser.add_argument("--task-gradient-weight", type=float, default=0.5)
    parser.add_argument("--task-gradient-rms", type=float, default=0.1)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--eval-families", type=int, default=57)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def set_typed_trainable(
    model: DisentangledProteinLM,
    train_index: bool,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for layer in model.layers:
        typed_modules = [
            layer.factor_norm,
            layer.factor_projection,
            layer.pair_norm,
            layer.pair_projection,
            layer.pair_decoder,
        ]
        if train_index:
            typed_modules.extend(
                [layer.index_norm, layer.index_left, layer.index_right]
            )
            layer.index_bias.requires_grad_(True)
            trainable.append(layer.index_bias)
        for module in typed_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                trainable.append(parameter)
    return trainable


def train_phase(
    model: DisentangledProteinLM,
    families,
    steps: int,
    learning_rate: float,
    gradient_weight: float,
    gradient_rms: float,
    mask_fraction: float,
    rng: np.random.Generator,
    device: torch.device,
    log_every: int,
    train_index: bool,
) -> tuple[list[dict[str, float | int]], float]:
    trainable = set_typed_trainable(model, train_index=train_index)
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=1e-4
    )
    history = []
    start = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        family = families[int(rng.integers(len(families)))]
        sequence = family.msa[int(rng.integers(len(family.msa)))]
        masked, selected = mask_sequence(sequence, mask_fraction, rng)
        tokens = torch.from_numpy(masked)[None].to(device)
        targets = torch.from_numpy(sequence[selected].astype(np.int64)).to(device)
        output = model(tokens)
        logits = output["logits"][0, selected]
        background = output["background_logits"][0, selected]
        interaction = output["layers"][-1]["interaction_logits"][0, selected]
        mlm_loss = F.cross_entropy(logits, targets)
        target_field = normalized_task_gradient_target(
            background, targets, target_rms=gradient_rms
        )
        task_gradient_loss = F.smooth_l1_loss(interaction, target_field)
        loss = mlm_loss + gradient_weight * task_gradient_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % log_every == 0:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "mlm_loss": float(mlm_loss.detach().cpu()),
                    "task_gradient_loss": float(
                        task_gradient_loss.detach().cpu()
                    ),
                    "interaction_rms": float(
                        torch.sqrt(interaction.square().mean()).detach().cpu()
                    ),
                }
            )
    if device.type == "mps":
        torch.mps.synchronize()
    return history, time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    frame = pd.read_csv(args.representations / "families.csv")
    train_frame = frame[frame["role"] == "train"]
    eval_frame = frame[frame["role"] == "validation"].head(args.eval_families)
    families = [
        load_seqmodels_family(args.benchmark, row.file, row.x_id)
        for row in train_frame.itertuples(index=False)
    ]
    model = DisentangledProteinLM(
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        index_dim=args.index_dim,
        pair_dim=args.pair_dim,
        neighbors=args.neighbors,
        layers=1,
        routing_mode="soft",
        soft_temperature=args.soft_temperature,
    ).to(device)
    load_local_background(model, args.local_checkpoint, args.hidden_dim, device)
    initial = evaluate(
        model,
        eval_frame,
        args.benchmark,
        args.mask_fraction,
        args.seed,
        device,
        "sparse",
    )
    soft_history, soft_elapsed = train_phase(
        model,
        families,
        args.soft_steps,
        args.learning_rate,
        args.task_gradient_weight,
        args.task_gradient_rms,
        args.mask_fraction,
        rng,
        device,
        args.log_every,
        train_index=True,
    )
    soft = evaluate(
        model,
        eval_frame,
        args.benchmark,
        args.mask_fraction,
        args.seed,
        device,
        "sparse",
    )
    model.set_routing_mode("topk")
    converted = evaluate(
        model,
        eval_frame,
        args.benchmark,
        args.mask_fraction,
        args.seed,
        device,
        "sparse",
    )
    sparse_history, sparse_elapsed = train_phase(
        model,
        families,
        args.sparse_steps,
        args.learning_rate,
        args.task_gradient_weight,
        args.task_gradient_rms,
        args.mask_fraction,
        rng,
        device,
        args.log_every,
        train_index=False,
    )
    adapted = evaluate(
        model,
        eval_frame,
        args.benchmark,
        args.mask_fraction,
        args.seed,
        device,
        "sparse",
    )
    summary = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "soft_steps": args.soft_steps,
        "sparse_steps": args.sparse_steps,
        "soft_elapsed_seconds": soft_elapsed,
        "sparse_elapsed_seconds": sparse_elapsed,
        "initial": initial,
        "trained_soft": soft,
        "converted_topk": converted,
        "adapted_topk": adapted,
    }
    pd.DataFrame(soft_history).to_csv(
        args.output / "soft_history.csv", index=False
    )
    pd.DataFrame(sparse_history).to_csv(
        args.output / "sparse_history.csv", index=False
    )
    torch.save(model.state_dict(), args.output / "model.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(vars(args) | {"device": str(device)}, default=str, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
