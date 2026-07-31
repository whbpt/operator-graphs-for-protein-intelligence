from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.train_end_to_end_demo import (
    current_memory,
    evaluate,
    mask_sequence,
)
from experiments.train_warmstarted_sparse_demo import load_local_background
from transformer_disentanglement.demo_language_models import DisentangledProteinLM
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import load_seqmodels_family


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
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--background-weight", type=float, default=0.2)
    parser.add_argument("--eval-families", type=int, default=57)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def train_phase(
    model: DisentangledProteinLM,
    families,
    steps: int,
    learning_rate: float,
    background_weight: float,
    mask_fraction: float,
    rng: np.random.Generator,
    device: torch.device,
    log_every: int,
    freeze_index: bool,
) -> tuple[list[dict[str, float | int]], float, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if freeze_index:
        for layer in model.layers:
            for module in [layer.index_norm, layer.index_left, layer.index_right]:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            layer.index_bias.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=1e-4
    )
    history = []
    processed_tokens = 0
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
        mlm_loss = F.cross_entropy(logits, targets)
        background_loss = F.cross_entropy(background, targets)
        loss = mlm_loss + background_weight * background_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        processed_tokens += len(sequence)
        if step == 1 or step % log_every == 0:
            layer = output["layers"][-1]
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "mlm_loss": float(mlm_loss.detach().cpu()),
                    "background_loss": float(background_loss.detach().cpu()),
                    "interaction_rms": float(
                        torch.sqrt(layer["interaction_logits"].square().mean())
                        .detach()
                        .cpu()
                    ),
                }
            )
    if device.type == "mps":
        torch.mps.synchronize()
    return history, time.perf_counter() - start, processed_tokens


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
    initial_soft = evaluate(
        model,
        eval_frame,
        args.benchmark,
        args.mask_fraction,
        args.seed,
        device,
        "sparse",
    )
    soft_history, soft_elapsed, soft_tokens = train_phase(
        model,
        families,
        args.soft_steps,
        args.learning_rate,
        args.background_weight,
        args.mask_fraction,
        rng,
        device,
        args.log_every,
        freeze_index=False,
    )
    soft_evaluation = evaluate(
        model,
        eval_frame,
        args.benchmark,
        args.mask_fraction,
        args.seed,
        device,
        "sparse",
    )
    model.set_routing_mode("topk")
    converted_evaluation = evaluate(
        model,
        eval_frame,
        args.benchmark,
        args.mask_fraction,
        args.seed,
        device,
        "sparse",
    )
    sparse_history, sparse_elapsed, sparse_tokens = train_phase(
        model,
        families,
        args.sparse_steps,
        args.learning_rate,
        args.background_weight,
        args.mask_fraction,
        rng,
        device,
        args.log_every,
        freeze_index=True,
    )
    sparse_evaluation = evaluate(
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
        "soft_train_tokens_per_second": soft_tokens / max(soft_elapsed, 1e-8),
        "sparse_train_tokens_per_second": sparse_tokens
        / max(sparse_elapsed, 1e-8),
        "peak_device_memory_bytes": current_memory(device),
        "initial_soft": initial_soft,
        "trained_soft": soft_evaluation,
        "converted_topk": converted_evaluation,
        "adapted_topk": sparse_evaluation,
    }
    pd.DataFrame(soft_history).to_csv(
        args.output / "soft_training_history.csv", index=False
    )
    pd.DataFrame(sparse_history).to_csv(
        args.output / "sparse_adaptation_history.csv", index=False
    )
    torch.save(model.state_dict(), args.output / "model.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(vars(args) | {"device": str(device)}, default=str, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
