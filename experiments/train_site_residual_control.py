from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.train_end_to_end_demo import mask_sequence
from transformer_disentanglement.demo_language_models import (
    MarginalOrthogonalResidualLM,
)
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
    parser.add_argument("--residual-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--task-gradient-weight", type=float, default=0.5)
    parser.add_argument("--task-gradient-rms", type=float, default=0.1)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--eval-families", type=int, default=57)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def evaluate(
    model: MarginalOrthogonalResidualLM,
    frame: pd.DataFrame,
    benchmark: Path,
    seed: int,
    mask_fraction: float,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for row in frame.itertuples(index=False):
            family = load_seqmodels_family(benchmark, row.file, row.x_id)
            rng = np.random.default_rng(seed + int(row.index) * 104729)
            masked, selected = mask_sequence(
                family.msa[0], mask_fraction, rng
            )
            tokens = torch.from_numpy(masked)[None].to(device)
            targets = torch.from_numpy(
                family.msa[0, selected].astype(np.int64)
            ).to(device)
            output = model(tokens)
            logits = output["logits"][0, selected]
            background = output["background_logits"][0, selected]
            residual = output["residual_logits"][0, selected]
            target_field = normalized_task_gradient_target(
                background, targets, target_rms=0.1
            )
            total_loss = float(F.cross_entropy(logits, targets).cpu())
            background_loss = float(F.cross_entropy(background, targets).cpu())
            rows.append(
                {
                    "x_id": row.x_id,
                    "total_cross_entropy": total_loss,
                    "background_cross_entropy": background_loss,
                    "residual_contribution": total_loss - background_loss,
                    "task_gradient_cosine": float(
                        F.cosine_similarity(
                            residual, target_field, dim=-1, eps=1e-8
                        )
                        .mean()
                        .cpu()
                    ),
                    "residual_rms": float(
                        torch.sqrt(residual.square().mean()).cpu()
                    ),
                }
            )
    return pd.DataFrame(rows)


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
    model = MarginalOrthogonalResidualLM(
        hidden_dim=args.hidden_dim,
        residual_dim=args.residual_dim,
    ).to(device)
    model.background.load_state_dict(
        torch.load(
            args.local_checkpoint, map_location=device, weights_only=True
        )
    )
    for parameter in model.background.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=1e-4
    )
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        family = families[int(rng.integers(len(families)))]
        sequence = family.msa[int(rng.integers(len(family.msa)))]
        masked, selected = mask_sequence(sequence, args.mask_fraction, rng)
        tokens = torch.from_numpy(masked)[None].to(device)
        targets = torch.from_numpy(sequence[selected].astype(np.int64)).to(device)
        output = model(tokens)
        logits = output["logits"][0, selected]
        background = output["background_logits"][0, selected]
        residual = output["residual_logits"][0, selected]
        mlm_loss = F.cross_entropy(logits, targets)
        target_field = normalized_task_gradient_target(
            background, targets, target_rms=args.task_gradient_rms
        )
        task_gradient_loss = F.smooth_l1_loss(residual, target_field)
        loss = mlm_loss + args.task_gradient_weight * task_gradient_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "mlm_loss": float(mlm_loss.detach().cpu()),
                    "task_gradient_loss": float(
                        task_gradient_loss.detach().cpu()
                    ),
                    "residual_rms": float(
                        torch.sqrt(residual.square().mean()).detach().cpu()
                    ),
                }
            )
    evaluation = evaluate(
        model,
        eval_frame,
        args.benchmark,
        args.seed,
        args.mask_fraction,
        device,
    )
    summary = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable
        ),
        "families": len(evaluation),
        "total_cross_entropy": float(evaluation["total_cross_entropy"].mean()),
        "background_cross_entropy": float(
            evaluation["background_cross_entropy"].mean()
        ),
        "residual_contribution": float(
            evaluation["residual_contribution"].mean()
        ),
        "task_gradient_cosine": float(
            evaluation["task_gradient_cosine"].mean()
        ),
        "residual_rms": float(evaluation["residual_rms"].mean()),
    }
    evaluation.to_csv(args.output / "per_family_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(
        args.output / "training_history.csv", index=False
    )
    torch.save(model.state_dict(), args.output / "model.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(vars(args) | {"device": str(device)}, default=str, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
