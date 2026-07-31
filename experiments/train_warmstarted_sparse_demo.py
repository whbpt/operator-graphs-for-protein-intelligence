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
from experiments.train_seqmodels_factor_head import (
    fit_family_null_models,
    load_representation,
    sample_pairs,
)
from experiments.train_seqmodels_gate_baseline import (
    pairwise_ranking_loss,
    robust_teacher_score,
)
from transformer_disentanglement.demo_language_models import (
    DisentangledProteinLM,
    LocalProteinLM,
)
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import (
    connected_pair_blocks,
    load_seqmodels_family,
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
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--adaptation-steps", type=int, default=500)
    parser.add_argument("--index-learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--teacher-pairs", type=int, default=128)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--background-weight", type=float, default=0.2)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--eval-families", type=int, default=57)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_local_background(
    sparse: DisentangledProteinLM,
    checkpoint: Path,
    hidden_dim: int,
    device: torch.device,
) -> None:
    local = LocalProteinLM(hidden_dim=hidden_dim, layers=1).to(device)
    local.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    sparse.token_embedding.load_state_dict(local.token_embedding.state_dict())
    sparse.position_embedding.load_state_dict(local.position_embedding.state_dict())
    layer = sparse.layers[0]
    layer.local_norm.load_state_dict(local.norms[0].state_dict())
    layer.local_depthwise.load_state_dict(local.depthwise[0].state_dict())
    layer.local_pointwise.load_state_dict(local.pointwise[0].state_dict())
    layer.background_norm.load_state_dict(local.output_norm.state_dict())
    layer.background_projection.load_state_dict(local.output.state_dict())


def set_index_trainable(model: DisentangledProteinLM, trainable: bool) -> list:
    for parameter in model.parameters():
        parameter.requires_grad_(not trainable)
    index_parameters = []
    for layer in model.layers:
        modules = [layer.index_norm, layer.index_left, layer.index_right]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
                index_parameters.append(parameter)
        layer.index_bias.requires_grad_(trainable)
        index_parameters.append(layer.index_bias)
    return index_parameters


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    teacher_rng = np.random.default_rng(args.seed + 1)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    frame = pd.read_csv(args.representations / "families.csv")
    train_frame = frame[frame["role"] == "train"].reset_index(drop=True)
    eval_frame = frame[frame["role"] == "validation"].head(args.eval_families)
    train_families = [
        load_seqmodels_family(args.benchmark, row.file, row.x_id)
        for row in train_frame.itertuples(index=False)
    ]
    null_models, _ = fit_family_null_models(
        train_frame,
        args.benchmark,
        args.representations,
        512,
        args.seed,
        3.0,
    )
    model = DisentangledProteinLM(
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        index_dim=args.index_dim,
        pair_dim=args.pair_dim,
        neighbors=args.neighbors,
        layers=1,
    ).to(device)
    load_local_background(
        model, args.local_checkpoint, args.hidden_dim, device
    )

    warmup_history = []
    index_parameters = set_index_trainable(model, True)
    optimizer = torch.optim.AdamW(
        index_parameters, lr=args.index_learning_rate, weight_decay=1e-4
    )
    model.train()
    for step in range(1, args.warmup_steps + 1):
        family_index = int(rng.integers(len(train_families)))
        family = train_families[family_index]
        row = train_frame.iloc[family_index]
        sequence_index = int(rng.integers(len(family.msa)))
        sequence = family.msa[sequence_index]
        masked, _ = mask_sequence(sequence, args.mask_fraction, rng)
        tokens = torch.from_numpy(masked)[None].to(device)
        output = model(tokens)
        pairs = sample_pairs(
            len(family.query), args.teacher_pairs, teacher_rng
        )
        _, pssm = load_representation(args.representations, row)
        blocks = connected_pair_blocks(
            family.msa, family.weights, pairs, pssm=pssm
        )
        teacher_score = robust_teacher_score(
            blocks, pairs, pssm, null_models[row.x_id]
        )
        target = torch.from_numpy(np.clip(teacher_score, -3.0, 10.0)).to(device)
        pair_tensor = torch.from_numpy(pairs).to(device)
        score_map = output["layers"][-1]["index_scores"][0]
        predicted = score_map[pair_tensor[:, 0], pair_tensor[:, 1]]
        regression_loss = F.smooth_l1_loss(predicted, target)
        ranking_loss = pairwise_ranking_loss(predicted, target)
        loss = regression_loss + args.ranking_weight * ranking_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(index_parameters, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            warmup_history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "regression_loss": float(regression_loss.detach().cpu()),
                    "ranking_loss": float(ranking_loss.detach().cpu()),
                }
            )

    pre_adaptation = evaluate(
        model,
        eval_frame,
        args.benchmark,
        args.mask_fraction,
        args.seed,
        device,
        "sparse",
    )

    set_index_trainable(model, False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=1e-4
    )
    adaptation_history = []
    processed_tokens = 0
    peak_memory = current_memory(device)
    start_time = time.perf_counter()
    for step in range(1, args.adaptation_steps + 1):
        family_index = int(rng.integers(len(train_families)))
        family = train_families[family_index]
        sequence_index = int(rng.integers(len(family.msa)))
        sequence = family.msa[sequence_index]
        masked, selected = mask_sequence(sequence, args.mask_fraction, rng)
        tokens = torch.from_numpy(masked)[None].to(device)
        targets = torch.from_numpy(sequence[selected].astype(np.int64)).to(device)
        output = model(tokens)
        logits = output["logits"][0, selected]
        background = output["background_logits"][0, selected]
        mlm_loss = F.cross_entropy(logits, targets)
        background_loss = F.cross_entropy(background, targets)
        loss = mlm_loss + args.background_weight * background_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        processed_tokens += len(sequence)
        peak_memory = max(peak_memory, current_memory(device))
        if step == 1 or step % args.log_every == 0:
            adaptation_history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "mlm_loss": float(mlm_loss.detach().cpu()),
                    "background_loss": float(background_loss.detach().cpu()),
                }
            )

    if device.type == "mps":
        torch.mps.synchronize()
    adaptation_elapsed = time.perf_counter() - start_time
    post_adaptation = evaluate(
        model,
        eval_frame,
        args.benchmark,
        args.mask_fraction,
        args.seed,
        device,
        "sparse",
    )
    summary = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "warmup_steps": args.warmup_steps,
        "adaptation_steps": args.adaptation_steps,
        "adaptation_tokens_per_second": processed_tokens
        / max(adaptation_elapsed, 1e-8),
        "peak_device_memory_bytes": peak_memory,
        "pre_adaptation": pre_adaptation,
        "post_adaptation": post_adaptation,
    }
    pd.DataFrame(warmup_history).to_csv(
        args.output / "index_warmup_history.csv", index=False
    )
    pd.DataFrame(adaptation_history).to_csv(
        args.output / "adaptation_history.csv", index=False
    )
    torch.save(model.state_dict(), args.output / "model.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(vars(args) | {"device": str(device)}, default=str, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
