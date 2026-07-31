from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
    TransformerProteinLM,
)
from transformer_disentanglement.metrics import binary_contact_precision
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import (
    connected_pair_blocks,
    load_seqmodels_family,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["sparse", "local", "transformer"], required=True
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--index-dim", type=int, default=16)
    parser.add_argument("--pair-dim", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument(
        "--routing-mode", choices=["topk", "soft"], default="topk"
    )
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--index-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--background-weight", type=float, default=0.2)
    parser.add_argument("--index-weight", type=float, default=0.1)
    parser.add_argument("--ranking-weight", type=float, default=0.05)
    parser.add_argument("--teacher-pairs", type=int, default=128)
    parser.add_argument("--eval-role", default="validation")
    parser.add_argument("--eval-families", type=int, default=57)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def make_model(args: argparse.Namespace) -> torch.nn.Module:
    common = {
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
    }
    if args.model == "sparse":
        return DisentangledProteinLM(
            **common,
            rank=args.rank,
            index_dim=args.index_dim,
            pair_dim=args.pair_dim,
            neighbors=args.neighbors,
            routing_mode=args.routing_mode,
            soft_temperature=args.soft_temperature,
        )
    if args.model == "local":
        return LocalProteinLM(**common)
    return TransformerProteinLM(**common, heads=4)


def mask_sequence(
    sequence: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.flatnonzero(sequence < 20)
    count = max(1, int(round(len(valid) * fraction)))
    selected = rng.choice(valid, size=min(count, len(valid)), replace=False)
    masked = sequence.astype(np.int64).copy()
    masked[selected] = 21
    return masked, selected.astype(np.int64)


def current_memory(device: torch.device) -> int:
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def evaluate(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    benchmark: Path,
    mask_fraction: float,
    seed: int,
    device: torch.device,
    model_kind: str,
) -> dict[str, float]:
    losses = []
    background_losses = []
    accuracies = []
    interaction_rms = []
    routing_entropy = []
    contact_precision = []
    token_count = 0
    elapsed = 0.0
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
            start = time.perf_counter()
            output = model(tokens)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed += time.perf_counter() - start
            logits = output["logits"][0, selected]
            background = output["background_logits"][0, selected]
            losses.append(float(F.cross_entropy(logits, targets).cpu()))
            background_losses.append(
                float(F.cross_entropy(background, targets).cpu())
            )
            accuracies.append(
                float((logits.argmax(dim=-1) == targets).float().mean().cpu())
            )
            token_count += len(family.query)
            if model_kind == "sparse":
                layer = output["layers"][-1]
                interaction_rms.append(
                    float(
                        torch.sqrt(
                            layer["interaction_logits"][0, selected]
                            .square()
                            .mean()
                        ).cpu()
                    )
                )
                weights = layer.get("routing_weights")
                if weights is not None:
                    entropy_value = -torch.sum(
                        weights * torch.log(weights.clamp_min(1e-8)), dim=-1
                    )
                    routing_entropy.append(float(entropy_value.mean().cpu()))
                scores = layer["index_scores"][0].cpu().numpy()
                contact_precision.append(
                    binary_contact_precision(
                        scores,
                        family.contacts,
                        family.contact_mask,
                        min_separation=24,
                    )
                )
    return {
        "families": int(len(frame)),
        "masked_cross_entropy": float(np.mean(losses)),
        "masked_perplexity": float(np.exp(np.mean(losses))),
        "background_cross_entropy": float(np.mean(background_losses)),
        "masked_accuracy": float(np.mean(accuracies)),
        "interaction_rms": float(np.mean(interaction_rms))
        if interaction_rms
        else 0.0,
        "routing_entropy": float(np.mean(routing_entropy))
        if routing_entropy
        else 0.0,
        "index_contact_p_at_l": float(np.mean(contact_precision))
        if contact_precision
        else float("nan"),
        "forward_tokens_per_second": float(token_count / max(elapsed, 1e-8)),
    }


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
    eval_frame = frame[frame["role"] == args.eval_role].head(
        args.eval_families
    )
    train_families = [
        load_seqmodels_family(args.benchmark, row.file, row.x_id)
        for row in train_frame.itertuples(index=False)
    ]
    use_index_teacher = args.model == "sparse" and (
        args.index_weight > 0.0 or args.ranking_weight > 0.0
    )
    if use_index_teacher:
        null_models, _ = fit_family_null_models(
            train_frame,
            args.benchmark,
            args.representations,
            512,
            args.seed,
            3.0,
        )
    else:
        null_models = None

    model = make_model(args).to(device)
    if args.model == "sparse":
        index_parameters = []
        for layer in model.layers:
            index_parameters.extend(layer.index_norm.parameters())
            index_parameters.extend(layer.index_left.parameters())
            index_parameters.extend(layer.index_right.parameters())
            index_parameters.append(layer.index_bias)
        index_parameter_ids = {id(parameter) for parameter in index_parameters}
        base_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in index_parameter_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": base_parameters, "lr": args.learning_rate},
                {"params": index_parameters, "lr": args.index_learning_rate},
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
    history = []
    start_time = time.perf_counter()
    processed_tokens = 0
    peak_memory = current_memory(device)
    model.train()
    for step in range(1, args.steps + 1):
        family_index = int(rng.integers(len(train_families)))
        family = train_families[family_index]
        row = train_frame.iloc[family_index]
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
        index_loss = mlm_loss.new_zeros(())
        ranking_loss = mlm_loss.new_zeros(())
        if use_index_teacher:
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
            target_score = torch.from_numpy(
                np.clip(teacher_score, -3.0, 10.0)
            ).to(device)
            pair_tensor = torch.from_numpy(pairs).to(device)
            layer_scores = output["layers"][-1]["index_scores"][0]
            predicted_score = layer_scores[
                pair_tensor[:, 0], pair_tensor[:, 1]
            ]
            index_loss = F.smooth_l1_loss(predicted_score, target_score)
            ranking_loss = pairwise_ranking_loss(
                predicted_score, target_score
            )
        loss = (
            mlm_loss
            + args.background_weight * background_loss
            + args.index_weight * index_loss
            + args.ranking_weight * ranking_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        processed_tokens += len(sequence)
        peak_memory = max(peak_memory, current_memory(device))
        if step == 1 or step % args.log_every == 0:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "mlm_loss": float(mlm_loss.detach().cpu()),
                    "background_loss": float(background_loss.detach().cpu()),
                    "index_loss": float(index_loss.detach().cpu()),
                    "ranking_loss": float(ranking_loss.detach().cpu()),
                }
            )

    if device.type == "mps":
        torch.mps.synchronize()
    train_elapsed = time.perf_counter() - start_time
    evaluation = evaluate(
        model,
        eval_frame,
        args.benchmark,
        args.mask_fraction,
        args.seed,
        device,
        args.model,
    )
    summary = {
        "model": args.model,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_steps": args.steps,
        "train_tokens_per_second": processed_tokens / max(train_elapsed, 1e-8),
        "peak_device_memory_bytes": peak_memory,
        "evaluation": evaluation,
    }
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    torch.save(model.state_dict(), args.output / "model.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(vars(args) | {"device": str(device)}, default=str, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
