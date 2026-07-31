from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.train_dual_stream_epistasis_demo import load_local_initialization
from experiments.train_epistasis_identifiability import (
    load_frozen_models,
    teacher_probe_losses,
)
from transformer_disentanglement.demo_language_models import (
    DualStreamProteinLM,
    TransformerProteinLM,
)
from transformer_disentanglement.epistasis import (
    robust_standardize,
    weighted_double_center,
)
from transformer_disentanglement.metrics import binary_average_precision, safe_spearman
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import load_seqmodels_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--consensus-teacher-checkpoint", type=Path)
    parser.add_argument("--background-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stable-dim", type=int, default=64)
    parser.add_argument("--task-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--rank-mode", choices=["fixed", "adaptive"], default="fixed"
    )
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--index-dim", type=int, default=16)
    parser.add_argument("--pair-dim", type=int, default=16)
    parser.add_argument("--pair-mlp-dim", type=int)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--train-families", type=int, default=64)
    parser.add_argument("--eval-families", type=int, default=24)
    parser.add_argument("--targets-per-family", type=int, default=4)
    parser.add_argument("--contexts-per-target", type=int, default=16)
    parser.add_argument("--min-separation", type=int, default=6)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--sparse-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--index-weight", type=float, default=0.2)
    parser.add_argument("--shape-weight", type=float, default=0.2)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def masked_target_sequence(
    sequence: np.ndarray, target: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one masked example without mutating the NumPy query sequence."""
    base = torch.tensor(sequence, dtype=torch.long, device=device)
    true_target = base[target].clone()
    base[target] = 21
    return base, true_target


@torch.no_grad()
def build_target_examples(
    frame: pd.DataFrame,
    count: int,
    args: argparse.Namespace,
    teacher: torch.nn.Module,
    background: torch.nn.Module,
    rng: np.random.Generator,
    device: torch.device,
    consensus_teachers: list[torch.nn.Module] | None = None,
) -> list[dict[str, torch.Tensor | str]]:
    examples = []
    amino_acids = torch.arange(20, device=device)
    for row in frame.head(count).itertuples(index=False):
        family = load_seqmodels_family(args.benchmark, row.file, row.x_id)
        sequence = family.msa[0].astype(np.int64)
        valid = np.flatnonzero(sequence < 20)
        targets = rng.choice(
            valid,
            size=min(args.targets_per_family, len(valid)),
            replace=False,
        )
        for target in targets:
            candidates = valid[np.abs(valid - target) >= args.min_separation]
            contexts = rng.choice(
                candidates,
                size=min(args.contexts_per_target, len(candidates)),
                replace=False,
            ).astype(np.int64)
            base, true_target = masked_target_sequence(
                sequence, int(target), device
            )
            base_tokens = base[None]
            all_teachers = [teacher] + (consensus_teachers or [])
            base_teacher_logits = [
                current_teacher(base_tokens)["logits"][0, target]
                for current_teacher in all_teachers
            ]
            teacher_probabilities = torch.stack(
                [logits.softmax(dim=-1) for logits in base_teacher_logits]
            ).mean(dim=0)
            teacher_logits = teacher_probabilities.clamp_min(1e-8).log()
            background_output = background(base_tokens)
            probabilities = background_output["logits"][0].softmax(dim=-1)

            variants = base_tokens.repeat(len(contexts) * 20, 1)
            variant_rows = torch.arange(len(variants), device=device)
            context_tensor = torch.from_numpy(contexts).to(device)
            variants[
                variant_rows,
                context_tensor.repeat_interleave(20),
            ] = amino_acids.repeat(len(contexts))
            teacher_variant_logits = []
            for current_teacher in all_teachers:
                variant_logits = []
                for start in range(0, len(variants), args.teacher_batch_size):
                    output = current_teacher(
                        variants[start : start + args.teacher_batch_size]
                    )
                    variant_logits.append(output["logits"][:, target])
                teacher_variant_logits.append(torch.cat(variant_logits))
            consensus_variant_probabilities = torch.stack(
                [logits.softmax(dim=-1) for logits in teacher_variant_logits]
            ).mean(dim=0)
            consensus_variant_logits = consensus_variant_probabilities.clamp_min(
                1e-8
            ).log()
            logits_by_context = consensus_variant_logits.reshape(
                len(contexts), 20, 20
            )
            blocks = logits_by_context.transpose(1, 2)
            left_probabilities = probabilities[target][None].expand(len(contexts), -1)
            right_probabilities = probabilities[context_tensor]
            blocks = weighted_double_center(
                blocks, left_probabilities, right_probabilities
            )
            shape_scale = blocks.square().mean().sqrt().clamp_min(1e-8)
            shape_target = (blocks / shape_scale).contiguous()
            strength = blocks.square().mean(dim=(-2, -1)).sqrt().clamp_min(1e-10)
            index_target, _, _ = robust_standardize(torch.log(strength))
            pairs = torch.stack(
                [
                    torch.full_like(context_tensor, int(target)),
                    context_tensor,
                ],
                dim=-1,
            )
            examples.append(
                {
                    "identifier": f"{row.x_id}:{int(target)}",
                    "base_tokens": base.cpu(),
                    "target_position": torch.tensor(int(target)),
                    "true_target": true_target.cpu(),
                    "teacher_logits": teacher_logits.cpu(),
                    "pairs": pairs.cpu(),
                    "context_tokens": torch.from_numpy(sequence[contexts]).long(),
                    "index_target": index_target.cpu(),
                    "shape_target": shape_target.cpu(),
                }
            )
        print(f"built {row.x_id}: {len(examples)} target examples", flush=True)
    return examples


def freeze_background(model: DualStreamProteinLM) -> None:
    modules = [
        model.stable_embedding,
        model.position_embedding,
        model.stable_norm,
        model.stable_depthwise,
        model.stable_pointwise,
        model.interaction.background_norm,
        model.interaction.background_projection,
    ]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def top_metrics(scores: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    count = max(1, int(round(0.1 * len(target))))
    truth = np.argsort(target)[-count:]
    predicted = np.argsort(scores)[-count:]
    labels = np.zeros(len(target), dtype=bool)
    labels[truth] = True
    return (
        binary_average_precision(scores, labels),
        len(np.intersect1d(truth, predicted)) / count,
    )


def sampled_outputs(
    model: DualStreamProteinLM,
    output: dict[str, torch.Tensor | None],
    pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pair_batch = pairs[None]
    scores = model.interaction.sampled_index_scores(
        output["encoded_task"], pair_batch  # type: ignore[arg-type]
    )[0]
    blocks, gates, effective_rank = model.interaction.sampled_pair_outputs(
        output["factors"],  # type: ignore[arg-type]
        output["value_state"],  # type: ignore[arg-type]
        output["encoded_task"],  # type: ignore[arg-type]
        pair_batch,
        output["marginal_probabilities"],  # type: ignore[arg-type]
    )
    scores = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-6)
    blocks = blocks[0]
    blocks = blocks / blocks.square().mean().sqrt().clamp_min(1e-6)
    return scores, blocks, gates[0], effective_rank[0]


def candidate_message(
    model: DualStreamProteinLM,
    scores: torch.Tensor,
    blocks: torch.Tensor,
    context_tokens: torch.Tensor,
) -> torch.Tensor:
    count = min(model.interaction.neighbors, len(scores))
    values, indices = torch.topk(scores, k=count)
    weights = torch.softmax(values / model.interaction.routing_temperature, dim=-1)
    selected_blocks = blocks[indices]
    selected_tokens = context_tokens[indices]
    messages = selected_blocks[
        torch.arange(count, device=blocks.device), :, selected_tokens
    ]
    return model.interaction.interaction_scale * torch.sum(
        weights[:, None] * messages, dim=0
    )


def example_objective(
    model: DualStreamProteinLM,
    example: dict[str, torch.Tensor | str],
    args: argparse.Namespace,
    device: torch.device,
    use_message: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
    target_position = int(example["target_position"])
    true_target = example["true_target"].to(device)  # type: ignore[union-attr]
    pairs = example["pairs"].to(device)  # type: ignore[union-attr]
    context_tokens = example["context_tokens"].to(device)  # type: ignore[union-attr]
    index_target = example["index_target"].to(device)  # type: ignore[union-attr]
    shape_target = example["shape_target"].to(device)  # type: ignore[union-attr]
    teacher_logits = example["teacher_logits"].to(device)  # type: ignore[union-attr]
    output = model(tokens, use_interaction=False)
    background_logits = output["background_logits"][0, target_position]  # type: ignore[index]
    scores, blocks, gates, effective_rank = sampled_outputs(
        model, output, pairs
    )
    index_loss = F.smooth_l1_loss(scores, index_target)
    shape_loss = F.smooth_l1_loss(blocks.contiguous(), shape_target.contiguous())
    rank_loss = effective_rank.mean() / model.interaction.rank
    logits = background_logits
    if use_message:
        logits = logits + candidate_message(model, scores, blocks, context_tokens)
    teacher_probabilities = teacher_logits.softmax(dim=-1)
    distill_loss = F.kl_div(
        logits.log_softmax(dim=-1), teacher_probabilities, reduction="sum"
    )
    ce_loss = F.cross_entropy(logits[None], true_target[None])
    loss = (
        args.index_weight * index_loss
        + args.shape_weight * shape_loss
        + args.rank_weight * rank_loss
    )
    if use_message:
        loss = loss + args.distill_weight * distill_loss + args.ce_weight * ce_loss
    return loss, {
        "index_loss": float(index_loss.detach().cpu()),
        "shape_loss": float(shape_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "effective_rank": float(effective_rank.mean().detach().cpu()),
        "active_modes": float((gates >= 0.5).sum(dim=-1).float().mean().cpu()),
        "distill_loss": float(distill_loss.detach().cpu()),
        "ce_loss": float(ce_loss.detach().cpu()),
    }


def train_phase(
    model: DualStreamProteinLM,
    examples: list[dict[str, torch.Tensor | str]],
    steps: int,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    use_message: bool,
) -> list[dict[str, float | int]]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    history = []
    model.train()
    for step in range(1, steps + 1):
        example = examples[int(rng.integers(len(examples)))]
        loss, metrics = example_objective(model, example, args, device, use_message)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == steps:
            history.append(
                {"step": step, "loss": float(loss.detach().cpu())} | metrics
            )
    return history


@torch.no_grad()
def evaluate(
    model: DualStreamProteinLM,
    examples: list[dict[str, torch.Tensor | str]],
    device: torch.device,
    use_message: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    model.eval()
    for example in examples:
        tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
        target_position = int(example["target_position"])
        true_target = example["true_target"].to(device)  # type: ignore[union-attr]
        pairs = example["pairs"].to(device)  # type: ignore[union-attr]
        context_tokens = example["context_tokens"].to(device)  # type: ignore[union-attr]
        index_target = example["index_target"].to(device)  # type: ignore[union-attr]
        shape_target = example["shape_target"].to(device)  # type: ignore[union-attr]
        teacher_logits = example["teacher_logits"].to(device)  # type: ignore[union-attr]
        output = model(tokens, use_interaction=False)
        background = output["background_logits"][0, target_position]  # type: ignore[index]
        scores, blocks, gates, effective_rank = sampled_outputs(
            model, output, pairs
        )
        logits = background
        if use_message:
            logits = logits + candidate_message(model, scores, blocks, context_tokens)
        score_np = scores.cpu().numpy()
        target_np = index_target.cpu().numpy()
        average_precision, recall = top_metrics(score_np, target_np)
        teacher_probabilities = teacher_logits.softmax(dim=-1)
        rows.append(
            {
                "example": example["identifier"],
                "cross_entropy": float(F.cross_entropy(logits[None], true_target[None]).cpu()),
                "background_cross_entropy": float(
                    F.cross_entropy(background[None], true_target[None]).cpu()
                ),
                "teacher_cross_entropy": float(
                    F.cross_entropy(teacher_logits[None], true_target[None]).cpu()
                ),
                "teacher_kl": float(
                    F.kl_div(
                        logits.log_softmax(dim=-1),
                        teacher_probabilities,
                        reduction="sum",
                    ).cpu()
                ),
                "background_teacher_kl": float(
                    F.kl_div(
                        background.log_softmax(dim=-1),
                        teacher_probabilities,
                        reduction="sum",
                    ).cpu()
                ),
                "index_spearman": safe_spearman(score_np, target_np),
                "index_top_ap": average_precision,
                "index_top_recall": recall,
                "shape_mse": float(
                    F.mse_loss(blocks.contiguous(), shape_target.contiguous()).cpu()
                ),
                "shape_correlation": safe_spearman(
                    blocks.cpu().numpy().ravel(), shape_target.cpu().numpy().ravel()
                ),
                "interaction_rms": float((logits - background).square().mean().sqrt().cpu()),
                "effective_rank": float(effective_rank.mean().cpu()),
                "active_modes": float(
                    (gates >= 0.5).sum(dim=-1).float().mean().cpu()
                ),
                "gate_mean": float(gates.mean().cpu()),
            }
            | (
                {
                    "marginal_kl_to_msa": float(
                        F.kl_div(
                            output["marginal_probabilities"][0, target_position]
                            .clamp_min(1e-8)
                            .log(),
                            example["msa_target_probabilities"].to(device),  # type: ignore[union-attr]
                            reduction="sum",
                        ).cpu()
                    )
                }
                if "msa_target_probabilities" in example
                else {}
            )
        )
    frame = pd.DataFrame(rows)
    return {
        column: float(frame[column].mean())
        for column in frame.columns
        if column != "example"
    }, frame


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    args.hidden_dim = args.stable_dim
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    teacher, background = load_frozen_models(args, device)
    consensus_teachers = []
    if args.consensus_teacher_checkpoint is not None:
        consensus_teacher = TransformerProteinLM(
            hidden_dim=args.hidden_dim, layers=1, heads=4
        ).to(device)
        consensus_teacher.load_state_dict(
            torch.load(
                args.consensus_teacher_checkpoint,
                map_location=device,
                weights_only=True,
            )
        )
        consensus_teacher.eval()
        for parameter in consensus_teacher.parameters():
            parameter.requires_grad_(False)
        consensus_teachers.append(consensus_teacher)
    frame = pd.read_csv(args.representations / "families.csv")
    train_frame = frame[frame.role == "train"].reset_index(drop=True)
    eval_frame = frame[frame.role == "validation"].reset_index(drop=True)
    train_examples = build_target_examples(
        train_frame,
        args.train_families,
        args,
        teacher,
        background,
        np.random.default_rng(args.seed + 1000),
        device,
        consensus_teachers,
    )
    eval_examples = build_target_examples(
        eval_frame,
        args.eval_families,
        args,
        teacher,
        background,
        np.random.default_rng(args.seed + 3000),
        device,
        consensus_teachers,
    )
    del teacher, background, consensus_teachers
    model = DualStreamProteinLM(
        stable_dim=args.stable_dim,
        task_dim=args.task_dim,
        rank=args.rank,
        index_dim=args.index_dim,
        pair_dim=args.pair_dim,
        pair_mlp_dim=args.pair_mlp_dim,
        neighbors=args.neighbors,
        routing_mode="topk",
        rank_mode=args.rank_mode,
        gate_temperature=args.gate_temperature,
    ).to(device)
    load_local_initialization(model, args.background_checkpoint, device)
    freeze_background(model)
    initial, _ = evaluate(model, eval_examples, device, use_message=False)
    rng = np.random.default_rng(args.seed + 4000)
    warmup_history = train_phase(
        model, train_examples, args.warmup_steps, args, device, rng, False
    )
    warmup, _ = evaluate(model, eval_examples, device, use_message=False)
    converted, converted_frame = evaluate(model, eval_examples, device, use_message=True)
    sparse_history = train_phase(
        model, train_examples, args.sparse_steps, args, device, rng, True
    )
    adapted, adapted_frame = evaluate(model, eval_examples, device, use_message=True)
    summary = {
        "seed": args.seed,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "rank": args.rank,
        "rank_mode": args.rank_mode,
        "rank_weight": args.rank_weight,
        "pair_mlp_dim": model.interaction.mode_decoder[0].out_features,
        "train_examples": len(train_examples),
        "validation_examples": len(eval_examples),
        "initial": initial,
        "warmup": warmup,
        "converted_topk": converted,
        "adapted_topk": adapted,
    }
    pd.DataFrame(warmup_history).to_csv(args.output / "warmup_history.csv", index=False)
    pd.DataFrame(sparse_history).to_csv(args.output / "sparse_history.csv", index=False)
    converted_frame.to_csv(args.output / "converted_topk_per_example.csv", index=False)
    adapted_frame.to_csv(args.output / "adapted_topk_per_example.csv", index=False)
    torch.save(model.state_dict(), args.output / "model.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
