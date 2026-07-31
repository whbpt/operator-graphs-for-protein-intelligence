from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.evaluate_content_tile_variants_split import make_router
from experiments.evaluate_lsh_candidate_router import load_model, teacher_args
from experiments.train_conditional_response_demo import build_target_examples
from experiments.train_epistasis_identifiability import load_frozen_models
from transformer_disentanglement.gauge_set_aggregation import (
    MarginalOrthogonalSetAggregator,
)
from transformer_disentanglement.demo_language_models import TransformerProteinLM
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-budget", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--correction-weight", type=float, default=1e-3)
    parser.add_argument("--max-correction-ratio", type=float, default=0.5)
    parser.add_argument(
        "--teacher-reliability",
        choices=["all", "better_than_background"],
        default="all",
    )
    parser.add_argument("--consensus-teacher-checkpoint", type=Path)
    parser.add_argument("--train-families", type=int)
    parser.add_argument("--eval-families", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def routed_message_inputs(
    model: torch.nn.Module,
    router: torch.nn.Module,
    output: dict[str, torch.Tensor | None],
    tokens: torch.Tensor,
    target_position: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    task = output["encoded_task"]
    stable = output["encoded_stable"]
    exact_query, exact_key = model.interaction.index_features(task)
    valid = model.interaction.valid_pair_mask(tokens.shape[1], tokens.device)
    routed = router(
        stable,
        task,
        torch.tensor([target_position], device=tokens.device),
        exact_query[:, target_position : target_position + 1],
        exact_key,
        valid[target_position : target_position + 1],
        score_scale=model.interaction.index_dim**-0.5,
        score_bias=model.interaction.index_bias,
    )
    neighbors = routed.neighbor_indices[0, 0]
    scores = routed.neighbor_scores[0, 0]
    target = torch.full_like(neighbors, target_position)
    pairs = torch.stack([target, neighbors], dim=-1)[None]
    blocks = model.interaction.sampled_pair_blocks(
        output["factors"],
        output["value_state"],
        output["encoded_task"],
        pairs,
    )[0]
    context_tokens = tokens[0, neighbors]
    messages = blocks[
        torch.arange(len(neighbors), device=tokens.device), :, context_tokens
    ]
    messages = model.interaction.interaction_scale.detach() * messages
    probabilities = output["marginal_probabilities"][0, target_position]
    return messages.detach(), scores.detach(), probabilities.detach()


def example_outputs(
    model: torch.nn.Module,
    router: torch.nn.Module,
    aggregator: MarginalOrthogonalSetAggregator,
    example: dict[str, torch.Tensor | str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
    target_position = int(example["target_position"])
    output = model(tokens, use_interaction=False)
    messages, scores, probabilities = routed_message_inputs(
        model, router, output, tokens, target_position
    )
    aggregation = aggregator(messages, scores, probabilities)
    background = output["background_logits"][0, target_position]
    return {
        "background_logits": background,
        "additive_logits": background + aggregation["additive"],
        "nonlinear_logits": background + aggregation["interaction"],
        "probabilities": probabilities,
        "additive": aggregation["additive"],
        "correction": aggregation["correction"],
        "interaction": aggregation["interaction"],
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    router: torch.nn.Module,
    aggregator: MarginalOrthogonalSetAggregator,
    examples: list[dict[str, torch.Tensor | str]],
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for example in examples:
        result = example_outputs(model, router, aggregator, example, device)
        target = example["true_target"].to(device)  # type: ignore[union-attr]
        teacher = example["teacher_logits"].to(device)  # type: ignore[union-attr]
        teacher_probabilities = teacher.softmax(dim=-1)
        ce = {
            name: F.cross_entropy(result[f"{name}_logits"][None], target[None])
            for name in ("background", "additive", "nonlinear")
        }
        teacher_ce = F.cross_entropy(teacher[None], target[None])
        kl = {
            name: F.kl_div(
                result[f"{name}_logits"].log_softmax(dim=-1),
                teacher_probabilities,
                reduction="sum",
            )
            for name in ("background", "additive", "nonlinear")
        }
        additive_rms = result["additive"].square().mean().sqrt()
        correction_rms = result["correction"].square().mean().sqrt()
        gauge_error = torch.abs(
            torch.sum(result["probabilities"] * result["interaction"])
        )
        teacher_probabilities = teacher.softmax(dim=-1)
        background_probabilities = result["background_logits"].softmax(dim=-1)
        target_index = int(target.item())
        identifier = str(example["identifier"])
        rows.append(
            {
                "example": identifier,
                "family": identifier.split(":", 1)[0],
                "additive_ce_gain": float(ce["background"] - ce["additive"]),
                "nonlinear_ce_gain": float(ce["background"] - ce["nonlinear"]),
                "nonlinear_minus_additive_ce_gain": float(
                    ce["additive"] - ce["nonlinear"]
                ),
                "teacher_ce_gain": float(ce["background"] - teacher_ce),
                "teacher_better_than_background": float(teacher_ce < ce["background"]),
                "teacher_top1_correct": float(int(teacher.argmax()) == target_index),
                "teacher_true_probability": float(teacher_probabilities[target_index]),
                "background_true_probability": float(
                    background_probabilities[target_index]
                ),
                "teacher_entropy": float(
                    -torch.sum(
                        teacher_probabilities
                        * teacher_probabilities.clamp_min(1e-8).log()
                    )
                ),
                "background_entropy": float(
                    -torch.sum(
                        background_probabilities
                        * background_probabilities.clamp_min(1e-8).log()
                    )
                ),
                "additive_teacher_kl_gain": float(
                    kl["background"] - kl["additive"]
                ),
                "nonlinear_teacher_kl_gain": float(
                    kl["background"] - kl["nonlinear"]
                ),
                "nonlinear_minus_additive_teacher_kl_gain": float(
                    kl["additive"] - kl["nonlinear"]
                ),
                "additive_rms": float(additive_rms),
                "correction_rms": float(correction_rms),
                "correction_to_additive_rms": float(
                    correction_rms / additive_rms.clamp_min(1e-8)
                ),
                "gauge_error": float(gauge_error),
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    return {
        column: float(frame[column].mean())
        for column in frame.columns
        if column not in {"example", "family", "stage"}
    }


def load_consensus_teacher(
    checkpoint: Path, hidden_dim: int, device: torch.device
) -> TransformerProteinLM:
    teacher = TransformerProteinLM(hidden_dim=hidden_dim, layers=1, heads=4).to(device)
    teacher.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def replace_teacher_logits_with_consensus(
    examples: list[dict[str, torch.Tensor | str]],
    second_teacher: torch.nn.Module,
    device: torch.device,
) -> None:
    for example in examples:
        tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
        target_position = int(example["target_position"])
        second_logits = second_teacher(tokens)["logits"][0, target_position]
        first_logits = example["teacher_logits"].to(device)  # type: ignore[union-attr]
        consensus = 0.5 * (
            first_logits.softmax(dim=-1) + second_logits.softmax(dim=-1)
        )
        example["teacher_logits"] = consensus.clamp_min(1e-8).log().cpu()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    router_summary = json.loads((args.router_run / "summary.json").read_text())
    router_config = router_summary["configuration"]
    base_run = Path(router_config["run"])
    config = json.loads((base_run / "run.json").read_text())
    benchmark = Path(router_config["benchmark"])
    representations = Path(router_config["representations"])
    device = choose_device(args.device)
    path_args = argparse.Namespace(benchmark=benchmark, representations=representations)
    frozen_args = teacher_args(config, path_args)
    teacher, background = load_frozen_models(frozen_args, device)
    families = pd.read_csv(representations / "families.csv")
    train_frame = families[families.role == "train"].reset_index(drop=True)
    eval_frame = families[families.role == "validation"].reset_index(drop=True)
    seed = int(config["seed"])
    train_examples = build_target_examples(
        train_frame,
        args.train_families or int(config["train_families"]),
        frozen_args,
        teacher,
        background,
        np.random.default_rng(seed + 1000),
        device,
    )
    eval_examples = build_target_examples(
        eval_frame,
        args.eval_families or int(config["eval_families"]),
        frozen_args,
        teacher,
        background,
        np.random.default_rng(seed + 3000),
        device,
    )
    if args.consensus_teacher_checkpoint is not None:
        second_teacher = load_consensus_teacher(
            args.consensus_teacher_checkpoint, int(config["hidden_dim"]), device
        )
        replace_teacher_logits_with_consensus(train_examples, second_teacher, device)
        replace_teacher_logits_with_consensus(eval_examples, second_teacher, device)
        del second_teacher
    del teacher, background
    model = load_model(base_run, config, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    router = make_router(
        config,
        router_config,
        args.candidate_budget,
        args.router_run / "router.pt",
        device,
    )
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    aggregator = MarginalOrthogonalSetAggregator(
        states=20,
        hidden_dim=args.hidden_dim,
        routing_temperature=float(model.interaction.routing_temperature),
        max_correction_ratio=args.max_correction_ratio,
    ).to(device)
    initial = evaluate(model, router, aggregator, eval_examples, device)
    optimizer = torch.optim.AdamW(
        aggregator.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = np.random.default_rng(seed + 11000)
    history = []
    aggregator.train()
    for step in range(1, args.steps + 1):
        example = train_examples[int(rng.integers(len(train_examples)))]
        result = example_outputs(model, router, aggregator, example, device)
        target = example["true_target"].to(device)  # type: ignore[union-attr]
        teacher_logits = example["teacher_logits"].to(device)  # type: ignore[union-attr]
        distill_loss = F.kl_div(
            result["nonlinear_logits"].log_softmax(dim=-1),
            teacher_logits.softmax(dim=-1),
            reduction="sum",
        )
        ce_loss = F.cross_entropy(result["nonlinear_logits"][None], target[None])
        background_ce = F.cross_entropy(
            result["background_logits"][None], target[None]
        )
        teacher_ce = F.cross_entropy(teacher_logits[None], target[None])
        reliability = torch.ones((), device=device)
        if args.teacher_reliability == "better_than_background":
            reliability = (teacher_ce < background_ce).to(result["nonlinear_logits"].dtype)
        correction_loss = result["correction"].square().mean()
        loss = (
            args.distill_weight * reliability * distill_loss
            + args.ce_weight * ce_loss
            + args.correction_weight * correction_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(aggregator.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "distill_loss": float(distill_loss.detach().cpu()),
                    "ce_loss": float(ce_loss.detach().cpu()),
                    "correction_loss": float(correction_loss.detach().cpu()),
                    "teacher_reliability": float(reliability.detach().cpu()),
                }
            )
    aggregator.eval()
    trained = evaluate(model, router, aggregator, eval_examples, device)
    initial["stage"] = "initial"
    trained["stage"] = "trained"
    pd.concat([initial, trained], ignore_index=True).to_csv(
        args.output / "per_example.csv", index=False
    )
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    torch.save(aggregator.state_dict(), args.output / "aggregator.pt")
    result = {
        "model_seed": seed,
        "aggregator_parameters": sum(p.numel() for p in aggregator.parameters()),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "initial": summarize(initial),
        "trained": summarize(trained),
        "configuration": vars(args),
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
