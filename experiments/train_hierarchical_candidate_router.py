from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.evaluate_lsh_candidate_router import (
    interaction_message,
    load_model,
    overlap_recall,
    teacher_args,
)
from experiments.train_conditional_response_demo import build_target_examples
from experiments.train_epistasis_identifiability import load_frozen_models
from transformer_disentanglement.hierarchical_routing import (
    HierarchicalSegmentRouter,
)
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--node-dim", type=int, default=16)
    parser.add_argument("--branching", type=int, default=4)
    parser.add_argument("--leaf-size", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--candidate-budget", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--conditional-weight", type=float, default=1.0)
    parser.add_argument("--dense-temperature", type=float, default=0.5)
    parser.add_argument("--conditional-temperature", type=float, default=0.5)
    parser.add_argument("--train-families", type=int)
    parser.add_argument("--eval-families", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def target_distributions(
    model: torch.nn.Module,
    output: dict[str, torch.Tensor | None],
    example: dict[str, torch.Tensor | str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_position = int(example["target_position"])
    pairs = example["pairs"].to(device)  # type: ignore[union-attr]
    index_target = example["index_target"].to(device)  # type: ignore[union-attr]
    task_hidden = output["encoded_task"]  # type: ignore[assignment]
    dense_scores = model.interaction.index_scores(task_hidden)[0, target_position]
    valid = model.interaction.valid_pair_mask(task_hidden.shape[1], device)[
        target_position
    ]
    dense_logits = dense_scores.masked_fill(~valid, -torch.inf)
    dense_probabilities = torch.softmax(
        dense_logits / args.dense_temperature, dim=-1
    )
    conditional_probabilities = torch.zeros_like(dense_probabilities)
    conditional_probabilities[pairs[:, 1]] = torch.softmax(
        index_target / args.conditional_temperature, dim=-1
    )
    query_positions = torch.tensor([[target_position]], device=device)
    return (
        query_positions,
        dense_probabilities[None, None],
        conditional_probabilities[None, None],
    )


def objective(
    model: torch.nn.Module,
    router: HierarchicalSegmentRouter,
    example: dict[str, torch.Tensor | str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
    with torch.no_grad():
        output = model(tokens, use_interaction=False)
        task_hidden = output["encoded_task"].detach()
        query_positions, dense_target, conditional_target = target_distributions(
            model, output, example, args, device
        )
    dense_loss = router.hierarchical_kl_loss(
        task_hidden, query_positions, dense_target
    )
    conditional_loss = router.hierarchical_kl_loss(
        task_hidden, query_positions, conditional_target
    )
    loss = args.dense_weight * dense_loss + args.conditional_weight * conditional_loss
    return loss, {
        "dense_loss": float(dense_loss.detach().cpu()),
        "conditional_loss": float(conditional_loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    router: HierarchicalSegmentRouter,
    examples: list[dict[str, torch.Tensor | str]],
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for example in examples:
        tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
        target_position = int(example["target_position"])
        pairs = example["pairs"].to(device)  # type: ignore[union-attr]
        index_target = example["index_target"].to(device)  # type: ignore[union-attr]
        teacher_logits = example["teacher_logits"].to(device)  # type: ignore[union-attr]
        true_target = example["true_target"].to(device)  # type: ignore[union-attr]
        output = model(tokens, use_interaction=False)
        task_hidden = output["encoded_task"]  # type: ignore[assignment]
        exact_query, exact_key = model.interaction.index_features(task_hidden)
        valid = model.interaction.valid_pair_mask(tokens.shape[1], device)
        valid_row = valid[target_position : target_position + 1]
        dense_scores = model.interaction.index_scores(task_hidden)[
            0, target_position
        ].masked_fill(~valid[target_position], -torch.inf)
        count = min(model.interaction.neighbors, int(valid_row.sum()))
        dense = torch.topk(dense_scores, k=count)
        routed = router(
            task_hidden,
            torch.tensor([target_position], device=device),
            exact_query[:, target_position : target_position + 1],
            exact_key,
            valid_row,
            score_scale=model.interaction.index_dim**-0.5,
            score_bias=model.interaction.index_bias,
        )
        candidates = routed.candidate_indices[0, 0]
        neighbors = routed.neighbor_indices[0, 0]
        strong_count = max(1, int(round(0.1 * len(index_target))))
        strong_positions = pairs[torch.topk(index_target, k=strong_count).indices, 1]
        dense_message = interaction_message(
            model,
            output,
            tokens,
            target_position,
            dense.indices,
            dense.values,
        )
        routed_message = interaction_message(
            model,
            output,
            tokens,
            target_position,
            neighbors,
            routed.neighbor_scores[0, 0],
        )
        background_logits = output["background_logits"][0, target_position]  # type: ignore[index]
        dense_logits = background_logits + dense_message
        routed_logits = background_logits + routed_message
        teacher_probabilities = teacher_logits.softmax(dim=-1)
        background_ce = F.cross_entropy(background_logits[None], true_target[None])
        dense_ce = F.cross_entropy(dense_logits[None], true_target[None])
        routed_ce = F.cross_entropy(routed_logits[None], true_target[None])
        background_kl = F.kl_div(
            background_logits.log_softmax(dim=-1), teacher_probabilities, reduction="sum"
        )
        dense_kl = F.kl_div(
            dense_logits.log_softmax(dim=-1), teacher_probabilities, reduction="sum"
        )
        routed_kl = F.kl_div(
            routed_logits.log_softmax(dim=-1), teacher_probabilities, reduction="sum"
        )
        valid_pairs = int(valid_row.sum())
        evaluated_pairs = int(routed.evaluated_pairs[0, 0])
        evaluated_nodes = int(routed.evaluated_nodes[0, 0])
        exact_dim = exact_query.shape[-1]
        work_ratio = (
            evaluated_pairs * exact_dim + evaluated_nodes * router.node_dim
        ) / (valid_pairs * exact_dim)
        rows.append(
            {
                "example": example["identifier"],
                "family": str(example["identifier"]).split(":", 1)[0],
                "candidate_dense_recall": overlap_recall(candidates, dense.indices),
                "neighbor_dense_recall": overlap_recall(neighbors, dense.indices),
                "candidate_teacher_recall": overlap_recall(
                    candidates, strong_positions
                ),
                "neighbor_teacher_recall": overlap_recall(neighbors, strong_positions),
                "evaluated_pairs": evaluated_pairs,
                "evaluated_nodes": evaluated_nodes,
                "valid_pairs": valid_pairs,
                "pair_fraction": evaluated_pairs / valid_pairs,
                "work_ratio": work_ratio,
                "message_cosine": float(
                    F.cosine_similarity(dense_message[None], routed_message[None])[0]
                ),
                "message_relative_error": float(
                    torch.linalg.vector_norm(routed_message - dense_message)
                    / torch.linalg.vector_norm(dense_message).clamp_min(1e-8)
                ),
                "dense_ce_gain": float(background_ce - dense_ce),
                "routed_ce_gain": float(background_ce - routed_ce),
                "routed_minus_dense_ce_gain": float(dense_ce - routed_ce),
                "dense_teacher_kl_gain": float(background_kl - dense_kl),
                "routed_teacher_kl_gain": float(background_kl - routed_kl),
                "routed_minus_dense_teacher_kl_gain": float(dense_kl - routed_kl),
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    return {
        column: float(frame[column].mean())
        for column in frame.columns
        if column not in {"example", "family", "stage"}
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    device = choose_device(args.device)
    config = json.loads((args.run / "run.json").read_text())
    model_seed = int(config["seed"])
    frozen_args = teacher_args(config, args)
    teacher, background = load_frozen_models(frozen_args, device)
    family_frame = pd.read_csv(args.representations / "families.csv")
    train_frame = family_frame[family_frame.role == "train"].reset_index(drop=True)
    eval_frame = family_frame[family_frame.role == "validation"].reset_index(drop=True)
    train_families = args.train_families or int(config["train_families"])
    eval_families = args.eval_families or int(config["eval_families"])
    train_examples = build_target_examples(
        train_frame,
        train_families,
        frozen_args,
        teacher,
        background,
        np.random.default_rng(model_seed + 1000),
        device,
    )
    eval_examples = build_target_examples(
        eval_frame,
        eval_families,
        frozen_args,
        teacher,
        background,
        np.random.default_rng(model_seed + 3000),
        device,
    )
    del teacher, background
    model = load_model(args.run, config, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    router = HierarchicalSegmentRouter(
        task_dim=int(config["task_dim"]),
        node_dim=args.node_dim,
        branching=args.branching,
        leaf_size=args.leaf_size,
        beam_size=args.beam_size,
        candidate_budget=args.candidate_budget,
        neighbors=int(config["neighbors"]),
    ).to(device)
    initial_frame = evaluate(model, router, eval_examples, device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = np.random.default_rng(model_seed + 6000)
    history = []
    router.train()
    for step in range(1, args.steps + 1):
        example = train_examples[int(rng.integers(len(train_examples)))]
        loss, metrics = objective(model, router, example, args, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append(
                {"step": step, "loss": float(loss.detach().cpu())} | metrics
            )
    router.eval()
    trained_frame = evaluate(model, router, eval_examples, device)
    initial_frame["stage"] = "initial"
    trained_frame["stage"] = "trained"
    pd.concat([initial_frame, trained_frame], ignore_index=True).to_csv(
        args.output / "per_example.csv", index=False
    )
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    torch.save(router.state_dict(), args.output / "router.pt")
    result = {
        "model_seed": model_seed,
        "router_parameters": sum(p.numel() for p in router.parameters()),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "initial": summarize(initial_frame),
        "trained": summarize(trained_frame),
        "configuration": vars(args),
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
