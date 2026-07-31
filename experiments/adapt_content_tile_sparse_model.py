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
    teacher_args,
)
from experiments.train_conditional_response_demo import (
    build_target_examples,
    freeze_background,
    sampled_outputs,
)
from experiments.train_content_tile_router import evaluate, summarize
from experiments.train_epistasis_identifiability import load_frozen_models
from experiments.train_hierarchical_candidate_router import target_distributions
from transformer_disentanglement.content_tile_routing import ContentTileRouter
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-budget", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--index-weight", type=float, default=0.2)
    parser.add_argument("--shape-weight", type=float, default=0.2)
    parser.add_argument("--tile-weight", type=float, default=0.2)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--dense-temperature", type=float, default=0.5)
    parser.add_argument("--conditional-temperature", type=float, default=0.5)
    parser.add_argument("--train-families", type=int)
    parser.add_argument("--eval-families", type=int)
    parser.add_argument("--freeze-value", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def freeze_categorical_value(model: torch.nn.Module) -> None:
    modules = [
        model.interaction.factor_norm,
        model.interaction.factor_projection,
        model.interaction.value_norm,
        model.interaction.value_projection,
        model.interaction.mode_decoder,
        model.interaction.interaction_to_stable,
        model.interaction.interaction_to_task,
    ]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    model.interaction.interaction_scale.requires_grad_(False)


def sparse_objective(
    model: torch.nn.Module,
    router: ContentTileRouter,
    example: dict[str, torch.Tensor | str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
    target_position = int(example["target_position"])
    pairs = example["pairs"].to(device)  # type: ignore[union-attr]
    index_target = example["index_target"].to(device)  # type: ignore[union-attr]
    shape_target = example["shape_target"].to(device)  # type: ignore[union-attr]
    teacher_logits = example["teacher_logits"].to(device)  # type: ignore[union-attr]
    true_target = example["true_target"].to(device)  # type: ignore[union-attr]
    output = model(tokens, use_interaction=False)
    exact_query, exact_key = model.interaction.index_features(
        output["encoded_task"]  # type: ignore[arg-type]
    )
    valid = model.interaction.valid_pair_mask(tokens.shape[1], device)
    routed = router(
        output["encoded_stable"],  # type: ignore[arg-type]
        output["encoded_task"],  # type: ignore[arg-type]
        torch.tensor([target_position], device=device),
        exact_query[:, target_position : target_position + 1],
        exact_key,
        valid[target_position : target_position + 1],
        score_scale=model.interaction.index_dim**-0.5,
        score_bias=model.interaction.index_bias,
    )
    message = interaction_message(
        model,
        output,
        tokens,
        target_position,
        routed.neighbor_indices[0, 0],
        routed.neighbor_scores[0, 0],
    )
    background_logits = output["background_logits"][0, target_position]  # type: ignore[index]
    logits = background_logits + message
    teacher_probabilities = teacher_logits.softmax(dim=-1)
    distill_loss = F.kl_div(
        logits.log_softmax(dim=-1), teacher_probabilities, reduction="sum"
    )
    ce_loss = F.cross_entropy(logits[None], true_target[None])
    sampled_scores, sampled_blocks, _, _ = sampled_outputs(model, output, pairs)
    index_loss = F.smooth_l1_loss(sampled_scores, index_target)
    shape_loss = F.smooth_l1_loss(
        sampled_blocks.contiguous(), shape_target.contiguous()
    )
    distribution_args = argparse.Namespace(
        dense_temperature=args.dense_temperature,
        conditional_temperature=args.conditional_temperature,
    )
    query_positions, dense_target, conditional_target = target_distributions(
        model, output, example, distribution_args, device
    )
    dense_target = dense_target.detach()
    conditional_target = conditional_target.detach()
    tile_loss = 0.5 * (
        router.routing_kl_loss(
            output["encoded_stable"],  # type: ignore[arg-type]
            output["encoded_task"],  # type: ignore[arg-type]
            query_positions,
            dense_target,
        )
        + router.routing_kl_loss(
            output["encoded_stable"],  # type: ignore[arg-type]
            output["encoded_task"],  # type: ignore[arg-type]
            query_positions,
            conditional_target,
        )
    )
    loss = (
        args.index_weight * index_loss
        + args.shape_weight * shape_loss
        + args.tile_weight * tile_loss
        + args.distill_weight * distill_loss
        + args.ce_weight * ce_loss
    )
    return loss, {
        "index_loss": float(index_loss.detach().cpu()),
        "shape_loss": float(shape_loss.detach().cpu()),
        "tile_loss": float(tile_loss.detach().cpu()),
        "distill_loss": float(distill_loss.detach().cpu()),
        "ce_loss": float(ce_loss.detach().cpu()),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    saved = json.loads((args.router_run / "summary.json").read_text())
    router_config = saved["configuration"]
    run = Path(router_config["run"])
    config = json.loads((run / "run.json").read_text())
    device = choose_device(args.device)
    benchmark = Path(router_config["benchmark"])
    representations = Path(router_config["representations"])
    path_args = argparse.Namespace(
        benchmark=benchmark,
        representations=representations,
    )
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
    del teacher, background
    model = load_model(run, config, device)
    freeze_background(model)
    if args.freeze_value:
        freeze_categorical_value(model)
    router = ContentTileRouter(
        stable_dim=int(config["stable_dim"]),
        task_dim=int(config["task_dim"]),
        tile_dim=int(router_config["tile_dim"]),
        tiles=int(router_config["tiles"]),
        selected_tiles=int(router_config["selected_tiles"]),
        candidate_budget=args.candidate_budget,
        neighbors=int(config["neighbors"]),
    ).to(device)
    router.load_state_dict(
        torch.load(args.router_run / "router.pt", map_location=device, weights_only=True)
    )
    initial_frame = evaluate(model, router, eval_examples, device)
    trainable = [
        parameter
        for module in (model, router)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 9000)
    history = []
    model.train()
    router.train()
    for step in range(1, args.steps + 1):
        example = train_examples[int(rng.integers(len(train_examples)))]
        loss, metrics = sparse_objective(model, router, example, args, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append({"step": step, "loss": float(loss.detach().cpu())} | metrics)
    model.eval()
    router.eval()
    adapted_frame = evaluate(model, router, eval_examples, device)
    initial_frame["stage"] = "initial"
    adapted_frame["stage"] = "adapted"
    pd.concat([initial_frame, adapted_frame], ignore_index=True).to_csv(
        args.output / "per_example.csv", index=False
    )
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    torch.save(model.state_dict(), args.output / "model.pt")
    torch.save(router.state_dict(), args.output / "router.pt")
    result = {
        "model_seed": seed,
        "candidate_budget": args.candidate_budget,
        "initial": summarize(initial_frame),
        "adapted": summarize(adapted_frame),
        "configuration": vars(args),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
