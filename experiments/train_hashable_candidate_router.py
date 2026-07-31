from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.evaluate_lsh_candidate_router import (
    evaluate_configuration,
    load_model,
    teacher_args,
)
from experiments.train_conditional_response_demo import build_target_examples
from experiments.train_epistasis_identifiability import load_frozen_models
from transformer_disentanglement.candidate_routing import (
    HashableTaskAdapter,
    MultiTableLSHCandidateRouter,
)
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--hash-dim", type=int, default=32)
    parser.add_argument("--candidate-budget", type=int, default=32)
    parser.add_argument("--hamming-radius", type=int, default=1)
    parser.add_argument("--hash-seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--conditional-weight", type=float, default=1.0)
    parser.add_argument("--quantization-weight", type=float, default=0.1)
    parser.add_argument("--balance-weight", type=float, default=0.1)
    parser.add_argument("--decorrelation-weight", type=float, default=0.1)
    parser.add_argument("--start-temperature", type=float, default=1.0)
    parser.add_argument("--end-temperature", type=float, default=0.2)
    parser.add_argument("--table-temperature", type=float, default=0.1)
    parser.add_argument("--train-families", type=int)
    parser.add_argument("--eval-families", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def standardize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def adapter_objective(
    model: torch.nn.Module,
    adapter: HashableTaskAdapter,
    router: MultiTableLSHCandidateRouter,
    example: dict[str, torch.Tensor | str],
    temperature: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
    target_position = int(example["target_position"])
    pairs = example["pairs"].to(device)  # type: ignore[union-attr]
    index_target = example["index_target"].to(device)  # type: ignore[union-attr]
    with torch.no_grad():
        output = model(tokens, use_interaction=False)
        task_hidden = output["encoded_task"].detach()
        teacher_scores = model.interaction.index_scores(task_hidden)[
            0, target_position
        ]
        valid = model.interaction.valid_pair_mask(tokens.shape[1], device)[
            target_position
        ]
    hash_features = adapter(task_hidden)
    relaxed_codes = adapter.relaxed_codes(
        hash_features, router.hyperplanes, temperature
    )
    hard_codes = torch.where(
        relaxed_codes >= 0,
        torch.ones_like(relaxed_codes),
        -torch.ones_like(relaxed_codes),
    )
    straight_through = relaxed_codes + (hard_codes - relaxed_codes).detach()
    student_scores = adapter.relaxed_scores(
        straight_through,
        torch.tensor([[target_position]], device=device),
        table_temperature=args.table_temperature,
    )[0, 0]
    distill_loss = F.smooth_l1_loss(
        standardize(student_scores[valid]),
        standardize(teacher_scores[valid]),
    )
    context_positions = pairs[:, 1]
    conditional_loss = F.smooth_l1_loss(
        standardize(student_scores[context_positions]), index_target
    )
    quantization, balance, decorrelation = adapter.regularization(relaxed_codes)
    loss = (
        args.distill_weight * distill_loss
        + args.conditional_weight * conditional_loss
        + args.quantization_weight * quantization
        + args.balance_weight * balance
        + args.decorrelation_weight * decorrelation
    )
    return loss, {
        "distill_loss": float(distill_loss.detach().cpu()),
        "conditional_loss": float(conditional_loss.detach().cpu()),
        "quantization": float(quantization.detach().cpu()),
        "balance": float(balance.detach().cpu()),
        "decorrelation": float(decorrelation.detach().cpu()),
        "temperature": temperature,
    }


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    columns = [
        "candidate_dense_recall",
        "neighbor_dense_recall",
        "candidate_teacher_recall",
        "neighbor_teacher_recall",
        "evaluated_fraction",
        "message_cosine",
        "message_relative_error",
        "lsh_minus_dense_ce_gain",
        "lsh_minus_dense_teacher_kl_gain",
    ]
    return {column: float(frame[column].mean()) for column in columns}


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
    adapter = HashableTaskAdapter(
        task_dim=int(config["task_dim"]), hash_dim=args.hash_dim
    ).to(device)
    router = MultiTableLSHCandidateRouter(
        feature_dim=args.hash_dim,
        tables=args.tables,
        bits=args.bits,
        candidate_budget=args.candidate_budget,
        neighbors=int(config["neighbors"]),
        hamming_radius=args.hamming_radius,
        seed=args.hash_seed,
    ).to(device)
    initial_frame = pd.DataFrame(
        evaluate_configuration(
            model,
            eval_examples,
            router,
            model_seed,
            args.hash_seed,
            device,
            hash_adapter=adapter,
        )
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = np.random.default_rng(model_seed + 5000)
    history = []
    adapter.train()
    for step in range(1, args.steps + 1):
        progress = (step - 1) / max(args.steps - 1, 1)
        temperature = args.start_temperature * (
            args.end_temperature / args.start_temperature
        ) ** progress
        example = train_examples[int(rng.integers(len(train_examples)))]
        loss, metrics = adapter_objective(
            model, adapter, router, example, temperature, args, device
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append(
                {"step": step, "loss": float(loss.detach().cpu())} | metrics
            )
    adapter.eval()
    trained_frame = pd.DataFrame(
        evaluate_configuration(
            model,
            eval_examples,
            router,
            model_seed,
            args.hash_seed,
            device,
            hash_adapter=adapter,
        )
    )
    initial_frame["stage"] = "initial"
    trained_frame["stage"] = "trained"
    pd.concat([initial_frame, trained_frame], ignore_index=True).to_csv(
        args.output / "per_example.csv", index=False
    )
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    torch.save(adapter.state_dict(), args.output / "adapter.pt")
    result = {
        "model_seed": model_seed,
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
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
