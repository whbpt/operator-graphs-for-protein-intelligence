from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.train_conditional_response_demo import build_target_examples
from experiments.train_epistasis_identifiability import load_frozen_models
from transformer_disentanglement.candidate_routing import (
    HashableTaskAdapter,
    MultiTableLSHCandidateRouter,
)
from transformer_disentanglement.demo_language_models import DualStreamProteinLM
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tables", type=int, nargs="+", default=[4])
    parser.add_argument("--bits", type=int, nargs="+", default=[6])
    parser.add_argument("--budgets", type=int, nargs="+", default=[32])
    parser.add_argument("--radii", type=int, nargs="+", default=[0])
    parser.add_argument("--hash-seeds", type=int, nargs="+", default=[17, 23, 29])
    parser.add_argument("--max-examples", type=int)
    return parser.parse_args()


def load_model(
    run: Path, config: dict, device: torch.device
) -> DualStreamProteinLM:
    summary = json.loads((run / "summary.json").read_text())
    model = DualStreamProteinLM(
        stable_dim=int(config["stable_dim"]),
        task_dim=int(config["task_dim"]),
        rank=int(config["rank"]),
        index_dim=int(config["index_dim"]),
        pair_dim=int(config["pair_dim"]),
        pair_mlp_dim=summary.get("pair_mlp_dim"),
        neighbors=int(config["neighbors"]),
        rank_mode=config.get("rank_mode", "fixed"),
        gate_temperature=float(config.get("gate_temperature", 1.0)),
    ).to(device)
    state = torch.load(run / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def teacher_args(config: dict, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark=args.benchmark,
        representations=args.representations,
        teacher_checkpoint=Path(config["teacher_checkpoint"]),
        background_checkpoint=Path(config["background_checkpoint"]),
        hidden_dim=int(config["stable_dim"]),
        targets_per_family=int(config["targets_per_family"]),
        contexts_per_target=int(config["contexts_per_target"]),
        min_separation=int(config["min_separation"]),
        teacher_batch_size=int(config["teacher_batch_size"]),
    )


def overlap_recall(selected: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        len(np.intersect1d(selected.cpu().numpy(), target.cpu().numpy()))
        / max(len(target), 1)
    )


def interaction_message(
    model: DualStreamProteinLM,
    output: dict[str, torch.Tensor | None],
    tokens: torch.Tensor,
    target_position: int,
    neighbors: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    target = torch.full_like(neighbors, target_position)
    pairs = torch.stack([target, neighbors], dim=-1)[None]
    blocks = model.interaction.sampled_pair_blocks(
        output["factors"],  # type: ignore[arg-type]
        output["value_state"],  # type: ignore[arg-type]
        output["encoded_task"],  # type: ignore[arg-type]
        pairs,
    )[0]
    context_tokens = tokens[0, neighbors]
    messages = blocks[
        torch.arange(len(neighbors), device=tokens.device), :, context_tokens
    ]
    weights = torch.softmax(
        scores / model.interaction.routing_temperature, dim=-1
    )
    return model.interaction.interaction_scale * torch.sum(
        weights[:, None] * messages, dim=0
    )


@torch.no_grad()
def evaluate_configuration(
    model: DualStreamProteinLM,
    examples: list[dict[str, torch.Tensor | str]],
    router: MultiTableLSHCandidateRouter,
    model_seed: int,
    hash_seed: int,
    device: torch.device,
    hash_adapter: HashableTaskAdapter | None = None,
) -> list[dict[str, float | int | str]]:
    rows = []
    layer = model.interaction
    for example in examples:
        tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
        target_position = int(example["target_position"])
        sampled_pairs = example["pairs"].to(device)  # type: ignore[union-attr]
        index_target = example["index_target"].to(device)  # type: ignore[union-attr]
        teacher_logits = example["teacher_logits"].to(device)  # type: ignore[union-attr]
        true_target = example["true_target"].to(device)  # type: ignore[union-attr]
        output = model(tokens, use_interaction=False)
        query, key = layer.index_features(output["encoded_task"])  # type: ignore[arg-type]
        hash_features = (
            hash_adapter(output["encoded_task"])  # type: ignore[arg-type]
            if hash_adapter is not None
            else None
        )
        valid = layer.valid_pair_mask(tokens.shape[1], device)
        valid_row = valid[target_position : target_position + 1]
        dense_scores = layer.index_scores(output["encoded_task"])[  # type: ignore[arg-type]
            0, target_position
        ].masked_fill(~valid[target_position], -torch.inf)
        neighbor_count = min(layer.neighbors, int(valid_row.sum()))
        dense_neighbors = torch.topk(dense_scores, k=neighbor_count).indices
        dense_neighbor_scores = dense_scores[dense_neighbors]
        routed = router(
            query[:, target_position : target_position + 1],
            key,
            valid_row,
            score_scale=layer.index_dim**-0.5,
            score_bias=layer.index_bias,
            query_positions=torch.tensor([target_position]),
            hash_query_features=(
                hash_features[:, target_position : target_position + 1]
                if hash_features is not None
                else None
            ),
            hash_key_features=hash_features,
        )
        candidates = routed.candidate_indices[0, 0]
        neighbors = routed.neighbor_indices[0, 0]
        dense_message = interaction_message(
            model,
            output,
            tokens,
            target_position,
            dense_neighbors,
            dense_neighbor_scores,
        )
        lsh_message = interaction_message(
            model,
            output,
            tokens,
            target_position,
            neighbors,
            routed.neighbor_scores[0, 0],
        )
        background_logits = output["background_logits"][0, target_position]  # type: ignore[index]
        dense_logits = background_logits + dense_message
        lsh_logits = background_logits + lsh_message
        teacher_probabilities = teacher_logits.softmax(dim=-1)
        background_ce = F.cross_entropy(
            background_logits[None], true_target[None]
        )
        dense_ce = F.cross_entropy(dense_logits[None], true_target[None])
        lsh_ce = F.cross_entropy(lsh_logits[None], true_target[None])
        background_kl = F.kl_div(
            background_logits.log_softmax(dim=-1),
            teacher_probabilities,
            reduction="sum",
        )
        dense_kl = F.kl_div(
            dense_logits.log_softmax(dim=-1),
            teacher_probabilities,
            reduction="sum",
        )
        lsh_kl = F.kl_div(
            lsh_logits.log_softmax(dim=-1),
            teacher_probabilities,
            reduction="sum",
        )
        message_cosine = F.cosine_similarity(
            dense_message[None], lsh_message[None]
        )[0]
        message_relative_error = (
            torch.linalg.vector_norm(lsh_message - dense_message)
            / torch.linalg.vector_norm(dense_message).clamp_min(1e-8)
        )
        strong_count = max(1, int(round(0.1 * len(index_target))))
        strong_sampled = torch.topk(index_target, k=strong_count).indices
        strong_positions = sampled_pairs[strong_sampled, 1]
        dense_teacher_recall = overlap_recall(dense_neighbors, strong_positions)
        rows.append(
            {
                "model_seed": model_seed,
                "hash_seed": hash_seed,
                "example": example["identifier"],
                "length": tokens.shape[1],
                "tables": router.tables,
                "bits": router.bits,
                "candidate_budget": router.candidate_budget,
                "hamming_radius": router.hamming_radius,
                "candidate_dense_recall": overlap_recall(
                    candidates, dense_neighbors
                ),
                "neighbor_dense_recall": overlap_recall(
                    neighbors, dense_neighbors
                ),
                "dense_teacher_recall": dense_teacher_recall,
                "candidate_teacher_recall": overlap_recall(
                    candidates, strong_positions
                ),
                "neighbor_teacher_recall": overlap_recall(
                    neighbors, strong_positions
                ),
                "bucket_candidates": int(routed.bucket_candidates[0, 0]),
                "evaluated_pairs": int(routed.evaluated_pairs[0, 0]),
                "fallback_candidates": int(routed.fallback_candidates[0, 0]),
                "valid_pairs": int(valid_row.sum()),
                "evaluated_fraction": float(
                    routed.evaluated_pairs[0, 0] / valid_row.sum()
                ),
                "message_cosine": float(message_cosine),
                "message_relative_error": float(message_relative_error),
                "dense_ce_gain": float(background_ce - dense_ce),
                "lsh_ce_gain": float(background_ce - lsh_ce),
                "lsh_minus_dense_ce_gain": float(dense_ce - lsh_ce),
                "dense_teacher_kl_gain": float(background_kl - dense_kl),
                "lsh_teacher_kl_gain": float(background_kl - lsh_kl),
                "lsh_minus_dense_teacher_kl_gain": float(dense_kl - lsh_kl),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    device = choose_device(args.device)
    family_frame = pd.read_csv(args.representations / "families.csv")
    validation_frame = family_frame[family_frame.role == "validation"].reset_index(
        drop=True
    )
    first_config = json.loads((args.runs[0] / "run.json").read_text())
    frozen_args = teacher_args(first_config, args)
    teacher, background = load_frozen_models(frozen_args, device)
    rows = []
    for run in args.runs:
        config = json.loads((run / "run.json").read_text())
        model_seed = int(config["seed"])
        examples = build_target_examples(
            validation_frame,
            int(config["eval_families"]),
            frozen_args,
            teacher,
            background,
            np.random.default_rng(model_seed + 3000),
            device,
        )
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        model = load_model(run, config, device)
        feature_dim = int(config["index_dim"]) * 2
        for tables in args.tables:
            for bits in args.bits:
                for budget in args.budgets:
                    for radius in args.radii:
                        for hash_seed in args.hash_seeds:
                            router = MultiTableLSHCandidateRouter(
                                feature_dim=feature_dim,
                                tables=tables,
                                bits=bits,
                                candidate_budget=budget,
                                neighbors=int(config["neighbors"]),
                                hamming_radius=radius,
                                seed=hash_seed,
                            ).to(device)
                            rows.extend(
                                evaluate_configuration(
                                    model,
                                    examples,
                                    router,
                                    model_seed,
                                    hash_seed,
                                    device,
                                )
                            )
    frame = pd.DataFrame(rows)
    group_columns = [
        "tables",
        "bits",
        "candidate_budget",
        "hamming_radius",
    ]
    metric_columns = [
        "candidate_dense_recall",
        "neighbor_dense_recall",
        "dense_teacher_recall",
        "candidate_teacher_recall",
        "neighbor_teacher_recall",
        "bucket_candidates",
        "evaluated_pairs",
        "fallback_candidates",
        "valid_pairs",
        "evaluated_fraction",
        "message_cosine",
        "message_relative_error",
        "dense_ce_gain",
        "lsh_ce_gain",
        "lsh_minus_dense_ce_gain",
        "dense_teacher_kl_gain",
        "lsh_teacher_kl_gain",
        "lsh_minus_dense_teacher_kl_gain",
    ]
    summary = frame.groupby(group_columns, as_index=False)[metric_columns].mean()
    summary["pair_reduction"] = 1.0 - summary.evaluated_fraction
    summary = summary.sort_values(
        ["candidate_dense_recall", "pair_reduction"], ascending=[False, False]
    )
    frame.to_csv(args.output / "per_example.csv", index=False)
    summary.to_csv(args.output / "configuration_summary.csv", index=False)
    result = {
        "runs": [str(run) for run in args.runs],
        "examples": int(frame.example.nunique()),
        "rows": int(len(frame)),
        "configurations": summary.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
