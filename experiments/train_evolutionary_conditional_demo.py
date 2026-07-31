from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.train_conditional_response_demo import (
    evaluate,
    freeze_background,
    masked_target_sequence,
    train_phase,
)
from experiments.train_dual_stream_epistasis_demo import load_local_initialization
from transformer_disentanglement.demo_language_models import (
    DualStreamProteinLM,
    LocalProteinLM,
)
from transformer_disentanglement.epistasis import (
    robust_standardize,
    weighted_double_center,
)
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import (
    load_seqmodels_family,
    weighted_log_odds_blocks,
    weighted_pssm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--background-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stable-dim", type=int, default=64)
    parser.add_argument("--task-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--value-mode",
        choices=["site_shared", "pair_residual"],
        default="site_shared",
    )
    parser.add_argument("--adapter-count", type=int, default=8)
    parser.add_argument("--adapter-topk", type=int, default=2)
    parser.add_argument("--adapter-bias-update-speed", type=float, default=0.0)
    parser.add_argument("--index-dim", type=int, default=16)
    parser.add_argument("--pair-dim", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--train-families", type=int, default=64)
    parser.add_argument("--eval-families", type=int, default=24)
    parser.add_argument("--eval-on-train-examples", action="store_true")
    parser.add_argument("--targets-per-family", type=int, default=4)
    parser.add_argument("--contexts-per-target", type=int, default=16)
    parser.add_argument("--min-separation", type=int, default=6)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--conditional-temperature", type=float, default=0.5)
    parser.add_argument("--message-scale", type=float, default=0.25)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--sparse-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--index-weight", type=float, default=0.2)
    parser.add_argument("--shape-weight", type=float, default=0.2)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


@torch.no_grad()
def build_evolutionary_examples(
    frame: pd.DataFrame,
    count: int,
    args: argparse.Namespace,
    background: torch.nn.Module,
    rng: np.random.Generator,
    device: torch.device,
) -> list[dict[str, torch.Tensor | str]]:
    examples = []
    for row in frame.head(count).itertuples(index=False):
        family = load_seqmodels_family(args.benchmark, row.file, row.x_id)
        sequence = family.msa[0].astype(np.int64)
        valid = np.flatnonzero(sequence < 20)
        targets = rng.choice(
            valid, size=min(args.targets_per_family, len(valid)), replace=False
        )
        teacher_weights = family.weights.astype(np.float64).copy()
        teacher_weights[0] = 0.0
        pssm = weighted_pssm(
            family.msa, teacher_weights, states=20, pseudocount=1e-3
        )
        for target in targets:
            candidates = valid[np.abs(valid - target) >= args.min_separation]
            contexts = rng.choice(
                candidates,
                size=min(args.contexts_per_target, len(candidates)),
                replace=False,
            ).astype(np.int64)
            pairs = np.stack(
                [np.full(len(contexts), int(target), dtype=np.int64), contexts],
                axis=-1,
            )
            blocks = weighted_log_odds_blocks(
                family.msa,
                teacher_weights,
                pairs,
                pssm=pssm,
                states=20,
                prior_weight=args.prior_weight,
            )
            base, true_target = masked_target_sequence(
                sequence, int(target), device
            )
            background_output = background(base[None])
            background_probabilities = background_output["logits"][0].softmax(
                dim=-1
            )
            context_tensor = torch.from_numpy(contexts).long().to(device)
            context_tokens = torch.tensor(
                sequence[contexts], dtype=torch.long, device=device
            )
            if torch.any(context_tokens >= 20):
                raise ValueError("evolutionary context tokens must be amino acids")

            block_tensor = torch.from_numpy(blocks).to(device)
            block_tensor = weighted_double_center(
                block_tensor,
                background_probabilities[target][None].expand(len(contexts), -1),
                background_probabilities[context_tensor],
            )
            shape_scale = block_tensor.square().mean().sqrt().clamp_min(1e-8)
            shape_target = (block_tensor / shape_scale).contiguous()
            strength = block_tensor.square().mean(dim=(-2, -1)).sqrt().clamp_min(1e-10)
            index_target, _, _ = robust_standardize(torch.log(strength))

            background_logits = background_output["logits"][0, target]
            observed = block_tensor[
                torch.arange(len(contexts), device=device), :, context_tokens
            ]
            routing_weights = torch.softmax(
                index_target / args.conditional_temperature, dim=-1
            )
            evolutionary_message = torch.sum(
                routing_weights[:, None] * observed, dim=0
            )
            teacher_logits = background_logits + args.message_scale * evolutionary_message
            examples.append(
                {
                    "identifier": f"{row.x_id}:{int(target)}",
                    "base_tokens": base.cpu(),
                    "target_position": torch.tensor(int(target)),
                    "true_target": true_target.cpu(),
                    "teacher_logits": teacher_logits.cpu(),
                    "teacher_message": evolutionary_message.cpu(),
                    "pairs": torch.from_numpy(pairs),
                    "context_tokens": context_tokens.cpu(),
                    "index_target": index_target.cpu(),
                    "shape_target": shape_target.cpu(),
                    "msa_target_probabilities": torch.from_numpy(
                        pssm[target].astype(np.float32)
                    ),
                }
            )
        print(f"built {row.x_id}: {len(examples)} target examples", flush=True)
    return examples


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    args.hidden_dim = args.stable_dim
    args.rank_mode = "fixed"
    args.gate_temperature = 1.0
    args.pair_mlp_dim = None
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    background = LocalProteinLM(hidden_dim=args.hidden_dim, layers=1).to(device)
    background.load_state_dict(
        torch.load(
            args.background_checkpoint, map_location=device, weights_only=True
        )
    )
    background.eval()
    for parameter in background.parameters():
        parameter.requires_grad_(False)
    families = pd.read_csv(args.representations / "families.csv")
    train_frame = families[families.role == "train"].reset_index(drop=True)
    eval_frame = families[families.role == "validation"].reset_index(drop=True)
    train_examples = build_evolutionary_examples(
        train_frame,
        args.train_families,
        args,
        background,
        np.random.default_rng(args.seed + 1000),
        device,
    )
    if args.eval_on_train_examples:
        eval_examples = train_examples
    else:
        eval_examples = build_evolutionary_examples(
            eval_frame,
            args.eval_families,
            args,
            background,
            np.random.default_rng(args.seed + 3000),
            device,
        )
    del background
    model = DualStreamProteinLM(
        stable_dim=args.stable_dim,
        task_dim=args.task_dim,
        rank=args.rank,
        index_dim=args.index_dim,
        pair_dim=args.pair_dim,
        neighbors=args.neighbors,
        routing_mode="topk",
        value_mode=args.value_mode,
        adapter_count=args.adapter_count,
        adapter_topk=args.adapter_topk,
        adapter_bias_update_speed=args.adapter_bias_update_speed,
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
        "teacher": "leave-query-out weighted MSA log odds",
        "value_mode": args.value_mode,
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
