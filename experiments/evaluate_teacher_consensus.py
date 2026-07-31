from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from experiments.compare_conditional_response_demo import hierarchical_bootstrap
from transformer_disentanglement.demo_language_models import (
    LocalProteinLM,
    TransformerProteinLM,
)
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import load_seqmodels_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--teacher-checkpoints", type=Path, nargs=2, required=True)
    parser.add_argument("--background-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--targets-per-family", type=int, default=12)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[20260712, 20260713, 20260714])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=20000)
    return parser.parse_args()


def load_models(args: argparse.Namespace, device: torch.device) -> tuple[list[torch.nn.Module], torch.nn.Module]:
    teachers = []
    for checkpoint in args.teacher_checkpoints:
        model = TransformerProteinLM(hidden_dim=args.hidden_dim, layers=1, heads=4).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        teachers.append(model)
    background = LocalProteinLM(hidden_dim=args.hidden_dim, layers=1).to(device)
    background.load_state_dict(
        torch.load(args.background_checkpoint, map_location=device, weights_only=True)
    )
    background.eval()
    return teachers, background


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probabilities * probabilities.clamp_min(1e-8).log(), dim=-1)


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    middle = 0.5 * (left + right)
    return 0.5 * (
        torch.sum(left * (left.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log()), dim=-1)
        + torch.sum(right * (right.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log()), dim=-1)
    )


@torch.no_grad()
def collect_examples(
    args: argparse.Namespace,
    teachers: list[torch.nn.Module],
    background: torch.nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    families = pd.read_csv(args.representations / "families.csv")
    split = families[families.role == args.split].reset_index(drop=True)
    rows = []
    for sampling_seed in args.sampling_seeds:
        rng = np.random.default_rng(sampling_seed + 4000)
        for row in split.itertuples(index=False):
            family = load_seqmodels_family(args.benchmark, row.file, row.x_id)
            sequence = family.msa[0].astype(np.int64)
            valid = np.flatnonzero(sequence < 20)
            targets = rng.choice(
                valid,
                size=min(args.targets_per_family, len(valid)),
                replace=False,
            )
            tokens = torch.from_numpy(np.repeat(sequence[None], len(targets), axis=0)).to(device)
            target_tensor = torch.from_numpy(targets.astype(np.int64)).to(device)
            batch = torch.arange(len(targets), device=device)
            true_targets = tokens[batch, target_tensor].clone()
            tokens[batch, target_tensor] = 21
            background_logits = background(tokens)["logits"][batch, target_tensor]
            teacher_logits = [
                teacher(tokens)["logits"][batch, target_tensor] for teacher in teachers
            ]
            probabilities = [logits.softmax(dim=-1) for logits in teacher_logits]
            consensus_probabilities = 0.5 * (probabilities[0] + probabilities[1])
            background_probabilities = background_logits.softmax(dim=-1)
            log_probabilities = [p.clamp_min(1e-8).log() for p in probabilities]
            background_log_probabilities = background_probabilities.clamp_min(1e-8).log()
            residuals = [lp - background_log_probabilities for lp in log_probabilities]
            agreement_cosine = F.cosine_similarity(residuals[0], residuals[1], dim=-1)
            teacher_ce = [
                F.nll_loss(lp, true_targets, reduction="none") for lp in log_probabilities
            ]
            background_ce = F.cross_entropy(
                background_logits, true_targets, reduction="none"
            )
            consensus_ce = F.nll_loss(
                consensus_probabilities.clamp_min(1e-8).log(),
                true_targets,
                reduction="none",
            )
            js = js_divergence(probabilities[0], probabilities[1])
            for index, target in enumerate(targets):
                rows.append(
                    {
                        "seed": sampling_seed,
                        "family": row.x_id,
                        "example": f"{row.x_id}:{int(target)}",
                        "teacher1_ce_gain": float(background_ce[index] - teacher_ce[0][index]),
                        "teacher2_ce_gain": float(background_ce[index] - teacher_ce[1][index]),
                        "consensus_ce_gain": float(background_ce[index] - consensus_ce[index]),
                        "consensus_minus_teacher1_ce_gain": float(teacher_ce[0][index] - consensus_ce[index]),
                        "consensus_minus_teacher2_ce_gain": float(teacher_ce[1][index] - consensus_ce[index]),
                        "both_teachers_better": float(
                            teacher_ce[0][index] < background_ce[index]
                            and teacher_ce[1][index] < background_ce[index]
                        ),
                        "either_teacher_better": float(
                            teacher_ce[0][index] < background_ce[index]
                            or teacher_ce[1][index] < background_ce[index]
                        ),
                        "teachers_top1_agree": float(
                            int(probabilities[0][index].argmax())
                            == int(probabilities[1][index].argmax())
                        ),
                        "agreement_cosine": float(agreement_cosine[index]),
                        "js_divergence": float(js[index]),
                        "teacher1_entropy": float(entropy(probabilities[0][index])),
                        "teacher2_entropy": float(entropy(probabilities[1][index])),
                        "consensus_entropy": float(entropy(consensus_probabilities[index])),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    device = choose_device(args.device)
    teachers, background = load_models(args, device)
    data = collect_examples(args, teachers, background, device)
    metrics = [
        "teacher1_ce_gain",
        "teacher2_ce_gain",
        "consensus_ce_gain",
        "consensus_minus_teacher1_ce_gain",
        "consensus_minus_teacher2_ce_gain",
        "both_teachers_better",
        "either_teacher_better",
        "teachers_top1_agree",
        "agreement_cosine",
        "js_divergence",
    ]
    draws = hierarchical_bootstrap(
        data, metrics, args.bootstrap, np.random.default_rng(20260721)
    )
    family = data.groupby(["seed", "family"], as_index=False)[metrics].mean()
    estimates = family.groupby("seed")[metrics].mean().mean(axis=0)
    comparison = []
    for index, metric in enumerate(metrics):
        comparison.append(
            {
                "metric": metric,
                "estimate": float(estimates[metric]),
                "ci_low": float(np.quantile(draws[:, index], 0.025)),
                "ci_high": float(np.quantile(draws[:, index], 0.975)),
                "probability_positive": float(np.mean(draws[:, index] > 0)),
            }
        )
    correlations = {
        "agreement_cosine_vs_consensus_ce_gain": float(
            spearmanr(data.agreement_cosine, data.consensus_ce_gain).statistic
        ),
        "negative_js_vs_consensus_ce_gain": float(
            spearmanr(-data.js_divergence, data.consensus_ce_gain).statistic
        ),
    }
    data["agreement_quartile"] = data.groupby("seed").agreement_cosine.transform(
        lambda values: pd.qcut(values, 4, labels=False, duplicates="drop")
    )
    quartiles = data.groupby("agreement_quartile", as_index=False)[
        ["agreement_cosine", "js_divergence", "consensus_ce_gain"]
    ].mean()
    data.to_csv(args.output / "per_example.csv", index=False)
    pd.DataFrame(comparison).to_csv(args.output / "hierarchical_bootstrap.csv", index=False)
    quartiles.to_csv(args.output / "agreement_quartiles.csv", index=False)
    result = {
        "configuration": vars(args),
        "examples": int(len(data)),
        "comparison": comparison,
        "correlations": correlations,
        "quartiles": quartiles.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
