from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.train_seqmodels_factor_head import (
    excess_gate,
    fit_family_null_models,
    load_representation,
    sample_pairs,
)
from transformer_disentanglement.metrics import (
    binary_average_precision,
    binary_contact_precision,
    binary_contact_prevalence,
    binary_roc_auc,
    leading_mode_fraction,
    safe_spearman,
)
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.phylogeny import (
    randomized_phylogeny_loadings,
    residualize_phylogeny_modes,
    weighted_centered_features,
)
from transformer_disentanglement.relation_baselines import (
    SymmetricBilinearIndexer,
    SymmetricPairMLPIndexer,
)
from transformer_disentanglement.seqmodels_benchmark import (
    connected_pair_blocks,
    entropy_pair_scale,
    fit_entropy_null_model,
    load_seqmodels_family,
    residualize_entropy_background,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["bilinear", "pair_mlp"], required=True)
    parser.add_argument(
        "--objective", choices=["binary", "robust_z"], default="binary"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--index-dim", type=int, default=32)
    parser.add_argument("--mlp-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--updates-per-family", type=int, default=2)
    parser.add_argument("--pairs-per-update", type=int, default=256)
    parser.add_argument("--eval-pairs", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--null-reference-pairs", type=int, default=1024)
    parser.add_argument("--null-z-threshold", type=float, default=3.0)
    parser.add_argument("--phylogeny-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def make_model(args: argparse.Namespace, hidden_dim: int) -> torch.nn.Module:
    if args.model == "bilinear":
        return SymmetricBilinearIndexer(hidden_dim, index_dim=args.index_dim)
    return SymmetricPairMLPIndexer(
        hidden_dim,
        pair_dim=args.index_dim,
        mlp_dim=args.mlp_dim,
    )


def robust_teacher_score(blocks, pairs, pssm, null_model) -> np.ndarray:
    strength = np.sqrt(np.mean(np.square(blocks), axis=(-2, -1)))
    normalized = strength / entropy_pair_scale(pssm, pairs).clip(min=1e-8)
    return ((normalized - null_model.location) / null_model.scale).astype(
        np.float32
    )


def fit_phylogeny_teacher_models(
    frame: pd.DataFrame,
    benchmark: Path,
    representations: Path,
    pairs_per_family: int,
    rank: int,
    z_threshold: float,
    seed: int,
    device: torch.device,
):
    null_models = {}
    active_fractions = {}
    loadings = {}
    with torch.no_grad():
        for row in frame.itertuples(index=False):
            family = load_seqmodels_family(benchmark, row.file, row.x_id)
            _, pssm = load_representation(representations, row)
            features = weighted_centered_features(
                family.msa, family.weights, pssm, device
            )
            family_loadings = randomized_phylogeny_loadings(
                features, rank, seed + int(row.index) * 65537
            )
            loadings[row.x_id] = family_loadings
            rng = np.random.default_rng(seed + int(row.index) * 65537)
            pairs = sample_pairs(
                len(family.query), pairs_per_family, rng
            )
            blocks = connected_pair_blocks(
                family.msa, family.weights, pairs, pssm=pssm
            )
            blocks = residualize_phylogeny_modes(
                blocks,
                pairs,
                family_loadings,
                family.msa,
                family.weights,
                pssm,
            )
            model = fit_entropy_null_model(blocks, pairs, pssm)
            null_models[row.x_id] = model
            _, reliability = residualize_entropy_background(
                blocks, pairs, pssm, model, z_threshold=z_threshold
            )
            active_fractions[row.x_id] = float(np.mean(reliability > 0.0))
    return null_models, active_fractions, loadings


def transform_teacher_blocks(
    blocks,
    pairs,
    pssm,
    family,
    family_loadings,
):
    if family_loadings is None:
        return blocks
    return residualize_phylogeny_modes(
        blocks,
        pairs,
        family_loadings,
        family.msa,
        family.weights,
        pssm,
    )


def evidence_gate(score: torch.Tensor, threshold: float) -> torch.Tensor:
    return 1.0 - torch.exp(-torch.relu(score - threshold))


def pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    min_difference: float = 0.5,
) -> torch.Tensor:
    permutation = torch.randperm(len(target), device=target.device)
    target_difference = target - target[permutation]
    valid = torch.abs(target_difference) >= min_difference
    if not torch.any(valid):
        return prediction.new_zeros(())
    direction = torch.sign(target_difference[valid])
    predicted_difference = prediction[valid] - prediction[permutation][valid]
    return F.softplus(-direction * predicted_difference).mean()


def evaluate(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    benchmark: Path,
    representations: Path,
    null_models,
    z_threshold: float,
    gate_prior: float,
    eval_pairs: int,
    seed: int,
    device: torch.device,
    objective: str,
    phylogeny_loadings,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for row in frame.itertuples(index=False):
            rng = np.random.default_rng(seed + int(row.index) * 104729)
            family = load_seqmodels_family(benchmark, row.file, row.x_id)
            hidden, pssm = load_representation(representations, row)
            hidden_tensor = torch.from_numpy(hidden).to(device)
            score = model.full_logits(hidden_tensor)
            probabilities = torch.sigmoid(score)
            if objective == "binary":
                gates = excess_gate(probabilities, gate_prior)
            else:
                gates = evidence_gate(score, z_threshold)

            pairs = sample_pairs(len(family.query), eval_pairs, rng)
            blocks = connected_pair_blocks(
                family.msa, family.weights, pairs, pssm=pssm
            )
            blocks = transform_teacher_blocks(
                blocks,
                pairs,
                pssm,
                family,
                None
                if phylogeny_loadings is None
                else phylogeny_loadings[row.x_id],
            )
            _, reliability = residualize_entropy_background(
                blocks,
                pairs,
                pssm,
                null_models[row.x_id],
                z_threshold=z_threshold,
            )
            pair_tensor = torch.from_numpy(pairs).to(device)
            pair_score = model.pair_logits(hidden_tensor, pair_tensor).cpu().numpy()
            pair_probability = 1.0 / (1.0 + np.exp(-pair_score))
            teacher_score = robust_teacher_score(
                blocks, pairs, pssm, null_models[row.x_id]
            )
            active = reliability > 0.0
            probability_map = probabilities.cpu().numpy()
            score_map = score.cpu().numpy()
            gate_map = gates.cpu().numpy()
            rows.append(
                {
                    "x_id": row.x_id,
                    "role": row.role,
                    "length": len(family.query),
                    "teacher_active_fraction": float(np.mean(active)),
                    "teacher_average_precision": binary_average_precision(
                        pair_probability, active
                    ),
                    "teacher_roc_auc": binary_roc_auc(pair_probability, active),
                    "teacher_score_spearman": safe_spearman(
                        pair_score, teacher_score
                    ),
                    "contact_prevalence_long": binary_contact_prevalence(
                        family.contacts, family.contact_mask, min_separation=24
                    ),
                    "raw_long_p_at_l": binary_contact_precision(
                        score_map,
                        family.contacts,
                        family.contact_mask,
                        min_separation=24,
                    ),
                    "centered_long_p_at_l": binary_contact_precision(
                        gate_map,
                        family.contacts,
                        family.contact_mask,
                        min_separation=24,
                    ),
                    "raw_gate_mean": float(probability_map.mean()),
                    "centered_gate_mean": float(gate_map.mean()),
                    "centered_active_fraction": float(np.mean(gate_map > 0.0)),
                    "raw_leading_mode_fraction": leading_mode_fraction(
                        probability_map
                    ),
                    "centered_leading_mode_fraction": leading_mode_fraction(
                        gate_map
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    metrics = [
        column
        for column in frame.columns
        if column not in {"x_id", "role", "length"}
    ]
    summary = {}
    for role, group in frame.groupby("role"):
        values: dict[str, float | int] = {"families": int(len(group))}
        for metric in metrics:
            values[f"{metric}_mean"] = float(group[metric].mean())
            values[f"{metric}_median"] = float(group[metric].median())
        summary[role] = values
    return summary


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    frame = pd.read_csv(args.representations / "families.csv")
    train_frame = frame[frame["role"] == "train"].reset_index(drop=True)
    device = choose_device(args.device)
    if args.phylogeny_rank > 0:
        null_models, active_fractions, phylogeny_loadings = (
            fit_phylogeny_teacher_models(
                frame,
                args.benchmark,
                args.representations,
                args.null_reference_pairs,
                args.phylogeny_rank,
                args.null_z_threshold,
                args.seed,
                device,
            )
        )
    else:
        null_models, active_fractions = fit_family_null_models(
            frame,
            args.benchmark,
            args.representations,
            args.null_reference_pairs,
            args.seed,
            args.null_z_threshold,
        )
        phylogeny_loadings = None
    gate_prior = float(np.mean([active_fractions[x] for x in train_frame.x_id]))
    first_hidden, _ = load_representation(
        args.representations, train_frame.iloc[0]
    )
    model = make_model(args, first_hidden.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    history = []
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for family_index in rng.permutation(len(train_frame)):
            row = train_frame.iloc[int(family_index)]
            family = load_seqmodels_family(args.benchmark, row.file, row.x_id)
            hidden, pssm = load_representation(args.representations, row)
            hidden_tensor = torch.from_numpy(hidden).to(device)
            for _ in range(args.updates_per_family):
                step += 1
                pairs = sample_pairs(
                    len(family.query), args.pairs_per_update, rng
                )
                blocks = connected_pair_blocks(
                    family.msa, family.weights, pairs, pssm=pssm
                )
                blocks = transform_teacher_blocks(
                    blocks,
                    pairs,
                    pssm,
                    family,
                    None
                    if phylogeny_loadings is None
                    else phylogeny_loadings[row.x_id],
                )
                _, reliability = residualize_entropy_background(
                    blocks,
                    pairs,
                    pssm,
                    null_models[row.x_id],
                    z_threshold=args.null_z_threshold,
                )
                pairs_tensor = torch.from_numpy(pairs).to(device)
                score = model.pair_logits(hidden_tensor, pairs_tensor)
                if args.objective == "binary":
                    target = torch.from_numpy(
                        (reliability > 0.0).astype(np.float32)
                    ).to(device)
                    probability = torch.sigmoid(score)
                    regression_loss = F.binary_cross_entropy(
                        probability, target
                    )
                    ranking_loss = score.new_zeros(())
                else:
                    target_score = robust_teacher_score(
                        blocks,
                        pairs,
                        pssm,
                        null_models[row.x_id],
                    )
                    target = torch.from_numpy(
                        np.clip(target_score, -3.0, 10.0)
                    ).to(device)
                    regression_loss = F.smooth_l1_loss(score, target)
                    ranking_loss = pairwise_ranking_loss(score, target)
                    probability = torch.sigmoid(score)
                loss = regression_loss + args.ranking_weight * ranking_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step == 1 or step % args.log_every == 0:
                    history.append(
                        {
                            "step": step,
                            "epoch": epoch,
                            "x_id": row.x_id,
                            "loss": float(loss.detach().cpu()),
                            "regression_loss": float(
                                regression_loss.detach().cpu()
                            ),
                            "ranking_loss": float(ranking_loss.detach().cpu()),
                            "probability_mean": float(
                                probability.mean().detach().cpu()
                            ),
                            "target_mean": float(target.mean().detach().cpu()),
                        }
                    )

    evaluation = evaluate(
        model,
        frame,
        args.benchmark,
        args.representations,
        null_models,
        args.null_z_threshold,
        gate_prior,
        args.eval_pairs,
        args.seed,
        device,
        args.objective,
        phylogeny_loadings,
    )
    summary = summarize(evaluation)
    evaluation.to_csv(args.output / "per_family_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    torch.save(model.state_dict(), args.output / "model.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(
            vars(args)
            | {
                "device": str(device),
                "gate_prior": gate_prior,
                "parameters": sum(p.numel() for p in model.parameters()),
            },
            default=str,
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
