from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformer_disentanglement.interaction_model import MarginalInteractionHead
from transformer_disentanglement.metrics import (
    average_product_correction,
    binary_average_precision,
    binary_contact_precision,
    binary_contact_prevalence,
    binary_roc_auc,
    leading_mode_fraction,
    safe_spearman,
)
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import (
    EntropyNullModel,
    connected_pair_blocks,
    fit_entropy_null_model,
    load_seqmodels_family,
    residualize_entropy_background,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--gate-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--updates-per-family", type=int, default=2)
    parser.add_argument("--pairs-per-update", type=int, default=256)
    parser.add_argument("--eval-pairs", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--strength-weight", type=float, default=0.5)
    parser.add_argument("--gate-weight", type=float, default=0.5)
    parser.add_argument("--marginal-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    parser.add_argument("--sparse-weight", type=float, default=0.01)
    parser.add_argument("--null-reference-pairs", type=int, default=1024)
    parser.add_argument("--null-z-threshold", type=float, default=3.0)
    parser.add_argument("--gate-prior", type=float)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--train-role", default="train")
    return parser.parse_args()


def sample_pairs(
    length: int,
    count: int,
    rng: np.random.Generator,
    min_separation: int = 6,
) -> np.ndarray:
    medium_i, medium_j = np.triu_indices(length, k=min_separation)
    separations = medium_j - medium_i
    long_indices = np.flatnonzero(separations >= 24)
    medium_indices = np.flatnonzero((separations >= min_separation) & (separations < 24))
    if len(long_indices) == 0:
        selected = rng.choice(len(medium_i), size=count, replace=True)
    else:
        long_count = count // 2
        selected = np.concatenate(
            [
                rng.choice(long_indices, size=long_count, replace=True),
                rng.choice(
                    medium_indices if len(medium_indices) else long_indices,
                    size=count - long_count,
                    replace=True,
                ),
            ]
        )
    rng.shuffle(selected)
    return np.stack([medium_i[selected], medium_j[selected]], axis=-1).astype(np.int64)


def load_representation(
    representations: Path, row
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(
        representations / "families" / row.representation_file,
        allow_pickle=False,
    ) as data:
        return data["hidden"].astype(np.float32), data["pssm"].astype(np.float32)


def estimate_target_scale(
    frame: pd.DataFrame,
    benchmark: Path,
    representations: Path,
    rng: np.random.Generator,
    null_models: dict[str, EntropyNullModel],
    null_z_threshold: float,
    pairs_per_family: int = 128,
) -> float:
    squared_sum = 0.0
    element_count = 0
    for row in frame.itertuples(index=False):
        family = load_seqmodels_family(benchmark, row.file, row.x_id)
        _, pssm = load_representation(representations, row)
        pairs = sample_pairs(len(family.query), pairs_per_family, rng)
        blocks = connected_pair_blocks(
            family.msa, family.weights, pairs, pssm=pssm
        )
        blocks, _ = residualize_entropy_background(
            blocks,
            pairs,
            pssm,
            null_models[row.x_id],
            z_threshold=null_z_threshold,
        )
        squared_sum += float(np.sum(blocks.astype(np.float64) ** 2))
        element_count += blocks.size
    return float(np.sqrt(squared_sum / max(element_count, 1)))


def fit_family_null_models(
    frame: pd.DataFrame,
    benchmark: Path,
    representations: Path,
    pairs_per_family: int,
    seed: int,
    z_threshold: float,
) -> tuple[dict[str, EntropyNullModel], dict[str, float]]:
    models = {}
    active_fractions = {}
    for row in frame.itertuples(index=False):
        rng = np.random.default_rng(seed + int(row.index) * 65537)
        family = load_seqmodels_family(benchmark, row.file, row.x_id)
        _, pssm = load_representation(representations, row)
        pairs = sample_pairs(len(family.query), pairs_per_family, rng)
        blocks = connected_pair_blocks(
            family.msa, family.weights, pairs, pssm=pssm
        )
        model = fit_entropy_null_model(blocks, pairs, pssm)
        models[row.x_id] = model
        _, reliability = residualize_entropy_background(
            blocks, pairs, pssm, model, z_threshold=z_threshold
        )
        active_fractions[row.x_id] = float(np.mean(reliability > 0.0))
    return models, active_fractions


def excess_gate(probability: torch.Tensor, prior: float) -> torch.Tensor:
    """Remove the null activation prior from calibrated gate probabilities."""
    return torch.clamp(probability - prior, min=0.0) / max(1.0 - prior, 1e-8)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(probabilities + 1e-12), axis=-1)


def evaluate(
    head: MarginalInteractionHead,
    frame: pd.DataFrame,
    benchmark: Path,
    representations: Path,
    device: torch.device,
    target_scale: float,
    null_models: dict[str, EntropyNullModel],
    null_z_threshold: float,
    gate_prior: float,
    eval_pairs: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    head.eval()
    with torch.no_grad():
        for row in frame.itertuples(index=False):
            rng = np.random.default_rng(seed + int(row.index) * 104729)
            family = load_seqmodels_family(benchmark, row.file, row.x_id)
            hidden, pssm = load_representation(representations, row)
            hidden_tensor = torch.from_numpy(hidden).to(device)
            output = head(hidden_tensor)
            gate_probability = head.full_gates(
                output["gate_left"], output["gate_right"], output["gate_bias"]
            )
            gates = excess_gate(gate_probability, gate_prior)
            scores = head.full_score_map(
                output["factors"], output["mode_scale"], gates=gates
            ).cpu().numpy() * target_scale
            predicted_pssm = output["marginal_probabilities"].cpu().numpy()

            pairs = sample_pairs(len(family.query), eval_pairs, rng)
            teacher_blocks = connected_pair_blocks(
                family.msa, family.weights, pairs, pssm=pssm
            )
            teacher_blocks, teacher_reliability = residualize_entropy_background(
                teacher_blocks,
                pairs,
                pssm,
                null_models[row.x_id],
                z_threshold=null_z_threshold,
            )
            pair_tensor = torch.from_numpy(pairs).to(device)
            pair_gate_probability = head.pair_gates(
                output["gate_left"],
                output["gate_right"],
                output["gate_bias"],
                pair_tensor,
            )
            pair_gates = excess_gate(pair_gate_probability, gate_prior)
            predicted_blocks = head.pair_blocks(
                output["factors"],
                output["mode_scale"],
                pair_tensor,
                gates=pair_gates,
            ).cpu().numpy() * target_scale
            teacher_strength = np.sqrt(np.mean(teacher_blocks**2, axis=(-2, -1)))
            predicted_strength = np.sqrt(
                np.mean(predicted_blocks**2, axis=(-2, -1))
            )
            active_target = teacher_reliability > 0.0
            marginal_kl = float(
                np.mean(
                    np.sum(
                        pssm
                        * (
                            np.log(pssm + 1e-12)
                            - np.log(predicted_pssm + 1e-12)
                        ),
                        axis=-1,
                    )
                )
            )
            rows.append(
                {
                    "x_id": row.x_id,
                    "role": row.role,
                    "length": len(family.query),
                    "marginal_kl": marginal_kl,
                    "entropy_spearman": safe_spearman(
                        entropy(pssm), entropy(predicted_pssm)
                    ),
                    "teacher_strength_spearman": safe_spearman(
                        teacher_strength, predicted_strength
                    ),
                    "teacher_active_fraction": float(
                        np.mean(active_target)
                    ),
                    "gate_teacher_average_precision": binary_average_precision(
                        pair_gate_probability.cpu().numpy(), active_target
                    ),
                    "gate_teacher_roc_auc": binary_roc_auc(
                        pair_gate_probability.cpu().numpy(), active_target
                    ),
                    "contact_prevalence_long": binary_contact_prevalence(
                        family.contacts, family.contact_mask, min_separation=24
                    ),
                    "predicted_long_p_at_l": binary_contact_precision(
                        scores,
                        family.contacts,
                        family.contact_mask,
                        min_separation=24,
                    ),
                    "predicted_apc_long_p_at_l": binary_contact_precision(
                        average_product_correction(scores),
                        family.contacts,
                        family.contact_mask,
                        min_separation=24,
                    ),
                    "gate_long_p_at_l": binary_contact_precision(
                        gates.cpu().numpy(),
                        family.contacts,
                        family.contact_mask,
                        min_separation=24,
                    ),
                    "gate_mean": float(gates.mean().cpu()),
                    "raw_gate_mean": float(gate_probability.mean().cpu()),
                    "gate_active_fraction_0": float(
                        (gates > 0.0).float().mean().cpu()
                    ),
                    "gate_active_fraction_0_5": float(
                        (gates > 0.5).float().mean().cpu()
                    ),
                    "score_leading_mode_fraction": leading_mode_fraction(scores),
                }
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    metrics = [
        "marginal_kl",
        "entropy_spearman",
        "teacher_strength_spearman",
        "teacher_active_fraction",
        "gate_teacher_average_precision",
        "gate_teacher_roc_auc",
        "contact_prevalence_long",
        "predicted_long_p_at_l",
        "predicted_apc_long_p_at_l",
        "gate_long_p_at_l",
        "gate_mean",
        "raw_gate_mean",
        "gate_active_fraction_0",
        "gate_active_fraction_0_5",
        "score_leading_mode_fraction",
    ]
    summary = {}
    for role, group in frame.groupby("role"):
        values: dict[str, float | int] = {"families": int(len(group))}
        for metric in metrics:
            values[f"{metric}_mean"] = float(group[metric].mean())
            values[f"{metric}_median"] = float(group[metric].median())
        values["fraction_above_prevalence"] = float(
            np.mean(group["predicted_long_p_at_l"] > group["contact_prevalence_long"])
        )
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
    train_frame = frame[frame["role"] == args.train_role].reset_index(drop=True)
    if train_frame.empty:
        raise ValueError(f"No families found for train role {args.train_role}")

    null_models, null_active_fractions = fit_family_null_models(
        frame,
        args.benchmark,
        args.representations,
        args.null_reference_pairs,
        args.seed,
        args.null_z_threshold,
    )
    gate_prior = (
        args.gate_prior
        if args.gate_prior is not None
        else float(np.mean([null_active_fractions[x] for x in train_frame.x_id]))
    )
    target_scale = estimate_target_scale(
        train_frame,
        args.benchmark,
        args.representations,
        rng,
        null_models,
        args.null_z_threshold,
    )
    first_hidden, _ = load_representation(
        args.representations, train_frame.iloc[0]
    )
    device = choose_device(args.device)
    head = MarginalInteractionHead(
        hidden_dim=first_hidden.shape[-1],
        states=20,
        rank=args.rank,
        gate_dim=args.gate_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    history = []
    step = 0
    head.train()
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_frame))
        for family_index in order:
            row = train_frame.iloc[int(family_index)]
            family = load_seqmodels_family(args.benchmark, row.file, row.x_id)
            hidden, pssm = load_representation(args.representations, row)
            hidden_tensor = torch.from_numpy(hidden).to(device)
            pssm_tensor = torch.from_numpy(pssm).to(device)
            for _ in range(args.updates_per_family):
                step += 1
                pairs = sample_pairs(
                    len(family.query), args.pairs_per_update, rng
                )
                teacher = connected_pair_blocks(
                    family.msa, family.weights, pairs, pssm=pssm
                )
                teacher, teacher_reliability = residualize_entropy_background(
                    teacher,
                    pairs,
                    pssm,
                    null_models[row.x_id],
                    z_threshold=args.null_z_threshold,
                )
                teacher = teacher / max(target_scale, 1e-8)
                pairs_tensor = torch.from_numpy(pairs).to(device)
                teacher_tensor = torch.from_numpy(teacher).to(device)
                gate_target = torch.from_numpy(
                    (teacher_reliability > 0.0).astype(np.float32)
                ).to(device)
                output = head(hidden_tensor, gauge_probabilities=pssm_tensor)
                gate_probability = head.pair_gates(
                    output["gate_left"],
                    output["gate_right"],
                    output["gate_bias"],
                    pairs_tensor,
                )
                gates = excess_gate(gate_probability, gate_prior)
                predicted = head.pair_blocks(
                    output["factors"],
                    output["mode_scale"],
                    pairs_tensor,
                    gates=gates,
                )
                interaction_loss = torch.mean((predicted - teacher_tensor) ** 2)
                predicted_strength = torch.sqrt(
                    predicted.square().mean(dim=(-2, -1)) + 1e-8
                )
                teacher_strength = torch.sqrt(
                    teacher_tensor.square().mean(dim=(-2, -1)) + 1e-8
                )
                strength_loss = torch.mean(
                    (predicted_strength - teacher_strength) ** 2
                )
                gate_loss = F.binary_cross_entropy(
                    gate_probability, gate_target
                )
                log_probabilities = torch.log_softmax(
                    output["marginal_logits"], dim=-1
                )
                marginal_loss = -torch.mean(
                    torch.sum(pssm_tensor * log_probabilities, dim=-1)
                )
                predicted_entropy = -torch.sum(
                    output["marginal_probabilities"]
                    * torch.log(
                        output["marginal_probabilities"].clamp_min(1e-8)
                    ),
                    dim=-1,
                )
                target_entropy = -torch.sum(
                    pssm_tensor * torch.log(pssm_tensor.clamp_min(1e-8)), dim=-1
                )
                entropy_loss = torch.mean(
                    (predicted_entropy - target_entropy) ** 2
                )
                loss = (
                    interaction_loss
                    + args.strength_weight * strength_loss
                    + args.gate_weight * gate_loss
                    + args.marginal_weight * marginal_loss
                    + args.entropy_weight * entropy_loss
                    + args.sparse_weight * gates.mean()
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                if step == 1 or step % args.log_every == 0:
                    history.append(
                        {
                            "step": step,
                            "epoch": epoch,
                            "x_id": row.x_id,
                            "loss": float(loss.detach().cpu()),
                            "interaction_loss": float(
                                interaction_loss.detach().cpu()
                            ),
                            "strength_loss": float(strength_loss.detach().cpu()),
                            "gate_loss": float(gate_loss.detach().cpu()),
                            "marginal_loss": float(marginal_loss.detach().cpu()),
                            "entropy_loss": float(entropy_loss.detach().cpu()),
                            "gate_mean": float(gates.mean().detach().cpu()),
                        }
                    )

    evaluation = evaluate(
        head,
        frame,
        args.benchmark,
        args.representations,
        device,
        target_scale,
        null_models,
        args.null_z_threshold,
        gate_prior,
        args.eval_pairs,
        args.seed,
    )
    summary = summarize(evaluation)
    evaluation.to_csv(args.output / "per_family_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    torch.save(head.state_dict(), args.output / "factor_head.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(
            vars(args)
            | {
                "device": str(device),
                "target_scale": target_scale,
                "resolved_gate_prior": gate_prior,
            },
            default=str,
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
