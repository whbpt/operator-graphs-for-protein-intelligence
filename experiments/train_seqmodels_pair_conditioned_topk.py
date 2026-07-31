from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.train_seqmodels_factor_head import (
    entropy,
    fit_family_null_models,
    load_representation,
    sample_pairs,
)
from experiments.train_seqmodels_gate_baseline import robust_teacher_score
from experiments.train_seqmodels_gate_baseline import (
    fit_phylogeny_teacher_models,
    transform_teacher_blocks,
)
from experiments.train_seqmodels_topk_typed_head import (
    estimate_target_scale,
    normalized_blocks,
    summarize_topk,
    symmetric_topk_mask,
    top_fraction_mask,
)
from transformer_disentanglement.metrics import (
    binary_contact_precision,
    binary_contact_prevalence,
    leading_mode_fraction,
    safe_spearman,
)
from transformer_disentanglement.pair_conditioned_interaction import (
    PairConditionedInteractionHead,
)
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.relation_baselines import (
    SymmetricPairMLPIndexer,
)
from transformer_disentanglement.seqmodels_benchmark import (
    connected_pair_blocks,
    load_seqmodels_family,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--indexer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--pair-dim", type=int, default=32)
    parser.add_argument("--pair-mlp-dim", type=int, default=64)
    parser.add_argument("--index-dim", type=int, default=32)
    parser.add_argument("--index-mlp-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--updates-per-family", type=int, default=2)
    parser.add_argument("--pairs-per-update", type=int, default=256)
    parser.add_argument("--eval-pairs", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--strength-weight", type=float, default=0.5)
    parser.add_argument("--marginal-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    parser.add_argument("--teacher-top-fraction", type=float, default=0.1)
    parser.add_argument("--neighbors-fraction", type=float, default=0.05)
    parser.add_argument("--null-reference-pairs", type=int, default=1024)
    parser.add_argument("--null-z-threshold", type=float, default=3.0)
    parser.add_argument("--phylogeny-rank", type=int, default=0)
    parser.add_argument(
        "--selection-source", choices=["teacher", "indexer"], default="teacher"
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def fill_selected_score_map(
    head: PairConditionedInteractionHead,
    output: dict[str, torch.Tensor],
    selected_map: np.ndarray,
    device: torch.device,
    chunk_size: int = 1024,
) -> np.ndarray:
    length = len(selected_map)
    pairs = np.argwhere(np.triu(selected_map, k=1)).astype(np.int64)
    score_map = np.zeros((length, length), dtype=np.float32)
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start : start + chunk_size]
        pair_tensor = torch.from_numpy(chunk).to(device)
        amplitude, mode = head.pair_parameters(output["pair_state"], pair_tensor)
        blocks = head.pair_blocks(
            output["factors"], pair_tensor, amplitude, mode
        )
        strength = torch.sqrt(blocks.square().mean(dim=(-2, -1))).cpu().numpy()
        score_map[chunk[:, 0], chunk[:, 1]] = strength
        score_map[chunk[:, 1], chunk[:, 0]] = strength
    return score_map


def estimate_selected_target_scale(
    frame: pd.DataFrame,
    benchmark: Path,
    representations: Path,
    indexer: SymmetricPairMLPIndexer,
    phylogeny_loadings,
    device: torch.device,
    rng: np.random.Generator,
    top_fraction: float,
    pairs_per_family: int = 256,
) -> float:
    squared_sum = 0.0
    element_count = 0
    indexer.eval()
    with torch.no_grad():
        for row in frame.itertuples(index=False):
            family = load_seqmodels_family(benchmark, row.file, row.x_id)
            hidden, pssm = load_representation(representations, row)
            pairs = sample_pairs(len(family.query), pairs_per_family, rng)
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
            pair_tensor = torch.from_numpy(pairs).to(device)
            hidden_tensor = torch.from_numpy(hidden).to(device)
            score = indexer.pair_logits(
                hidden_tensor, pair_tensor
            ).cpu().numpy()
            selected = top_fraction_mask(score, top_fraction)
            target = normalized_blocks(blocks, pairs, pssm)[selected]
            squared_sum += float(np.sum(target.astype(np.float64) ** 2))
            element_count += target.size
    return float(np.sqrt(squared_sum / max(element_count, 1)))


def evaluate(
    head: PairConditionedInteractionHead,
    indexer: SymmetricPairMLPIndexer,
    frame: pd.DataFrame,
    benchmark: Path,
    representations: Path,
    null_models,
    target_scale: float,
    top_fraction: float,
    neighbors_fraction: float,
    eval_pairs: int,
    seed: int,
    device: torch.device,
    phylogeny_loadings,
    selection_source: str,
) -> pd.DataFrame:
    rows = []
    head.eval()
    indexer.eval()
    with torch.no_grad():
        for row in frame.itertuples(index=False):
            rng = np.random.default_rng(seed + int(row.index) * 104729)
            family = load_seqmodels_family(benchmark, row.file, row.x_id)
            hidden, pssm = load_representation(representations, row)
            hidden_tensor = torch.from_numpy(hidden).to(device)
            output = head(hidden_tensor)
            index_score = indexer.full_logits(hidden_tensor).cpu().numpy()
            neighbors = max(1, int(round(len(family.query) * neighbors_fraction)))
            selected_map = symmetric_topk_mask(index_score, neighbors)
            typed_score = fill_selected_score_map(
                head, output, selected_map, device
            )

            pairs = sample_pairs(len(family.query), eval_pairs, rng)
            raw_blocks = connected_pair_blocks(
                family.msa, family.weights, pairs, pssm=pssm
            )
            teacher_score = robust_teacher_score(
                raw_blocks, pairs, pssm, null_models[row.x_id]
            )
            blocks = transform_teacher_blocks(
                raw_blocks,
                pairs,
                pssm,
                family,
                None
                if phylogeny_loadings is None
                else phylogeny_loadings[row.x_id],
            )
            target_blocks = normalized_blocks(blocks, pairs, pssm) / max(
                target_scale, 1e-8
            )
            pair_tensor = torch.from_numpy(pairs).to(device)
            amplitude, mode = head.pair_parameters(
                output["pair_state"], pair_tensor
            )
            predicted_blocks = head.pair_blocks(
                output["factors"], pair_tensor, amplitude, mode
            ).cpu().numpy()
            pair_index_score = indexer.pair_logits(
                hidden_tensor, pair_tensor
            ).cpu().numpy()
            selection_score = (
                teacher_score if selection_source == "teacher" else pair_index_score
            )
            selected = top_fraction_mask(selection_score, top_fraction)
            teacher_strength = np.sqrt(
                np.mean(np.square(target_blocks[selected]), axis=(-2, -1))
            )
            predicted_strength = np.sqrt(
                np.mean(np.square(predicted_blocks[selected]), axis=(-2, -1))
            )
            relative_error = float(
                np.linalg.norm(predicted_blocks[selected] - target_blocks[selected])
                / max(np.linalg.norm(target_blocks[selected]), 1e-12)
            )
            predicted_pssm = output["marginal_probabilities"].cpu().numpy()
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
                    "teacher_active_fraction": float(selected.mean()),
                    "index_teacher_spearman": safe_spearman(
                        pair_index_score, teacher_score
                    ),
                    "categorical_relative_error": relative_error,
                    "contact_prevalence_long": binary_contact_prevalence(
                        family.contacts, family.contact_mask, min_separation=24
                    ),
                    "index_long_p_at_l": binary_contact_precision(
                        index_score,
                        family.contacts,
                        family.contact_mask,
                        min_separation=24,
                    ),
                    "typed_long_p_at_l": binary_contact_precision(
                        typed_score,
                        family.contacts,
                        family.contact_mask,
                        min_separation=24,
                    ),
                    "selected_pair_fraction": float(selected_map.mean()),
                    "typed_leading_mode_fraction": leading_mode_fraction(
                        typed_score
                    ),
                    "amplitude_mean": float(amplitude.mean().cpu()),
                    "mode_mean_absolute": float(mode.abs().mean().cpu()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    frame = pd.read_csv(args.representations / "families.csv")
    train_frame = frame[frame["role"] == "train"].reset_index(drop=True)
    null_models, _ = fit_family_null_models(
        frame,
        args.benchmark,
        args.representations,
        args.null_reference_pairs,
        args.seed,
        args.null_z_threshold,
    )
    first_hidden, _ = load_representation(
        args.representations, train_frame.iloc[0]
    )
    device = choose_device(args.device)
    indexer = SymmetricPairMLPIndexer(
        first_hidden.shape[-1],
        pair_dim=args.index_dim,
        mlp_dim=args.index_mlp_dim,
    ).to(device)
    indexer.load_state_dict(
        torch.load(args.indexer_checkpoint, map_location=device, weights_only=True)
    )
    indexer.eval()
    for parameter in indexer.parameters():
        parameter.requires_grad_(False)

    if args.phylogeny_rank > 0:
        _, _, phylogeny_loadings = fit_phylogeny_teacher_models(
            frame,
            args.benchmark,
            args.representations,
            args.null_reference_pairs,
            args.phylogeny_rank,
            args.null_z_threshold,
            args.seed,
            device,
        )
    else:
        phylogeny_loadings = None
    if args.selection_source == "indexer":
        target_scale = estimate_selected_target_scale(
            train_frame,
            args.benchmark,
            args.representations,
            indexer,
            phylogeny_loadings,
            device,
            rng,
            args.teacher_top_fraction,
        )
    else:
        target_scale = estimate_target_scale(
            train_frame,
            args.benchmark,
            args.representations,
            null_models,
            rng,
            args.teacher_top_fraction,
        )

    head = PairConditionedInteractionHead(
        hidden_dim=first_hidden.shape[-1],
        states=20,
        rank=args.rank,
        pair_dim=args.pair_dim,
        pair_mlp_dim=args.pair_mlp_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    history = []
    step = 0
    head.train()
    for epoch in range(args.epochs):
        for family_index in rng.permutation(len(train_frame)):
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
                raw_blocks = connected_pair_blocks(
                    family.msa, family.weights, pairs, pssm=pssm
                )
                teacher_score = robust_teacher_score(
                    raw_blocks, pairs, pssm, null_models[row.x_id]
                )
                blocks = transform_teacher_blocks(
                    raw_blocks,
                    pairs,
                    pssm,
                    family,
                    None
                    if phylogeny_loadings is None
                    else phylogeny_loadings[row.x_id],
                )
                if args.selection_source == "indexer":
                    with torch.no_grad():
                        selection_score = indexer.pair_logits(
                            hidden_tensor,
                            torch.from_numpy(pairs).to(device),
                        ).cpu().numpy()
                else:
                    selection_score = teacher_score
                selected = top_fraction_mask(
                    selection_score, args.teacher_top_fraction
                )
                target_blocks = normalized_blocks(
                    blocks, pairs, pssm
                ) / max(target_scale, 1e-8)
                pairs_tensor = torch.from_numpy(pairs).to(device)
                selected_tensor = torch.from_numpy(selected).to(device)
                target_blocks_tensor = torch.from_numpy(target_blocks).to(device)
                output = head(hidden_tensor, gauge_probabilities=pssm_tensor)
                amplitude, mode = head.pair_parameters(
                    output["pair_state"], pairs_tensor
                )
                predicted_blocks = head.pair_blocks(
                    output["factors"], pairs_tensor, amplitude, mode
                )
                interaction_loss = F.mse_loss(
                    predicted_blocks[selected_tensor],
                    target_blocks_tensor[selected_tensor],
                )
                predicted_strength = torch.sqrt(
                    predicted_blocks[selected_tensor]
                    .square()
                    .mean(dim=(-2, -1))
                    + 1e-8
                )
                teacher_strength = torch.sqrt(
                    target_blocks_tensor[selected_tensor]
                    .square()
                    .mean(dim=(-2, -1))
                    + 1e-8
                )
                strength_loss = F.mse_loss(
                    predicted_strength, teacher_strength
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
                    pssm_tensor * torch.log(pssm_tensor.clamp_min(1e-8)),
                    dim=-1,
                )
                entropy_loss = F.mse_loss(predicted_entropy, target_entropy)
                loss = (
                    interaction_loss
                    + args.strength_weight * strength_loss
                    + args.marginal_weight * marginal_loss
                    + args.entropy_weight * entropy_loss
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
                            "marginal_loss": float(marginal_loss.detach().cpu()),
                            "entropy_loss": float(entropy_loss.detach().cpu()),
                            "amplitude_mean": float(amplitude.mean().detach().cpu()),
                        }
                    )

    evaluation = evaluate(
        head,
        indexer,
        frame,
        args.benchmark,
        args.representations,
        null_models,
        target_scale,
        args.teacher_top_fraction,
        args.neighbors_fraction,
        args.eval_pairs,
        args.seed,
        device,
        phylogeny_loadings,
        args.selection_source,
    )
    summary = summarize_topk(evaluation)
    evaluation.to_csv(args.output / "per_family_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    torch.save(head.state_dict(), args.output / "interaction_head.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(
            vars(args)
            | {
                "device": str(device),
                "target_scale": target_scale,
                "trainable_parameters": sum(
                    p.numel() for p in head.parameters()
                ),
                "frozen_indexer_parameters": sum(
                    p.numel() for p in indexer.parameters()
                ),
            },
            default=str,
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
