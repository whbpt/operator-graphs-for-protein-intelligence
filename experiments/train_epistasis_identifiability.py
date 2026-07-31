from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformer_disentanglement.demo_language_models import (
    LocalProteinLM,
    TransformerProteinLM,
)
from transformer_disentanglement.epistasis import (
    PairEpistasisRegressor,
    SiteOnlyEpistasisControl,
    double_mutation_epistasis,
    weighted_double_center,
    weighted_gauge_error,
)
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import load_seqmodels_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--background-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--features", choices=["local", "esm", "teacher"], default="local"
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--pair-dim", type=int, default=16)
    parser.add_argument("--pair-mlp-dim", type=int, default=64)
    parser.add_argument("--train-families", type=int, default=24)
    parser.add_argument("--calibration-families", type=int, default=12)
    parser.add_argument("--eval-families", type=int, default=12)
    parser.add_argument("--pairs-per-family", type=int, default=8)
    parser.add_argument("--mutation-states", default="0,1,2,3,4")
    parser.add_argument("--probe-fraction", type=float, default=0.15)
    parser.add_argument(
        "--target-normalization",
        choices=["none", "family"],
        default="family",
    )
    parser.add_argument("--min-separation", type=int, default=6)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_frozen_models(
    args: argparse.Namespace, device: torch.device
) -> tuple[TransformerProteinLM, LocalProteinLM]:
    teacher = TransformerProteinLM(
        hidden_dim=args.hidden_dim, layers=1, heads=4
    ).to(device)
    background = LocalProteinLM(hidden_dim=args.hidden_dim, layers=1).to(device)
    teacher.load_state_dict(
        torch.load(args.teacher_checkpoint, map_location=device, weights_only=True)
    )
    background.load_state_dict(
        torch.load(args.background_checkpoint, map_location=device, weights_only=True)
    )
    teacher.eval()
    background.eval()
    for model in (teacher, background):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return teacher, background


def sample_pairs(
    positions: np.ndarray,
    count: int,
    min_separation: int,
    rng: np.random.Generator,
) -> np.ndarray:
    left, right = np.triu_indices(len(positions), k=1)
    pairs = np.stack([positions[left], positions[right]], axis=-1)
    pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) >= min_separation]
    if not len(pairs):
        raise ValueError("No valid residue pairs after applying min_separation")
    selected = rng.choice(len(pairs), size=min(count, len(pairs)), replace=False)
    return pairs[selected].astype(np.int64)


@torch.no_grad()
def teacher_probe_losses(
    teacher: TransformerProteinLM,
    variants: torch.Tensor,
    probe_positions: torch.Tensor,
    probe_targets: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    losses = []
    for start in range(0, len(variants), batch_size):
        logits = teacher(variants[start : start + batch_size])["logits"]
        probe_logits = logits[:, probe_positions]
        targets = probe_targets[None].expand(len(probe_logits), -1)
        losses.append(
            F.cross_entropy(
                probe_logits.reshape(-1, probe_logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            )
            .reshape(len(probe_logits), -1)
            .mean(dim=-1)
        )
    return torch.cat(losses)


def make_variants(
    base_tokens: torch.Tensor,
    positions: torch.Tensor,
    mutation_states: torch.Tensor,
) -> torch.Tensor:
    state_count = len(mutation_states)
    variants = base_tokens.repeat(len(positions) * state_count, 1)
    row = torch.arange(len(variants), device=base_tokens.device)
    variants[row, positions.repeat_interleave(state_count)] = mutation_states.repeat(
        len(positions)
    )
    return variants


def make_double_variants(
    base_tokens: torch.Tensor,
    pairs: torch.Tensor,
    mutation_states: torch.Tensor,
) -> torch.Tensor:
    state_count = len(mutation_states)
    variants = base_tokens.repeat(len(pairs) * state_count * state_count, 1)
    row = torch.arange(len(variants), device=base_tokens.device)
    repeated_pairs = pairs.repeat_interleave(state_count * state_count, dim=0)
    left_states = mutation_states.repeat_interleave(state_count).repeat(len(pairs))
    right_states = mutation_states.repeat(state_count).repeat(len(pairs))
    variants[row, repeated_pairs[:, 0]] = left_states
    variants[row, repeated_pairs[:, 1]] = right_states
    return variants


@torch.no_grad()
def build_family_example(
    teacher: TransformerProteinLM,
    background: LocalProteinLM,
    sequence: np.ndarray,
    identifier: str,
    feature_source: str,
    feature_hidden: torch.Tensor | None,
    mutation_states: torch.Tensor,
    pairs_per_family: int,
    probe_fraction: float,
    min_separation: int,
    teacher_batch_size: int,
    target_normalization: str,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor | str | float]:
    valid = np.flatnonzero(sequence < 20)
    probe_count = max(1, int(round(len(valid) * probe_fraction)))
    probes = np.sort(
        rng.choice(valid, size=min(probe_count, len(valid)), replace=False)
    ).astype(np.int64)
    candidates = np.setdiff1d(valid, probes, assume_unique=True)
    pairs_np = sample_pairs(candidates, pairs_per_family, min_separation, rng)

    base = torch.from_numpy(sequence.astype(np.int64)).to(device)
    probe_positions = torch.from_numpy(probes).to(device)
    probe_targets = base[probe_positions].clone()
    masked = base.clone()
    masked[probe_positions] = 21
    base_tokens = masked[None]
    pairs = torch.from_numpy(pairs_np).to(device)
    unique_positions = torch.unique(pairs)

    base_loss = teacher_probe_losses(
        teacher,
        base_tokens,
        probe_positions,
        probe_targets,
        teacher_batch_size,
    )[0]
    single_variants = make_variants(
        base_tokens, unique_positions, mutation_states
    )
    single_losses = teacher_probe_losses(
        teacher,
        single_variants,
        probe_positions,
        probe_targets,
        teacher_batch_size,
    ).reshape(len(unique_positions), len(mutation_states))
    position_to_row = {
        int(position): index
        for index, position in enumerate(unique_positions.cpu().tolist())
    }
    left_rows = torch.tensor(
        [
            position_to_row[int(position)]
            for position in pairs[:, 0].cpu().tolist()
        ],
        device=device,
    )
    right_rows = torch.tensor(
        [
            position_to_row[int(position)]
            for position in pairs[:, 1].cpu().tolist()
        ],
        device=device,
    )
    double_variants = make_double_variants(
        base_tokens, pairs, mutation_states
    )
    double_losses = teacher_probe_losses(
        teacher,
        double_variants,
        probe_positions,
        probe_targets,
        teacher_batch_size,
    ).reshape(len(pairs), len(mutation_states), len(mutation_states))
    raw_epistasis = double_mutation_epistasis(
        base_loss,
        single_losses[left_rows],
        single_losses[right_rows],
        double_losses,
    )

    background_output = background(base_tokens)
    if feature_source == "teacher":
        hidden = teacher(base_tokens)["hidden"][0]
    elif feature_hidden is None:
        hidden = background_output["hidden"][0]
    else:
        hidden = feature_hidden.to(device)
    probabilities = background_output["logits"][0].softmax(dim=-1)
    restricted = probabilities[:, mutation_states]
    restricted = restricted / restricted.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    left_probabilities = restricted[pairs[:, 0]]
    right_probabilities = restricted[pairs[:, 1]]
    physical_target = weighted_double_center(
        raw_epistasis, left_probabilities, right_probabilities
    )
    target_scale = physical_target.square().mean().sqrt().clamp_min(1e-8)
    target = (
        physical_target / target_scale
        if target_normalization == "family"
        else physical_target
    )
    return {
        "identifier": identifier,
        "base_loss": float(base_loss.cpu()),
        "base_tokens": base_tokens[0].cpu(),
        "probe_positions": probe_positions.cpu(),
        "probe_targets": probe_targets.cpu(),
        "left_hidden": hidden[pairs[:, 0]].cpu(),
        "right_hidden": hidden[pairs[:, 1]].cpu(),
        "left_probabilities": left_probabilities.cpu(),
        "right_probabilities": right_probabilities.cpu(),
        "target": target.cpu(),
        "physical_target": physical_target.cpu(),
        "target_scale": float(target_scale.cpu()),
        "raw_target": raw_epistasis.cpu(),
        "pairs": pairs.cpu(),
    }


def build_examples(
    frame: pd.DataFrame,
    count: int,
    args: argparse.Namespace,
    teacher: TransformerProteinLM,
    background: LocalProteinLM,
    mutation_states: torch.Tensor,
    rng: np.random.Generator,
    device: torch.device,
) -> list[dict[str, torch.Tensor | str | float]]:
    examples = []
    for row in frame.head(count).itertuples(index=False):
        family = load_seqmodels_family(args.benchmark, row.file, row.x_id)
        if len(family.query) > 320:
            continue
        if args.features == "esm":
            with np.load(
                args.representations / "families" / row.representation_file,
                allow_pickle=False,
            ) as data:
                feature_hidden = torch.from_numpy(
                    data["hidden"].astype(np.float32)
                )
        else:
            feature_hidden = None
        examples.append(
            build_family_example(
                teacher,
                background,
                family.msa[0],
                row.x_id,
                args.features,
                feature_hidden,
                mutation_states,
                args.pairs_per_family,
                args.probe_fraction,
                args.min_separation,
                args.teacher_batch_size,
                args.target_normalization,
                rng,
                device,
            )
        )
        print(
            f"built {len(examples)}/{min(count, len(frame))} examples: {row.x_id}",
            flush=True,
        )
    return examples


def concatenate_examples(
    examples: list[dict[str, torch.Tensor | str | float]],
) -> dict[str, torch.Tensor]:
    tensor_keys = [
        "left_hidden",
        "right_hidden",
        "left_probabilities",
        "right_probabilities",
        "target",
    ]
    return {
        key: torch.cat([example[key] for example in examples])  # type: ignore[arg-type]
        for key in tensor_keys
    }


def scale_summary(
    examples: list[dict[str, torch.Tensor | str | float]],
) -> dict[str, float]:
    values = np.asarray([float(example["target_scale"]) for example in examples])
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
    }


def train_model(
    model: torch.nn.Module,
    train: dict[str, torch.Tensor],
    target_rms: float,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> list[dict[str, float | int]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(train["target"]),
            (min(args.batch_size, len(train["target"])),),
            generator=generator,
        )
        batch = {key: value[indices].to(device) for key, value in train.items()}
        prediction = model(
            batch["left_hidden"],
            batch["right_hidden"],
            batch["left_probabilities"],
            batch["right_probabilities"],
        )
        loss = F.smooth_l1_loss(
            prediction, batch["target"] / target_rms
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
    return history


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().flatten().cpu().double()
    right = right.detach().flatten().cpu().double()
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if float(denominator) <= 1e-12:
        return float("nan")
    return float((left * right).sum() / denominator)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    examples: list[dict[str, torch.Tensor | str | float]],
    target_rms: float,
    device: torch.device,
    output_scale: float = 1.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows = []
    all_predictions = []
    all_targets = []
    gauge_errors = []
    for example in examples:
        left_hidden = example["left_hidden"].to(device)  # type: ignore[union-attr]
        right_hidden = example["right_hidden"].to(device)  # type: ignore[union-attr]
        left_probabilities = example["left_probabilities"].to(device)  # type: ignore[union-attr]
        right_probabilities = example["right_probabilities"].to(device)  # type: ignore[union-attr]
        target = example["target"].to(device)  # type: ignore[union-attr]
        prediction = (
            model(
                left_hidden,
                right_hidden,
                left_probabilities,
                right_probabilities,
            )
            * target_rms
            * output_scale
        )
        mse = float(F.mse_loss(prediction, target).cpu())
        zero_mse = float(target.square().mean().cpu())
        gauge = float(
            weighted_gauge_error(
                prediction, left_probabilities, right_probabilities
            ).cpu()
        )
        rows.append(
            {
                "family": example["identifier"],
                "pairs": len(target),
                "mse": mse,
                "zero_mse": zero_mse,
                "relative_mse": mse / max(zero_mse, 1e-12),
                "correlation": correlation(prediction, target),
                "gauge_error": gauge,
            }
        )
        all_predictions.append(prediction.cpu())
        all_targets.append(target.cpu())
        gauge_errors.append(gauge)
    prediction = torch.cat(all_predictions)
    target = torch.cat(all_targets)
    mse = float(F.mse_loss(prediction, target))
    zero_mse = float(target.square().mean())
    summary = {
        "mse": mse,
        "zero_mse": zero_mse,
        "relative_mse": mse / max(zero_mse, 1e-12),
        "explained_fraction": 1.0 - mse / max(zero_mse, 1e-12),
        "correlation": correlation(prediction, target),
        "max_gauge_error": max(gauge_errors),
    }
    return summary, pd.DataFrame(rows)


@torch.no_grad()
def fit_output_scale(
    model: torch.nn.Module,
    examples: list[dict[str, torch.Tensor | str | float]],
    target_rms: float,
    device: torch.device,
) -> float:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    for example in examples:
        prediction = (
            model(
                example["left_hidden"].to(device),  # type: ignore[union-attr]
                example["right_hidden"].to(device),  # type: ignore[union-attr]
                example["left_probabilities"].to(device),  # type: ignore[union-attr]
                example["right_probabilities"].to(device),  # type: ignore[union-attr]
            )
            * target_rms
        )
        target = example["target"].to(device)  # type: ignore[union-attr]
        numerator += float(torch.sum(prediction * target).cpu())
        denominator += float(torch.sum(prediction.square()).cpu())
    return max(0.0, numerator / max(denominator, 1e-20))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    mutation_state_values = [
        int(value) for value in args.mutation_states.split(",") if value.strip()
    ]
    if len(set(mutation_state_values)) < 2:
        raise ValueError("mutation-states must contain at least two unique states")
    if min(mutation_state_values) < 0 or max(mutation_state_values) >= 20:
        raise ValueError("mutation-states must be amino-acid indices in [0, 19]")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    mutation_states = torch.tensor(mutation_state_values, device=device)
    teacher, background = load_frozen_models(args, device)
    frame = pd.read_csv(args.representations / "families.csv")
    train_frame = frame[frame["role"] == "train"].reset_index(drop=True)
    calibration_frame = frame[frame["role"] == "calibration"].reset_index(
        drop=True
    )
    eval_frame = frame[frame["role"] == "validation"].reset_index(drop=True)

    start = time.perf_counter()
    train_examples = build_examples(
        train_frame,
        args.train_families,
        args,
        teacher,
        background,
        mutation_states,
        np.random.default_rng(args.seed + 1000),
        device,
    )
    calibration_examples = build_examples(
        calibration_frame,
        args.calibration_families,
        args,
        teacher,
        background,
        mutation_states,
        np.random.default_rng(args.seed + 2000),
        device,
    )
    eval_examples = build_examples(
        eval_frame,
        args.eval_families,
        args,
        teacher,
        background,
        mutation_states,
        np.random.default_rng(args.seed + 3000),
        device,
    )
    data_seconds = time.perf_counter() - start
    train = concatenate_examples(train_examples)
    target_rms = float(train["target"].square().mean().sqrt().clamp_min(1e-8))

    states = len(mutation_state_values)
    feature_dim = int(train["left_hidden"].shape[-1])
    torch.manual_seed(args.seed + 11)
    unprojected = PairEpistasisRegressor(
        hidden_dim=feature_dim,
        states=states,
        rank=args.rank,
        pair_dim=args.pair_dim,
        pair_mlp_dim=args.pair_mlp_dim,
        projected=False,
    )
    shared_initialization = unprojected.state_dict()
    projected = PairEpistasisRegressor(
        hidden_dim=feature_dim,
        states=states,
        rank=args.rank,
        pair_dim=args.pair_dim,
        pair_mlp_dim=args.pair_mlp_dim,
        projected=True,
    )
    projected.load_state_dict(shared_initialization)
    torch.manual_seed(args.seed + 17)
    site_only = SiteOnlyEpistasisControl(
        hidden_dim=feature_dim,
        states=states,
        residual_dim=args.pair_mlp_dim,
    )
    models = {
        "site_only": site_only,
        "unprojected_pair": unprojected,
        "projected_pair": projected,
    }

    summaries = {}
    for model_index, (name, model) in enumerate(models.items()):
        history = train_model(
            model,
            train,
            target_rms,
            args,
            device,
            args.seed + 100 + model_index,
        )
        train_metrics, _ = evaluate_model(
            model, train_examples, target_rms, device
        )
        raw_eval_metrics, _ = evaluate_model(
            model, eval_examples, target_rms, device
        )
        output_scale = fit_output_scale(
            model, calibration_examples, target_rms, device
        )
        calibration_metrics, _ = evaluate_model(
            model,
            calibration_examples,
            target_rms,
            device,
            output_scale=output_scale,
        )
        eval_metrics, per_family = evaluate_model(
            model,
            eval_examples,
            target_rms,
            device,
            output_scale=output_scale,
        )
        summaries[name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train": train_metrics,
            "output_scale": output_scale,
            "calibration": calibration_metrics,
            "raw_validation": raw_eval_metrics,
            "validation": eval_metrics,
        }
        pd.DataFrame(history).to_csv(
            args.output / f"{name}_history.csv", index=False
        )
        per_family.to_csv(
            args.output / f"{name}_per_family.csv", index=False
        )
        torch.save(model.state_dict(), args.output / f"{name}.pt")

    summary = {
        "seed": args.seed,
        "device": str(device),
        "train_families": len(train_examples),
        "calibration_families": len(calibration_examples),
        "validation_families": len(eval_examples),
        "pairs_per_family": args.pairs_per_family,
        "mutation_states": mutation_state_values,
        "target_rms": target_rms,
        "target_normalization": args.target_normalization,
        "physical_target_scale": {
            "train": scale_summary(train_examples),
            "calibration": scale_summary(calibration_examples),
            "validation": scale_summary(eval_examples),
        },
        "features": args.features,
        "feature_dim": feature_dim,
        "data_generation_seconds": data_seconds,
        "models": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(vars(args) | {"device": str(device)}, default=str, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
