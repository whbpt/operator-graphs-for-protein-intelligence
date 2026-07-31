from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from experiments.train_epistasis_identifiability import (
    build_examples,
    load_frozen_models,
)
from transformer_disentanglement.epistasis import robust_standardize
from transformer_disentanglement.metrics import (
    binary_average_precision,
    safe_spearman,
)
from transformer_disentanglement.protein_transformer import choose_device


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
        "--features", choices=["local", "esm", "teacher"], default="esm"
    )
    parser.add_argument("--train-families", type=int, default=64)
    parser.add_argument("--eval-families", type=int, default=24)
    parser.add_argument("--pairs-per-family", type=int, default=32)
    parser.add_argument("--mutation-states", default="0,1,2,3,4")
    parser.add_argument("--probe-fraction", type=float, default=0.15)
    parser.add_argument("--min-separation", type=int, default=6)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--pair-dim", type=int, default=32)
    parser.add_argument("--mlp-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


class EntropyStrengthRegressor(nn.Module):
    def __init__(self, feature_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class PairStrengthRegressor(nn.Module):
    def __init__(self, hidden_dim: int, pair_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, pair_dim)
        self.network = nn.Sequential(
            nn.Linear(pair_dim * 3, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, left_hidden: torch.Tensor, right_hidden: torch.Tensor
    ) -> torch.Tensor:
        left = self.projection(self.norm(left_hidden))
        right = self.projection(self.norm(right_hidden))
        features = torch.cat(
            [left + right, torch.abs(left - right), left * right], dim=-1
        )
        return self.network(features).squeeze(-1)


def entropy_features(
    left_probabilities: torch.Tensor,
    right_probabilities: torch.Tensor,
    pairs: torch.Tensor,
) -> torch.Tensor:
    left_entropy = -torch.sum(
        left_probabilities * torch.log(left_probabilities.clamp_min(1e-8)),
        dim=-1,
    )
    right_entropy = -torch.sum(
        right_probabilities * torch.log(right_probabilities.clamp_min(1e-8)),
        dim=-1,
    )
    left_diversity = 1.0 - torch.sum(left_probabilities.square(), dim=-1)
    right_diversity = 1.0 - torch.sum(right_probabilities.square(), dim=-1)
    separation = torch.abs(pairs[:, 1] - pairs[:, 0]).float()
    length = float(pairs.max() + 1)
    separation = torch.log1p(separation) / np.log1p(max(length, 2.0))
    return torch.stack(
        [
            left_entropy + right_entropy,
            torch.abs(left_entropy - right_entropy),
            left_entropy * right_entropy,
            left_diversity + right_diversity,
            torch.abs(left_diversity - right_diversity),
            left_diversity * right_diversity,
            separation,
        ],
        dim=-1,
    )


def flatten_strength_examples(
    examples: list[dict[str, torch.Tensor | str | float]],
) -> dict[str, torch.Tensor | list[str]]:
    left_hidden = []
    right_hidden = []
    entropy = []
    targets = []
    log_strengths = []
    families = []
    locations = []
    scales = []
    for example in examples:
        physical_target = example["physical_target"]  # type: ignore[assignment]
        strength = physical_target.square().mean(dim=(-2, -1)).sqrt().clamp_min(1e-10)
        log_strength = torch.log(strength)
        standardized, location, scale = robust_standardize(log_strength)
        pair_count = len(standardized)
        left_hidden.append(example["left_hidden"])  # type: ignore[arg-type]
        right_hidden.append(example["right_hidden"])  # type: ignore[arg-type]
        entropy.append(
            entropy_features(
                example["left_probabilities"],  # type: ignore[arg-type]
                example["right_probabilities"],  # type: ignore[arg-type]
                example["pairs"],  # type: ignore[arg-type]
            )
        )
        targets.append(standardized)
        log_strengths.append(log_strength)
        families.extend([str(example["identifier"])] * pair_count)
        locations.extend([float(location)] * pair_count)
        scales.extend([float(scale)] * pair_count)
    return {
        "left_hidden": torch.cat(left_hidden),
        "right_hidden": torch.cat(right_hidden),
        "entropy_features": torch.cat(entropy),
        "target": torch.cat(targets),
        "log_strength": torch.cat(log_strengths),
        "family": families,
        "family_location": torch.tensor(locations),
        "family_scale": torch.tensor(scales),
    }


def train_entropy_model(
    model: EntropyStrengthRegressor,
    data: dict[str, torch.Tensor | list[str]],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> list[dict[str, float | int]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    features = data["entropy_features"]  # type: ignore[assignment]
    targets = data["target"]  # type: ignore[assignment]
    history = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(targets),
            (min(args.batch_size, len(targets)),),
            generator=generator,
        )
        prediction = model(features[indices].to(device))
        loss = F.smooth_l1_loss(prediction, targets[indices].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
    return history


def train_pair_model(
    model: PairStrengthRegressor,
    data: dict[str, torch.Tensor | list[str]],
    targets: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> list[dict[str, float | int]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    left_hidden = data["left_hidden"]  # type: ignore[assignment]
    right_hidden = data["right_hidden"]  # type: ignore[assignment]
    history = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(targets),
            (min(args.batch_size, len(targets)),),
            generator=generator,
        )
        prediction = model(
            left_hidden[indices].to(device), right_hidden[indices].to(device)
        )
        loss = F.smooth_l1_loss(prediction, targets[indices].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
    return history


@torch.no_grad()
def predictions(
    entropy_model: EntropyStrengthRegressor,
    direct_model: PairStrengthRegressor,
    residual_model: PairStrengthRegressor,
    data: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    entropy_model.eval()
    direct_model.eval()
    residual_model.eval()
    entropy_prediction = entropy_model(
        data["entropy_features"].to(device)  # type: ignore[union-attr]
    ).cpu()
    direct_prediction = direct_model(
        data["left_hidden"].to(device),  # type: ignore[union-attr]
        data["right_hidden"].to(device),  # type: ignore[union-attr]
    ).cpu()
    residual_prediction = residual_model(
        data["left_hidden"].to(device),  # type: ignore[union-attr]
        data["right_hidden"].to(device),  # type: ignore[union-attr]
    ).cpu()
    return {
        "entropy_only": (
            entropy_prediction.numpy(),
            np.zeros(len(entropy_prediction), dtype=np.float32),
        ),
        "direct_pair": (
            direct_prediction.numpy(),
            (direct_prediction - entropy_prediction).numpy(),
        ),
        "entropy_plus_pair": (
            (entropy_prediction + residual_prediction).numpy(),
            residual_prediction.numpy(),
        ),
    }


def top_metrics(
    scores: np.ndarray, target: np.ndarray, fraction: float
) -> tuple[float, float]:
    count = max(1, int(round(len(target) * fraction)))
    if np.ptp(scores) == 0:
        baseline = count / len(target)
        return float(baseline), float(baseline)
    true_top = np.argsort(target)[-count:]
    predicted_top = np.argsort(scores)[-count:]
    labels = np.zeros(len(target), dtype=bool)
    labels[true_top] = True
    recall = len(np.intersect1d(true_top, predicted_top)) / count
    return binary_average_precision(scores, labels), float(recall)


def evaluate_predictions(
    prediction_map: dict[str, tuple[np.ndarray, np.ndarray]],
    data: dict[str, torch.Tensor | list[str]],
    entropy_prediction: np.ndarray,
    top_fraction: float,
) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    target = data["target"].numpy()  # type: ignore[union-attr]
    residual_target = target - entropy_prediction
    families = np.asarray(data["family"])
    summaries = {}
    frames = {}
    for name, (strength_prediction, residual_prediction) in prediction_map.items():
        rows = []
        for family in np.unique(families):
            selected = families == family
            strength_ap, strength_recall = top_metrics(
                strength_prediction[selected], target[selected], top_fraction
            )
            residual_ap, residual_recall = top_metrics(
                residual_prediction[selected],
                residual_target[selected],
                top_fraction,
            )
            rows.append(
                {
                    "family": family,
                    "pairs": int(selected.sum()),
                    "strength_mse": float(
                        np.mean((strength_prediction[selected] - target[selected]) ** 2)
                    ),
                    "strength_spearman": safe_spearman(
                        strength_prediction[selected], target[selected]
                    ),
                    "strength_top_ap": strength_ap,
                    "strength_top_recall": strength_recall,
                    "residual_spearman": safe_spearman(
                        residual_prediction[selected], residual_target[selected]
                    ),
                    "residual_top_ap": residual_ap,
                    "residual_top_recall": residual_recall,
                }
            )
        frame = pd.DataFrame(rows)
        frames[name] = frame
        summaries[name] = {
            column: float(frame[column].mean())
            for column in frame.columns
            if column not in {"family", "pairs"}
        }
    return summaries, frames


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    args.target_normalization = "none"
    mutation_values = [int(value) for value in args.mutation_states.split(",")]
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    teacher, background = load_frozen_models(args, device)
    frame = pd.read_csv(args.representations / "families.csv")
    train_frame = frame[frame["role"] == "train"].reset_index(drop=True)
    eval_frame = frame[frame["role"] == "validation"].reset_index(drop=True)
    mutation_states = torch.tensor(mutation_values, device=device)

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
    train_data = flatten_strength_examples(train_examples)
    eval_data = flatten_strength_examples(eval_examples)

    entropy_model = EntropyStrengthRegressor(
        feature_dim=train_data["entropy_features"].shape[-1],  # type: ignore[union-attr]
        mlp_dim=args.mlp_dim,
    )
    feature_dim = train_data["left_hidden"].shape[-1]  # type: ignore[union-attr]
    torch.manual_seed(args.seed + 11)
    direct_model = PairStrengthRegressor(feature_dim, args.pair_dim, args.mlp_dim)
    initial_pair_state = direct_model.state_dict()
    residual_model = PairStrengthRegressor(feature_dim, args.pair_dim, args.mlp_dim)
    residual_model.load_state_dict(initial_pair_state)

    entropy_history = train_entropy_model(
        entropy_model, train_data, args, device, args.seed + 101
    )
    with torch.no_grad():
        train_entropy_prediction = entropy_model(
            train_data["entropy_features"].to(device)  # type: ignore[union-attr]
        ).cpu()
    direct_history = train_pair_model(
        direct_model,
        train_data,
        train_data["target"],  # type: ignore[arg-type]
        args,
        device,
        args.seed + 102,
    )
    residual_history = train_pair_model(
        residual_model,
        train_data,
        train_data["target"] - train_entropy_prediction,  # type: ignore[operator]
        args,
        device,
        args.seed + 103,
    )

    train_prediction_map = predictions(
        entropy_model, direct_model, residual_model, train_data, device
    )
    eval_prediction_map = predictions(
        entropy_model, direct_model, residual_model, eval_data, device
    )
    train_entropy_np = train_prediction_map["entropy_only"][0]
    eval_entropy_np = eval_prediction_map["entropy_only"][0]
    train_summary, _ = evaluate_predictions(
        train_prediction_map,
        train_data,
        train_entropy_np,
        args.top_fraction,
    )
    eval_summary, eval_frames = evaluate_predictions(
        eval_prediction_map,
        eval_data,
        eval_entropy_np,
        args.top_fraction,
    )

    pd.DataFrame(entropy_history).to_csv(
        args.output / "entropy_history.csv", index=False
    )
    pd.DataFrame(direct_history).to_csv(
        args.output / "direct_pair_history.csv", index=False
    )
    pd.DataFrame(residual_history).to_csv(
        args.output / "residual_pair_history.csv", index=False
    )
    for name, frame_value in eval_frames.items():
        frame_value.to_csv(args.output / f"{name}_per_family.csv", index=False)
    torch.save(entropy_model.state_dict(), args.output / "entropy_model.pt")
    torch.save(direct_model.state_dict(), args.output / "direct_pair_model.pt")
    torch.save(residual_model.state_dict(), args.output / "residual_pair_model.pt")

    summary = {
        "seed": args.seed,
        "device": str(device),
        "features": args.features,
        "train_families": len(train_examples),
        "validation_families": len(eval_examples),
        "pairs_per_family": args.pairs_per_family,
        "mutation_states": mutation_values,
        "data_generation_seconds": data_seconds,
        "parameters": {
            "entropy_only": sum(p.numel() for p in entropy_model.parameters()),
            "direct_pair": sum(p.numel() for p in direct_model.parameters()),
            "entropy_plus_pair": sum(
                p.numel() for p in entropy_model.parameters()
            )
            + sum(p.numel() for p in residual_model.parameters()),
        },
        "train": train_summary,
        "validation": eval_summary,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
