from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.train_evolutionary_conditional_demo import build_evolutionary_examples
from transformer_disentanglement.demo_language_models import LocalProteinLM
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--background-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets-per-family", type=int, default=12)
    parser.add_argument("--contexts-per-target", type=int, default=16)
    parser.add_argument("--min-separation", type=int, default=6)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--conditional-temperature", type=float, default=0.5)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--sampling-seeds", type=int, nargs="+", default=[20260712, 20260713, 20260714]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=20000)
    return parser.parse_args()


@torch.no_grad()
def evaluate_lambdas(
    background: torch.nn.Module,
    examples: list[dict[str, torch.Tensor | str]],
    lambdas: list[float],
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for example in examples:
        tokens = example["base_tokens"][None].to(device)  # type: ignore[index,union-attr]
        position = int(example["target_position"])
        target = example["true_target"].to(device)  # type: ignore[union-attr]
        message = example["teacher_message"].to(device)  # type: ignore[union-attr]
        background_logits = background(tokens)["logits"][0, position]
        background_ce = F.cross_entropy(background_logits[None], target[None])
        identifier = str(example["identifier"])
        for scale in lambdas:
            logits = background_logits + scale * message
            ce = F.cross_entropy(logits[None], target[None])
            rows.append(
                {
                    "example": identifier,
                    "family": identifier.split(":", 1)[0],
                    "lambda": scale,
                    "ce_gain": float(background_ce - ce),
                    "message_rms": float(message.square().mean().sqrt()),
                }
            )
    return pd.DataFrame(rows)


def family_sampling_bootstrap(
    frame: pd.DataFrame,
    columns: list[str],
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cluster by family; sampling repetitions remain nested within family."""
    families = [
        family_frame[columns].to_numpy(dtype=float)
        for _, family_frame in frame.groupby("family")
    ]
    draws = np.empty((samples, len(columns)), dtype=float)
    for sample in range(samples):
        family_means = []
        for family_index in rng.integers(len(families), size=len(families)):
            values = families[family_index]
            row_indices = rng.integers(len(values), size=len(values))
            family_means.append(values[row_indices].mean(axis=0))
        draws[sample] = np.mean(family_means, axis=0)
    return draws


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    device = choose_device(args.device)
    background = LocalProteinLM(hidden_dim=64, layers=1).to(device)
    background.load_state_dict(
        torch.load(args.background_checkpoint, map_location=device, weights_only=True)
    )
    background.eval()
    families = pd.read_csv(args.representations / "families.csv")
    validation = families[families.role == "validation"].reset_index(drop=True)
    test = families[families.role == "test"].reset_index(drop=True)
    build_args = argparse.Namespace(
        benchmark=args.benchmark,
        targets_per_family=args.targets_per_family,
        contexts_per_target=args.contexts_per_target,
        min_separation=args.min_separation,
        prior_weight=args.prior_weight,
        conditional_temperature=args.conditional_temperature,
        message_scale=1.0,
    )
    validation_rows = []
    test_rows = []
    selections = []
    for seed in args.sampling_seeds:
        validation_examples = build_evolutionary_examples(
            validation,
            len(validation),
            build_args,
            background,
            np.random.default_rng(seed + 3000),
            device,
        )
        validation_frame = evaluate_lambdas(
            background, validation_examples, args.lambdas, device
        )
        validation_frame["seed"] = seed
        means = validation_frame.groupby("lambda").ce_gain.mean()
        selected = float(means.idxmax())
        selections.append(
            {
                "seed": seed,
                "selected_lambda": selected,
                "validation_ce_gain": float(means.loc[selected]),
            }
        )
        validation_rows.append(validation_frame)
        test_examples = build_evolutionary_examples(
            test,
            len(test),
            build_args,
            background,
            np.random.default_rng(seed + 4000),
            device,
        )
        test_frame = evaluate_lambdas(background, test_examples, [selected], device)
        test_frame["seed"] = seed
        test_rows.append(test_frame)
    validation_data = pd.concat(validation_rows, ignore_index=True)
    test_data = pd.concat(test_rows, ignore_index=True)
    draws = family_sampling_bootstrap(
        test_data,
        ["ce_gain", "message_rms"],
        args.bootstrap,
        np.random.default_rng(20260727),
    )
    family = test_data.groupby("family", as_index=False)[
        ["ce_gain", "message_rms"]
    ].mean()
    estimates = family[["ce_gain", "message_rms"]].mean(axis=0)
    comparison = []
    for index, metric in enumerate(["ce_gain", "message_rms"]):
        comparison.append(
            {
                "metric": metric,
                "estimate": float(estimates[metric]),
                "ci_low": float(np.quantile(draws[:, index], 0.025)),
                "ci_high": float(np.quantile(draws[:, index], 0.975)),
                "probability_positive": float(np.mean(draws[:, index] > 0)),
            }
        )
    validation_data.to_csv(args.output / "validation_grid.csv", index=False)
    test_data.to_csv(args.output / "test_per_example.csv", index=False)
    pd.DataFrame(selections).to_csv(args.output / "selected_lambdas.csv", index=False)
    result = {
        "configuration": vars(args),
        "selections": selections,
        "comparison": comparison,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
