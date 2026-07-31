from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.evaluate_lsh_candidate_router import teacher_args
from experiments.train_epistasis_identifiability import load_frozen_models
from transformer_disentanglement.block_structure import (
    block_concentration_metrics,
    top_position_fragmentation,
)
from transformer_disentanglement.epistasis import weighted_double_center
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import load_seqmodels_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--families", type=int, default=24)
    parser.add_argument("--targets-per-family", type=int, default=2)
    parser.add_argument("--min-separation", type=int)
    parser.add_argument("--teacher-batch-size", type=int, default=128)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument(
        "--block-modes", nargs="+", choices=["partition", "sliding"],
        default=["partition", "sliding"]
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


@torch.no_grad()
def conditional_strengths(
    sequence: np.ndarray,
    target: int,
    min_separation: int,
    teacher: torch.nn.Module,
    background: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    valid_positions = np.flatnonzero(sequence < 20)
    contexts = valid_positions[np.abs(valid_positions - target) >= min_separation]
    base = torch.from_numpy(sequence.astype(np.int64)).to(device)
    base[target] = 21
    base_tokens = base[None]
    background_probabilities = background(base_tokens)["logits"][0].softmax(dim=-1)
    amino_acids = torch.arange(20, device=device)
    variants = base_tokens.repeat(len(contexts) * 20, 1)
    rows = torch.arange(len(variants), device=device)
    context_tensor = torch.from_numpy(contexts).to(device)
    variants[rows, context_tensor.repeat_interleave(20)] = amino_acids.repeat(
        len(contexts)
    )
    variant_logits = []
    for start in range(0, len(variants), batch_size):
        variant_logits.append(teacher(variants[start : start + batch_size])["logits"][:, target])
    blocks = torch.cat(variant_logits).reshape(len(contexts), 20, 20).transpose(1, 2)
    left = background_probabilities[target][None].expand(len(contexts), -1)
    right = background_probabilities[context_tensor]
    projected = weighted_double_center(blocks, left, right)
    strengths = projected.square().mean(dim=(-2, -1)).sqrt().cpu().numpy()
    return contexts, strengths


def positive_excess_energy(strengths: np.ndarray) -> np.ndarray:
    energy = np.square(strengths)
    return np.maximum(energy - np.median(energy), 0.0)


def clustered_bootstrap(
    frame: pd.DataFrame,
    columns: list[str],
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    families = [part[columns].to_numpy(float) for _, part in frame.groupby("family")]
    draws = np.empty((samples, len(columns)), dtype=float)
    for sample in range(samples):
        means = []
        for family_index in rng.integers(len(families), size=len(families)):
            values = families[family_index]
            rows = rng.integers(len(values), size=len(values))
            means.append(values[rows].mean(axis=0))
        draws[sample] = np.mean(means, axis=0)
    return draws


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    observed = summary[summary.signal == "excess_energy"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = {32: "#287271", 64: "#D97706"}
    styles = {"partition": "--", "sliding": "-"}
    markers = {"partition": "s", "sliding": "o"}
    for (block_mode, budget), frame in observed.groupby(["block_mode", "budget"]):
        frame = frame.sort_values("block_size")
        label = f"{block_mode}, budget {int(budget)}"
        for axis, metric in zip(
            axes, ["block_to_token_mass", "excess_over_permuted"]
        ):
            low = frame[metric] - frame[f"{metric}_ci_low"]
            high = frame[f"{metric}_ci_high"] - frame[metric]
            axis.errorbar(
                frame.block_size,
                frame[metric],
                yerr=np.vstack([low, high]),
                color=colors[int(budget)],
                linestyle=styles[block_mode],
                marker=markers[block_mode],
                linewidth=1.8,
                capsize=3,
                label=label,
            )
    axes[0].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Block mass / oracle token mass")
    axes[0].set_xlabel("Block width (residues)")
    widths = sorted(observed.block_size.unique())
    axes[0].set_xticks(widths)
    axes[0].set_ylim(0, 1.05)
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Observed block mass / permuted null")
    axes[1].set_xlabel("Block width (residues)")
    axes[1].set_xticks(widths)
    axes[1].set_ylim(0.82, 1.06)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Conditional-response block concentration")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    config = json.loads((args.run / "run.json").read_text())
    benchmark = Path(config["benchmark"])
    representations = Path(config["representations"])
    frozen_args = teacher_args(
        config,
        argparse.Namespace(benchmark=benchmark, representations=representations),
    )
    frozen_args.teacher_batch_size = args.teacher_batch_size
    device = choose_device(args.device)
    teacher, background = load_frozen_models(frozen_args, device)
    family_frame = pd.read_csv(representations / "families.csv")
    validation = family_frame[family_frame.role == "validation"].head(args.families)
    min_separation = args.min_separation or int(config["min_separation"])
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    strength_rows: list[dict[str, float | int | str]] = []
    for family_index, row in enumerate(validation.itertuples(index=False), start=1):
        family = load_seqmodels_family(benchmark, row.file, row.x_id)
        sequence = family.msa[0].astype(np.int64)
        valid_targets = np.flatnonzero(sequence < 20)
        targets = rng.choice(
            valid_targets,
            size=min(args.targets_per_family, len(valid_targets)),
            replace=False,
        )
        for target in targets:
            contexts, strengths = conditional_strengths(
                sequence,
                int(target),
                min_separation,
                teacher,
                background,
                args.teacher_batch_size,
                device,
            )
            valid = np.zeros(len(sequence), dtype=bool)
            valid[contexts] = True
            energy = np.zeros(len(sequence), dtype=float)
            energy[contexts] = np.square(strengths)
            excess = np.zeros(len(sequence), dtype=float)
            excess[contexts] = positive_excess_energy(strengths)
            for position, strength in zip(contexts, strengths):
                strength_rows.append(
                    {
                        "family": row.x_id,
                        "target": int(target),
                        "position": int(position),
                        "strength": float(strength),
                    }
                )
            for signal_name, signal in (("energy", energy), ("excess_energy", excess)):
                for budget in args.budgets:
                    fragmentation = top_position_fragmentation(signal, valid, budget)
                    for block_size in args.block_sizes:
                        if budget < block_size:
                            continue
                        for block_mode in args.block_modes:
                            metrics = block_concentration_metrics(
                                signal, valid, block_size, budget, block_mode
                            )
                            null_mass = []
                            valid_values = signal[valid].copy()
                            for _ in range(args.permutations):
                                permuted = np.zeros_like(signal)
                                permuted[valid] = rng.permutation(valid_values)
                                null_mass.append(
                                    block_concentration_metrics(
                                        permuted,
                                        valid,
                                        block_size,
                                        budget,
                                        block_mode,
                                    )["block_mass_fraction"]
                                )
                            rows.append(
                                {
                                    "family": row.x_id,
                                    "target": int(target),
                                    "signal": signal_name,
                                    "block_mode": block_mode,
                                    "block_size": block_size,
                                    "budget": budget,
                                    **metrics,
                                    **fragmentation,
                                    "permuted_block_mass_fraction": float(np.mean(null_mass)),
                                    "excess_over_permuted": metrics["block_mass_fraction"]
                                    / max(float(np.mean(null_mass)), 1e-12),
                                }
                            )
        print(
            f"processed {family_index}/{len(validation)} families: {row.x_id}",
            flush=True,
        )
    data = pd.DataFrame(rows)
    metric_columns = [
        "token_mass_fraction",
        "block_mass_fraction",
        "block_to_token_mass",
        "top_token_recall",
        "component_fraction",
        "span_ratio",
        "permuted_block_mass_fraction",
        "excess_over_permuted",
    ]
    summary_rows = []
    bootstrap_rng = np.random.default_rng(args.seed + 1)
    for keys, frame in data.groupby(
        ["signal", "block_mode", "block_size", "budget"]
    ):
        draws = clustered_bootstrap(frame, metric_columns, args.bootstrap, bootstrap_rng)
        estimates = frame.groupby("family")[metric_columns].mean().mean(axis=0)
        result: dict[str, float | int | str] = {
            "signal": keys[0],
            "block_mode": keys[1],
            "block_size": int(keys[2]),
            "budget": int(keys[3]),
            "families": int(frame.family.nunique()),
            "examples": int(len(frame)),
        }
        for index, metric in enumerate(metric_columns):
            result[metric] = float(estimates[metric])
            result[f"{metric}_ci_low"] = float(np.quantile(draws[:, index], 0.025))
            result[f"{metric}_ci_high"] = float(np.quantile(draws[:, index], 0.975))
        summary_rows.append(result)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["signal", "block_mode", "budget", "block_size"]
    )
    data.to_csv(args.output / "per_example.csv", index=False)
    pd.DataFrame(strength_rows).to_csv(args.output / "strengths.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    plot_summary(summary, args.output / "block_concentration.png")
    result = {
        "configuration": vars(args),
        "device": str(device),
        "families": int(data.family.nunique()),
        "examples": int(data[["family", "target"]].drop_duplicates().shape[0]),
        "summary": summary.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
