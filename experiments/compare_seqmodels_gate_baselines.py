from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    return parser.parse_args()


def load_runs(results: Path) -> pd.DataFrame:
    specifications = [
        (
            "Typed categorical",
            "seqmodels_factor_head_rank32_v3_prior_centered",
            {
                "teacher_ap": "gate_teacher_average_precision",
                "teacher_auc": "gate_teacher_roc_auc",
                "contact_p_at_l": "gate_long_p_at_l",
                "leading_mode_fraction": "score_leading_mode_fraction",
            },
        ),
        (
            "Bilinear indexer",
            "seqmodels_gate_bilinear_v1",
            {
                "teacher_ap": "teacher_average_precision",
                "teacher_auc": "teacher_roc_auc",
                "contact_p_at_l": "centered_long_p_at_l",
                "leading_mode_fraction": "centered_leading_mode_fraction",
            },
        ),
        (
            "Pair MLP indexer",
            "seqmodels_gate_pair_mlp_v1",
            {
                "teacher_ap": "teacher_average_precision",
                "teacher_auc": "teacher_roc_auc",
                "contact_p_at_l": "centered_long_p_at_l",
                "leading_mode_fraction": "centered_leading_mode_fraction",
            },
        ),
    ]
    frames = []
    for model, directory, mapping in specifications:
        source = pd.read_csv(results / directory / "per_family_metrics.csv")
        selected = source[["x_id", "role", *mapping.values()]].rename(
            columns={value: key for key, value in mapping.items()}
        )
        selected["model"] = model
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def paired_bootstrap(
    pivot: pd.DataFrame,
    left: str,
    right: str,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float, float]:
    difference = (pivot[left] - pivot[right]).to_numpy()
    indices = rng.integers(0, len(difference), size=(iterations, len(difference)))
    means = difference[indices].mean(axis=1)
    return (
        float(difference.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_runs(args.results)
    frame.to_csv(args.output / "gate_baseline_per_family.csv", index=False)
    test = frame[frame["role"] == "test"]
    summary = test.groupby("model").agg(
        families=("x_id", "count"),
        teacher_ap=("teacher_ap", "mean"),
        teacher_auc=("teacher_auc", "mean"),
        contact_p_at_l=("contact_p_at_l", "mean"),
        leading_mode_fraction=("leading_mode_fraction", "mean"),
    )
    summary.to_csv(args.output / "gate_baseline_summary.csv")

    rng = np.random.default_rng(args.seed)
    comparisons = []
    models = list(summary.index)
    for metric in ["teacher_ap", "teacher_auc", "contact_p_at_l"]:
        pivot = test.pivot(index="x_id", columns="model", values=metric)
        for left, right in [
            ("Bilinear indexer", "Typed categorical"),
            ("Pair MLP indexer", "Typed categorical"),
            ("Pair MLP indexer", "Bilinear indexer"),
        ]:
            mean, low, high = paired_bootstrap(
                pivot, left, right, rng, args.bootstrap
            )
            comparisons.append(
                {
                    "metric": metric,
                    "left": left,
                    "right": right,
                    "mean_difference": mean,
                    "ci_2_5": low,
                    "ci_97_5": high,
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(args.output / "gate_baseline_paired_bootstrap.csv", index=False)

    order = ["Typed categorical", "Bilinear indexer", "Pair MLP indexer"]
    colors = ["#d1495b", "#2a9d8f", "#8f5d9f"]
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for axis, metric, title in zip(
        axes,
        ["teacher_auc", "teacher_ap", "contact_p_at_l"],
        ["Sparse-teacher ROC AUC", "Sparse-teacher AP", "Held-out contact P@L"],
    ):
        values = summary.loc[order, metric]
        axis.bar(order, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=28)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(0.02643, color="black", linestyle="--", linewidth=1)
    figure.tight_layout()
    figure.savefig(args.output / "gate_baseline_comparison.png", dpi=200)
    plt.close(figure)

    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print()
    print(comparison_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
