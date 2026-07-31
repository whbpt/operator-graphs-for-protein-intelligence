from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ["entropy_only", "direct_pair", "entropy_plus_pair"]
METRICS = ["strength_mse", "strength_spearman", "strength_top_ap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_runs(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summaries = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        seed = int(summary["seed"])
        for model in MODELS:
            frame = pd.read_csv(run / f"{model}_per_family.csv")
            frame["seed"] = seed
            frame["model"] = model
            frames.append(frame)
            row = {"seed": seed, "model": model}
            row.update(summary["validation"][model])
            summaries.append(row)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def paired_bootstrap(
    frame: pd.DataFrame,
    model: str,
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float | str | int]:
    pivot = frame.pivot_table(
        index=["family", "seed"], columns="model", values=metric
    )
    if metric == "strength_mse":
        effect = pivot["entropy_only"] - pivot[model]
    else:
        effect = pivot[model] - pivot["entropy_only"]
    family_effect = effect.unstack("seed").dropna().mean(axis=1).to_numpy()
    draws = rng.choice(
        family_effect,
        size=(samples, len(family_effect)),
        replace=True,
    ).mean(axis=1)
    return {
        "model": model,
        "metric": metric,
        "families": int(len(family_effect)),
        "mean_improvement": float(family_effect.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0)),
    }


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    labels = {
        "entropy_only": "Entropy only",
        "direct_pair": "Direct pair",
        "entropy_plus_pair": "Entropy + pair",
    }
    colors = {
        "entropy_only": "#777777",
        "direct_pair": "#d95f02",
        "entropy_plus_pair": "#1b9e77",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    seeds = sorted(summary.seed.unique())
    x = np.arange(len(MODELS))
    width = 0.32
    for axis, metric, title in zip(
        axes,
        ["strength_spearman", "strength_top_ap"],
        ["Within-family Spearman", "Top-10% average precision"],
    ):
        for seed_index, seed in enumerate(seeds):
            values = [
                float(
                    summary[(summary.seed == seed) & (summary.model == model)][
                        metric
                    ].iloc[0]
                )
                for model in MODELS
            ]
            axis.bar(
                x + (seed_index - 0.5) * width,
                values,
                width,
                color=[colors[model] for model in MODELS],
                alpha=0.65 + 0.25 * seed_index,
                edgecolor="black",
                linewidth=0.5,
            )
        axis.set_xticks(x, [labels[model] for model in MODELS], rotation=20)
        axis.set_title(title)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    family_frame, summary = load_runs(args.runs)
    rng = np.random.default_rng(args.seed)
    comparisons = [
        paired_bootstrap(family_frame, model, metric, args.bootstrap, rng)
        for model in ["direct_pair", "entropy_plus_pair"]
        for metric in METRICS
    ]
    family_frame.to_csv(args.output / "per_family_metrics.csv", index=False)
    summary.to_csv(args.output / "seed_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        args.output / "paired_bootstrap.csv", index=False
    )
    plot_summary(summary, args.output / "epistasis_strength_replication.png")
    result = {
        "runs": [str(run) for run in args.runs],
        "seed_summary": summary.to_dict(orient="records"),
        "comparisons": comparisons,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
