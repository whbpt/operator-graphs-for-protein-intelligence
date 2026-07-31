from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        frame = pd.read_csv(run / "adapted_topk_per_family.csv")
        frame["seed"] = seed
        frame["interaction_improvement"] = (
            frame["background_cross_entropy"] - frame["cross_entropy"]
        )
        frames.append(frame)
        for stage in [
            "initial_background",
            "trained_background",
            "converted_topk",
            "adapted_topk",
        ]:
            row = {"seed": seed, "stage": stage}
            row.update(summary[stage])
            summaries.append(row)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def clustered_bootstrap(
    frame: pd.DataFrame, samples: int, rng: np.random.Generator
) -> dict[str, float | int]:
    pivot = frame.pivot_table(
        index="family", columns="seed", values="interaction_improvement"
    ).dropna()
    family_effect = pivot.mean(axis=1).to_numpy()
    draws = rng.choice(
        family_effect,
        size=(samples, len(family_effect)),
        replace=True,
    ).mean(axis=1)
    return {
        "families": int(len(family_effect)),
        "mean_interaction_improvement": float(family_effect.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0)),
    }


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    adapted = summary[summary.stage == "adapted_topk"].sort_values("seed")
    figure, axes = plt.subplots(1, 3, figsize=(8.0, 3.0))
    seeds = adapted.seed.astype(str).tolist()
    contribution = (
        adapted.background_cross_entropy - adapted.cross_entropy
    ).to_numpy()
    axes[0].bar(seeds, contribution, color=["#6baed6", "#2171b5"])
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Interaction CE gain")
    axes[1].bar(
        seeds, adapted.index_spearman, color=["#74c476", "#238b45"]
    )
    axes[1].set_title("Index Spearman")
    axes[2].bar(
        seeds, adapted.shape_correlation, color=["#fd8d3c", "#d94801"]
    )
    axes[2].set_title("Shape correlation")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
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
    comparison = clustered_bootstrap(
        family_frame, args.bootstrap, np.random.default_rng(args.seed)
    )
    family_frame.to_csv(args.output / "per_family_metrics.csv", index=False)
    summary.to_csv(args.output / "stage_summary.csv", index=False)
    pd.DataFrame([comparison]).to_csv(
        args.output / "paired_bootstrap.csv", index=False
    )
    plot_summary(summary, args.output / "dual_stream_demo_replication.png")
    result = {
        "runs": [str(run) for run in args.runs],
        "interaction_comparison": comparison,
        "stage_summary": summary.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
