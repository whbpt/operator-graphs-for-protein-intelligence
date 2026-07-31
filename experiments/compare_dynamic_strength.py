from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["strength_mse", "strength_spearman", "strength_top_ap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--dynamic-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_source(runs: list[Path], source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_frames = []
    summary_rows = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        seed = int(summary["seed"])
        for model in ["entropy_only", "direct_pair"]:
            frame = pd.read_csv(run / f"{model}_per_family.csv")
            frame["seed"] = seed
            frame["source"] = source
            frame["model"] = model
            family_frames.append(frame)
            row = {"seed": seed, "source": source, "model": model}
            row.update(summary["validation"][model])
            summary_rows.append(row)
    return pd.concat(family_frames, ignore_index=True), pd.DataFrame(summary_rows)


def clustered_comparison(
    frame: pd.DataFrame,
    left: tuple[str, str],
    right: tuple[str, str],
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    keyed = frame.copy()
    keyed["system"] = keyed["source"] + ":" + keyed["model"]
    pivot = keyed.pivot_table(
        index=["family", "seed"], columns="system", values=metric
    )
    left_key = f"{left[0]}:{left[1]}"
    right_key = f"{right[0]}:{right[1]}"
    if metric == "strength_mse":
        effect = pivot[right_key] - pivot[left_key]
    else:
        effect = pivot[left_key] - pivot[right_key]
    family_effect = effect.unstack("seed").dropna().mean(axis=1).to_numpy()
    draws = rng.choice(
        family_effect,
        size=(samples, len(family_effect)),
        replace=True,
    ).mean(axis=1)
    return {
        "comparison": f"{left_key}_over_{right_key}",
        "metric": metric,
        "families": int(len(family_effect)),
        "mean_improvement": float(family_effect.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0)),
    }


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    systems = [
        ("static", "entropy_only", "Entropy only", "#777777"),
        ("static", "direct_pair", "Static pair", "#d95f02"),
        ("dynamic", "direct_pair", "Dynamic pair", "#1b9e77"),
    ]
    seeds = sorted(summary.seed.unique())
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    x = np.arange(len(systems))
    width = 0.32
    for axis, metric, title in zip(
        axes,
        ["strength_spearman", "strength_top_ap"],
        ["Within-family Spearman", "Top-10% average precision"],
    ):
        for seed_index, seed in enumerate(seeds):
            values = []
            for source, model, _, _ in systems:
                row = summary[
                    (summary.seed == seed)
                    & (summary.source == source)
                    & (summary.model == model)
                ]
                values.append(float(row[metric].iloc[0]))
            axis.bar(
                x + (seed_index - 0.5) * width,
                values,
                width,
                color=[system[3] for system in systems],
                alpha=0.65 + 0.25 * seed_index,
                edgecolor="black",
                linewidth=0.5,
            )
        axis.set_xticks(x, [system[2] for system in systems], rotation=18)
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
    static_family, static_summary = load_source(args.static_runs, "static")
    dynamic_family, dynamic_summary = load_source(args.dynamic_runs, "dynamic")
    family_frame = pd.concat([static_family, dynamic_family], ignore_index=True)
    summary = pd.concat([static_summary, dynamic_summary], ignore_index=True)
    rng = np.random.default_rng(args.seed)
    comparisons = []
    for metric in METRICS:
        comparisons.append(
            clustered_comparison(
                family_frame,
                ("dynamic", "direct_pair"),
                ("static", "direct_pair"),
                metric,
                args.bootstrap,
                rng,
            )
        )
        comparisons.append(
            clustered_comparison(
                family_frame,
                ("dynamic", "direct_pair"),
                ("static", "entropy_only"),
                metric,
                args.bootstrap,
                rng,
            )
        )
    family_frame.to_csv(args.output / "per_family_metrics.csv", index=False)
    summary.to_csv(args.output / "seed_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        args.output / "paired_bootstrap.csv", index=False
    )
    plot_summary(summary, args.output / "dynamic_strength_replication.png")
    result = {
        "static_runs": [str(path) for path in args.static_runs],
        "dynamic_runs": [str(path) for path in args.dynamic_runs],
        "comparisons": comparisons,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
