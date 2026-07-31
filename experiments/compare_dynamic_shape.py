from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--dynamic-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_source(runs: list[Path], source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summaries = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        seed = int(summary["seed"])
        for model in ["site_only", "projected_pair"]:
            frame = pd.read_csv(run / f"{model}_per_family.csv")
            frame["seed"] = seed
            frame["source"] = source
            frame["model"] = model
            frames.append(frame)
            metrics = summary["models"][model]["validation"]
            summaries.append(
                {
                    "seed": seed,
                    "source": source,
                    "model": model,
                    "explained_fraction": metrics["explained_fraction"],
                    "correlation": metrics["correlation"],
                    "gauge_error": metrics["max_gauge_error"],
                }
            )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def bootstrap(
    frame: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    keyed = frame.copy()
    keyed["system"] = keyed["source"] + ":" + keyed["model"]
    pivot = keyed.pivot_table(
        index=["family", "seed"], columns="system", values=metric
    )
    effect = (
        pivot[right] - pivot[left]
        if metric == "mse"
        else pivot[left] - pivot[right]
    )
    family_effect = effect.unstack("seed").dropna().mean(axis=1).to_numpy()
    draws = rng.choice(
        family_effect,
        size=(samples, len(family_effect)),
        replace=True,
    ).mean(axis=1)
    return {
        "comparison": f"{left}_over_{right}",
        "metric": metric,
        "families": int(len(family_effect)),
        "mean_improvement": float(family_effect.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0)),
    }


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    systems = [
        ("static", "site_only", "Site-only", "#777777"),
        ("static", "projected_pair", "Stable value", "#d95f02"),
        ("dynamic", "projected_pair", "Dynamic value", "#1b9e77"),
    ]
    seeds = sorted(summary.seed.unique())
    x = np.arange(len(systems))
    width = 0.32
    figure, axis = plt.subplots(figsize=(6.2, 3.4))
    for seed_index, seed in enumerate(seeds):
        values = []
        for source, model, _, _ in systems:
            row = summary[
                (summary.seed == seed)
                & (summary.source == source)
                & (summary.model == model)
            ]
            values.append(float(row.explained_fraction.iloc[0]))
        axis.bar(
            x + (seed_index - 0.5) * width,
            values,
            width,
            color=[system[3] for system in systems],
            alpha=0.65 + 0.25 * seed_index,
            edgecolor="black",
            linewidth=0.5,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, [system[2] for system in systems])
    axis.set_ylabel("Held-out-family explained fraction")
    axis.set_title("Static versus task-conditioned interaction value")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    static_frame, static_summary = load_source(args.static_runs, "static")
    dynamic_frame, dynamic_summary = load_source(args.dynamic_runs, "dynamic")
    frame = pd.concat([static_frame, dynamic_frame], ignore_index=True)
    summary = pd.concat([static_summary, dynamic_summary], ignore_index=True)
    rng = np.random.default_rng(args.seed)
    comparisons = []
    for metric in ["mse", "correlation"]:
        comparisons.append(
            bootstrap(
                frame,
                "dynamic:projected_pair",
                "static:projected_pair",
                metric,
                args.bootstrap,
                rng,
            )
        )
        comparisons.append(
            bootstrap(
                frame,
                "static:projected_pair",
                "static:site_only",
                metric,
                args.bootstrap,
                rng,
            )
        )
    frame.to_csv(args.output / "per_family_metrics.csv", index=False)
    summary.to_csv(args.output / "seed_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        args.output / "paired_bootstrap.csv", index=False
    )
    plot_summary(summary, args.output / "dynamic_shape_replication.png")
    result = {"comparisons": comparisons}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
