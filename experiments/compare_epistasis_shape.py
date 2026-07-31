from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_NAMES = ["site_only", "unprojected_pair", "projected_pair"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_runs(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_frames = []
    summaries = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        seed = int(summary["seed"])
        for model in MODEL_NAMES:
            frame = pd.read_csv(run / f"{model}_per_family.csv")
            frame["seed"] = seed
            frame["model"] = model
            family_frames.append(frame)
            validation = summary["models"][model]["validation"]
            summaries.append(
                {
                    "seed": seed,
                    "model": model,
                    "explained_fraction": validation["explained_fraction"],
                    "correlation": validation["correlation"],
                    "gauge_error": validation["max_gauge_error"],
                    "output_scale": summary["models"][model]["output_scale"],
                }
            )
    return pd.concat(family_frames, ignore_index=True), pd.DataFrame(summaries)


def clustered_bootstrap(
    family_frame: pd.DataFrame,
    left_model: str,
    right_model: str,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    pivot = family_frame.pivot_table(
        index=["family", "seed"], columns="model", values="mse"
    )
    paired = (pivot[right_model] - pivot[left_model]).unstack("seed")
    paired = paired.dropna()
    family_effect = paired.mean(axis=1).to_numpy()
    draws = rng.choice(
        family_effect,
        size=(samples, len(family_effect)),
        replace=True,
    ).mean(axis=1)
    return {
        "comparison": f"{left_model}_improvement_over_{right_model}",
        "families": int(len(family_effect)),
        "mean_mse_improvement": float(family_effect.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0)),
    }


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    labels = {
        "site_only": "Site-only",
        "unprojected_pair": "Pair",
        "projected_pair": "Projected pair",
    }
    colors = {
        "site_only": "#777777",
        "unprojected_pair": "#d95f02",
        "projected_pair": "#1b9e77",
    }
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(MODEL_NAMES))
    width = 0.32
    seeds = sorted(summary.seed.unique())
    for seed_index, seed in enumerate(seeds):
        values = [
            float(
                summary[(summary.seed == seed) & (summary.model == model)][
                    "explained_fraction"
                ].iloc[0]
            )
            for model in MODEL_NAMES
        ]
        axis.bar(
            x + (seed_index - (len(seeds) - 1) / 2) * width,
            values,
            width=width,
            color=[colors[model] for model in MODEL_NAMES],
            alpha=0.65 + 0.25 * seed_index,
            edgecolor="black",
            linewidth=0.5,
            label=f"seed {seed}",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, [labels[model] for model in MODEL_NAMES])
    axis.set_ylabel("Held-out-family explained fraction")
    axis.set_title("Epistasis shape reconstruction after family-scale removal")
    axis.legend(frameon=False, fontsize=8)
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
        clustered_bootstrap(
            family_frame,
            "unprojected_pair",
            "site_only",
            args.bootstrap,
            rng,
        ),
        clustered_bootstrap(
            family_frame,
            "projected_pair",
            "site_only",
            args.bootstrap,
            rng,
        ),
        clustered_bootstrap(
            family_frame,
            "projected_pair",
            "unprojected_pair",
            args.bootstrap,
            rng,
        ),
    ]
    summary.to_csv(args.output / "seed_summary.csv", index=False)
    family_frame.to_csv(args.output / "per_family_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        args.output / "paired_bootstrap.csv", index=False
    )
    plot_summary(summary, args.output / "epistasis_shape_replication.png")
    result = {
        "runs": [str(run) for run in args.runs],
        "seed_summary": summary.to_dict(orient="records"),
        "comparisons": comparisons,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
