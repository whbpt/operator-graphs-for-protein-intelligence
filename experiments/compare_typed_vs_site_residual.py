from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--typed", type=Path, required=True)
    parser.add_argument("--site12", type=Path, required=True)
    parser.add_argument("--site13", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    return parser.parse_args()


def bootstrap(values: np.ndarray, seed: int, iterations: int):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    return values.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    typed = pd.read_csv(args.typed)
    typed["seed"] = typed["seed"].astype(str)
    site_frames = []
    for seed, path in [("20260712", args.site12), ("20260713", args.site13)]:
        frame = pd.read_csv(path)
        frame["seed"] = seed
        site_frames.append(frame)
    site = pd.concat(site_frames, ignore_index=True)
    merged = typed.merge(
        site[["seed", "x_id", "residual_contribution"]],
        on=["seed", "x_id"],
        how="inner",
    ).rename(columns={"residual_contribution": "site_contribution"})
    merged["typed_minus_site"] = (
        merged["interaction_contribution"] - merged["site_contribution"]
    )
    merged.to_csv(args.output / "per_family_comparison.csv", index=False)

    rows = []
    for seed, group in merged.groupby("seed"):
        mean, low, high = bootstrap(
            group["typed_minus_site"].to_numpy(), int(seed), args.bootstrap
        )
        rows.append(
            {
                "seed": seed,
                "typed_contribution": group["interaction_contribution"].mean(),
                "site_contribution": group["site_contribution"].mean(),
                "typed_minus_site": mean,
                "ci_2_5": low,
                "ci_97_5": high,
            }
        )
    family_mean = merged.groupby("x_id").mean(numeric_only=True)
    mean, low, high = bootstrap(
        family_mean["typed_minus_site"].to_numpy(), 20260714, args.bootstrap
    )
    rows.append(
        {
            "seed": "family-mean",
            "typed_contribution": family_mean[
                "interaction_contribution"
            ].mean(),
            "site_contribution": family_mean["site_contribution"].mean(),
            "typed_minus_site": mean,
            "ci_2_5": low,
            "ci_97_5": high,
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    seeds = ["20260712", "20260713", "family-mean"]
    ordered = summary.set_index("seed").loc[seeds]
    positions = np.arange(len(seeds))
    width = 0.35
    axes[0].bar(
        positions - width / 2,
        ordered["typed_contribution"],
        width,
        label="Typed pair",
        color="#d1495b",
    )
    axes[0].bar(
        positions + width / 2,
        ordered["site_contribution"],
        width,
        label="Site residual",
        color="#2a9d8f",
    )
    axes[0].set_xticks(positions, seeds, rotation=20)
    axes[0].set_title("Residual contribution")
    axes[0].set_ylabel("Total CE - background CE")
    axes[0].legend(frameon=False, fontsize=8)
    error_low = ordered["typed_minus_site"] - ordered["ci_2_5"]
    error_high = ordered["ci_97_5"] - ordered["typed_minus_site"]
    axes[1].bar(
        seeds,
        ordered["typed_minus_site"],
        yerr=np.stack([error_low, error_high]),
        color=["#6b7280", "#d1495b", "#8f5d9f"],
        capsize=4,
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_title("Typed minus site-only")
    axes[1].set_ylabel("Cross-entropy difference")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output / "typed_vs_site_residual.png", dpi=200)
    plt.close(figure)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
