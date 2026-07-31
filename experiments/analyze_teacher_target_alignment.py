from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.compare_conditional_response_demo import hierarchical_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def bootstrap_rows(
    frame: pd.DataFrame,
    metrics: list[str],
    samples: int,
    rng: np.random.Generator,
    analysis: str,
) -> list[dict[str, float | int | str]]:
    draws = hierarchical_bootstrap(frame, metrics, samples, rng)
    family = frame.groupby(["seed", "family"], as_index=False)[metrics].mean()
    estimates = family.groupby("seed")[metrics].mean().mean(axis=0)
    rows = []
    for index, metric in enumerate(metrics):
        rows.append(
            {
                "analysis": analysis,
                "metric": metric,
                "estimate": float(estimates[metric]),
                "ci_low": float(np.quantile(draws[:, index], 0.025)),
                "ci_high": float(np.quantile(draws[:, index], 0.975)),
                "probability_positive": float(np.mean(draws[:, index] > 0)),
                "seeds": int(frame.seed.nunique()),
                "examples": int(len(frame)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    frames = []
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        frame = pd.read_csv(run / "per_example.csv")
        frame["seed"] = int(summary["model_seed"])
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    effect_metrics = [
        "nonlinear_minus_additive_ce_gain",
        "nonlinear_minus_additive_teacher_kl_gain",
    ]
    rng = np.random.default_rng(args.seed)
    rows = bootstrap_rows(
        data,
        ["teacher_ce_gain", "teacher_better_than_background"] + effect_metrics,
        args.bootstrap,
        rng,
        "all_targets",
    )
    good = data[data.teacher_better_than_background == 1].copy()
    bad = data[data.teacher_better_than_background == 0].copy()
    rows.extend(
        bootstrap_rows(good, effect_metrics, args.bootstrap, rng, "teacher_better")
    )
    rows.extend(
        bootstrap_rows(bad, effect_metrics, args.bootstrap, rng, "teacher_worse")
    )

    grouped = data.groupby(
        ["seed", "family", "teacher_better_than_background"], as_index=False
    )[effect_metrics].mean()
    good_family = grouped[grouped.teacher_better_than_background == 1].drop(
        columns="teacher_better_than_background"
    )
    bad_family = grouped[grouped.teacher_better_than_background == 0].drop(
        columns="teacher_better_than_background"
    )
    paired = good_family.merge(
        bad_family,
        on=["seed", "family"],
        suffixes=("_good", "_bad"),
    )
    difference_metrics = []
    for metric in effect_metrics:
        column = f"teacher_good_minus_bad_{metric}"
        paired[column] = paired[f"{metric}_good"] - paired[f"{metric}_bad"]
        difference_metrics.append(column)
    rows.extend(
        bootstrap_rows(
            paired,
            difference_metrics,
            args.bootstrap,
            rng,
            "teacher_good_minus_bad",
        )
    )

    correlations = []
    for seed, frame in data.groupby("seed"):
        for metric in effect_metrics:
            correlations.append(
                {
                    "seed": int(seed),
                    "metric": metric,
                    "spearman": float(
                        spearmanr(frame.teacher_ce_gain, frame[metric]).statistic
                    ),
                }
            )
    data["teacher_gain_quartile"] = data.groupby("seed").teacher_ce_gain.transform(
        lambda values: pd.qcut(values, 4, labels=False, duplicates="drop")
    )
    quartiles = data.groupby("teacher_gain_quartile", as_index=False)[
        ["teacher_ce_gain"] + effect_metrics
    ].mean()
    comparison = pd.DataFrame(rows)
    data.to_csv(args.output / "per_example.csv", index=False)
    paired.to_csv(args.output / "paired_family_strata.csv", index=False)
    comparison.to_csv(args.output / "hierarchical_bootstrap.csv", index=False)
    pd.DataFrame(correlations).to_csv(args.output / "seed_correlations.csv", index=False)
    quartiles.to_csv(args.output / "teacher_gain_quartiles.csv", index=False)
    result = {
        "runs": [str(run) for run in args.runs],
        "bootstrap": args.bootstrap,
        "comparison": comparison.to_dict(orient="records"),
        "correlations": correlations,
        "quartiles": quartiles.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
