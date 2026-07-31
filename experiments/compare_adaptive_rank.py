from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_conditional_response_demo import hierarchical_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--adaptive-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--fixed-label", default="fixed")
    parser.add_argument("--adaptive-label", default="adaptive")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_runs(runs: list[Path], architecture: str) -> pd.DataFrame:
    frames = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        frame = pd.read_csv(run / "adapted_topk_per_example.csv")
        frame["seed"] = int(summary["seed"])
        frame["family"] = frame.example.str.split(":", n=1).str[0]
        frame["architecture"] = architecture
        frame["ce_gain"] = frame.background_cross_entropy - frame.cross_entropy
        frame["teacher_kl_gain"] = (
            frame.background_teacher_kl - frame.teacher_kl
        )
        if "effective_rank" not in frame:
            frame["effective_rank"] = float(summary.get("rank", 8))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def paired_frame(fixed: pd.DataFrame, adaptive: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "seed",
        "family",
        "example",
        "ce_gain",
        "teacher_kl_gain",
        "index_spearman",
        "index_top_ap",
        "shape_correlation",
        "shape_mse",
        "effective_rank",
    ]
    paired = fixed[columns].merge(
        adaptive[columns],
        on=["seed", "family", "example"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    paired["ce_gain_delta"] = paired.ce_gain_candidate - paired.ce_gain_baseline
    paired["teacher_kl_gain_delta"] = (
        paired.teacher_kl_gain_candidate - paired.teacher_kl_gain_baseline
    )
    paired["index_spearman_delta"] = (
        paired.index_spearman_candidate - paired.index_spearman_baseline
    )
    paired["index_top_ap_delta"] = (
        paired.index_top_ap_candidate - paired.index_top_ap_baseline
    )
    paired["shape_correlation_delta"] = (
        paired.shape_correlation_candidate - paired.shape_correlation_baseline
    )
    paired["shape_mse_improvement"] = (
        paired.shape_mse_baseline - paired.shape_mse_candidate
    )
    paired["effective_rank_reduction"] = (
        paired.effective_rank_baseline - paired.effective_rank_candidate
    )
    return paired


def summarize_deltas(
    frame: pd.DataFrame,
    columns: list[str],
    draws: np.ndarray,
) -> pd.DataFrame:
    family_means = frame.groupby(["seed", "family"], as_index=False)[columns].mean()
    estimates = family_means.groupby("seed")[columns].mean().mean(axis=0)
    rows = []
    for index, column in enumerate(columns):
        rows.append(
            {
                "metric": column,
                "estimate": float(estimates[column]),
                "ci_low": float(np.quantile(draws[:, index], 0.025)),
                "ci_high": float(np.quantile(draws[:, index], 0.975)),
                "probability_candidate_better": float(
                    np.mean(draws[:, index] > 0)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    fixed = load_runs(args.fixed_runs, args.fixed_label)
    adaptive = load_runs(args.adaptive_runs, args.adaptive_label)
    paired = paired_frame(fixed, adaptive)
    delta_columns = [
        "ce_gain_delta",
        "teacher_kl_gain_delta",
        "index_spearman_delta",
        "index_top_ap_delta",
        "shape_correlation_delta",
        "shape_mse_improvement",
        "effective_rank_reduction",
    ]
    draws = hierarchical_bootstrap(
        paired, delta_columns, args.bootstrap, np.random.default_rng(args.seed)
    )
    comparison = summarize_deltas(paired, delta_columns, draws)
    absolute = pd.concat([fixed, adaptive], ignore_index=True).groupby(
        ["architecture", "seed"], as_index=False
    )[
        [
            "ce_gain",
            "teacher_kl_gain",
            "index_spearman",
            "index_top_ap",
            "shape_correlation",
            "shape_mse",
            "effective_rank",
        ]
    ].mean()
    paired.to_csv(args.output / "paired_examples.csv", index=False)
    absolute.to_csv(args.output / "absolute_metrics.csv", index=False)
    comparison.to_csv(args.output / "paired_bootstrap.csv", index=False)
    result = {
        "fixed_runs": [str(path) for path in args.fixed_runs],
        "adaptive_runs": [str(path) for path in args.adaptive_runs],
        "fixed_label": args.fixed_label,
        "adaptive_label": args.adaptive_label,
        "paired_examples": int(len(paired)),
        "comparison": comparison.to_dict(orient="records"),
        "absolute_metrics": absolute.to_dict(orient="records"),
        "compute_note": (
            "Soft mode gates still evaluate all maximum-rank modes; effective-rank "
            "reduction is not a measured runtime reduction."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
