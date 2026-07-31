from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_conditional_response_demo import hierarchical_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--segment-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-budget", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def load_tile_runs(runs: list[Path]) -> pd.DataFrame:
    frames = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        frame = pd.read_csv(run / "per_example.csv")
        frame = frame[frame.stage == "trained"].drop(columns="stage")
        frame["seed"] = int(summary["model_seed"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_segment_runs(runs: list[Path], candidate_budget: int) -> pd.DataFrame:
    frames = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        router_run = Path(summary["router_run"])
        router_summary = json.loads((router_run / "summary.json").read_text())
        frame = pd.read_csv(run / "per_example.csv")
        frame = frame[frame.candidate_budget == candidate_budget].copy()
        frame["seed"] = int(router_summary["model_seed"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def bootstrap_summary(
    frame: pd.DataFrame,
    columns: list[str],
    samples: int,
    rng: np.random.Generator,
    analysis: str,
) -> pd.DataFrame:
    draws = hierarchical_bootstrap(frame, columns, samples, rng)
    family = frame.groupby(["seed", "family"], as_index=False)[columns].mean()
    estimates = family.groupby("seed")[columns].mean().mean(axis=0)
    rows = []
    for index, column in enumerate(columns):
        seed_values = family.groupby("seed")[column].mean()
        rows.append(
            {
                "analysis": analysis,
                "metric": column,
                "estimate": float(estimates[column]),
                "ci_low": float(np.quantile(draws[:, index], 0.025)),
                "ci_high": float(np.quantile(draws[:, index], 0.975)),
                "probability_positive": float(np.mean(draws[:, index] > 0)),
                "positive_seeds": int((seed_values > 0).sum()),
                "seeds": int(len(seed_values)),
                "examples": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    tile = load_tile_runs(args.tile_runs)
    segment = load_segment_runs(args.segment_runs, args.candidate_budget)
    keys = ["seed", "example", "family"]
    metrics = [
        "neighbor_dense_recall",
        "candidate_teacher_recall",
        "message_cosine",
        "message_relative_error",
        "work_ratio",
        "routed_ce_gain",
        "routed_teacher_kl_gain",
        "routed_minus_dense_ce_gain",
        "routed_minus_dense_teacher_kl_gain",
    ]
    paired = tile[keys + metrics].merge(
        segment[keys + metrics], on=keys, suffixes=("_tile", "_segment")
    )
    differences = paired[keys].copy()
    for metric in metrics:
        differences[f"tile_minus_segment_{metric}"] = (
            paired[f"{metric}_tile"] - paired[f"{metric}_segment"]
        )
    differences["segment_minus_tile_work_ratio"] = (
        paired["work_ratio_segment"] - paired["work_ratio_tile"]
    )
    absolute_metrics = metrics
    difference_metrics = [
        "tile_minus_segment_neighbor_dense_recall",
        "tile_minus_segment_candidate_teacher_recall",
        "tile_minus_segment_message_cosine",
        "tile_minus_segment_message_relative_error",
        "segment_minus_tile_work_ratio",
        "tile_minus_segment_routed_ce_gain",
        "tile_minus_segment_routed_teacher_kl_gain",
    ]
    rng = np.random.default_rng(args.seed)
    comparison = pd.concat(
        [
            bootstrap_summary(tile, absolute_metrics, args.bootstrap, rng, "tile_absolute"),
            bootstrap_summary(
                differences,
                difference_metrics,
                args.bootstrap,
                rng,
                "tile_minus_segment",
            ),
        ],
        ignore_index=True,
    )
    seed_summary = tile.groupby("seed", as_index=False)[metrics].mean()
    tile.to_csv(args.output / "tile_per_example.csv", index=False)
    segment.to_csv(args.output / "segment_per_example.csv", index=False)
    differences.to_csv(args.output / "paired_differences.csv", index=False)
    seed_summary.to_csv(args.output / "tile_seed_summary.csv", index=False)
    comparison.to_csv(args.output / "hierarchical_bootstrap.csv", index=False)
    result = {
        "tile_runs": [str(run) for run in args.tile_runs],
        "segment_runs": [str(run) for run in args.segment_runs],
        "candidate_budget": args.candidate_budget,
        "bootstrap": args.bootstrap,
        "tile_seed_summary": seed_summary.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
