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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def load_adapted(runs: list[Path]) -> pd.DataFrame:
    frames = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        frame = pd.read_csv(run / "per_example.csv")
        frame = frame[frame.stage == "adapted"].drop(columns="stage")
        frame["seed"] = int(summary["model_seed"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    tile = load_adapted(args.tile_runs)
    segment = load_adapted(args.segment_runs)
    keys = ["seed", "example", "family"]
    metrics = [
        "neighbor_dense_recall",
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
    columns = []
    for metric in metrics:
        column = f"tile_minus_segment_{metric}"
        differences[column] = paired[f"{metric}_tile"] - paired[f"{metric}_segment"]
        columns.append(column)
    differences["segment_minus_tile_work_ratio"] = (
        paired["work_ratio_segment"] - paired["work_ratio_tile"]
    )
    columns.remove("tile_minus_segment_work_ratio")
    columns.append("segment_minus_tile_work_ratio")
    rng = np.random.default_rng(args.seed)
    draws = hierarchical_bootstrap(differences, columns, args.bootstrap, rng)
    family = differences.groupby(["seed", "family"], as_index=False)[columns].mean()
    estimates = family.groupby("seed")[columns].mean().mean(axis=0)
    rows = []
    for index, column in enumerate(columns):
        seed_values = family.groupby("seed")[column].mean()
        rows.append(
            {
                "metric": column,
                "estimate": float(estimates[column]),
                "ci_low": float(np.quantile(draws[:, index], 0.025)),
                "ci_high": float(np.quantile(draws[:, index], 0.975)),
                "probability_positive": float(np.mean(draws[:, index] > 0)),
                "positive_seeds": int((seed_values > 0).sum()),
                "seeds": int(len(seed_values)),
            }
        )
    comparison = pd.DataFrame(rows)
    differences.to_csv(args.output / "paired_differences.csv", index=False)
    comparison.to_csv(args.output / "hierarchical_bootstrap.csv", index=False)
    result = {
        "tile_runs": [str(run) for run in args.tile_runs],
        "segment_runs": [str(run) for run in args.segment_runs],
        "bootstrap": args.bootstrap,
        "comparison": comparison.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
