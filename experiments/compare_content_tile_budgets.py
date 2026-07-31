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


def load_runs(runs: list[Path], kind: str) -> pd.DataFrame:
    frames = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        frame = pd.read_csv(run / "per_example.csv")
        if kind == "tile":
            seed = int(summary["model_seed"])
        else:
            router_summary = json.loads(
                (Path(summary["router_run"]) / "summary.json").read_text()
            )
            seed = int(router_summary["model_seed"])
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_group(
    frame: pd.DataFrame,
    columns: list[str],
    samples: int,
    rng: np.random.Generator,
    budget: int,
    analysis: str,
) -> list[dict[str, float | int | str]]:
    draws = hierarchical_bootstrap(frame, columns, samples, rng)
    family = frame.groupby(["seed", "family"], as_index=False)[columns].mean()
    estimates = family.groupby("seed")[columns].mean().mean(axis=0)
    rows = []
    for index, column in enumerate(columns):
        seed_values = family.groupby("seed")[column].mean()
        rows.append(
            {
                "analysis": analysis,
                "candidate_budget": budget,
                "metric": column,
                "estimate": float(estimates[column]),
                "ci_low": float(np.quantile(draws[:, index], 0.025)),
                "ci_high": float(np.quantile(draws[:, index], 0.975)),
                "probability_positive": float(np.mean(draws[:, index] > 0)),
                "positive_seeds": int((seed_values > 0).sum()),
                "seeds": int(len(seed_values)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    tile = load_runs(args.tile_runs, "tile")
    segment = load_runs(args.segment_runs, "segment")
    metrics = [
        "neighbor_dense_recall",
        "candidate_teacher_recall",
        "message_cosine",
        "message_relative_error",
        "work_ratio",
        "routed_minus_dense_ce_gain",
        "routed_minus_dense_teacher_kl_gain",
    ]
    common_budgets = sorted(
        set(tile.candidate_budget.astype(int))
        & set(segment.candidate_budget.astype(int))
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    paired_frames = []
    keys = ["seed", "example", "family"]
    for budget in common_budgets:
        tile_budget = tile[tile.candidate_budget == budget]
        segment_budget = segment[segment.candidate_budget == budget]
        rows.extend(
            summarize_group(
                tile_budget, metrics, args.bootstrap, rng, budget, "tile_absolute"
            )
        )
        paired = tile_budget[keys + metrics].merge(
            segment_budget[keys + metrics],
            on=keys,
            suffixes=("_tile", "_segment"),
        )
        differences = paired[keys].copy()
        differences["candidate_budget"] = budget
        for metric in metrics:
            differences[f"tile_minus_segment_{metric}"] = (
                paired[f"{metric}_tile"] - paired[f"{metric}_segment"]
            )
        differences["segment_minus_tile_work_ratio"] = (
            paired["work_ratio_segment"] - paired["work_ratio_tile"]
        )
        difference_metrics = [
            "tile_minus_segment_neighbor_dense_recall",
            "tile_minus_segment_candidate_teacher_recall",
            "tile_minus_segment_message_cosine",
            "tile_minus_segment_message_relative_error",
            "segment_minus_tile_work_ratio",
            "tile_minus_segment_routed_minus_dense_ce_gain",
            "tile_minus_segment_routed_minus_dense_teacher_kl_gain",
        ]
        rows.extend(
            summarize_group(
                differences,
                difference_metrics,
                args.bootstrap,
                rng,
                budget,
                "tile_minus_segment",
            )
        )
        paired_frames.append(differences)
    comparison = pd.DataFrame(rows)
    tile.to_csv(args.output / "tile_per_example.csv", index=False)
    segment.to_csv(args.output / "segment_per_example.csv", index=False)
    pd.concat(paired_frames, ignore_index=True).to_csv(
        args.output / "paired_differences.csv", index=False
    )
    comparison.to_csv(args.output / "hierarchical_bootstrap.csv", index=False)
    result = {
        "tile_runs": [str(run) for run in args.tile_runs],
        "segment_runs": [str(run) for run in args.segment_runs],
        "budgets": common_budgets,
        "bootstrap": args.bootstrap,
        "comparison": comparison.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
