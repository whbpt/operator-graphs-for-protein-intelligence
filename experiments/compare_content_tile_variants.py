from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_conditional_response_demo import hierarchical_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--frozen-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--full-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def load_runs(runs: list[Path], variant: str) -> pd.DataFrame:
    frames = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        frame = pd.read_csv(run / "per_example.csv")
        frame["seed"] = int(summary["model_seed"])
        frame["variant"] = variant
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(
    frame: pd.DataFrame,
    columns: list[str],
    samples: int,
    rng: np.random.Generator,
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
    return rows


def paired_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    metrics: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    keys = ["seed", "example", "family"]
    merged = left[keys + metrics].merge(
        right[keys + metrics], on=keys, suffixes=("_left", "_right")
    )
    result = merged[keys].copy()
    columns = []
    for metric in metrics:
        column = f"{left_name}_minus_{right_name}_{metric}"
        result[column] = merged[f"{metric}_left"] - merged[f"{metric}_right"]
        columns.append(column)
    return result, columns


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    base = load_runs(args.base_runs, "base")
    frozen = load_runs(args.frozen_runs, "frozen_value")
    full = load_runs(args.full_runs, "full")
    metrics = [
        "routed_ce_gain",
        "routed_teacher_kl_gain",
        "routed_minus_dense_ce_gain",
        "routed_minus_dense_teacher_kl_gain",
        "neighbor_dense_recall",
        "message_cosine",
    ]
    rng = np.random.default_rng(args.seed)
    rows = []
    for name, frame in (("base", base), ("frozen_value", frozen), ("full", full)):
        rows.extend(summarize(frame, metrics, args.bootstrap, rng, name))
    paired_frames = []
    for left, right, left_name, right_name in (
        (frozen, base, "frozen", "base"),
        (full, base, "full", "base"),
        (frozen, full, "frozen", "full"),
    ):
        differences, columns = paired_difference(
            left, right, left_name, right_name, metrics
        )
        rows.extend(
            summarize(
                differences,
                columns,
                args.bootstrap,
                rng,
                f"{left_name}_minus_{right_name}",
            )
        )
        differences["comparison"] = f"{left_name}_minus_{right_name}"
        paired_frames.append(differences)
    all_data = pd.concat([base, frozen, full], ignore_index=True)
    comparison = pd.DataFrame(rows)
    all_data.to_csv(args.output / "per_example.csv", index=False)
    pd.concat(paired_frames, ignore_index=True).to_csv(
        args.output / "paired_differences.csv", index=False
    )
    comparison.to_csv(args.output / "hierarchical_bootstrap.csv", index=False)
    result = {
        "base_runs": [str(run) for run in args.base_runs],
        "frozen_runs": [str(run) for run in args.frozen_runs],
        "full_runs": [str(run) for run in args.full_runs],
        "bootstrap": args.bootstrap,
        "comparison": comparison.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
