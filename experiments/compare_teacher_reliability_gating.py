from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_conditional_response_demo import hierarchical_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ungated-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--gated-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def load_runs(runs: list[Path]) -> pd.DataFrame:
    frames = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        frame = pd.read_csv(run / "per_example.csv")
        frame["seed"] = int(summary["model_seed"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    ungated = load_runs(args.ungated_runs)
    gated = load_runs(args.gated_runs)
    keys = ["seed", "family", "example"]
    metrics = [
        "nonlinear_ce_gain",
        "nonlinear_minus_additive_ce_gain",
        "nonlinear_teacher_kl_gain",
        "nonlinear_minus_additive_teacher_kl_gain",
        "correction_to_additive_rms",
    ]
    merged = gated[keys + metrics].merge(
        ungated[keys + metrics], on=keys, suffixes=("_gated", "_ungated")
    )
    differences = merged[keys].copy()
    columns = []
    for metric in metrics:
        column = f"gated_minus_ungated_{metric}"
        differences[column] = (
            merged[f"{metric}_gated"] - merged[f"{metric}_ungated"]
        )
        columns.append(column)
    draws = hierarchical_bootstrap(
        differences, columns, args.bootstrap, np.random.default_rng(args.seed)
    )
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
                "examples": int(len(differences)),
            }
        )
    comparison = pd.DataFrame(rows)
    differences.to_csv(args.output / "paired_differences.csv", index=False)
    comparison.to_csv(args.output / "hierarchical_bootstrap.csv", index=False)
    result = {
        "ungated_runs": [str(run) for run in args.ungated_runs],
        "gated_runs": [str(run) for run in args.gated_runs],
        "bootstrap": args.bootstrap,
        "comparison": comparison.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
