from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_conditional_response_demo import hierarchical_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def summarize(
    frame: pd.DataFrame,
    columns: list[str],
    samples: int,
    rng: np.random.Generator,
    analysis: str,
) -> pd.DataFrame:
    draws = hierarchical_bootstrap(frame, columns, samples, rng)
    per_family = frame.groupby(["seed", "family"], as_index=False)[columns].mean()
    estimates = per_family.groupby("seed")[columns].mean().mean(axis=0)
    rows = []
    for index, column in enumerate(columns):
        seed_values = per_family.groupby("seed")[column].mean()
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
                "families_per_seed": int(
                    frame.groupby("seed").family.nunique().min()
                ),
                "examples": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def load_runs(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_stages = []
    all_deltas = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        seed = int(summary["model_seed"])
        frame = pd.read_csv(run / "per_example.csv")
        frame["seed"] = seed
        all_stages.append(frame)
        initial = frame[frame.stage == "initial"].drop(columns="stage")
        adapted = frame[frame.stage == "adapted"].drop(columns="stage")
        keys = ["seed", "example", "family"]
        merged = initial.merge(adapted, on=keys, suffixes=("_initial", "_adapted"))
        delta = merged[keys].copy()
        numeric = [
            column
            for column in initial.columns
            if column not in keys and pd.api.types.is_numeric_dtype(initial[column])
        ]
        for column in numeric:
            delta[f"delta_{column}"] = (
                merged[f"{column}_adapted"] - merged[f"{column}_initial"]
            )
        all_deltas.append(delta)
    return pd.concat(all_stages, ignore_index=True), pd.concat(all_deltas, ignore_index=True)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    stages, deltas = load_runs(args.runs)
    adapted = stages[stages.stage == "adapted"].copy()
    absolute_columns = [
        "routed_ce_gain",
        "dense_ce_gain",
        "routed_teacher_kl_gain",
        "dense_teacher_kl_gain",
        "routed_minus_dense_ce_gain",
        "routed_minus_dense_teacher_kl_gain",
        "neighbor_dense_recall",
        "message_cosine",
        "work_ratio",
    ]
    delta_columns = [
        "delta_routed_ce_gain",
        "delta_dense_ce_gain",
        "delta_routed_teacher_kl_gain",
        "delta_dense_teacher_kl_gain",
        "delta_neighbor_dense_recall",
        "delta_message_cosine",
    ]
    rng = np.random.default_rng(args.seed)
    comparison = pd.concat(
        [
            summarize(adapted, absolute_columns, args.bootstrap, rng, "adapted"),
            summarize(deltas, delta_columns, args.bootstrap, rng, "adaptation_delta"),
        ],
        ignore_index=True,
    )
    seed_summary = adapted.groupby("seed", as_index=False)[absolute_columns].mean()
    stages.to_csv(args.output / "per_example_stages.csv", index=False)
    deltas.to_csv(args.output / "per_example_deltas.csv", index=False)
    seed_summary.to_csv(args.output / "seed_summary.csv", index=False)
    comparison.to_csv(args.output / "hierarchical_bootstrap.csv", index=False)
    result = {
        "runs": [str(run) for run in args.runs],
        "bootstrap": args.bootstrap,
        "seed_summary": seed_summary.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
