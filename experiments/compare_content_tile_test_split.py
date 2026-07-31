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
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


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
    columns = [
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
    rng = np.random.default_rng(args.seed)
    draws = hierarchical_bootstrap(data, columns, args.bootstrap, rng)
    family = data.groupby(["seed", "family"], as_index=False)[columns].mean()
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
                "families_per_seed": int(
                    data.groupby("seed").family.nunique().min()
                ),
                "examples": int(len(data)),
            }
        )
    comparison = pd.DataFrame(rows)
    data.to_csv(args.output / "per_example.csv", index=False)
    comparison.to_csv(args.output / "hierarchical_bootstrap.csv", index=False)
    result = {
        "runs": [str(run) for run in args.runs],
        "bootstrap": args.bootstrap,
        "comparison": comparison.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
