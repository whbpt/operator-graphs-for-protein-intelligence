from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_conditional_response_demo import hierarchical_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--consensus-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def load_runs(runs: list[Path]) -> pd.DataFrame:
    frames = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        frame = pd.read_csv(run / "adapted_topk_per_example.csv")
        frame["seed"] = int(summary["seed"])
        frame["family"] = frame.example.str.split(":", n=1).str[0]
        frame["ce_gain"] = frame.background_cross_entropy - frame.cross_entropy
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    single = load_runs(args.single_runs)
    consensus = load_runs(args.consensus_runs)
    keys = ["seed", "family", "example"]
    metrics = ["ce_gain", "cross_entropy", "interaction_rms"]
    merged = consensus[keys + metrics].merge(
        single[keys + metrics], on=keys, suffixes=("_consensus", "_single")
    )
    differences = merged[keys].copy()
    differences["consensus_minus_single_ce_gain"] = (
        merged.ce_gain_consensus - merged.ce_gain_single
    )
    differences["single_minus_consensus_cross_entropy"] = (
        merged.cross_entropy_single - merged.cross_entropy_consensus
    )
    differences["consensus_minus_single_interaction_rms"] = (
        merged.interaction_rms_consensus - merged.interaction_rms_single
    )
    columns = [
        "consensus_minus_single_ce_gain",
        "single_minus_consensus_cross_entropy",
        "consensus_minus_single_interaction_rms",
    ]
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
        "single_runs": [str(run) for run in args.single_runs],
        "consensus_runs": [str(run) for run in args.consensus_runs],
        "bootstrap": args.bootstrap,
        "comparison": comparison.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
