from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_runs(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    examples = []
    summaries = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        seed = int(summary["seed"])
        frame = pd.read_csv(run / "adapted_topk_per_example.csv")
        frame["seed"] = seed
        frame["family"] = frame["example"].str.split(":", n=1).str[0]
        frame["ce_gain"] = frame["background_cross_entropy"] - frame["cross_entropy"]
        frame["teacher_kl_gain"] = (
            frame["background_teacher_kl"] - frame["teacher_kl"]
        )
        examples.append(frame)
        for stage in ["initial", "warmup", "converted_topk", "adapted_topk"]:
            row = {"seed": seed, "stage": stage}
            row.update(summary[stage])
            summaries.append(row)
    return pd.concat(examples, ignore_index=True), pd.DataFrame(summaries)


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    columns: list[str],
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    seeds = frame.seed.unique()
    grouped = {
        seed: [
            family_frame[columns].to_numpy(dtype=float)
            for _, family_frame in frame[frame.seed == seed].groupby("family")
        ]
        for seed in seeds
    }
    draws = np.empty((samples, len(columns)), dtype=float)
    for sample in range(samples):
        seed_means = []
        for seed_index in rng.integers(len(seeds), size=len(seeds)):
            families = grouped[seeds[seed_index]]
            family_means = []
            for family_index in rng.integers(len(families), size=len(families)):
                values = families[family_index]
                example_indices = rng.integers(len(values), size=len(values))
                family_means.append(values[example_indices].mean(axis=0))
            seed_means.append(np.mean(family_means, axis=0))
        draws[sample] = np.mean(seed_means, axis=0)
    return draws


def summarize_metrics(
    frame: pd.DataFrame,
    columns: list[str],
    draws: np.ndarray,
) -> pd.DataFrame:
    per_family = frame.groupby(["seed", "family"], as_index=False)[columns].mean()
    estimates = per_family.groupby("seed")[columns].mean().mean(axis=0)
    rows = []
    for index, column in enumerate(columns):
        result: dict[str, float | int | str] = {
            "metric": column,
            "seeds": int(frame.seed.nunique()),
            "families_per_seed": int(frame.groupby("seed").family.nunique().min()),
            "examples": int(len(frame)),
            "estimate": float(estimates[column]),
            "ci_low": float(np.quantile(draws[:, index], 0.025)),
            "ci_high": float(np.quantile(draws[:, index], 0.975)),
        }
        if column in {"ce_gain", "teacher_kl_gain"}:
            result["probability_positive"] = float(np.mean(draws[:, index] > 0))
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    examples, stages = load_runs(args.runs)
    rng = np.random.default_rng(args.seed)
    metrics = [
        "ce_gain",
        "teacher_kl_gain",
        "index_spearman",
        "index_top_ap",
        "shape_correlation",
        "shape_mse",
        "interaction_rms",
    ]
    draws = hierarchical_bootstrap(examples, metrics, args.bootstrap, rng)
    comparison = summarize_metrics(examples, metrics, draws)
    serializable_comparison = comparison.astype(object).where(
        pd.notna(comparison), None
    )
    examples.to_csv(args.output / "per_example_metrics.csv", index=False)
    stages.to_csv(args.output / "stage_summary.csv", index=False)
    comparison.to_csv(args.output / "hierarchical_bootstrap.csv", index=False)
    result = {
        "runs": [str(run) for run in args.runs],
        "bootstrap": args.bootstrap,
        "comparison": serializable_comparison.to_dict(orient="records"),
        "stage_summary": stages.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
