from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_conditional_response_demo import hierarchical_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--ce-margin", type=float, default=5e-4)
    parser.add_argument("--kl-margin", type=float, default=1e-4)
    return parser.parse_args()


def equal_weight_estimate(frame: pd.DataFrame, column: str) -> float:
    family = frame.groupby(["seed", "family"], as_index=False)[column].mean()
    return float(family.groupby("seed")[column].mean().mean())


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    raw = pd.read_csv(args.input / "per_example.csv")
    raw["family"] = raw.example.str.split(":", n=1).str[0]
    raw = raw.rename(columns={"model_seed": "seed"})
    config_columns = [
        "tables",
        "bits",
        "candidate_budget",
        "hamming_radius",
    ]
    metric_columns = [
        "lsh_minus_dense_ce_gain",
        "lsh_minus_dense_teacher_kl_gain",
        "message_cosine",
        "message_relative_error",
        "neighbor_dense_recall",
        "evaluated_fraction",
    ]
    per_example = raw.groupby(
        config_columns + ["seed", "family", "example"], as_index=False
    )[metric_columns].mean()
    rng = np.random.default_rng(args.seed)
    rows = []
    for config, frame in per_example.groupby(config_columns):
        draws = hierarchical_bootstrap(
            frame, metric_columns, args.bootstrap, rng
        )
        row = dict(zip(config_columns, config, strict=True))
        for index, metric in enumerate(metric_columns):
            row[metric] = equal_weight_estimate(frame, metric)
            row[f"{metric}_ci_low"] = float(
                np.quantile(draws[:, index], 0.025)
            )
            row[f"{metric}_ci_high"] = float(
                np.quantile(draws[:, index], 0.975)
            )
        ce_index = metric_columns.index("lsh_minus_dense_ce_gain")
        kl_index = metric_columns.index("lsh_minus_dense_teacher_kl_gain")
        row["ce_noninferiority_probability"] = float(
            np.mean(draws[:, ce_index] >= -args.ce_margin)
        )
        row["kl_noninferiority_probability"] = float(
            np.mean(draws[:, kl_index] >= -args.kl_margin)
        )
        raw_config = raw
        for column, value in zip(config_columns, config, strict=True):
            raw_config = raw_config[raw_config[column] == value]
        row["message_cosine_median"] = float(raw_config.message_cosine.median())
        row["message_cosine_05"] = float(
            raw_config.message_cosine.quantile(0.05)
        )
        row["message_relative_error_95"] = float(
            raw_config.message_relative_error.quantile(0.95)
        )
        family_ce = frame.groupby(["seed", "family"])[
            "lsh_minus_dense_ce_gain"
        ].mean()
        row["worst_family_ce_delta"] = float(family_ce.min())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["ce_noninferiority_probability", "evaluated_fraction"],
        ascending=[False, True],
    )
    per_example.to_csv(args.output / "per_example_hash_averaged.csv", index=False)
    summary.to_csv(args.output / "clustered_summary.csv", index=False)
    result = {
        "input": str(args.input),
        "ce_margin": args.ce_margin,
        "kl_margin": args.kl_margin,
        "configurations": summary.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
