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
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--ce-margin", type=float, default=5e-4)
    parser.add_argument("--kl-margin", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    frames = []
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        router_run = Path(summary["router_run"])
        router_summary = json.loads((router_run / "summary.json").read_text())
        frame = pd.read_csv(run / "per_example.csv")
        frame["seed"] = int(router_summary["model_seed"])
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    metric_columns = [
        "routed_minus_dense_ce_gain",
        "routed_minus_dense_teacher_kl_gain",
        "message_cosine",
        "message_relative_error",
        "neighbor_dense_recall",
        "candidate_teacher_recall",
        "work_ratio",
        "routed_ce_gain",
        "routed_teacher_kl_gain",
    ]
    rows = []
    rng = np.random.default_rng(args.seed)
    for beam_size, frame in data.groupby("beam_size"):
        draws = hierarchical_bootstrap(frame, metric_columns, args.bootstrap, rng)
        family = frame.groupby(["seed", "family"], as_index=False)[
            metric_columns
        ].mean()
        estimates = family.groupby("seed")[metric_columns].mean().mean(axis=0)
        row = {
            "beam_size": int(beam_size),
            "candidate_budget": int(frame.candidate_budget.iloc[0]),
        }
        for index, metric in enumerate(metric_columns):
            row[metric] = float(estimates[metric])
            row[f"{metric}_ci_low"] = float(np.quantile(draws[:, index], 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(draws[:, index], 0.975))
        ce_index = metric_columns.index("routed_minus_dense_ce_gain")
        kl_index = metric_columns.index("routed_minus_dense_teacher_kl_gain")
        row["ce_noninferiority_probability"] = float(
            np.mean(draws[:, ce_index] >= -args.ce_margin)
        )
        row["kl_noninferiority_probability"] = float(
            np.mean(draws[:, kl_index] >= -args.kl_margin)
        )
        row["message_cosine_05"] = float(frame.message_cosine.quantile(0.05))
        family_ce = frame.groupby(["seed", "family"])[
            "routed_minus_dense_ce_gain"
        ].mean()
        row["worst_family_ce_delta"] = float(family_ce.min())
        rows.append(row)
    result_frame = pd.DataFrame(rows).sort_values("beam_size")
    data.to_csv(args.output / "per_example.csv", index=False)
    result_frame.to_csv(args.output / "clustered_summary.csv", index=False)
    result = {
        "runs": [str(run) for run in args.runs],
        "ce_margin": args.ce_margin,
        "kl_margin": args.kl_margin,
        "configurations": result_frame.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
