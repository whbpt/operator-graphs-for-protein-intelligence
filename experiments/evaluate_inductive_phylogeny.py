from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from transformer_disentanglement.inductive_phylogeny import (
    SubstitutionModel,
    TreeEdge,
    TreeNode,
    canonical_newick,
    reconstruct_alignment,
    robinson_foulds_distance,
    simulate_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[25, 50, 100, 250])
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def random_quartet(rng: np.random.Generator) -> TreeNode:
    labels = np.asarray(["A", "B", "C", "D"])
    rng.shuffle(labels)
    pendant = rng.uniform(0.05, 0.35, size=4)
    middle = rng.uniform(0.005, 0.05, size=2)
    left = TreeNode(
        children=(
            TreeEdge(TreeNode(label=str(labels[0])), float(pendant[0])),
            TreeEdge(TreeNode(label=str(labels[1])), float(pendant[1])),
        )
    )
    right = TreeNode(
        children=(
            TreeEdge(TreeNode(label=str(labels[2])), float(pendant[2])),
            TreeEdge(TreeNode(label=str(labels[3])), float(pendant[3])),
        )
    )
    return TreeNode(
        children=(
            TreeEdge(left, float(middle[0])),
            TreeEdge(right, float(middle[1])),
        )
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.replicates <= 0 or min(args.depths) <= 0:
        raise ValueError("depths and replicates must be positive")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    model = SubstitutionModel.jukes_cantor()
    rows = []
    configurations = {
        "greedy": {"beam_size": 1, "expansions_per_state": 1},
        "beam": {"beam_size": 8, "expansions_per_state": 6},
    }
    for depth in args.depths:
        for replicate in range(args.replicates):
            truth = random_quartet(rng)
            alignment, labels = simulate_alignment(truth, depth, model, rng)
            for name, config in configurations.items():
                result = reconstruct_alignment(
                    alignment,
                    labels,
                    model=model,
                    **config,
                )
                rf = robinson_foulds_distance(truth, result.tree)
                rows.append(
                    {
                        "depth": depth,
                        "replicate": replicate,
                        "method": name,
                        "recovered": int(rf == 0),
                        "rf_distance": rf,
                        "log_likelihood": result.log_likelihood,
                        "candidate_trees": result.candidate_trees,
                        "truth": canonical_newick(truth),
                        "estimate": canonical_newick(result.tree),
                    }
                )

    with (args.output / "per_replicate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for depth in args.depths:
        for method in configurations:
            selected = [
                row for row in rows if row["depth"] == depth and row["method"] == method
            ]
            summary.append(
                {
                    "depth": depth,
                    "method": method,
                    "replicates": len(selected),
                    "recovery_rate": float(
                        np.mean([row["recovered"] for row in selected])
                    ),
                    "mean_rf_distance": float(
                        np.mean([row["rf_distance"] for row in selected])
                    ),
                    "mean_candidate_trees": float(
                        np.mean([row["candidate_trees"] for row in selected])
                    ),
                }
            )
    payload = {
        "seed": args.seed,
        "model": "four-state Jukes-Cantor",
        "topology": "randomized quartet",
        "summary": summary,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
