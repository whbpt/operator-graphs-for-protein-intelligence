from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summary_value(results: Path, directory: str, key: str) -> float:
    summary = json.loads((results / directory / "summary.json").read_text())
    return float(summary["test"][f"{key}_mean"])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    indexer_rows = [
        {
            "model": "Bilinear binary",
            "contact_p_at_l": summary_value(
                args.results, "seqmodels_gate_bilinear_v1", "centered_long_p_at_l"
            ),
        },
        {
            "model": "Pair MLP binary",
            "contact_p_at_l": summary_value(
                args.results, "seqmodels_gate_pair_mlp_v1", "centered_long_p_at_l"
            ),
        },
        {
            "model": "Bilinear continuous",
            "contact_p_at_l": summary_value(
                args.results,
                "seqmodels_score_bilinear_robust_z_v1",
                "raw_long_p_at_l",
            ),
        },
        {
            "model": "Pair MLP continuous",
            "contact_p_at_l": summary_value(
                args.results,
                "seqmodels_score_pair_mlp_robust_z_v1",
                "raw_long_p_at_l",
            ),
        },
    ]
    typed_rows = [
        {
            "model": "Continuous indexer",
            "contact_p_at_l": indexer_rows[-1]["contact_p_at_l"],
        },
        {
            "model": "Global modes",
            "contact_p_at_l": summary_value(
                args.results,
                "seqmodels_topk_typed_rank32_v1",
                "typed_long_p_at_l",
            ),
        },
        {
            "model": "Pair-conditioned",
            "contact_p_at_l": summary_value(
                args.results,
                "seqmodels_pair_conditioned_topk_rank32_v1",
                "typed_long_p_at_l",
            ),
        },
        {
            "model": "Pair-conditioned phylo-4",
            "contact_p_at_l": summary_value(
                args.results,
                "seqmodels_pair_conditioned_topk_phylo4_v2",
                "typed_long_p_at_l",
            ),
        },
    ]
    phylogeny = pd.read_csv(
        args.results / "phylogeny_residual_teacher_test57_v1" / "summary.csv"
    )
    indexer = pd.DataFrame(indexer_rows)
    typed = pd.DataFrame(typed_rows)
    indexer.to_csv(args.output / "indexer_objective_comparison.csv", index=False)
    typed.to_csv(args.output / "typed_operator_comparison.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))
    colors = ["#6b7280", "#8f5d9f", "#2a9d8f", "#d1495b"]
    axes[0].bar(indexer["model"], indexer["contact_p_at_l"], color=colors)
    axes[0].axhline(0.02643, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Indexer teacher objective")
    axes[0].set_ylabel("Held-out contact P@L")
    axes[1].bar(typed["model"], typed["contact_p_at_l"], color=colors)
    axes[1].axhline(0.02643, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Typed operator attribution")
    axes[1].set_ylabel("Held-out contact P@L")
    axes[2].bar(
        phylogeny["rank_removed"].astype(str),
        phylogeny["contact_average_precision"],
        color=["#6b7280", "#2a9d8f", "#457b9d", "#d1495b"],
    )
    axes[2].set_title("Phylogeny-mode removal")
    axes[2].set_xlabel("Removed rank")
    axes[2].set_ylabel("Teacher contact AP")
    for axis in axes[:2]:
        axis.tick_params(axis="x", rotation=30)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output / "architecture_revision.png", dpi=200)
    plt.close(figure)

    print(indexer.to_string(index=False))
    print()
    print(typed.to_string(index=False))
    print()
    print(phylogeny.to_string(index=False))


if __name__ == "__main__":
    main()
