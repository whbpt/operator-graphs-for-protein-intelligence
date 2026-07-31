from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.analyze_adaptive_mode_gates import load_model, validation_pairs
from transformer_disentanglement.candidate_routing import (
    BoundingBoxMIPSTreeRouter,
)
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--leaf-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    device = choose_device(args.device)
    frame = pd.read_csv(args.representations / "families.csv")
    validation = frame[frame.role == "validation"].reset_index(drop=True)
    rows = []
    for run in args.runs:
        model, config = load_model(run, device)
        seed = int(config["seed"])
        examples = validation_pairs(
            validation,
            args.benchmark,
            seed,
            int(config["eval_families"]),
            int(config["targets_per_family"]),
            int(config["contexts_per_target"]),
            int(config["min_separation"]),
        )
        for example in examples:
            tokens = example["tokens"][None].to(device)  # type: ignore[index,union-attr]
            pairs = example["pairs"]  # type: ignore[assignment]
            target_position = int(pairs[0, 0])
            output = model(tokens, use_interaction=False)
            query, key = model.interaction.index_features(
                output["encoded_task"]  # type: ignore[arg-type]
            )
            valid = model.interaction.valid_pair_mask(tokens.shape[1], device)
            valid_row = valid[target_position : target_position + 1]
            dense = model.interaction.index_scores(
                output["encoded_task"]  # type: ignore[arg-type]
            )[0, target_position].masked_fill(~valid[target_position], -torch.inf)
            count = min(model.interaction.neighbors, int(valid_row.sum()))
            expected = torch.topk(dense, k=count)
            for leaf_size in args.leaf_sizes:
                router = BoundingBoxMIPSTreeRouter(
                    neighbors=model.interaction.neighbors,
                    leaf_size=leaf_size,
                )
                routed = router(
                    query[:, target_position : target_position + 1],
                    key,
                    valid_row,
                    score_scale=model.interaction.index_dim**-0.5,
                    score_bias=model.interaction.index_bias,
                )
                exact = torch.equal(
                    routed.neighbor_indices[0, 0], expected.indices
                ) and torch.allclose(
                    routed.neighbor_scores[0, 0], expected.values, atol=1e-5
                )
                valid_pairs = int(valid_row.sum())
                evaluated = int(routed.evaluated_pairs[0, 0])
                rows.append(
                    {
                        "seed": seed,
                        "example": example["example"],
                        "family": str(example["example"]).split(":", 1)[0],
                        "length": tokens.shape[1],
                        "leaf_size": leaf_size,
                        "exact": bool(exact),
                        "valid_pairs": valid_pairs,
                        "evaluated_pairs": evaluated,
                        "evaluated_fraction": evaluated / valid_pairs,
                        "visited_nodes": int(routed.visited_nodes[0, 0]),
                    }
                )
    result_frame = pd.DataFrame(rows)
    summary = result_frame.groupby("leaf_size", as_index=False).agg(
        exact_rate=("exact", "mean"),
        evaluated_pairs=("evaluated_pairs", "mean"),
        valid_pairs=("valid_pairs", "mean"),
        evaluated_fraction=("evaluated_fraction", "mean"),
        evaluated_fraction_95=("evaluated_fraction", lambda x: np.quantile(x, 0.95)),
        visited_nodes=("visited_nodes", "mean"),
    )
    summary["pair_reduction"] = 1.0 - summary.evaluated_fraction
    result_frame.to_csv(args.output / "per_example.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    result = {
        "runs": [str(run) for run in args.runs],
        "examples": int(result_frame[["seed", "example"]].drop_duplicates().shape[0]),
        "configurations": summary.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
