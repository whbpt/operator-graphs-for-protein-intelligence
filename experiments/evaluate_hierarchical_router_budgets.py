from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.evaluate_lsh_candidate_router import load_model, teacher_args
from experiments.train_conditional_response_demo import build_target_examples
from experiments.train_epistasis_identifiability import load_frozen_models
from experiments.train_hierarchical_candidate_router import evaluate, summarize
from transformer_disentanglement.hierarchical_routing import (
    HierarchicalSegmentRouter,
)
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam-sizes", type=int, nargs="+", default=[8, 12, 16, 24])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    saved = json.loads((args.router_run / "summary.json").read_text())
    router_config = saved["configuration"]
    run = Path(router_config["run"])
    config = json.loads((run / "run.json").read_text())
    device = choose_device(args.device)
    frozen_args = teacher_args(
        config,
        argparse.Namespace(
            benchmark=Path(router_config["benchmark"]),
            representations=Path(router_config["representations"]),
        ),
    )
    teacher, background = load_frozen_models(frozen_args, device)
    families = pd.read_csv(Path(router_config["representations"]) / "families.csv")
    validation = families[families.role == "validation"].reset_index(drop=True)
    seed = int(config["seed"])
    examples = build_target_examples(
        validation,
        int(config["eval_families"]),
        frozen_args,
        teacher,
        background,
        np.random.default_rng(seed + 3000),
        device,
    )
    model = load_model(run, config, device)
    rows = []
    summaries = []
    for beam_size in args.beam_sizes:
        candidate_budget = beam_size * int(router_config["leaf_size"])
        router = HierarchicalSegmentRouter(
            task_dim=int(config["task_dim"]),
            node_dim=int(router_config["node_dim"]),
            branching=int(router_config["branching"]),
            leaf_size=int(router_config["leaf_size"]),
            beam_size=beam_size,
            candidate_budget=candidate_budget,
            neighbors=int(config["neighbors"]),
        ).to(device)
        router.load_state_dict(
            torch.load(
                args.router_run / "router.pt", map_location=device, weights_only=True
            )
        )
        router.eval()
        frame = evaluate(model, router, examples, device)
        frame["beam_size"] = beam_size
        frame["candidate_budget"] = candidate_budget
        rows.append(frame)
        summaries.append(
            {"beam_size": beam_size, "candidate_budget": candidate_budget}
            | summarize(frame)
        )
    pd.concat(rows, ignore_index=True).to_csv(
        args.output / "per_example.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
    result = {
        "router_run": str(args.router_run),
        "configurations": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
