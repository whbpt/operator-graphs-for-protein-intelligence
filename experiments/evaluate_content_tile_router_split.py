from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.evaluate_lsh_candidate_router import load_model, teacher_args
from experiments.train_conditional_response_demo import build_target_examples
from experiments.train_content_tile_router import evaluate, summarize
from experiments.train_epistasis_identifiability import load_frozen_models
from transformer_disentanglement.content_tile_routing import ContentTileRouter
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--families", type=int)
    parser.add_argument("--targets-per-family", type=int)
    parser.add_argument("--candidate-budget", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    saved = json.loads((args.router_run / "summary.json").read_text())
    router_config = saved["configuration"]
    base_run = Path(router_config["run"])
    config = json.loads((base_run / "run.json").read_text())
    benchmark = Path(router_config["benchmark"])
    representations = Path(router_config["representations"])
    device = choose_device(args.device)
    path_args = argparse.Namespace(
        benchmark=benchmark,
        representations=representations,
    )
    frozen_args = teacher_args(config, path_args)
    if args.targets_per_family is not None:
        frozen_args.targets_per_family = args.targets_per_family
    teacher, background = load_frozen_models(frozen_args, device)
    families = pd.read_csv(representations / "families.csv")
    split_frame = families[families.role == args.split].reset_index(drop=True)
    count = args.families or len(split_frame)
    seed = int(config["seed"])
    examples = build_target_examples(
        split_frame,
        count,
        frozen_args,
        teacher,
        background,
        np.random.default_rng(seed + 4000),
        device,
    )
    del teacher, background
    model = load_model(base_run, config, device)
    router = ContentTileRouter(
        stable_dim=int(config["stable_dim"]),
        task_dim=int(config["task_dim"]),
        tile_dim=int(router_config["tile_dim"]),
        tiles=int(router_config["tiles"]),
        selected_tiles=int(router_config["selected_tiles"]),
        candidate_budget=args.candidate_budget,
        neighbors=int(config["neighbors"]),
    ).to(device)
    router.load_state_dict(
        torch.load(args.router_run / "router.pt", map_location=device, weights_only=True)
    )
    router.eval()
    frame = evaluate(model, router, examples, device)
    frame.to_csv(args.output / "per_example.csv", index=False)
    result = {
        "router_run": str(args.router_run),
        "model_seed": seed,
        "split": args.split,
        "families": int(frame.family.nunique()),
        "examples": int(len(frame)),
        "targets_per_family": int(frozen_args.targets_per_family),
        "candidate_budget": args.candidate_budget,
        "metrics": summarize(frame),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
