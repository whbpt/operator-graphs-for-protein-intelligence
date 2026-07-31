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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--budgets", type=int, nargs="+", default=[24, 32, 48, 64])
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
    benchmark = Path(router_config["benchmark"])
    representations = Path(router_config["representations"])
    path_args = argparse.Namespace(
        benchmark=benchmark,
        representations=representations,
    )
    frozen_args = teacher_args(config, path_args)
    device = choose_device(args.device)
    teacher, background = load_frozen_models(frozen_args, device)
    families = pd.read_csv(representations / "families.csv")
    eval_frame = families[families.role == "validation"].reset_index(drop=True)
    seed = int(config["seed"])
    eval_examples = build_target_examples(
        eval_frame,
        int(config["eval_families"]),
        frozen_args,
        teacher,
        background,
        np.random.default_rng(seed + 3000),
        device,
    )
    del teacher, background
    model = load_model(run, config, device)
    state = torch.load(
        args.router_run / "router.pt", map_location=device, weights_only=True
    )
    frames = []
    summaries = []
    for budget in args.budgets:
        router = ContentTileRouter(
            stable_dim=int(config["stable_dim"]),
            task_dim=int(config["task_dim"]),
            tile_dim=int(router_config["tile_dim"]),
            tiles=int(router_config["tiles"]),
            selected_tiles=int(router_config["selected_tiles"]),
            candidate_budget=budget,
            neighbors=int(config["neighbors"]),
        ).to(device)
        router.load_state_dict(state)
        router.eval()
        frame = evaluate(model, router, eval_examples, device)
        frame["candidate_budget"] = budget
        frames.append(frame)
        summaries.append({"candidate_budget": budget} | summarize(frame))
    data = pd.concat(frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    data.to_csv(args.output / "per_example.csv", index=False)
    summary_frame.to_csv(args.output / "summary.csv", index=False)
    result = {
        "router_run": str(args.router_run),
        "model_seed": seed,
        "configurations": summary_frame.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
