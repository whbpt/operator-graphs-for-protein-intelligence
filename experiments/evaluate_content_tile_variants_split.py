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
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--targets-per-family", type=int, default=12)
    parser.add_argument("--candidate-budget", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def make_router(
    config: dict,
    router_config: dict,
    candidate_budget: int,
    state_path: Path,
    device: torch.device,
) -> ContentTileRouter:
    router = ContentTileRouter(
        stable_dim=int(config["stable_dim"]),
        task_dim=int(config["task_dim"]),
        tile_dim=int(router_config["tile_dim"]),
        tiles=int(router_config["tiles"]),
        selected_tiles=int(router_config["selected_tiles"]),
        candidate_budget=candidate_budget,
        neighbors=int(config["neighbors"]),
    ).to(device)
    router.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    router.eval()
    return router


def write_variant(
    output: Path,
    variant: str,
    seed: int,
    frame: pd.DataFrame,
) -> dict:
    directory = output / variant
    directory.mkdir(parents=True)
    frame.to_csv(directory / "per_example.csv", index=False)
    result = {
        "model_seed": seed,
        "variant": variant,
        "families": int(frame.family.nunique()),
        "examples": int(len(frame)),
        "metrics": summarize(frame),
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    router_summary = json.loads((args.router_run / "summary.json").read_text())
    router_config = router_summary["configuration"]
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
    frozen_args.targets_per_family = args.targets_per_family
    teacher, background = load_frozen_models(frozen_args, device)
    families = pd.read_csv(representations / "families.csv")
    split_frame = families[families.role == args.split].reset_index(drop=True)
    seed = int(config["seed"])
    examples = build_target_examples(
        split_frame,
        len(split_frame),
        frozen_args,
        teacher,
        background,
        np.random.default_rng(seed + 4000),
        device,
    )
    del teacher, background
    results = []
    base_model = load_model(base_run, config, device)
    base_router = make_router(
        config,
        router_config,
        args.candidate_budget,
        args.router_run / "router.pt",
        device,
    )
    results.append(
        write_variant(
            args.output,
            "base",
            seed,
            evaluate(base_model, base_router, examples, device),
        )
    )
    for variant, run in (("frozen_value", args.frozen_run), ("full", args.full_run)):
        model = load_model(base_run, config, device)
        model.load_state_dict(
            torch.load(run / "model.pt", map_location=device, weights_only=True)
        )
        model.eval()
        router = make_router(
            config,
            router_config,
            args.candidate_budget,
            run / "router.pt",
            device,
        )
        results.append(
            write_variant(
                args.output,
                variant,
                seed,
                evaluate(model, router, examples, device),
            )
        )
    result = {
        "model_seed": seed,
        "split": args.split,
        "targets_per_family": args.targets_per_family,
        "candidate_budget": args.candidate_budget,
        "variants": results,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
