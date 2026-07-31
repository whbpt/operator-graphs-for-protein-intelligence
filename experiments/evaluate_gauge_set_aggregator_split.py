from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.evaluate_content_tile_variants_split import make_router
from experiments.evaluate_lsh_candidate_router import load_model, teacher_args
from experiments.train_conditional_response_demo import build_target_examples
from experiments.train_epistasis_identifiability import load_frozen_models
from experiments.train_gauge_set_aggregator import (
    evaluate,
    load_consensus_teacher,
    replace_teacher_logits_with_consensus,
    summarize,
)
from transformer_disentanglement.gauge_set_aggregation import (
    MarginalOrthogonalSetAggregator,
)
from transformer_disentanglement.protein_transformer import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregator-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--targets-per-family", type=int, default=12)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    saved = json.loads((args.aggregator_run / "summary.json").read_text())
    aggregator_config = saved["configuration"]
    router_run = Path(aggregator_config["router_run"])
    router_summary = json.loads((router_run / "summary.json").read_text())
    router_config = router_summary["configuration"]
    base_run = Path(router_config["run"])
    config = json.loads((base_run / "run.json").read_text())
    benchmark = Path(router_config["benchmark"])
    representations = Path(router_config["representations"])
    device = choose_device(args.device)
    path_args = argparse.Namespace(benchmark=benchmark, representations=representations)
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
    consensus_checkpoint = aggregator_config.get("consensus_teacher_checkpoint")
    if consensus_checkpoint is not None:
        second_teacher = load_consensus_teacher(
            Path(consensus_checkpoint), int(config["hidden_dim"]), device
        )
        replace_teacher_logits_with_consensus(examples, second_teacher, device)
        del second_teacher
    del teacher, background
    model = load_model(base_run, config, device)
    model.eval()
    router = make_router(
        config,
        router_config,
        int(aggregator_config["candidate_budget"]),
        router_run / "router.pt",
        device,
    )
    aggregator = MarginalOrthogonalSetAggregator(
        states=20,
        hidden_dim=int(aggregator_config["hidden_dim"]),
        routing_temperature=float(model.interaction.routing_temperature),
        max_correction_ratio=float(aggregator_config["max_correction_ratio"]),
    ).to(device)
    aggregator.load_state_dict(
        torch.load(
            args.aggregator_run / "aggregator.pt",
            map_location=device,
            weights_only=True,
        )
    )
    aggregator.eval()
    frame = evaluate(model, router, aggregator, examples, device)
    frame.to_csv(args.output / "per_example.csv", index=False)
    result = {
        "aggregator_run": str(args.aggregator_run),
        "model_seed": seed,
        "split": args.split,
        "families": int(frame.family.nunique()),
        "examples": int(len(frame)),
        "metrics": summarize(frame),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
