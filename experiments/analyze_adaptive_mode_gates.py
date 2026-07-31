from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformer_disentanglement.demo_language_models import DualStreamProteinLM
from transformer_disentanglement.protein_transformer import choose_device
from transformer_disentanglement.seqmodels_benchmark import load_seqmodels_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def validation_pairs(
    frame: pd.DataFrame,
    benchmark: Path,
    seed: int,
    family_count: int,
    targets_per_family: int,
    contexts_per_target: int,
    min_separation: int,
) -> list[dict[str, torch.Tensor | str]]:
    rng = np.random.default_rng(seed + 3000)
    examples = []
    for row in frame.head(family_count).itertuples(index=False):
        family = load_seqmodels_family(benchmark, row.file, row.x_id)
        sequence = family.msa[0].astype(np.int64)
        valid = np.flatnonzero(sequence < 20)
        targets = rng.choice(
            valid, size=min(targets_per_family, len(valid)), replace=False
        )
        for target in targets:
            candidates = valid[np.abs(valid - target) >= min_separation]
            contexts = rng.choice(
                candidates,
                size=min(contexts_per_target, len(candidates)),
                replace=False,
            ).astype(np.int64)
            tokens = torch.from_numpy(sequence).long()
            tokens[target] = 21
            context_tensor = torch.from_numpy(contexts).long()
            pairs = torch.stack(
                [torch.full_like(context_tensor, int(target)), context_tensor], dim=-1
            )
            examples.append(
                {
                    "example": f"{row.x_id}:{int(target)}",
                    "tokens": tokens,
                    "pairs": pairs,
                }
            )
    return examples


def load_model(run: Path, device: torch.device) -> tuple[DualStreamProteinLM, dict]:
    config = json.loads((run / "run.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    model = DualStreamProteinLM(
        stable_dim=int(config["stable_dim"]),
        task_dim=int(config["task_dim"]),
        rank=int(config["rank"]),
        index_dim=int(config["index_dim"]),
        pair_dim=int(config["pair_dim"]),
        pair_mlp_dim=summary.get("pair_mlp_dim"),
        neighbors=int(config["neighbors"]),
        rank_mode=config.get("rank_mode", "fixed"),
        gate_temperature=float(config.get("gate_temperature", 1.0)),
    ).to(device)
    state = torch.load(run / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config


@torch.no_grad()
def collect_gates(
    model: DualStreamProteinLM,
    examples: list[dict[str, torch.Tensor | str]],
    device: torch.device,
) -> tuple[np.ndarray, pd.DataFrame]:
    gates = []
    rows = []
    for example in examples:
        tokens = example["tokens"][None].to(device)  # type: ignore[index,union-attr]
        pairs = example["pairs"][None].to(device)  # type: ignore[index,union-attr]
        output = model(tokens, use_interaction=False)
        _, pair_gates, effective_rank = model.interaction.sampled_pair_outputs(
            output["factors"],  # type: ignore[arg-type]
            output["value_state"],  # type: ignore[arg-type]
            output["encoded_task"],  # type: ignore[arg-type]
            pairs,
        )
        pair_gates = pair_gates[0].cpu().numpy()
        pair_rank = effective_rank[0].cpu().numpy()
        gates.append(pair_gates)
        for index, rank in enumerate(pair_rank):
            rows.append(
                {
                    "example": example["example"],
                    "pair_index": index,
                    "effective_rank": float(rank),
                    "top_mode": int(np.argmax(pair_gates[index])),
                }
            )
    return np.concatenate(gates), pd.DataFrame(rows)


def summarize(gates: np.ndarray, pairs: pd.DataFrame) -> dict:
    normalized = gates / np.linalg.norm(gates, axis=1, keepdims=True).clip(1e-8)
    template = normalized.mean(axis=0)
    template = template / max(float(np.linalg.norm(template)), 1e-8)
    template_cosine = normalized @ template
    centered = gates - gates.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    centered_energy = np.square(singular_values)
    top_counts = pairs.top_mode.value_counts().sort_index()
    top_frequency = {
        str(index): float(top_counts.get(index, 0) / len(pairs))
        for index in range(gates.shape[1])
    }
    rank_values = pairs.effective_rank.to_numpy()
    return {
        "pairs": int(len(gates)),
        "rank_mean": float(rank_values.mean()),
        "rank_std": float(rank_values.std()),
        "rank_05": float(np.quantile(rank_values, 0.05)),
        "rank_95": float(np.quantile(rank_values, 0.95)),
        "template_cosine_mean": float(template_cosine.mean()),
        "template_cosine_05": float(np.quantile(template_cosine, 0.05)),
        "centered_gate_pc1_fraction": float(
            centered_energy[0] / max(float(centered_energy.sum()), 1e-8)
        ),
        "per_mode_mean": gates.mean(axis=0).tolist(),
        "per_mode_std": gates.std(axis=0).tolist(),
        "top_mode_frequency": top_frequency,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    device = choose_device(args.device)
    frame = pd.read_csv(args.representations / "families.csv")
    validation = frame[frame.role == "validation"].reset_index(drop=True)
    results = []
    pair_frames = []
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
        gates, pairs = collect_gates(model, examples, device)
        pairs["seed"] = seed
        pairs["run"] = str(run)
        pair_frames.append(pairs)
        results.append({"run": str(run), "seed": seed} | summarize(gates, pairs))
    pd.concat(pair_frames, ignore_index=True).to_csv(
        args.output / "pair_effective_rank.csv", index=False
    )
    (args.output / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
