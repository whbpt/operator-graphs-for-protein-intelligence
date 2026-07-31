from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.train_seqmodels_factor_head import (
    fit_family_null_models,
    load_representation,
)
from transformer_disentanglement.metrics import (
    binary_average_precision,
    binary_contact_precision,
    binary_contact_prevalence,
    binary_roc_auc,
    leading_mode_fraction,
)
from transformer_disentanglement.seqmodels_benchmark import (
    connected_pair_blocks,
    entropy_pair_scale,
    load_seqmodels_family,
    residualize_entropy_background,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", default="test")
    parser.add_argument("--min-separation", type=int, default=24)
    parser.add_argument("--null-reference-pairs", type=int, default=1024)
    parser.add_argument("--null-z-threshold", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def score_map(length: int, pairs: np.ndarray, values: np.ndarray) -> np.ndarray:
    matrix = np.zeros((length, length), dtype=np.float32)
    matrix[pairs[:, 0], pairs[:, 1]] = values
    matrix[pairs[:, 1], pairs[:, 0]] = values
    return matrix


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    frame = pd.read_csv(args.representations / "families.csv")
    role_frame = frame[frame["role"] == args.role]
    null_models, _ = fit_family_null_models(
        frame,
        args.benchmark,
        args.representations,
        args.null_reference_pairs,
        args.seed,
        args.null_z_threshold,
    )

    rows = []
    for row in role_frame.itertuples(index=False):
        family = load_seqmodels_family(args.benchmark, row.file, row.x_id)
        _, pssm = load_representation(args.representations, row)
        left, right = np.triu_indices(
            len(family.query), k=args.min_separation
        )
        pairs = np.stack([left, right], axis=-1).astype(np.int64)
        blocks = connected_pair_blocks(
            family.msa, family.weights, pairs, pssm=pssm
        )
        residual, reliability = residualize_entropy_background(
            blocks,
            pairs,
            pssm,
            null_models[row.x_id],
            z_threshold=args.null_z_threshold,
        )
        raw_strength = np.sqrt(np.mean(np.square(blocks), axis=(-2, -1)))
        normalized_strength = raw_strength / entropy_pair_scale(
            pssm, pairs
        ).clip(min=1e-8)
        residual_strength = np.sqrt(
            np.mean(np.square(residual), axis=(-2, -1))
        )
        raw_map = score_map(len(family.query), pairs, raw_strength)
        normalized_map = score_map(
            len(family.query), pairs, normalized_strength
        )
        residual_map = score_map(
            len(family.query), pairs, residual_strength
        )
        valid = family.contact_mask[left, right]
        contact = family.contacts[left, right] > 0.01
        active = reliability > 0.0
        rows.append(
            {
                "x_id": row.x_id,
                "role": row.role,
                "length": len(family.query),
                "contact_prevalence": binary_contact_prevalence(
                    family.contacts,
                    family.contact_mask,
                    min_separation=args.min_separation,
                ),
                "raw_contact_p_at_l": binary_contact_precision(
                    raw_map,
                    family.contacts,
                    family.contact_mask,
                    min_separation=args.min_separation,
                ),
                "normalized_contact_p_at_l": binary_contact_precision(
                    normalized_map,
                    family.contacts,
                    family.contact_mask,
                    min_separation=args.min_separation,
                ),
                "residual_contact_p_at_l": binary_contact_precision(
                    residual_map,
                    family.contacts,
                    family.contact_mask,
                    min_separation=args.min_separation,
                ),
                "raw_contact_average_precision": binary_average_precision(
                    raw_strength[valid], contact[valid]
                ),
                "normalized_contact_average_precision": binary_average_precision(
                    normalized_strength[valid], contact[valid]
                ),
                "residual_contact_average_precision": binary_average_precision(
                    residual_strength[valid], contact[valid]
                ),
                "raw_contact_roc_auc": binary_roc_auc(
                    raw_strength[valid], contact[valid]
                ),
                "normalized_contact_roc_auc": binary_roc_auc(
                    normalized_strength[valid], contact[valid]
                ),
                "residual_contact_roc_auc": binary_roc_auc(
                    residual_strength[valid], contact[valid]
                ),
                "active_fraction": float(np.mean(active)),
                "active_contact_precision": float(
                    np.mean(contact[valid & active])
                    if np.any(valid & active)
                    else np.nan
                ),
                "raw_leading_mode_fraction": leading_mode_fraction(raw_map),
                "residual_leading_mode_fraction": leading_mode_fraction(
                    residual_map
                ),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(args.output / "per_family_metrics.csv", index=False)
    summary = {
        column: {
            "mean": float(result[column].mean()),
            "median": float(result[column].median()),
        }
        for column in result.columns
        if column not in {"x_id", "role"}
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    print(
        result.drop(columns=["x_id", "role"])
        .mean()
        .to_string(float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
