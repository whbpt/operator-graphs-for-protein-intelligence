from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformer_disentanglement.protein_transformer import choose_device, load_model
from transformer_disentanglement.seqmodels_benchmark import (
    load_seqmodels_family,
    weighted_pssm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["esm2_8m", "esm2_35m"], default="esm2_8m")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    representations_dir = args.output / "families"
    representations_dir.mkdir()

    frame = pd.read_csv(args.splits / "families_with_splits.csv")
    device = choose_device(args.device)
    model, alphabet, architecture = load_model(args.model, device)
    if architecture != "single_sequence":
        raise ValueError("Representation cache requires a single-sequence model")
    layer = int(model.num_layers)
    batch_converter = alphabet.get_batch_converter()
    output_rows = []
    for start in range(0, len(frame), args.batch_size):
        batch_frame = frame.iloc[start : start + args.batch_size]
        families = [
            load_seqmodels_family(args.benchmark, row.file, row.x_id)
            for row in batch_frame.itertuples(index=False)
        ]
        batch = [(family.identifier, family.query) for family in families]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            model_output = model(tokens, repr_layers=[layer])
        representations = model_output["representations"][layer]
        for batch_index, (row, family) in enumerate(
            zip(batch_frame.itertuples(index=False), families)
        ):
            hidden = representations[
                batch_index, 1 : len(family.query) + 1
            ].float().cpu().numpy()
            pssm = weighted_pssm(family.msa, family.weights)
            filename = f"{int(row.index):04d}_{row.x_id}.npz"
            np.savez_compressed(
                representations_dir / filename,
                hidden=hidden.astype(np.float16),
                pssm=pssm.astype(np.float32),
                query_positions=family.query_positions,
            )
            output_rows.append(
                {
                    **row._asdict(),
                    "representation_file": filename,
                    "query_length": len(family.query),
                }
            )

    pd.DataFrame(output_rows).to_csv(args.output / "families.csv", index=False)
    metadata = {
        "schema_version": 1,
        "model": args.model,
        "layer": layer,
        "device": str(device),
        "family_count": len(output_rows),
        "hidden_dtype": "float16",
        "pssm_dtype": "float32",
        "query_gap_columns_removed": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
