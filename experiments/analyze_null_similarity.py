from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from transformer_disentanglement.contacts import load_3cnba_distances
from transformer_disentanglement.metrics import (
    average_product_correction,
    contact_precision,
    safe_spearman,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def matrix_correlation(a: np.ndarray, b: np.ndarray, separation: int) -> float:
    i, j = np.triu_indices(a.shape[0], k=separation)
    return safe_spearman(a[i, j], b[i, j])


def main() -> None:
    args = parse_args()
    arrays = np.load(args.arrays)
    distances = load_3cnba_distances(args.distances, args.reference)
    conditions = sorted(
        key.removesuffix("__contacts")
        for key in arrays.files
        if key.endswith("__contacts") and not key.startswith("real__")
    )
    real_contacts = arrays["real__contacts"]
    real_attention = arrays["real__attention_heads"]
    real_mi = arrays["real__mi"]
    real_mi_apc = arrays["real__mi_apc"]

    rows = []
    head_rows = []
    for condition in conditions:
        contacts = arrays[f"{condition}__contacts"]
        attention = arrays[f"{condition}__attention_heads"]
        mi = arrays[f"{condition}__mi"]
        mi_apc = arrays[f"{condition}__mi_apc"]
        correlations_raw = []
        correlations_apc = []
        for layer in range(real_attention.shape[0]):
            for head in range(real_attention.shape[1]):
                real_head = real_attention[layer, head]
                null_head = attention[layer, head]
                raw_correlation = matrix_correlation(real_head, null_head, 24)
                apc_correlation = matrix_correlation(
                    average_product_correction(real_head),
                    average_product_correction(null_head),
                    24,
                )
                correlations_raw.append(raw_correlation)
                correlations_apc.append(apc_correlation)
                head_rows.append(
                    {
                        "condition": condition,
                        "layer": layer,
                        "head": head,
                        "raw_long_correlation": raw_correlation,
                        "apc_long_correlation": apc_correlation,
                        "real_apc_long_p_at_l": contact_precision(
                            average_product_correction(real_head),
                            distances,
                            min_separation=24,
                        ),
                        "null_apc_long_p_at_l": contact_precision(
                            average_product_correction(null_head),
                            distances,
                            min_separation=24,
                        ),
                    }
                )
        correlations_raw = np.asarray(correlations_raw)
        correlations_apc = np.asarray(correlations_apc)
        rows.append(
            {
                "condition": condition,
                "contacts_long_correlation": matrix_correlation(
                    real_contacts, contacts, 24
                ),
                "contacts_apc_long_correlation": matrix_correlation(
                    average_product_correction(real_contacts),
                    average_product_correction(contacts),
                    24,
                ),
                "mi_long_correlation": matrix_correlation(real_mi, mi, 24),
                "mi_apc_long_correlation": matrix_correlation(
                    real_mi_apc, mi_apc, 24
                ),
                "attention_head_raw_median_correlation": float(
                    np.median(correlations_raw)
                ),
                "attention_head_raw_min_correlation": float(np.min(correlations_raw)),
                "attention_head_apc_median_correlation": float(
                    np.median(correlations_apc)
                ),
                "attention_head_apc_min_correlation": float(np.min(correlations_apc)),
                "attention_head_apc_fraction_below_0_9": float(
                    np.mean(correlations_apc < 0.9)
                ),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    heads = pd.DataFrame(head_rows)
    summary.to_csv(args.output / "null_similarity.csv", index=False)
    heads.to_csv(args.output / "head_similarity.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

