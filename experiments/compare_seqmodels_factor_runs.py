from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUNS = {
    "Raw connected": "seqmodels_factor_head_rank32_v1",
    "Entropy null": "seqmodels_factor_head_rank32_v2_entropy_null",
    "Prior centered": "seqmodels_factor_head_rank32_v3_prior_centered",
    "Prior centered, 10 epochs": "seqmodels_factor_head_rank32_v4_10epoch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, directory in RUNS.items():
        path = args.results / directory
        summary = json.loads((path / "summary.json").read_text())
        run = json.loads((path / "run.json").read_text())
        for role, values in summary.items():
            rows.append(
                {
                    "model": label,
                    "role": role,
                    "epochs": run["epochs"],
                    "marginal_kl": values["marginal_kl_mean"],
                    "entropy_spearman": values["entropy_spearman_mean"],
                    "teacher_strength_spearman": values[
                        "teacher_strength_spearman_mean"
                    ],
                    "contact_prevalence": values[
                        "contact_prevalence_long_mean"
                    ],
                    "contact_p_at_l": values["predicted_long_p_at_l_mean"],
                    "gate_contact_p_at_l": values["gate_long_p_at_l_mean"],
                    "gate_mean": values["gate_mean_mean"],
                    "leading_mode_fraction": values[
                        "score_leading_mode_fraction_mean"
                    ],
                    "gate_teacher_ap": values.get(
                        "gate_teacher_average_precision_mean"
                    ),
                    "gate_teacher_auc": values.get("gate_teacher_roc_auc_mean"),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "seqmodels_run_comparison.csv", index=False)

    test = frame[frame["role"] == "test"].set_index("model").loc[list(RUNS)]
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    colors = ["#6b7280", "#2a9d8f", "#d1495b", "#8f5d9f"]
    axes[0].bar(test.index, test["contact_p_at_l"], color=colors)
    axes[0].axhline(
        test["contact_prevalence"].mean(), color="black", linestyle="--", linewidth=1
    )
    axes[0].set_title("Held-out contact P@L")
    axes[0].set_ylabel("Precision")
    axes[1].bar(test.index, test["leading_mode_fraction"], color=colors)
    axes[1].set_title("Leading-mode fraction")
    axes[1].set_ylabel("Fraction")
    axes[2].bar(test.index, test["gate_teacher_auc"].fillna(0), color=colors)
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Sparse-teacher ROC AUC")
    axes[2].set_ylabel("AUC")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output / "seqmodels_run_comparison.png", dpi=200)
    plt.close(figure)

    columns = [
        "contact_prevalence",
        "contact_p_at_l",
        "gate_contact_p_at_l",
        "teacher_strength_spearman",
        "gate_teacher_ap",
        "gate_teacher_auc",
        "leading_mode_fraction",
        "marginal_kl",
        "entropy_spearman",
    ]
    print(test[columns].to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
