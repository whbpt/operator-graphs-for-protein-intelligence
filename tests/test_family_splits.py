import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "make_family_splits.py"
SPEC = importlib.util.spec_from_file_location("make_family_splits", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cluster_split_has_no_group_leakage() -> None:
    frame = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            "length": [100, 105, 120, 140, 160, 180, 200, 220, 240, 260],
            "gap_fraction": [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19],
            "positive_pair_fraction_sep6": [
                0.02,
                0.021,
                0.023,
                0.025,
                0.027,
                0.029,
                0.031,
                0.033,
                0.035,
                0.037,
            ],
        }
    )
    assignment = MODULE.assign_cluster_splits(frame, 0.6, 0.2, seed=3)
    assert set(assignment) == set(frame["cluster_id"])
    assert set(assignment.values()) == {"train", "validation", "test"}


def test_alignment_identity_for_related_sequences() -> None:
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner(mode="local")
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    identity, coverage = MODULE.alignment_identity(
        aligner, "ACDEFGHIK", "ACDEYGHIK"
    )
    assert identity > 0.85
    assert coverage == 1.0


def test_calibration_selection_keeps_clusters_whole() -> None:
    frame = pd.DataFrame(
        {
            "index": list(range(8)),
            "cluster_id": [0, 0, 1, 2, 3, 4, 5, 6],
            "length": [100, 102, 120, 140, 160, 180, 200, 220],
            "gap_fraction": [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
            "positive_pair_fraction_sep6": [
                0.02,
                0.021,
                0.023,
                0.025,
                0.027,
                0.029,
                0.031,
                0.033,
            ],
        }
    )
    selected = MODULE.farthest_point_calibration_clusters(frame, count=3, seed=5)
    selected_count = frame["cluster_id"].isin(selected).sum()
    assert selected_count == 3
