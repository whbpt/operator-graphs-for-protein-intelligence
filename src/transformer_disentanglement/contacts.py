from __future__ import annotations

from pathlib import Path

import numpy as np


def load_3cnba_distances(
    dist_path: str | Path, reference_path: str | Path
) -> np.ndarray:
    reference_lines = Path(reference_path).read_text().splitlines()
    if len(reference_lines) < 2:
        raise ValueError("Reference mapping requires original and aligned sequences")
    aligned_reference = reference_lines[1].strip()
    coordinate_to_alignment: dict[int, int] = {}
    aligned_index = 0
    for coordinate, amino_acid in enumerate(aligned_reference):
        if amino_acid != "-":
            coordinate_to_alignment[coordinate] = aligned_index
            aligned_index += 1

    distances = np.full((aligned_index, aligned_index), np.nan, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    for line in Path(dist_path).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        coordinate_i, coordinate_j = int(fields[0]), int(fields[2])
        if coordinate_i not in coordinate_to_alignment or coordinate_j not in coordinate_to_alignment:
            continue
        i = coordinate_to_alignment[coordinate_i]
        j = coordinate_to_alignment[coordinate_j]
        distance = float(fields[4])
        current = distances[i, j]
        if np.isnan(current) or distance < current:
            distances[i, j] = distance
            distances[j, i] = distance
    return distances

