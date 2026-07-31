from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


ALPHABET = "-ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INDEX = {aa: index for index, aa in enumerate(ALPHABET)}


def load_aln(path: str | Path) -> list[str]:
    sequences = [line.strip().upper() for line in Path(path).read_text().splitlines()]
    sequences = [sequence for sequence in sequences if sequence]
    if not sequences:
        raise ValueError(f"No sequences found in {path}")
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError(f"Alignment contains unequal sequence lengths: {sorted(lengths)}")
    invalid = sorted(set("".join(sequences)) - set(ALPHABET))
    if invalid:
        raise ValueError(f"Alignment contains unsupported symbols: {invalid}")
    return sequences


def encode_alignment(sequences: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[AA_TO_INDEX[aa] for aa in sequence] for sequence in sequences],
        dtype=np.int16,
    )


def subsample_alignment(
    sequences: Sequence[str], depth: int, rng: np.random.Generator
) -> list[str]:
    if depth < 1:
        raise ValueError("depth must be positive")
    if depth > len(sequences):
        raise ValueError(f"Requested depth {depth} exceeds MSA depth {len(sequences)}")
    if depth == 1:
        return [sequences[0]]
    selected = rng.choice(np.arange(1, len(sequences)), size=depth - 1, replace=False)
    return [sequences[0], *(sequences[index] for index in selected)]


def site_frequencies(encoded: np.ndarray, states: int = len(ALPHABET)) -> np.ndarray:
    if encoded.ndim != 2:
        raise ValueError("encoded alignment must have shape [depth, length]")
    frequencies = np.zeros((encoded.shape[1], states), dtype=np.float64)
    for state in range(states):
        frequencies[:, state] = np.mean(encoded == state, axis=0)
    return frequencies


def sample_pssm_null(
    reference_sequences: Sequence[str],
    depth: int,
    rng: np.random.Generator,
    keep_query: bool = True,
) -> list[str]:
    encoded = encode_alignment(reference_sequences)
    frequencies = site_frequencies(encoded)
    length = encoded.shape[1]
    sampled = np.empty((depth, length), dtype=np.int16)
    for position in range(length):
        sampled[:, position] = rng.choice(
            len(ALPHABET), size=depth, replace=True, p=frequencies[position]
        )
    if keep_query:
        sampled[0] = encoded[0]
    return ["".join(ALPHABET[index] for index in row) for row in sampled]


def shuffle_columns_null(
    reference_sequences: Sequence[str], rng: np.random.Generator
) -> list[str]:
    encoded = encode_alignment(reference_sequences)
    shuffled = encoded.copy()
    for position in range(shuffled.shape[1]):
        shuffled[:, position] = rng.permutation(shuffled[:, position])
    return ["".join(ALPHABET[index] for index in row) for row in shuffled]


def sample_global_composition_null(
    reference_sequences: Sequence[str],
    depth: int,
    rng: np.random.Generator,
) -> list[str]:
    encoded = encode_alignment(reference_sequences)
    composition = np.bincount(encoded.ravel(), minlength=len(ALPHABET)).astype(np.float64)
    composition /= composition.sum()
    sampled = rng.choice(
        len(ALPHABET),
        size=(depth, encoded.shape[1]),
        replace=True,
        p=composition,
    )
    return ["".join(ALPHABET[index] for index in row) for row in sampled]
