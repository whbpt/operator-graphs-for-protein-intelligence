from __future__ import annotations

import numpy as np


def oracle_token_mask(
    signal: np.ndarray,
    valid: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Select the highest-signal valid positions under an exact token budget."""
    signal = np.asarray(signal, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    selected = np.zeros_like(valid)
    candidates = np.flatnonzero(valid)
    count = min(int(budget), len(candidates))
    if count:
        order = np.argsort(signal[candidates], kind="stable")[-count:]
        selected[candidates[order]] = True
    return selected


def oracle_partition_block_mask(
    signal: np.ndarray,
    valid: np.ndarray,
    block_size: int,
    budget: int,
) -> np.ndarray:
    """Select fixed, sequence-aligned blocks with the largest valid signal mass."""
    signal = np.asarray(signal, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if budget < block_size:
        raise ValueError("budget must cover at least one block")
    length = len(signal)
    starts = np.arange(0, length, block_size)
    scores = np.array(
        [
            np.sum(signal[start : start + block_size] * valid[start : start + block_size])
            for start in starts
        ]
    )
    block_count = min(budget // block_size, len(starts))
    selected = np.zeros_like(valid)
    if block_count:
        chosen = np.argsort(scores, kind="stable")[-block_count:]
        for block_index in chosen:
            start = starts[block_index]
            selected[start : start + block_size] = True
    return selected & valid


def oracle_sliding_block_mask(
    signal: np.ndarray,
    valid: np.ndarray,
    block_size: int,
    budget: int,
) -> np.ndarray:
    """Exactly select non-overlapping fixed-width windows with arbitrary starts."""
    signal = np.asarray(signal, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if budget < block_size:
        raise ValueError("budget must cover at least one block")
    length = len(signal)
    block_count = min(budget // block_size, length // block_size)
    weighted = signal * valid
    prefix = np.concatenate([[0.0], np.cumsum(weighted)])
    block_scores = prefix[block_size:] - prefix[:-block_size]
    scores = np.zeros((block_count + 1, length + 1), dtype=float)
    choices = np.zeros((block_count + 1, length + 1), dtype=bool)
    for count in range(1, block_count + 1):
        for end in range(1, length + 1):
            scores[count, end] = scores[count, end - 1]
            if end >= block_size:
                candidate = (
                    scores[count - 1, end - block_size]
                    + block_scores[end - block_size]
                )
                if candidate > scores[count, end]:
                    scores[count, end] = candidate
                    choices[count, end] = True
    selected = np.zeros_like(valid)
    count = block_count
    end = length
    while count and end:
        if choices[count, end]:
            selected[end - block_size : end] = True
            end -= block_size
            count -= 1
        else:
            end -= 1
    return selected & valid


def block_concentration_metrics(
    signal: np.ndarray,
    valid: np.ndarray,
    block_size: int,
    budget: int,
    block_mode: str = "partition",
) -> dict[str, float]:
    """Compare oracle contiguous blocks with oracle individual-token selection."""
    signal = np.maximum(np.asarray(signal, dtype=float), 0.0)
    valid = np.asarray(valid, dtype=bool)
    token_mask = oracle_token_mask(signal, valid, budget)
    if block_mode == "partition":
        block_mask = oracle_partition_block_mask(signal, valid, block_size, budget)
    elif block_mode == "sliding":
        block_mask = oracle_sliding_block_mask(signal, valid, block_size, budget)
    else:
        raise ValueError(f"Unknown block mode: {block_mode}")
    total_mass = float(np.sum(signal[valid]))
    token_mass = float(np.sum(signal[token_mask]))
    block_mass = float(np.sum(signal[block_mask]))
    top_count = int(token_mask.sum())
    overlap = int(np.sum(token_mask & block_mask))
    return {
        "total_mass": total_mass,
        "token_mass_fraction": token_mass / max(total_mass, 1e-12),
        "block_mass_fraction": block_mass / max(total_mass, 1e-12),
        "block_to_token_mass": block_mass / max(token_mass, 1e-12),
        "top_token_recall": overlap / max(top_count, 1),
        "selected_valid_positions": float(block_mask.sum()),
    }


def top_position_fragmentation(
    signal: np.ndarray,
    valid: np.ndarray,
    budget: int,
) -> dict[str, float]:
    """Measure how fragmented the strongest positions are along the sequence."""
    selected = np.flatnonzero(oracle_token_mask(signal, valid, budget))
    if not len(selected):
        return {"components": 0.0, "component_fraction": 0.0, "span_ratio": 0.0}
    components = 1 + int(np.sum(np.diff(selected) > 1))
    span = int(selected[-1] - selected[0] + 1)
    return {
        "components": float(components),
        "component_fraction": components / len(selected),
        "span_ratio": span / len(selected),
    }
