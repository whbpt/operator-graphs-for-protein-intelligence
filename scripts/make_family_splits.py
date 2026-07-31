from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices


AA_ORDER = np.asarray(list("ARNDCQEGHILKMFPSTWYV-"))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.25)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def load_queries(benchmark: Path, manifest: pd.DataFrame) -> list[str]:
    queries = []
    for filename in manifest["file"]:
        with np.load(benchmark / "families" / filename, allow_pickle=False) as data:
            sequence = "".join(AA_ORDER[data["x"][0]])
        queries.append(sequence.replace("-", ""))
    return queries


def alignment_identity(
    aligner: Align.PairwiseAligner, left: str, right: str
) -> tuple[float, float]:
    alignment = aligner.align(left, right)[0]
    matches = 0
    alignment_columns = 0
    coordinates = alignment.coordinates
    for segment in range(coordinates.shape[1] - 1):
        left_start, left_end = coordinates[0, segment : segment + 2]
        right_start, right_end = coordinates[1, segment : segment + 2]
        left_step = int(left_end - left_start)
        right_step = int(right_end - right_start)
        alignment_columns += max(left_step, right_step)
        if left_step and right_step:
            left_segment = left[left_start:left_end]
            right_segment = right[right_start:right_end]
            matches += sum(a == b for a, b in zip(left_segment, right_segment))
    if alignment_columns == 0:
        return 0.0, 0.0
    identity = matches / alignment_columns
    left_coverage = (coordinates[0, -1] - coordinates[0, 0]) / len(left)
    right_coverage = (coordinates[1, -1] - coordinates[1, 0]) / len(right)
    coverage = min(left_coverage, right_coverage)
    return float(identity), float(coverage)


def build_homology_clusters(
    sequences: list[str], identity_threshold: float, coverage_threshold: float
) -> tuple[list[int], list[dict[str, float | int]]]:
    aligner = Align.PairwiseAligner(mode="local")
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    union_find = UnionFind(len(sequences))
    edges = []
    for left in range(len(sequences)):
        for right in range(left):
            identity, coverage = alignment_identity(
                aligner, sequences[left], sequences[right]
            )
            if identity >= identity_threshold and coverage >= coverage_threshold:
                union_find.union(left, right)
                edges.append(
                    {
                        "left_index": left,
                        "right_index": right,
                        "identity": identity,
                        "coverage": coverage,
                    }
                )
    roots = [union_find.find(index) for index in range(len(sequences))]
    root_to_cluster = {
        root: cluster for cluster, root in enumerate(sorted(set(roots)))
    }
    return [root_to_cluster[root] for root in roots], edges


def quantile_bins(values: pd.Series, bins: int) -> np.ndarray:
    ranks = values.rank(method="first", pct=True).to_numpy()
    return np.minimum((ranks * bins).astype(int), bins - 1)


def assign_cluster_splits(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[int, str]:
    split_names = ("train", "validation", "test")
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": 1.0 - train_fraction - validation_fraction,
    }
    if min(fractions.values()) <= 0:
        raise ValueError("Split fractions must all be positive")

    frame = frame.copy()
    frame["stratum"] = (
        quantile_bins(frame["length"], 4).astype(str)
        + "_"
        + quantile_bins(frame["gap_fraction"], 3).astype(str)
        + "_"
        + quantile_bins(frame["positive_pair_fraction_sep6"], 3).astype(str)
    )
    total_strata = Counter(frame["stratum"])
    target_size = {name: len(frame) * fraction for name, fraction in fractions.items()}
    target_strata = {
        name: {stratum: count * fractions[name] for stratum, count in total_strata.items()}
        for name in split_names
    }
    cluster_members = {
        int(cluster): members
        for cluster, members in frame.groupby("cluster_id", sort=False)
    }
    rng = np.random.default_rng(seed)
    cluster_order = list(cluster_members)
    rng.shuffle(cluster_order)
    cluster_order.sort(key=lambda cluster: len(cluster_members[cluster]), reverse=True)
    assigned_size = Counter()
    assigned_strata: dict[str, Counter[str]] = defaultdict(Counter)
    assignment: dict[int, str] = {}
    for cluster in cluster_order:
        members = cluster_members[cluster]
        cluster_strata = Counter(members["stratum"])
        best_name = ""
        best_cost = float("inf")
        for name in split_names:
            size_fill = (assigned_size[name] + len(members)) / max(
                target_size[name], 1.0
            )
            stratum_fill = []
            for stratum, count in cluster_strata.items():
                target = max(target_strata[name][stratum], 1.0)
                stratum_fill.append(
                    (assigned_strata[name][stratum] + count) / target
                )
            overfill = max(
                0.0,
                assigned_size[name] + len(members) - np.ceil(target_size[name]),
            )
            cost = 2.0 * size_fill + float(np.mean(stratum_fill)) + 4.0 * overfill**2
            if cost < best_cost:
                best_cost = cost
                best_name = name
        assignment[cluster] = best_name
        assigned_size[best_name] += len(members)
        assigned_strata[best_name].update(cluster_strata)
    return assignment


def farthest_point_calibration_clusters(
    frame: pd.DataFrame, count: int, seed: int
) -> set[int]:
    if count > len(frame):
        raise ValueError("Calibration count exceeds training families")
    clusters = (
        frame.groupby("cluster_id", as_index=False)
        .agg(
            index=("index", "min"),
            length=("length", "mean"),
            gap_fraction=("gap_fraction", "mean"),
            positive_pair_fraction_sep6=("positive_pair_fraction_sep6", "mean"),
            cluster_size=("index", "size"),
        )
        .reset_index(drop=True)
    )
    features = np.stack(
        [
            np.log(clusters["length"].to_numpy()),
            clusters["gap_fraction"].to_numpy(),
            clusters["positive_pair_fraction_sep6"].to_numpy(),
        ],
        axis=-1,
    )
    features = (features - features.mean(axis=0)) / features.std(axis=0).clip(min=1e-8)
    rng = np.random.default_rng(seed)
    median = np.median(features, axis=0)
    first_candidates = np.flatnonzero(
        np.isclose(np.linalg.norm(features - median, axis=-1), np.min(np.linalg.norm(features - median, axis=-1)))
    )
    candidate_order = [int(rng.choice(first_candidates))]
    min_distance = np.linalg.norm(features - features[candidate_order[0]], axis=-1)
    while len(candidate_order) < len(clusters):
        min_distance[candidate_order] = -1.0
        next_index = int(np.argmax(min_distance))
        candidate_order.append(next_index)
        distance = np.linalg.norm(features - features[next_index], axis=-1)
        min_distance = np.minimum(min_distance, distance)

    selected_clusters: set[int] = set()
    selected_families = 0
    for row_index in candidate_order:
        cluster_size = int(clusters.iloc[row_index]["cluster_size"])
        if selected_families + cluster_size > count:
            continue
        selected_clusters.add(int(clusters.iloc[row_index]["cluster_id"]))
        selected_families += cluster_size
        if selected_families == count:
            break
    if selected_families != count:
        raise ValueError(f"Could not select exactly {count} calibration families by cluster")
    return selected_clusters


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    manifest = pd.read_csv(args.benchmark / "families.csv")
    queries = load_queries(args.benchmark, manifest)
    clusters, edges = build_homology_clusters(
        queries, args.identity_threshold, args.coverage_threshold
    )
    manifest["cluster_id"] = clusters
    assignment = assign_cluster_splits(
        manifest, args.train_fraction, args.validation_fraction, args.seed
    )
    manifest["split"] = [assignment[cluster] for cluster in clusters]
    calibration_clusters = farthest_point_calibration_clusters(
        manifest[manifest["split"] == "train"], args.calibration_count, args.seed
    )
    manifest["calibration"] = manifest["cluster_id"].isin(calibration_clusters)
    manifest["role"] = manifest["split"]
    manifest.loc[manifest["calibration"], "role"] = "calibration"

    args.output.mkdir(parents=True)
    manifest.to_csv(args.output / "families_with_splits.csv", index=False)
    with (args.output / "queries.fasta").open("w") as handle:
        for identifier, sequence in zip(manifest["x_id"], queries):
            handle.write(f">{identifier}\n{sequence}\n")
    edge_rows = []
    for edge in edges:
        edge_rows.append(
            {
                **edge,
                "left_id": manifest.iloc[int(edge["left_index"])]["x_id"],
                "right_id": manifest.iloc[int(edge["right_index"])]["x_id"],
            }
        )
    if edge_rows:
        write_csv(args.output / "homology_edges.csv", edge_rows)
    else:
        (args.output / "homology_edges.csv").write_text(
            "left_index,right_index,identity,coverage,left_id,right_id\n"
        )

    role_summary = {}
    for name, group in manifest.groupby("role"):
        role_summary[name] = {
            "families": int(len(group)),
            "clusters": int(group["cluster_id"].nunique()),
            "length_mean": float(group["length"].mean()),
            "gap_fraction_mean": float(group["gap_fraction"].mean()),
            "contact_fraction_mean": float(
                group["positive_pair_fraction_sep6"].mean()
            ),
        }
    metadata = {
        "schema_version": 1,
        "seed": args.seed,
        "identity_threshold": args.identity_threshold,
        "coverage_threshold": args.coverage_threshold,
        "homology_edges": len(edges),
        "homology_clusters": int(manifest["cluster_id"].nunique()),
        "largest_cluster": int(manifest.groupby("cluster_id").size().max()),
        "calibration_count": int(manifest["calibration"].sum()),
        "roles": role_summary,
    }
    (args.output / "split_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
