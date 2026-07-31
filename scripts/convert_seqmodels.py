from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickletools
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npformat


ALLOWED_PICKLE_GLOBALS = {
    "numpy.core.multiarray _reconstruct",
    "numpy.core.multiarray scalar",
    "numpy._core.multiarray _reconstruct",
    "numpy._core.multiarray scalar",
    "numpy ndarray",
    "numpy dtype",
    "_codecs encode",
}
REQUIRED_FIELDS = {"x", "x_w", "x_true", "x_mask", "x_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_numpy_pickle(path: Path) -> dict[str, Any]:
    """Inspect pickle globals embedded in an object NPY without executing it."""
    with path.open("rb") as handle:
        version = npformat.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = npformat.read_array_header_1_0(handle)
        else:
            shape, fortran_order, dtype = npformat.read_array_header_2_0(handle)
        if not dtype.hasobject:
            raise ValueError("Expected an object NPY requiring one-time conversion")
        globals_seen: set[str] = set()
        stack_globals = 0
        opcode_count = 0
        for opcode, argument, _ in pickletools.genops(handle):
            opcode_count += 1
            if opcode.name == "GLOBAL":
                globals_seen.add(str(argument))
            elif opcode.name == "STACK_GLOBAL":
                stack_globals += 1
    unexpected = sorted(globals_seen - ALLOWED_PICKLE_GLOBALS)
    if stack_globals or unexpected:
        raise ValueError(
            "Unsafe or unexpected pickle globals: "
            f"stack_globals={stack_globals}, unexpected={unexpected}"
        )
    return {
        "npy_version": list(version),
        "shape": list(shape),
        "fortran_order": bool(fortran_order),
        "dtype": str(dtype),
        "opcode_count": opcode_count,
        "globals": sorted(globals_seen),
    }


def slugify(identifier: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier).strip("._")
    return slug or "family"


def validate_family(family: Any) -> dict[str, Any]:
    if not isinstance(family, dict):
        raise TypeError(f"Expected family dictionary, found {type(family)!r}")
    missing = REQUIRED_FIELDS - set(family)
    if missing:
        raise ValueError(f"Family is missing fields: {sorted(missing)}")

    identifier = str(family["x_id"])
    x = np.asarray(family["x"])
    weights = np.asarray(family["x_w"])
    contacts = np.asarray(family["x_true"])
    mask = np.asarray(family["x_mask"])
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.integer):
        raise ValueError(f"{identifier}: x must be a 2D integer array")
    depth, length = x.shape
    if depth < 1 or length < 1 or np.min(x) < 0 or np.max(x) > 20:
        raise ValueError(f"{identifier}: invalid MSA shape or state range")
    if weights.shape != (depth,) or not np.all(np.isfinite(weights)):
        raise ValueError(f"{identifier}: invalid sequence weights")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(f"{identifier}: sequence weights must be non-negative")
    if contacts.shape != (length, length) or mask.shape != (length, length):
        raise ValueError(f"{identifier}: contact and mask shapes must be [L, L]")
    if not np.all(np.isfinite(contacts)):
        raise ValueError(f"{identifier}: contact matrix contains non-finite values")

    return {
        "x_id": identifier,
        "x": x.astype(np.uint8, copy=False),
        "x_w": weights.astype(np.float32, copy=False),
        "x_true": contacts.astype(np.float32, copy=False),
        "x_mask": mask.astype(bool, copy=False),
    }


def family_manifest_row(index: int, filename: str, family: dict[str, Any]) -> dict[str, Any]:
    x = family["x"]
    weights = family["x_w"]
    contacts = family["x_true"]
    mask = family["x_mask"]
    length = x.shape[1]
    i, j = np.triu_indices(length, k=6)
    valid = mask[i, j]
    contact_values = contacts[i[valid], j[valid]] if np.any(valid) else np.asarray([])
    return {
        "index": index,
        "x_id": family["x_id"],
        "file": filename,
        "depth": int(x.shape[0]),
        "length": int(length),
        "neff": float(weights.sum()),
        "gap_fraction": float(np.mean(x == 20)),
        "valid_pairs_sep6": int(np.sum(valid)),
        "positive_pairs_sep6": int(np.sum(contact_values > 0.01)),
        "positive_pair_fraction_sep6": (
            float(np.mean(contact_values > 0.01)) if len(contact_values) else 0.0
        ),
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_hash = sha256(args.input)
    audit = audit_numpy_pickle(args.input)
    print(json.dumps({"source_sha256": source_hash, "audit": audit}, indent=2))
    if args.audit_only:
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    staging = args.output.with_name(f".{args.output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    families_dir = staging / "families"
    families_dir.mkdir(parents=True)

    # This is the only intentionally permitted pickle load in the project.
    raw = np.load(args.input, allow_pickle=True)
    families = list(raw)
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw_family in enumerate(families):
        family = validate_family(raw_family)
        if family["x_id"] in identifiers:
            raise ValueError(f"Duplicate family identifier: {family['x_id']}")
        identifiers.add(family["x_id"])
        filename = f"{index:04d}_{slugify(family['x_id'])}.npz"
        destination = families_dir / filename
        np.savez_compressed(
            destination,
            x=family["x"],
            x_w=family["x_w"],
            x_true=family["x_true"],
            x_mask=family["x_mask"],
        )
        rows.append(family_manifest_row(index, filename, family))

    write_manifest(staging / "families.csv", rows)
    metadata = {
        "schema_version": 1,
        "family_count": len(rows),
        "source_file": args.input.name,
        "source_sha256": source_hash,
        "pickle_audit": audit,
        "states": 21,
        "gap_state": 20,
        "fields": {
            "x": "uint8 [depth, length]",
            "x_w": "float32 [depth]",
            "x_true": "float32 [length, length]",
            "x_mask": "bool [length, length]",
        },
    }
    (staging / "manifest.json").write_text(json.dumps(metadata, indent=2))
    staging.rename(args.output)
    print(f"Converted {len(rows)} families into {args.output}")


if __name__ == "__main__":
    main()
