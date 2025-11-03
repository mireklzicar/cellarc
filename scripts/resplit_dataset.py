#!/usr/bin/env python3
"""
Rebuild the CellARc dataset splits with coverage-aware metadata.

Steps performed:
  * Load all existing split records from the extended (meta-rich) dataset.
  * Compute query-window statistics and compression-based difficulty metrics.
  * Promote `schema_version` and rule table `format_version` to 1.0.2.
  * Assign the lowest-coverage episodes to `test_extrapolation`.
  * Shuffle the remaining pool deterministically and slice into train/val/test_interpolation.
  * Emit refreshed JSONL files containing the updated metadata.

The resulting directory can be fed into `scripts/build_hf_dataset.py`
to regenerate the lightweight and extended Hugging Face packages.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_SPLITS: Tuple[str, ...] = ("train", "val", "test_interpolation", "test_extrapolation")
SCHEMA_VERSION = "1.0.2"
RULE_TABLE_VERSION = "1.0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/datasets/cellarc_100k_meta/data"),
        help="Directory that contains the existing extended dataset JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/processing/resplit_splits"),
        help="Where to write the refreshed split JSONL files.",
    )
    parser.add_argument(
        "--test-extrapolation-size",
        type=int,
        default=1000,
        help="Number of lowest coverage episodes assigned to test_extrapolation.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=100_000,
        help="Number of episodes to allocate to the train split.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1_000,
        help="Number of episodes to allocate to the validation split.",
    )
    parser.add_argument(
        "--test-interpolation-size",
        type=int,
        default=1_000,
        help="Number of episodes to allocate to the test_interpolation split.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=12345,
        help="Seed used when shuffling the remaining pool.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    return parser.parse_args()


def sequence_to_bytes(seq: Sequence[int]) -> bytes:
    """Convert an integer sequence into bytes for compression."""

    if not seq:
        return b""
    return bytes(int(v) % 256 for v in seq)


def compression_size(payload: bytes) -> int:
    """Return the compressed size of the payload using zlib."""

    return len(zlib.compress(payload, level=9))


def normalized_compression_distance(a: Sequence[int], b: Sequence[int]) -> Optional[float]:
    """Compute the Normalized Compression Distance between two sequences."""

    a_bytes = sequence_to_bytes(a)
    b_bytes = sequence_to_bytes(b)
    if not a_bytes and not b_bytes:
        return None

    c_x = compression_size(a_bytes)
    c_y = compression_size(b_bytes)
    c_xy = compression_size(a_bytes + b_bytes)

    denom = max(c_x, c_y)
    if denom == 0:
        return None
    return (c_xy - min(c_x, c_y)) / denom


def centered_window(seq: Sequence[int], idx: int, width: int, wrap: bool) -> Tuple[int, ...]:
    """Return the neighbourhood window centred on idx."""

    half = width // 2
    n = len(seq)
    if n == 0 or width <= 0:
        return ()
    if wrap:
        return tuple(seq[(idx - half + j) % n] for j in range(width))
    window: List[int] = []
    for j in range(idx - half, idx + half + 1):
        if 0 <= j < n:
            window.append(seq[j])
        else:
            window.append(0)
    return tuple(window)


def collect_train_windows(train_pairs: Sequence[Dict[str, Sequence[int]]], width: int) -> Counter:
    """Count centred windows observed across the training inputs."""

    counts: Counter = Counter()
    if width <= 0:
        return counts
    half = width // 2
    for pair in train_pairs:
        seq = pair.get("input", [])
        n = len(seq)
        if n < width:
            continue
        for idx in range(half, n - half):
            win = centered_window(seq, idx, width, wrap=True)
            counts[win] += 1
    return counts


def collect_query_windows(query: Sequence[int], width: int, wrap: bool) -> Counter:
    """Count centred windows present in the query sequence."""

    counts: Counter = Counter()
    if width <= 0:
        return counts
    n = len(query)
    if n == 0:
        return counts
    for idx in range(n):
        win = centered_window(query, idx, width, wrap=wrap)
        counts[win] += 1
    return counts


@dataclass
class Episode:
    """In-memory representation of an episode with derived metrics."""

    payload: Dict[str, object]
    coverage_weighted: float


def load_records(source_dir: Path) -> List[Dict[str, object]]:
    """Load all records across splits into memory."""

    records: List[Dict[str, object]] = []
    for split in DEFAULT_SPLITS:
        path = source_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source split at {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse {path} line {line_number}: {exc}") from exc
                record["_original_split"] = split
                records.append(record)
    return records


def compute_metrics(record: Dict[str, object]) -> float:
    """Augment record metadata with coverage and compression statistics."""

    meta = record.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Record missing 'meta' payload required for augmentation.")

    width = int(meta.get("window", 0))
    wrap = bool(meta.get("wrap", True))

    train_pairs: Sequence[Dict[str, Sequence[int]]] = record.get("train") or []
    query_seq: Sequence[int] = record.get("query") or []
    solution_seq: Sequence[int] = record.get("solution") or []

    train_counts = collect_train_windows(train_pairs, width)
    query_counts = collect_query_windows(query_seq, width, wrap=wrap)

    train_unique = set(train_counts.keys())
    query_unique = set(query_counts.keys())

    unique_cov: Optional[float]
    if query_unique:
        covered_unique = len(train_unique & query_unique)
        unique_cov = covered_unique / len(query_unique)
    else:
        unique_cov = None

    total_query_windows = sum(query_counts.values())
    if total_query_windows > 0:
        covered_weighted = sum(count for win, count in query_counts.items() if win in train_counts)
        weighted_cov = covered_weighted / total_query_windows
        avg_depth = sum(train_counts.get(win, 0) for win in query_counts) / len(query_counts)
    else:
        weighted_cov = None
        avg_depth = None

    flattened: List[int] = []
    for pair in train_pairs:
        flattened.extend(pair.get("input", []))
        flattened.extend(pair.get("output", []))
    flattened.extend(query_seq)

    ncd_value = normalized_compression_distance(flattened, solution_seq)

    meta["schema_version"] = SCHEMA_VERSION
    coverage = meta.get("coverage") or {}
    coverage_windows = coverage.get("windows")
    meta["coverage_windows"] = coverage_windows
    meta["query_window_coverage_weighted"] = weighted_cov
    meta["query_window_coverage_unique"] = unique_cov
    meta["query_window_avg_depth"] = avg_depth
    meta["ncd_train_query_solution"] = ncd_value

    # Keep the duplicated rule table versions aligned with the schema.
    rule_table_meta = meta.get("rule_table")
    if isinstance(rule_table_meta, dict):
        rule_table_meta["format_version"] = RULE_TABLE_VERSION
    rule_table = record.get("rule_table")
    if isinstance(rule_table, dict):
        rule_table["format_version"] = RULE_TABLE_VERSION

    return float(weighted_cov) if weighted_cov is not None else math.inf


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                raise FileExistsError(f"Refusing to remove directory subtree inside {path}: {child}")
    else:
        path.mkdir(parents=True, exist_ok=True)


def assign_splits(
    episodes: List[Episode],
    *,
    test_extrapolation_size: int,
    train_size: int,
    val_size: int,
    test_interpolation_size: int,
    shuffle_seed: int,
) -> Dict[str, List[Dict[str, object]]]:
    """Partition the episodes into their refreshed splits."""

    if test_extrapolation_size > len(episodes):
        raise ValueError("Requested test_extrapolation_size exceeds total episodes.")

    episodes_sorted = sorted(episodes, key=lambda ep: (ep.coverage_weighted, ep.payload["meta"]["fingerprint"]))
    assigned: Dict[str, List[Dict[str, object]]] = {split: [] for split in DEFAULT_SPLITS}

    assigned["test_extrapolation"] = [ep.payload for ep in episodes_sorted[:test_extrapolation_size]]

    remaining = episodes_sorted[test_extrapolation_size:]
    rng = random.Random(shuffle_seed)
    rng.shuffle(remaining)

    required_total = train_size + val_size + test_interpolation_size
    if required_total != len(remaining):
        raise ValueError(
            "Remainder does not match requested split sizes "
            f"(remaining={len(remaining)}, requested={required_total})."
        )

    idx = 0
    assigned["train"] = [ep.payload for ep in remaining[idx : idx + train_size]]
    idx += train_size
    assigned["val"] = [ep.payload for ep in remaining[idx : idx + val_size]]
    idx += val_size
    assigned["test_interpolation"] = [ep.payload for ep in remaining[idx : idx + test_interpolation_size]]

    return assigned


def write_splits(splits: Dict[str, List[Dict[str, object]]], output_dir: Path) -> None:
    """Persist refreshed splits to JSONL files."""

    for split, records in splits.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                record.pop("_original_split", None)
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    records = load_records(source_dir)
    if not records:
        raise ValueError(f"No records found under {source_dir}")

    episodes: List[Episode] = []
    for record in records:
        coverage_weighted = compute_metrics(record)
        meta = record.get("meta") or {}
        fingerprint = meta.get("fingerprint")
        if not fingerprint:
            raise ValueError("Episode missing fingerprint field after augmentation.")
        episodes.append(Episode(payload=record, coverage_weighted=coverage_weighted))

    ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    splits = assign_splits(
        episodes,
        test_extrapolation_size=args.test_extrapolation_size,
        train_size=args.train_size,
        val_size=args.val_size,
        test_interpolation_size=args.test_interpolation_size,
        shuffle_seed=args.shuffle_seed,
    )

    write_splits(splits, args.output_dir)


if __name__ == "__main__":
    main()
