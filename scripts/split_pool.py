#!/usr/bin/env python3
"""Split the enriched pool into high-coverage train/val/test partitions."""

from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def sequence_to_bytes(seq: Sequence[int]) -> bytes:
    if not seq:
        return b""
    return bytes(int(v) % 256 for v in seq)


def compression_size(payload: bytes) -> int:
    return len(zlib.compress(payload, level=9))


def normalized_compression_distance(a: Sequence[int], b: Sequence[int]) -> Optional[float]:
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


def collect_train_windows(train_pairs: Sequence[Dict[str, Sequence[int]]], width: int) -> Dict[Tuple[int, ...], int]:
    counts: Dict[Tuple[int, ...], int] = {}
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
            counts[win] = counts.get(win, 0) + 1
    return counts


def collect_query_windows(query: Sequence[int], width: int, wrap: bool) -> Dict[Tuple[int, ...], int]:
    counts: Dict[Tuple[int, ...], int] = {}
    if width <= 0:
        return counts
    n = len(query)
    if n == 0:
        return counts
    for idx in range(n):
        win = centered_window(query, idx, width, wrap=wrap)
        counts[win] = counts.get(win, 0) + 1
    return counts


@dataclass
class EpisodeRecord:
    payload: Dict[str, object]
    weighted_cov: float
    lambda_value: float
    entropy: float
    combined_score: float = 0.0


def iter_records(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def flatten_train_query(train_pairs: Sequence[Dict[str, Sequence[int]]], query: Sequence[int]) -> List[int]:
    flattened: List[int] = []
    for pair in train_pairs:
        flattened.extend(pair.get("input", []))
        flattened.extend(pair.get("output", []))
    flattened.extend(query)
    return flattened


def augment_record(record: Dict[str, object], coverage_threshold: float) -> Optional[EpisodeRecord]:
    meta = record.get("meta", {}) or {}
    width = int(meta.get("window", 0))
    wrap = bool(meta.get("wrap", True))

    train_pairs = record.get("train") or []
    query_seq = record.get("query") or []
    solution_seq = record.get("solution") or []

    train_counts = collect_train_windows(train_pairs, width)
    query_counts = collect_query_windows(query_seq, width, wrap)

    total_query_windows = sum(query_counts.values())
    if total_query_windows > 0:
        covered_weighted = sum(cnt for win, cnt in query_counts.items() if win in train_counts)
        weighted_cov = covered_weighted / total_query_windows
        avg_depth = sum(train_counts.get(win, 0) for win in query_counts) / len(query_counts)
    else:
        weighted_cov = 0.0
        avg_depth = math.nan

    if weighted_cov < coverage_threshold:
        return None

    if query_counts:
        unique_cov = len(set(query_counts) & set(train_counts)) / len(set(query_counts))
    else:
        unique_cov = math.nan

    flattened = flatten_train_query(train_pairs, query_seq)
    ncd_value = normalized_compression_distance(flattened, solution_seq)

    coverage = meta.get("coverage") or {}
    meta["schema_version"] = "1.0.2"
    meta["coverage_windows"] = coverage.get("windows")
    meta["query_window_coverage_weighted"] = weighted_cov
    meta["query_window_coverage_unique"] = None if math.isnan(unique_cov) else unique_cov
    meta["query_window_avg_depth"] = None if math.isnan(avg_depth) else avg_depth
    meta["ncd_train_query_solution"] = ncd_value

    rule_table_meta = meta.get("rule_table")
    if isinstance(rule_table_meta, dict):
        rule_table_meta["format_version"] = "1.0.2"
    rule_table_payload = record.get("rule_table")
    if isinstance(rule_table_payload, dict):
        rule_table_payload["format_version"] = "1.0.2"

    lambda_value = float(meta.get("lambda", 0.0) or 0.0)
    entropy = float(meta.get("avg_cell_entropy", 0.0) or 0.0)
    return EpisodeRecord(payload=record, weighted_cov=weighted_cov, lambda_value=lambda_value, entropy=entropy)


def assign_normalised_ranks(values: List[float], *, reverse: bool) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: (float("inf") if math.isnan(item[1]) else item[1]), reverse=reverse)
    ranks = [0] * n
    for rank, (idx, _) in enumerate(indexed):
        ranks[idx] = rank
    denom = float(n - 1)
    return [rank / denom for rank in ranks]


def write_split(name: str, records: List[Dict[str, object]], out_dir: Path) -> None:
    data_path = out_dir / f"{name}.jsonl"
    meta_path = out_dir / f"{name}_meta.jsonl"
    with data_path.open("w", encoding="utf-8") as data_out, meta_path.open("w", encoding="utf-8") as meta_out:
        for rec in records:
            meta = rec.get("meta", {}) or {}
            fingerprint = meta.get("fingerprint")
            if fingerprint is None:
                raise ValueError("Record missing fingerprint in meta block.")
            data_out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            meta_out.write(
                json.dumps({"fingerprint": fingerprint, "meta": meta}, separators=(",", ":")) + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/pool_downsampled/downsampled_enriched.jsonl"),
        help="Path to the enriched downsampled JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pool_downsampled/splits"),
        help="Directory where split JSONL files will be written.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for shuffling splits.")
    parser.add_argument("--train-count", type=int, default=100_000)
    parser.add_argument("--val-count", type=int, default=1_000)
    parser.add_argument("--test-interp-count", type=int, default=1_000)
    parser.add_argument("--test-extra-count", type=int, default=1_000)
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.9,
        help="Minimum query-window coverage (weighted) required to keep an episode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episodes: List[EpisodeRecord] = []
    for record in iter_records(args.input):
        augmented = augment_record(record, args.coverage_threshold)
        if augmented is not None:
            episodes.append(augmented)

    if not episodes:
        raise ValueError("No episodes satisfied the coverage threshold; cannot split.")

    coverage_norm = assign_normalised_ranks([ep.weighted_cov for ep in episodes], reverse=False)
    lambda_norm = assign_normalised_ranks([ep.lambda_value for ep in episodes], reverse=True)
    entropy_norm = assign_normalised_ranks([ep.entropy for ep in episodes], reverse=True)

    for idx, ep in enumerate(episodes):
        ep.combined_score = (coverage_norm[idx] + lambda_norm[idx] + entropy_norm[idx]) / 3.0

    ordered_indices = sorted(range(len(episodes)), key=lambda idx: episodes[idx].combined_score)

    if args.test_extra_count > len(ordered_indices):
        raise ValueError(
            f"Requested {args.test_extra_count} test extrapolation episodes but only {len(ordered_indices)} "
            "episodes satisfied the coverage threshold."
        )

    test_extra_indices = ordered_indices[: args.test_extra_count]
    remaining_indices = ordered_indices[args.test_extra_count :]

    remaining_required = args.train_count + args.val_count + args.test_interp_count
    if remaining_required > len(remaining_indices):
        raise ValueError(
            "Not enough high-coverage episodes to satisfy split sizes: "
            f"required {remaining_required}, available {len(remaining_indices)}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(remaining_indices)

    def collect(indices: List[int]) -> List[Dict[str, object]]:
        return [episodes[idx].payload for idx in indices]

    test_extra_records = collect(test_extra_indices)
    train_records = collect(remaining_indices[: args.train_count])
    val_records = collect(remaining_indices[args.train_count : args.train_count + args.val_count])
    test_interp_records = collect(
        remaining_indices[
            args.train_count + args.val_count : args.train_count + args.val_count + args.test_interp_count
        ]
    )
    leftover_indices = remaining_indices[
        args.train_count + args.val_count + args.test_interp_count :
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_split("train", train_records, args.output_dir)
    write_split("val", val_records, args.output_dir)
    write_split("test_interpolation", test_interp_records, args.output_dir)
    write_split("test_extrapolation", test_extra_records, args.output_dir)

    if leftover_indices:
        info_path = args.output_dir / "unused.jsonl"
        with info_path.open("w", encoding="utf-8") as fh:
            for idx in leftover_indices:
                fh.write(json.dumps(episodes[idx].payload, separators=(",", ":")) + "\n")

    print(
        f"Kept {len(episodes)} high-coverage episodes → "
        f"{len(train_records)} train / {len(val_records)} val / "
        f"{len(test_interp_records)} test_interpolation / {len(test_extra_records)} test_extrapolation."
    )
    if leftover_indices:
        print(f"{len(leftover_indices)} records written to unused.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
