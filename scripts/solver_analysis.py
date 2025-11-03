#!/usr/bin/env python3

"""Analyse solver accuracy against dataset metadata and generate plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from cellarc.data import EpisodeDataset
from cellarc.solver import learn_from_record


def load_meta_lookup(meta_path: Path) -> Dict[str, Dict[str, object]]:
    """Load metadata records keyed by episode id."""

    lookup: Dict[str, Dict[str, object]] = {}
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id = record.get("id") or record.get("meta", {}).get("fingerprint")
            meta = record.get("meta")
            if not record_id or not isinstance(meta, dict):
                continue
            lookup[str(record_id)] = meta
    return lookup


def numeric_value(value: object) -> Optional[float]:
    """Return a float if the value is numeric, otherwise ``None``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return float(value)
    return None


def sequence_to_bytes(seq: Sequence[int]) -> bytes:
    """Convert a sequence of integers to bytes for compression."""

    if not seq:
        return b""
    return bytes(int(v) % 256 for v in seq)


def compression_size(payload: bytes) -> int:
    """Return the size of the compressed payload."""

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


def centered_window(seq: Sequence[int], idx: int, W: int, wrap: bool) -> Tuple[int, ...]:
    half = W // 2
    n = len(seq)
    if wrap:
        return tuple(seq[(idx - half + j) % n] for j in range(W))
    window = []
    for j in range(idx - half, idx + half + 1):
        if 0 <= j < n:
            window.append(seq[j])
        else:
            window.append(0)
    return tuple(window)


def collect_train_windows(train_pairs: Sequence[Dict[str, Sequence[int]]], W: int) -> Counter:
    counts: Counter = Counter()
    half = W // 2
    for pair in train_pairs:
        x = pair.get("input", [])
        n = len(x)
        if n < W:
            continue
        for i in range(half, n - half):
            win = centered_window(x, i, W, wrap=True)
            counts[win] += 1
    return counts


def collect_query_windows(seq: Sequence[int], W: int, wrap: bool) -> Counter:
    counts: Counter = Counter()
    n = len(seq)
    if n == 0:
        return counts
    for i in range(n):
        win = centered_window(seq, i, W, wrap=wrap)
        counts[win] += 1
    return counts


def gather_rows(
    splits: Sequence[str],
    data_dir: Path,
    meta_dir: Path,
    runs: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for split in splits:
        data_path = data_dir / f"{split}.jsonl"
        meta_path = meta_dir / f"{split}.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data file for split '{split}': {data_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file for split '{split}': {meta_path}")

        dataset = EpisodeDataset(paths=[data_path], fmt="jsonl")
        meta_lookup = load_meta_lookup(meta_path)

        for record in dataset:
            record_id = record.get("id")
            if record_id is None:
                continue
            meta = meta_lookup.get(str(record_id))
            if not meta:
                continue

            record["meta"] = meta
            W = int(meta["window"])

            model = learn_from_record(record)
            query = list(record["query"])
            solution = list(record["solution"])

            wrap = bool(meta.get("wrap", True))

            train_window_counts = collect_train_windows(record.get("train", []), W=W)
            query_window_counts = collect_query_windows(query, W=W, wrap=wrap)

            train_unique = set(train_window_counts.keys())
            query_unique = set(query_window_counts.keys())
            if query_unique:
                unique_covered = len(train_unique & query_unique)
                query_unique_cov = unique_covered / len(query_unique)
            else:
                query_unique_cov = None

            total_query_windows = sum(query_window_counts.values())
            if total_query_windows:
                covered_weighted = sum(
                    count for win, count in query_window_counts.items() if win in train_window_counts
                )
                query_weighted_cov = covered_weighted / total_query_windows
                avg_depth = sum(
                    train_window_counts.get(win, 0) for win in query_window_counts
                ) / len(query_window_counts)
            else:
                query_weighted_cov = None
                avg_depth = None

            total = len(solution)
            accuracies: List[float] = []
            matches_list: List[int] = []
            exact_list: List[int] = []

            base_seed = meta.get("episode_seed")
            if base_seed is None:
                digest = hashlib.sha256(str(record_id).encode("utf-8")).hexdigest()
                base_seed = int(digest[:16], 16)

            for run_idx in range(runs):
                rng_seed = int(base_seed) + run_idx
                rng = random.Random(rng_seed)
                prediction = model.predict(query, rng=rng)

                matches = sum(int(p == t) for p, t in zip(prediction, solution))
                accuracy = matches / total if total else 0.0

                matches_list.append(matches)
                accuracies.append(accuracy)
                exact_list.append(int(matches == total))

            def mean(values: Sequence[float]) -> float:
                return sum(values) / len(values) if values else 0.0

            def std(values: Sequence[float], avg: float) -> float:
                if len(values) < 2:
                    return 0.0
                variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
                return math.sqrt(variance)

            accuracy_mean = mean(accuracies)
            accuracy_std = std(accuracies, accuracy_mean)
            matches_mean = mean(matches_list)
            matches_std = std(matches_list, matches_mean)
            exact_rate = mean(exact_list)

            flattened_train: List[int] = []
            for pair in record.get("train", []):
                flattened_train.extend(pair.get("input", []))
                flattened_train.extend(pair.get("output", []))
            flattened_train.extend(query)

            ncd_value = normalized_compression_distance(flattened_train, solution)

            row: Dict[str, object] = {
                "split": split,
                "id": record_id,
                "alphabet_size": meta.get("alphabet_size"),
                "radius": meta.get("radius"),
                "steps": meta.get("steps"),
                "window": meta.get("window"),
                "windows_total": meta.get("windows_total"),
                "lambda": meta.get("lambda"),
                "lambda_bin": meta.get("lambda_bin"),
                "avg_cell_entropy": meta.get("avg_cell_entropy"),
                "entropy_bin": meta.get("entropy_bin"),
                "avg_mutual_information_d1": meta.get("avg_mutual_information_d1"),
                "family": meta.get("family"),
                "family_params": json.dumps(meta.get("family_params", {}), sort_keys=True),
                "hamming_accuracy": accuracy_mean,
                "hamming_accuracy_std": accuracy_std,
                "matches": matches_mean,
                "matches_std": matches_std,
                "exact_match_rate": exact_rate,
                "query_length": total,
                "ncd_train_query_solution": ncd_value,
                "solver_runs": runs,
                "query_windows_unique": len(query_unique),
                "train_windows_unique": len(train_unique),
                "query_window_coverage_unique": query_unique_cov,
                "query_window_coverage_weighted": query_weighted_cov,
                "query_window_avg_depth": avg_depth,
            }

            coverage = meta.get("coverage") or {}
            for key, value in coverage.items():
                column = f"coverage_{key}"
                if isinstance(value, (int, float, bool)):
                    row[column] = float(value) if not isinstance(value, bool) else int(value)
                else:
                    row[column] = value

            morphology = meta.get("morphology") or {}
            if isinstance(morphology, dict):
                for key, value in morphology.items():
                    column = f"morphology_{key}"
                    if isinstance(value, (int, float, bool)):
                        row[column] = float(value) if not isinstance(value, bool) else int(value)
                    else:
                        row[column] = value

            rows.append(row)
    return rows


def write_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames: List[str] = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_mean = x.mean()
    y_mean = y.mean()
    x_diff = x - x_mean
    y_diff = y - y_mean
    denom = np.linalg.norm(x_diff) * np.linalg.norm(y_diff)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x_diff, y_diff) / denom)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=float)
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    return pearsonr(x_rank, y_rank)


def is_numeric_column(values: Iterable[object]) -> bool:
    has_value = False
    for value in values:
        numeric = numeric_value(value)
        if numeric is None:
            continue
        has_value = True
    return has_value


def generate_plots(
    rows: Sequence[Dict[str, object]],
    output_dir: Path,
    accuracy_key: str = "hamming_accuracy",
    plot_format: str = "png",
) -> None:
    accuracy_values = []
    for row in rows:
        value = numeric_value(row.get(accuracy_key))
        if value is not None:
            accuracy_values.append(value)
    if not accuracy_values:
        raise ValueError(f"No numeric values found for accuracy column '{accuracy_key}'.")

    numeric_columns: Dict[str, List[float]] = defaultdict(list)
    for column in rows[0]:
        if column in {"id", accuracy_key, "family_params"}:
            continue
        column_values: List[float] = []
        for row in rows:
            numeric = numeric_value(row.get(column))
            if numeric is None:
                continue
            column_values.append(numeric)
        if column_values and len(column_values) >= 2:
            numeric_columns[column] = column_values

    y = np.array(accuracy_values, dtype=float)

    for column, values in numeric_columns.items():
        # Align values and accuracies by dropping rows with missing data.
        xy_pairs: List[Tuple[float, float]] = []
        for row in rows:
            x_val = numeric_value(row.get(column))
            y_val = numeric_value(row.get(accuracy_key))
            if x_val is None or y_val is None:
                continue
            xy_pairs.append((x_val, y_val))
        if len(xy_pairs) < 3:
            continue

        x_array = np.array([p[0] for p in xy_pairs], dtype=float)
        y_array = np.array([p[1] for p in xy_pairs], dtype=float)

        pearson = pearsonr(x_array, y_array)
        spearman = spearmanr(x_array, y_array)

        plt.figure(figsize=(6, 4))
        plt.scatter(x_array, y_array, alpha=0.4, edgecolors="none")
        plt.xlabel(column.replace("_", " "))
        plt.ylabel("Hamming accuracy")
        plt.title(
            f"{column} vs. accuracy\n"
            f"Pearson: {pearson:.3f} | Spearman: {spearman:.3f}"
        )
        plt.grid(True, alpha=0.2)

        sanitized = column.replace(" ", "_").replace("/", "_")
        output_path = output_dir / f"{sanitized}_vs_accuracy.{plot_format}"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute solver accuracies per episode and correlate with metadata.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/datasets/cellarc_100k/data"),
        help="Directory containing dataset JSONL files.",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path("artifacts/datasets/cellarc_100k_meta/data"),
        help="Directory containing metadata JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/solver"),
        help="Directory to store CSV and plots.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_interpolation", "test_extrapolation"],
        help="Dataset splits to evaluate.",
    )
    parser.add_argument(
        "--plot-format",
        type=str,
        default="png",
        help="Image format for scatter plots (e.g., png, pdf).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of stochastic solver runs per episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = gather_rows(
        args.splits,
        data_dir=args.data_dir,
        meta_dir=args.meta_dir,
        runs=args.runs,
    )

    csv_path = args.output_dir / "solver_metadata_accuracy.csv"
    write_csv(rows, csv_path)

    generate_plots(rows, output_dir=args.output_dir, plot_format=args.plot_format)
    print(f"Wrote CSV to {csv_path}")
    print(f"Scatter plots saved in {args.output_dir}")


if __name__ == "__main__":
    main()
