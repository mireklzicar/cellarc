#!/usr/bin/env python3

"""Evaluate the baseline CA solver on CellARC test splits."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cellarc.data import EpisodeDataset
from cellarc.solver import learn_from_record


@dataclass
class SplitMetrics:
    split: str
    episodes: int
    predictions: int
    exact_matches: int
    matched_cells: int
    total_cells: int
    per_episode_accuracy_sum: float
    failures: List[Tuple[Optional[str], str]]

    @property
    def exact_match_accuracy(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.exact_matches / self.episodes

    @property
    def hamming_accuracy(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return self.matched_cells / self.total_cells

    @property
    def mean_episode_accuracy(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.per_episode_accuracy_sum / self.episodes

    @property
    def failure_count(self) -> int:
        return len(self.failures)


def load_meta_lookup(meta_path: Path) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id = record.get("id") or record.get("meta", {}).get("fingerprint")
            if not record_id:
                continue
            meta = record.get("meta")
            if not isinstance(meta, dict):
                continue
            lookup[str(record_id)] = meta
    return lookup


def evaluate_split(
    split: str,
    *,
    data_dir: Path,
    meta_dir: Path,
    limit: Optional[int] = None,
) -> SplitMetrics:
    data_path = data_dir / f"{split}.jsonl"
    meta_path = meta_dir / f"{split}.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file for split '{split}' not found: {data_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file for split '{split}' not found: {meta_path}")

    meta_lookup = load_meta_lookup(meta_path)

    dataset = EpisodeDataset(paths=[data_path], fmt="jsonl")

    episodes = 0
    predictions = 0
    exact_matches = 0
    matched_cells = 0
    total_cells = 0
    per_episode_accuracy_sum = 0.0
    failures: List[Tuple[Optional[str], str]] = []

    for idx, record in enumerate(dataset, start=1):
        if limit is not None and idx > limit:
            break

        episodes += 1
        record_id = record.get("id")
        query = record.get("query")
        solution = record.get("solution")

        if not isinstance(query, Iterable) or not isinstance(solution, Iterable):
            raise ValueError(f"Episode {record_id!r} is missing 'query' or 'solution' sequences.")

        query_list = list(query)
        solution_list = list(solution)

        if len(query_list) != len(solution_list):
            raise ValueError(
                f"Episode {record_id!r} has mismatched query/solution lengths: "
                f"{len(query_list)} vs {len(solution_list)}"
            )

        total_cells += len(solution_list)

        meta = meta_lookup.get(str(record_id))
        if not meta:
            failures.append((record_id, "missing metadata"))
            continue

        record["meta"] = meta

        try:
            model = learn_from_record(record)
            prediction = model.predict(query_list)
        except Exception as exc:  # noqa: BLE001 - propagate solver failures as evaluation stats
            failures.append((record_id, str(exc)))
            continue

        predictions += 1

        matches = sum(int(p == t) for p, t in zip(prediction, solution_list))
        matched_cells += matches

        episode_accuracy = matches / len(solution_list) if solution_list else 0.0
        per_episode_accuracy_sum += episode_accuracy

        if matches == len(solution_list):
            exact_matches += 1

    return SplitMetrics(
        split=split,
        episodes=episodes,
        predictions=predictions,
        exact_matches=exact_matches,
        matched_cells=matched_cells,
        total_cells=total_cells,
        per_episode_accuracy_sum=per_episode_accuracy_sum,
        failures=failures,
    )


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_metrics(metrics: SplitMetrics, show_failures: bool = False, failure_limit: int = 5) -> None:
    print(f"Split: {metrics.split}")
    print(f"  Episodes evaluated: {metrics.episodes}")
    print(
        f"  Predictions succeeded: {metrics.predictions} "
        f"({format_percentage(metrics.predictions / metrics.episodes) if metrics.episodes else '0.00%'})"
    )
    print(
        f"  Exact matches: {metrics.exact_matches} "
        f"({format_percentage(metrics.exact_match_accuracy)})"
    )
    print(f"  Hamming accuracy (cell-level): {format_percentage(metrics.hamming_accuracy)}")
    print(
        f"  Mean per-episode accuracy: {format_percentage(metrics.mean_episode_accuracy)}"
    )
    if metrics.failure_count:
        print(f"  Solver failures: {metrics.failure_count}")
        if show_failures:
            to_show = metrics.failures[:failure_limit]
            for record_id, error in to_show:
                label = record_id or "<missing id>"
                print(f"    - {label}: {error}")
            if metrics.failure_count > failure_limit:
                print(f"    ... and {metrics.failure_count - failure_limit} more.")
    print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the CellARC solver on specified dataset splits.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/datasets/cellarc_100k/data"),
        help="Directory containing the main dataset JSONL files.",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path("artifacts/datasets/cellarc_100k_meta/data"),
        help="Directory containing the metadata JSONL files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_interpolation", "test_extrapolation"],
        help="Dataset splits to evaluate (without file extension).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of episodes to evaluate per split.",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print the first few solver failures for each split.",
    )
    parser.add_argument(
        "--failure-limit",
        type=int,
        default=5,
        help="Maximum number of failure examples to display per split when --show-failures is set.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    all_metrics: List[SplitMetrics] = []
    for split in args.splits:
        metrics = evaluate_split(
            split,
            data_dir=args.data_dir,
            meta_dir=args.meta_dir,
            limit=args.limit,
        )
        all_metrics.append(metrics)
        print_metrics(metrics, show_failures=args.show_failures, failure_limit=args.failure_limit)

    # Provide a simple aggregate summary across splits.
    if len(all_metrics) > 1:
        total_episodes = sum(m.episodes for m in all_metrics)
        total_matched = sum(m.matched_cells for m in all_metrics)
        total_cells = sum(m.total_cells for m in all_metrics)
        total_exact = sum(m.exact_matches for m in all_metrics)

        aggregate_hamming = total_matched / total_cells if total_cells else 0.0
        aggregate_exact = total_exact / total_episodes if total_episodes else 0.0

        print("Aggregate across splits:")
        print(f"  Episodes: {total_episodes}")
        print(f"  Exact matches: {total_exact} ({format_percentage(aggregate_exact)})")
        print(f"  Hamming accuracy (cell-level): {format_percentage(aggregate_hamming)}")


if __name__ == "__main__":
    main()
