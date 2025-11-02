"""Minimal utilities for reading ARC-style episode records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


@dataclass
class EpisodeRecord:
    """Container bundling a parsed episode with provenance metadata."""

    record: Dict[str, Any]
    source: Path
    manifest: Optional[Path] = None


@dataclass
class PreparedEpisode:
    """Structured episode representation for visualization helpers."""

    train_pairs: List[Tuple[List[List[int]], List[List[int]]]]
    query: List[List[int]]
    solution: Optional[List[List[int]]]
    meta: Dict[str, Any]


def _ensure_2d(grid: Any) -> List[List[int]]:
    """Convert a scalar, vector, or grid-like object into a 2-D list of ints."""
    if grid is None:
        return [[]]
    if isinstance(grid, Sequence) and grid:
        first = grid[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            return [[int(value) for value in row] for row in grid]  # type: ignore[arg-type]
        return [[int(value) for value in grid]]  # type: ignore[arg-type]
    if isinstance(grid, Sequence):
        return [[]]
    return [[int(grid)]]


def load_records(inputs: Sequence[str]) -> Iterator[EpisodeRecord]:
    """Yield EpisodeRecord objects from JSONL shards."""
    for raw in inputs:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Input source not found: {path}")

        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    yield EpisodeRecord(record=record, source=path)
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            episodes = data.get("episodes") if isinstance(data, dict) else data
            if not isinstance(episodes, list):
                raise ValueError(f"Unsupported JSON manifest format: {path}")
            for record in episodes:
                if not isinstance(record, dict):
                    continue
                yield EpisodeRecord(record=record, source=path, manifest=path)
        else:
            raise ValueError(f"Unsupported input format for {path}. Expected .jsonl or .json")


def prepare_episode(record: Dict[str, Any]) -> PreparedEpisode:
    """Normalise a raw record dictionary for visualization or analysis."""
    train_pairs: List[Tuple[List[List[int]], List[List[int]]]] = []
    for pair in record.get("train", []):
        if not isinstance(pair, dict):
            continue
        inp = _ensure_2d(pair.get("input", []))
        out = _ensure_2d(pair.get("output", []))
        train_pairs.append((inp, out))

    query = _ensure_2d(record.get("query", []))
    solution_raw = record.get("solution")
    solution = _ensure_2d(solution_raw) if solution_raw is not None else None

    meta = dict(record.get("meta", {}) or {})

    return PreparedEpisode(
        train_pairs=train_pairs,
        query=query,
        solution=solution,
        meta=meta,
    )


__all__ = ["EpisodeRecord", "PreparedEpisode", "load_records", "prepare_episode"]
