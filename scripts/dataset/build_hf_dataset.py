#!/usr/bin/env python3
"""
Convert the processed pool splits into Hugging Face friendly dataset packages.

We produce two artefacts:
  * `<target_root>/<dataset_name>` – light JSONL + Parquet (id/train/query/solution).
  * `<target_root>/<dataset_name><extended_suffix>` – identical Parquet plus full JSONL metadata.

The Parquet schema is shared between both packages so they can be uploaded as-is
to the Hugging Face Hub or consumed locally with `datasets.load_dataset`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import statistics

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Sequence, Value


SPLITS: Tuple[str, ...] = ("train", "val", "test_interpolation", "test_extrapolation")
DEFAULT_EXPORT_ROOT = Path(os.getenv("CELLARC_HOME", Path.home() / ".cache" / "cellarc")) / "exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/processing/pool_downsampled/splits"),
        help="Directory containing the input *.jsonl files.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="Root directory for the generated Hugging Face packages "
        "(defaults to ${CELLARC_HOME:-~/.cache/cellarc}/exports).",
    )
    parser.add_argument(
        "--dataset-name",
        default="cellarc_100k",
        help="Base name for the lightweight dataset package.",
    )
    parser.add_argument(
        "--extended-suffix",
        default="_extended",
        help="Suffix appended to the base dataset name for the extended package.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="How many episodes to buffer before flushing to Parquet.",
    )
    parser.add_argument(
        "--parquet-compression",
        default="snappy",
        help="Compression codec passed to pyarrow.parquet.ParquetWriter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output directories.",
    )
    return parser.parse_args()


class NumericStats:
    """Running summary for a numeric field."""

    __slots__ = ("count", "total", "minimum", "maximum")

    def __init__(self) -> None:
        self.count: int = 0
        self.total: float = 0.0
        self.minimum: Optional[float] = None
        self.maximum: Optional[float] = None

    def update(self, value: Optional[float]) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        v = float(value)
        self.count += 1
        self.total += v
        self.minimum = v if self.minimum is None else min(self.minimum, v)
        self.maximum = v if self.maximum is None else max(self.maximum, v)

    def as_dict(self) -> Dict[str, Optional[float]]:
        mean = self.total / self.count if self.count else None
        return {
            "count": self.count,
            "mean": mean,
            "min": self.minimum,
            "max": self.maximum,
        }


class DistributionStats:
    """Aggregate summary that also tracks median."""

    __slots__ = ("values", "count", "total", "minimum", "maximum")

    def __init__(self) -> None:
        self.values: List[int] = []
        self.count: int = 0
        self.total: float = 0.0
        self.minimum: Optional[int] = None
        self.maximum: Optional[int] = None

    def update(self, value: Optional[int]) -> None:
        if value is None:
            return
        v = int(value)
        self.values.append(v)
        self.count += 1
        self.total += v
        self.minimum = v if self.minimum is None else min(self.minimum, v)
        self.maximum = v if self.maximum is None else max(self.maximum, v)

    def as_dict(self) -> Dict[str, Optional[float]]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }
        median = float(statistics.median(self.values))
        mean = self.total / self.count
        return {
            "count": self.count,
            "mean": mean,
            "median": median,
            "min": self.minimum,
            "max": self.maximum,
        }


def _new_counter() -> Counter:
    return Counter()


@dataclass
class SplitStats:
    num_episodes: int = 0
    alphabet_sizes: Counter = field(default_factory=_new_counter)
    radii: Counter = field(default_factory=_new_counter)
    steps: Counter = field(default_factory=_new_counter)
    train_context: Counter = field(default_factory=_new_counter)
    train_example_counts: Counter = field(default_factory=_new_counter)
    window_sizes: Counter = field(default_factory=_new_counter)
    families: Counter = field(default_factory=_new_counter)
    lambda_stats: NumericStats = field(default_factory=NumericStats)
    entropy_stats: NumericStats = field(default_factory=NumericStats)
    coverage_fraction_stats: NumericStats = field(default_factory=NumericStats)
    sample_lengths: DistributionStats = field(default_factory=DistributionStats)
    episode_lengths: DistributionStats = field(default_factory=DistributionStats)

    def register_episode(self, record: Dict) -> None:
        self.num_episodes += 1
        meta = record.get("meta", {})
        alphabet_size = meta.get("alphabet_size")
        if alphabet_size is not None:
            self.alphabet_sizes[alphabet_size] += 1
        radius = meta.get("radius")
        if radius is not None:
            self.radii[radius] += 1
        steps = meta.get("steps")
        if steps is not None:
            self.steps[steps] += 1
        train_ctx = meta.get("train_context")
        if train_ctx is not None:
            self.train_context[train_ctx] += 1
        window = meta.get("window")
        if window is not None:
            self.window_sizes[window] += 1
        family = meta.get("family")
        if family is not None:
            self.families[family] += 1
        train_examples = len(record.get("train", []))
        self.train_example_counts[train_examples] += 1
        self.lambda_stats.update(meta.get("lambda"))
        self.entropy_stats.update(meta.get("avg_cell_entropy"))
        coverage = meta.get("coverage") or {}
        self.coverage_fraction_stats.update(coverage.get("fraction"))

        train_records = record.get("train") or []
        for example in train_records:
            sample_len = len(example.get("input", []))
            self.sample_lengths.update(sample_len)

        if train_records or record.get("query") or record.get("solution"):
            episode_total = 0
            for example in train_records:
                episode_total += len(example.get("input", []))
                episode_total += len(example.get("output", []))
            episode_total += len(record.get("query", []))
            episode_total += len(record.get("solution", []))
            self.episode_lengths.update(episode_total)

    def as_dict(self) -> Dict[str, object]:
        return {
            "num_episodes": self.num_episodes,
            "alphabet_sizes": counter_to_series(self.alphabet_sizes),
            "radii": counter_to_series(self.radii),
            "steps": counter_to_series(self.steps),
            "train_context": counter_to_series(self.train_context),
            "train_example_counts": counter_to_series(self.train_example_counts),
            "window_sizes": counter_to_series(self.window_sizes),
            "families": counter_to_series(self.families),
            "lambda": self.lambda_stats.as_dict(),
            "avg_cell_entropy": self.entropy_stats.as_dict(),
            "coverage_fraction": self.coverage_fraction_stats.as_dict(),
            "sample_length": self.sample_lengths.as_dict(),
            "episode_total_length": self.episode_lengths.as_dict(),
        }


def counter_to_series(counter: Counter) -> List[Dict[str, object]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: item[0])
    ]


BASE_FEATURES = Features(
    {
        "id": Value("string"),
        "train": Sequence(
            {
                "input": Sequence(Value("int32")),
                "output": Sequence(Value("int32")),
            }
        ),
        "query": Sequence(Value("int32")),
        "solution": Sequence(Value("int32")),
    }
)


PARQUET_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field(
            "train",
            pa.list_(
                pa.struct(
                    [
                        pa.field("input", pa.list_(pa.int32())),
                        pa.field("output", pa.list_(pa.int32())),
                    ]
                )
            ),
        ),
        pa.field("query", pa.list_(pa.int32())),
        pa.field("solution", pa.list_(pa.int32())),
    ]
)


def ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_parquet_chunk(
    writer: Optional[pq.ParquetWriter],
    path: Path,
    chunk: List[Dict[str, object]],
    compression: str,
) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(chunk, schema=PARQUET_SCHEMA)
    if writer is None:
        try:
            writer = pq.ParquetWriter(str(path), table.schema, compression=compression)
        except (ValueError, TypeError):
            # Fallback to snappy if the requested codec is unavailable.
            if compression != "snappy":
                writer = pq.ParquetWriter(str(path), table.schema, compression="snappy")
            else:
                raise
    writer.write_table(table)
    return writer


def main() -> None:
    args = parse_args()

    source_dir = args.source_dir.expanduser()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    base_dir = (args.target_root / args.dataset_name).resolve()
    extended_dir = (args.target_root / f"{args.dataset_name}{args.extended_suffix}").resolve()

    ensure_clean_dir(base_dir, overwrite=args.overwrite)
    ensure_clean_dir(extended_dir, overwrite=args.overwrite)
    base_data_dir = base_dir / "data"
    base_data_dir.mkdir(parents=True, exist_ok=True)
    extended_data_dir = extended_dir / "data"
    extended_data_dir.mkdir(parents=True, exist_ok=True)

    split_stats: Dict[str, SplitStats] = {split: SplitStats() for split in SPLITS}
    global_stats = SplitStats()

    base_manifest: Dict[str, Dict[str, Dict[str, object]]] = {}
    extended_manifest: Dict[str, Dict[str, Dict[str, object]]] = {}

    for split in SPLITS:
        source_path = source_dir / f"{split}.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source file for split '{split}': {source_path}")

        base_json_path = base_data_dir / f"{split}.jsonl"
        extended_json_path = extended_data_dir / f"{split}.jsonl"
        base_parquet_path = base_data_dir / f"{split}.parquet"
        extended_parquet_path = extended_data_dir / f"{split}.parquet"

        parquet_writer: Optional[pq.ParquetWriter] = None
        buffered_records: List[Dict[str, object]] = []

        with (
            source_path.open("r", encoding="utf-8") as source_handle,
            base_json_path.open("w", encoding="utf-8") as base_json_out,
            extended_json_path.open("w", encoding="utf-8") as extended_json_out,
        ):
            for line_number, line in enumerate(source_handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                fingerprint = (
                    record.get("meta", {}).get("fingerprint")
                    or record.get("fingerprint")
                )
                if not fingerprint:
                    raise ValueError(
                        f"Missing fingerprint for split '{split}' at line {line_number}"
                    )

                base_record = {
                    "id": fingerprint,
                    "train": record.get("train", []),
                    "query": record.get("query", []),
                    "solution": record.get("solution", []),
                }
                extended_record = dict(record)
                extended_record["id"] = fingerprint

                base_json_out.write(json.dumps(base_record, separators=(",", ":")) + "\n")
                extended_json_out.write(json.dumps(extended_record, separators=(",", ":")) + "\n")
                buffered_records.append(base_record)

                # Update stats after we know metadata.
                split_stats[split].register_episode(record)
                global_stats.register_episode(record)

                if len(buffered_records) >= args.chunk_size:
                    parquet_writer = write_parquet_chunk(
                        parquet_writer,
                        base_parquet_path,
                        buffered_records,
                        args.parquet_compression,
                    )
                    buffered_records.clear()

        if buffered_records:
            parquet_writer = write_parquet_chunk(
                parquet_writer,
                base_parquet_path,
                buffered_records,
                args.parquet_compression,
            )
        if parquet_writer is not None:
            parquet_writer.close()
        else:
            # Ensure an empty Parquet file is created even if the split had no data.
            empty_table = pa.Table.from_pylist([], schema=PARQUET_SCHEMA)
            pq.write_table(empty_table, base_parquet_path)

        shutil.copy2(base_parquet_path, extended_parquet_path)

        base_manifest[split] = {
            "jsonl": {
                "path": str(base_json_path.relative_to(base_dir)),
                "records": split_stats[split].num_episodes,
                "bytes": base_json_path.stat().st_size,
            },
            "parquet": {
                "path": str(base_parquet_path.relative_to(base_dir)),
                "records": split_stats[split].num_episodes,
                "bytes": base_parquet_path.stat().st_size,
            },
        }
        extended_manifest[split] = {
            "jsonl": {
                "path": str(extended_json_path.relative_to(extended_dir)),
                "records": split_stats[split].num_episodes,
                "bytes": extended_json_path.stat().st_size,
            },
            "parquet": {
                "path": str(extended_parquet_path.relative_to(extended_dir)),
                "records": split_stats[split].num_episodes,
                "bytes": extended_parquet_path.stat().st_size,
            },
        }

    # Persist metadata artefacts
    stats_payload = {
        "description": "Summary statistics collected while exporting the dataset.",
        "splits": {split: stats.as_dict() for split, stats in split_stats.items()},
        "global": global_stats.as_dict(),
    }
    (base_dir / "dataset_stats.json").write_text(
        json.dumps(stats_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (extended_dir / "dataset_stats.json").write_text(
        json.dumps(stats_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    features_payload = {
        "format": "parquet",
        "features": BASE_FEATURES.to_dict(),
    }
    (base_dir / "features.json").write_text(
        json.dumps(features_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (extended_dir / "features.json").write_text(
        json.dumps(features_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (base_dir / "data_files.json").write_text(
        json.dumps(base_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (extended_dir / "data_files.json").write_text(
        json.dumps(extended_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
