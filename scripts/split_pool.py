#!/usr/bin/env python3
"""Split the enriched downsampled pool into train/val/test subsets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def iter_records(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def coverage_lambda_key(record: Dict[str, object]) -> Tuple[float, float]:
    meta = record.get("meta", {}) or {}
    coverage = meta.get("coverage") or {}
    obs_frac = coverage.get("observed_fraction", 1.0)
    try:
        obs_frac = float(obs_frac)
    except (TypeError, ValueError):
        obs_frac = 1.0
    lam = meta.get("lambda", 0.0)
    try:
        lam = float(lam)
    except (TypeError, ValueError):
        lam = 0.0
    return obs_frac, -lam


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


def main() -> int:
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
    args = parser.parse_args()

    records = list(iter_records(args.input))
    total_required = (
        args.train_count + args.val_count + args.test_interp_count + args.test_extra_count
    )
    if len(records) < total_required:
        raise ValueError(
            f"Not enough records ({len(records)}) to satisfy requested split sizes ({total_required})."
        )

    records.sort(key=coverage_lambda_key)
    test_extra = records[: args.test_extra_count]
    remaining = records[args.test_extra_count :]

    rng = random.Random(args.seed)
    rng.shuffle(remaining)

    train_end = args.train_count
    val_end = train_end + args.val_count
    test_interp_end = val_end + args.test_interp_count

    train_records = remaining[:train_end]
    val_records = remaining[train_end:val_end]
    test_interp_records = remaining[val_end:test_interp_end]

    if len(train_records) != args.train_count:
        raise AssertionError("Did not obtain expected number of training records.")
    if len(val_records) != args.val_count:
        raise AssertionError("Did not obtain expected number of validation records.")
    if len(test_interp_records) != args.test_interp_count:
        raise AssertionError("Did not obtain expected number of test_interpolation records.")

    leftover = remaining[test_interp_end:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_split("train", train_records, args.output_dir)
    write_split("val", val_records, args.output_dir)
    write_split("test_interpolation", test_interp_records, args.output_dir)
    write_split("test_extrapolation", test_extra, args.output_dir)

    if leftover:
        info_path = args.output_dir / "unused.jsonl"
        with info_path.open("w", encoding="utf-8") as fh:
            for rec in leftover:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    print(
        f"Split {len(records)} records into "
        f"{len(train_records)} train / {len(val_records)} val / "
        f"{len(test_interp_records)} test_interpolation / {len(test_extra)} test_extrapolation."
    )
    if leftover:
        print(f"{len(leftover)} records written to unused.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
