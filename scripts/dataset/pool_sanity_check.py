#!/usr/bin/env python3
"""Sanity checks for downsampled pool shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


def is_pathological(meta: Dict[str, object]) -> bool:
    """Return True if the episode metadata satisfies the pathological criteria."""
    morphology = meta.get("morphology") or {}
    lam = float(meta.get("lambda", 0.0) or 0.0)
    entropy = float(meta.get("avg_cell_entropy", 0.0) or 0.0)
    coverage = meta.get("coverage") or {}
    observed_windows = int(coverage.get("observed_windows", 0) or 0)
    observed_fraction = float(coverage.get("observed_fraction", 0.0) or 0.0)

    if morphology.get("absorbing") and lam < 0.02 and entropy < 0.02:
        return True
    if observed_windows < 128 and observed_fraction < 1e-4:
        return True
    return False


def iter_records(path: Path) -> Iterable[Dict[str, object]]:
    """Yield parsed dictionaries from a JSONL shard."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_meta_index(meta_path: Path | None) -> Dict[str, Dict[str, object]]:
    if meta_path is None:
        return {}
    index: Dict[str, Dict[str, object]] = {}
    with meta_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            fingerprint = payload.get("fingerprint")
            meta = payload.get("meta") or {}
            if isinstance(fingerprint, str) and isinstance(meta, dict):
                index[fingerprint] = meta
    return index


def run_checks(path: Path, meta_index: Dict[str, Dict[str, object]]) -> Tuple[int, int]:
    """Run uniqueness and pathology checks on a JSONL shard."""
    seen_fp = set()
    seen_probe = set()
    pathological = 0
    total = 0

    for record in iter_records(path):
        meta = record.get("meta", {}) or {}
        fp = meta.get("fingerprint")
        if fp is None and isinstance(record.get("fingerprint"), str):
            fp = record["fingerprint"]
        if not isinstance(fp, str):
            raise AssertionError("Record is missing fingerprint.")

        merged_meta = dict(meta_index.get(fp, {}))
        merged_meta.update(meta)
        probe = merged_meta.get("probe_fingerprint")
        if not isinstance(probe, str):
            raise AssertionError(f"Record {fp} is missing probe_fingerprint.")

        if fp in seen_fp:
            raise AssertionError(f"Duplicate fingerprint detected: {fp}")
        if probe in seen_probe:
            raise AssertionError(f"Duplicate probe_fingerprint detected: {probe}")

        seen_fp.add(fp)
        seen_probe.add(probe)

        if is_pathological(merged_meta):
            pathological += 1

        total += 1

    if pathological:
        raise AssertionError(f"Detected {pathological} pathological episodes.")

    return total, pathological


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/pool_downsampled/downsampled_enriched.jsonl"),
        help="Path to the JSONL shard to check (default: artifacts/pool_downsampled/downsampled_enriched.jsonl).",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        help="Optional companion *_meta.jsonl file providing full episode metadata.",
    )
    args = parser.parse_args()

    meta_index = load_meta_index(args.meta)
    total, _ = run_checks(args.input, meta_index)
    print(f"Sanity checks passed for {total} episodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
