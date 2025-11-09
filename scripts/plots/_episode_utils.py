"""Shared helpers for episode-based plotting scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def load_jsonl(path: Path) -> List[dict]:
    """Load all records from a JSONL file."""
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def select_records(records: Sequence[dict], count: int, rng: random.Random) -> Iterable[dict]:
    """Return up to `count` records sampled without replacement."""
    if count >= len(records):
        return records
    return rng.sample(list(records), count)


def load_meta_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load metadata records keyed by episode fingerprint or id."""
    lookup: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            meta = record.get("meta")
            if not isinstance(meta, dict):
                continue
            record_id = (
                record.get("id")
                or meta.get("fingerprint")
                or record.get("fingerprint")
            )
            if not record_id:
                continue
            lookup[str(record_id)] = meta
    return lookup


def merge_metadata(records: Sequence[dict], meta_lookup: Dict[str, Dict[str, Any]]) -> None:
    """Attach metadata from a lookup table to each record in place."""
    if not meta_lookup:
        return
    for record in records:
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        record_id = (
            record.get("id")
            or meta.get("fingerprint")
            or record.get("fingerprint")
        )
        if record_id is None:
            continue
        lookup_meta = meta_lookup.get(str(record_id))
        if not lookup_meta:
            continue
        merged = dict(lookup_meta)
        if isinstance(meta, dict):
            merged.update(meta)
        record["meta"] = merged


def filter_records(
    records: Sequence[dict],
    *,
    splits: Optional[Sequence[str]] = None,
    families: Optional[Sequence[str]] = None,
    alphabet_sizes: Optional[Sequence[int]] = None,
    lambda_min: Optional[float] = None,
    lambda_max: Optional[float] = None,
) -> List[dict]:
    """Filter records according to split, family, alphabet size, and λ thresholds."""
    split_set = {str(split).lower() for split in splits} if splits else None
    family_set = {str(fam).lower() for fam in families} if families else None
    alphabet_set = {int(size) for size in alphabet_sizes} if alphabet_sizes else None

    filtered: List[dict] = []
    for record in records:
        meta = record.get("meta")
        if not isinstance(meta, dict):
            meta = {}

        if split_set is not None:
            record_split = record.get("split") or meta.get("split")
            if record_split is None or str(record_split).lower() not in split_set:
                continue

        if family_set is not None:
            family = meta.get("family")
            if family is None or str(family).lower() not in family_set:
                continue

        if alphabet_set is not None:
            alphabet_value = meta.get("alphabet_size")
            try:
                alphabet_int = int(alphabet_value)
            except (TypeError, ValueError):
                continue
            if alphabet_int not in alphabet_set:
                continue

        lambda_value = meta.get("lambda")
        if lambda_min is not None:
            try:
                if lambda_value is None or float(lambda_value) < float(lambda_min):
                    continue
            except (TypeError, ValueError):
                continue
        if lambda_max is not None:
            try:
                if lambda_value is None or float(lambda_value) > float(lambda_max):
                    continue
            except (TypeError, ValueError):
                continue

        filtered.append(record)

    return filtered


__all__ = [
    "filter_records",
    "load_jsonl",
    "load_meta_lookup",
    "merge_metadata",
    "select_records",
]

