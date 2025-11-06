#!/usr/bin/env python3
"""Render episode cards for a JSONL split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

# Force a non-interactive backend so the script runs in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pylint: disable=wrong-import-position

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellarc.visualization import show_episode_card


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


def render_cards(
    records: Iterable[dict],
    *,
    output_dir: Path,
    prefix: str,
    rng: random.Random,
    tau_max: Optional[int] = None,
    show_metadata: bool = False,
    metadata_fields: Optional[Sequence[str]] = None,
) -> None:
    """Render and save episode cards."""

    metadata_flag = show_metadata or bool(metadata_fields)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(records):
        fig = show_episode_card(
            record,
            rng_seed=rng.randint(0, 2**31 - 1),
            tau_max=tau_max,
            show_metadata=metadata_flag,
            metadata_fields=metadata_fields,
        )
        fingerprint = (
            record.get("meta", {}).get("fingerprint")
            or record.get("fingerprint")
            or "record"
        )
        suffix = str(fingerprint)[:10]
        filename = f"{prefix}_{idx:02d}_{suffix}.png"
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the JSONL split file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the episode card images to.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of records to sample (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for sampling (default: 0).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix (defaults to input stem).",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional path to companion metadata JSONL file.",
    )
    parser.add_argument(
        "--split",
        dest="splits",
        nargs="+",
        default=None,
        help="Filter to episodes whose split matches any of the provided names.",
    )
    parser.add_argument(
        "--family",
        dest="families",
        nargs="+",
        default=None,
        help="Filter to CA families (case-insensitive).",
    )
    parser.add_argument(
        "--alphabet-size",
        dest="alphabet_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Filter to automata with the given alphabet sizes.",
    )
    parser.add_argument(
        "--lambda-min",
        "--lambda-threshold",
        dest="lambda_min",
        type=float,
        default=None,
        help="Minimum lambda value (inclusive).",
    )
    parser.add_argument(
        "--lambda-max",
        dest="lambda_max",
        type=float,
        default=None,
        help="Maximum lambda value (inclusive).",
    )
    parser.add_argument(
        "--tau-max",
        type=int,
        default=None,
        help="Maximum rollout depth passed to the renderer.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Render a metadata footer below each card.",
    )
    parser.add_argument(
        "--metadata-fields",
        nargs="+",
        default=None,
        help="Explicit list of metadata keys to display in the footer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    records = load_jsonl(args.input)
    if not records:
        raise ValueError(f"No records found in {args.input}")

    meta_path = args.meta
    if meta_path is None:
        candidate = args.input.with_name(f"{args.input.stem}_meta.jsonl")
        if candidate.exists():
            meta_path = candidate
    if meta_path is not None:
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        meta_lookup = load_meta_lookup(meta_path)
        merge_metadata(records, meta_lookup)

    default_split = args.input.stem
    for record in records:
        record.setdefault("split", default_split)
        meta = record.get("meta")
        if isinstance(meta, dict):
            meta.setdefault("split", default_split)

    filtered = filter_records(
        records,
        splits=args.splits,
        families=args.families,
        alphabet_sizes=args.alphabet_sizes,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
    )
    if not filtered:
        raise ValueError("No records matched the provided filters.")
    if len(filtered) < len(records):
        print(f"[plot_episode_cards] Filtered {len(filtered)} / {len(records)} episodes.", flush=True)

    selected = list(select_records(filtered, args.count, rng))
    prefix = args.prefix or args.input.stem
    render_cards(
        selected,
        output_dir=args.output_dir,
        prefix=prefix,
        rng=rng,
        tau_max=args.tau_max,
        show_metadata=args.include_metadata,
        metadata_fields=args.metadata_fields,
    )


if __name__ == "__main__":
    main()
