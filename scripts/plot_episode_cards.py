#!/usr/bin/env python3
"""Render episode cards for a JSONL split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence

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


def render_cards(
    records: Iterable[dict],
    *,
    output_dir: Path,
    prefix: str,
    rng: random.Random,
) -> None:
    """Render and save episode cards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(records):
        fig = show_episode_card(record, rng_seed=rng.randint(0, 2**31 - 1))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    records = load_jsonl(args.input)
    if not records:
        raise ValueError(f"No records found in {args.input}")
    selected = select_records(records, args.count, rng)
    prefix = args.prefix or args.input.stem
    render_cards(selected, output_dir=args.output_dir, prefix=prefix, rng=rng)


if __name__ == "__main__":
    main()
