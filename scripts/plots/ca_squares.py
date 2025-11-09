#!/usr/bin/env python3
"""Render square-only CA unrollings without metadata or I/O bands."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
import numpy as np

# Force a non-interactive backend so the script runs in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pylint: disable=wrong-import-position

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellarc.visualization import BG_COLOR, PALETTE, space_time_from_record
from scripts.plots._episode_utils import (
    filter_records,
    load_jsonl,
    load_meta_lookup,
    merge_metadata,
    select_records,
)


def _square_history(history: np.ndarray) -> np.ndarray:
    """Crop or tile the spatial axis so width matches the number of timesteps."""
    if history.ndim != 2:
        raise ValueError("Expected 2D space-time array.")
    height, width = history.shape
    if height == 0 or width == 0:
        return history
    if width == height:
        return history
    if width > height:
        start = max(0, (width - height) // 2)
        end = start + height
        if end > width:
            end = width
            start = end - height
        return history[:, start:end]
    # width < height: tile the spatial dimension until we can crop to square
    reps = int(np.ceil(height / width))
    tiled = np.tile(history, (1, reps))
    return tiled[:, :height]


def render_square_panels(
    records: Sequence[dict],
    *,
    output_dir: Path,
    prefix: str,
    rng: random.Random,
    tau_max: Optional[int],
    size: float,
    dpi: int,
) -> List[Dict[str, object]]:
    """Render each record as a square CA unrolling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, object]] = []
    for idx, record in enumerate(records):
        history = space_time_from_record(
            record,
            tau_max=tau_max,
            rng_seed=rng.randint(0, 2**31 - 1),
        )
        history = _square_history(history)
        fig, ax = plt.subplots(figsize=(size, size), squeeze=True, dpi=dpi, facecolor=BG_COLOR)
        ax.imshow(history, aspect="equal", interpolation="nearest", cmap=PALETTE)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        metadata = record.get("meta") or {}
        fingerprint = (
            metadata.get("fingerprint")
            or record.get("fingerprint")
            or record.get("id")
            or "record"
        )
        filename = f"{prefix}_{idx:02d}_{str(fingerprint)[:10]}.png"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=dpi)
        plt.close(fig)

        summary.append(
            {
                "file": str(filepath),
                "fingerprint": fingerprint,
                "split": metadata.get("split") or record.get("split"),
                "family": metadata.get("family"),
                "alphabet_size": metadata.get("alphabet_size"),
                "lambda": metadata.get("lambda"),
                "lambda_bin": metadata.get("lambda_bin"),
                "avg_cell_entropy": metadata.get("avg_cell_entropy"),
                "radius": metadata.get("radius"),
                "steps": metadata.get("steps"),
                "window": metadata.get("window"),
                "morphology": metadata.get("morphology"),
            }
        )
    return summary


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
        help="Directory to write the square CA images to.",
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
        "--size",
        type=float,
        default=5.0,
        help="Square figure size in inches (default: 5).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Output resolution (default: 240 dpi).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of generated panels.",
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
        print(f"[ca_squares] Filtered {len(filtered)} / {len(records)} episodes.", flush=True)

    selected = list(select_records(filtered, args.count, rng))
    prefix = args.prefix or f"{args.input.stem}_square"
    summary = render_square_panels(
        selected,
        output_dir=args.output_dir,
        prefix=prefix,
        rng=rng,
        tau_max=args.tau_max,
        size=args.size,
        dpi=args.dpi,
    )

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        payload = {"count": len(summary), "records": summary}
        with args.summary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
