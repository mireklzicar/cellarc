#!/usr/bin/env python3
"""Render wide CA panels plus per-episode patch tiles."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def _ensure_2d_tile(data: Any) -> Optional[np.ndarray]:
    """Normalize train/query slices to 2D arrays suitable for imshow."""
    if data is None:
        return None
    array = np.asarray(data, dtype=int)
    if array.size == 0:
        return None
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[None, :]
    elif array.ndim > 2:
        # Collapse higher dimensions while preserving the leading axis.
        array = array.reshape(array.shape[0], -1)
    return array


def _save_tile_image(
    tile: Any,
    path: Path,
    *,
    height: float,
    dpi: int,
) -> bool:
    """Write a single train/query tile to disk."""
    array = _ensure_2d_tile(tile)
    if array is None:
        return False
    tile_height, tile_width = array.shape
    if tile_height <= 0 or tile_width <= 0:
        return False
    width_inches = max(height, (tile_width / max(1, tile_height)) * height)
    fig, ax = plt.subplots(figsize=(width_inches, height), dpi=dpi, facecolor=BG_COLOR)
    ax.imshow(array, aspect="equal", interpolation="nearest", cmap=PALETTE)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def render_patch_tiles(
    record: Dict[str, Any],
    *,
    patch_dir: Path,
    patch_height: float,
    patch_dpi: int,
) -> List[Dict[str, Any]]:
    """Render every train/query/solution patch for a single record."""
    patch_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Dict[str, Any]] = []

    train_pairs = record.get("train") or []
    for idx, pair in enumerate(train_pairs):
        if not isinstance(pair, dict):
            continue
        input_tile = pair.get("input")
        if input_tile is not None:
            input_path = patch_dir / f"input_{idx:02d}.png"
            if _save_tile_image(input_tile, input_path, height=patch_height, dpi=patch_dpi):
                saved.append({"type": "train_input", "index": idx, "file": str(input_path)})
        output_tile = pair.get("output")
        if output_tile is not None:
            output_path = patch_dir / f"output_{idx:02d}.png"
            if _save_tile_image(output_tile, output_path, height=patch_height, dpi=patch_dpi):
                saved.append({"type": "train_output", "index": idx, "file": str(output_path)})

    query = record.get("query")
    if query is not None:
        query_path = patch_dir / "query.png"
        if _save_tile_image(query, query_path, height=patch_height, dpi=patch_dpi):
            saved.append({"type": "query", "file": str(query_path)})

    solution = record.get("solution")
    if solution is not None:
        solution_path = patch_dir / "solution.png"
        if _save_tile_image(solution, solution_path, height=patch_height, dpi=patch_dpi):
            saved.append({"type": "solution", "file": str(solution_path)})

    return saved


def _panel_figsize(history: np.ndarray, *, height: float, min_width: float) -> tuple[float, float]:
    """Compute a figure size that keeps cells square while emphasizing width."""
    if history.ndim != 2:
        raise ValueError("Expected 2D space-time array.")
    hist_height, hist_width = history.shape
    if hist_height <= 0 or hist_width <= 0:
        return (max(min_width, height), height)
    width_inches = max(min_width, (hist_width / max(1, hist_height)) * height)
    return (width_inches, height)


def render_wide_panels(
    records: Sequence[dict],
    *,
    output_dir: Path,
    prefix: str,
    rng: random.Random,
    tau_max: Optional[int],
    height: float,
    min_width: float,
    dpi: int,
    patch_height: float,
    patch_dpi: int,
) -> List[Dict[str, Any]]:
    """Render each record as a wide CA panel and dump its patch tiles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        history = space_time_from_record(
            record,
            tau_max=tau_max,
            rng_seed=rng.randint(0, 2**31 - 1),
        )
        fig_width, fig_height = _panel_figsize(history, height=height, min_width=min_width)
        fig, ax = plt.subplots(
            figsize=(fig_width, fig_height),
            squeeze=True,
            dpi=dpi,
            facecolor=BG_COLOR,
        )
        ax.imshow(history, aspect="equal", interpolation="nearest", cmap=PALETTE)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)

        metadata = record.get("meta") or {}
        fingerprint = (
            metadata.get("fingerprint")
            or record.get("fingerprint")
            or record.get("id")
            or f"record_{idx:02d}"
        )
        suffix = str(fingerprint)[:10]
        stub = f"{prefix}_{idx:02d}_{suffix}"
        panel_path = output_dir / f"{stub}.png"
        fig.savefig(panel_path, dpi=dpi)
        plt.close(fig)

        patch_dir = output_dir / stub
        patch_entries = render_patch_tiles(
            record,
            patch_dir=patch_dir,
            patch_height=patch_height,
            patch_dpi=patch_dpi,
        )

        summary.append(
            {
                "panel": str(panel_path),
                "patch_dir": str(patch_dir),
                "patch_files": patch_entries,
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
        help="Directory to write the CA images to.",
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
        "--height",
        type=float,
        default=20.0,
        help="Figure height in inches for the CA panels (default: 20).",
    )
    parser.add_argument(
        "--min-width",
        type=float,
        default=6.0,
        help="Minimum figure width in inches for CA panels (default: 6).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Output resolution for CA panels (default: 240 dpi).",
    )
    parser.add_argument(
        "--patch-height",
        type=float,
        default=1.4,
        help="Figure height in inches for patch tiles (default: 1.4).",
    )
    parser.add_argument(
        "--patch-dpi",
        type=int,
        default=240,
        help="Output resolution for patch tiles (default: 240 dpi).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of generated artifacts.",
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
        print(f"[plot_wide_ca_patches] Filtered {len(filtered)} / {len(records)} episodes.", flush=True)

    selected = list(select_records(filtered, args.count, rng))
    prefix = args.prefix or f"{args.input.stem}_wide"
    summary = render_wide_panels(
        selected,
        output_dir=args.output_dir,
        prefix=prefix,
        rng=rng,
        tau_max=args.tau_max,
        height=args.height,
        min_width=args.min_width,
        dpi=args.dpi,
        patch_height=args.patch_height,
        patch_dpi=args.patch_dpi,
    )

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        payload = {"count": len(summary), "records": summary}
        with args.summary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
