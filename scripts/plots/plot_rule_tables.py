#!/usr/bin/env python3
"""Visualize CA rule tables for a sample of episodes.

For each selected episode, this script decodes the rule table payload and
renders a grid of heatmaps showing the mapping from neighbourhoods to output
states. Each panel fixes the center cell value (c in 0..k-1) and plots the
output over all combinations of left and right contexts:

    rows   = left context index (k^r combinations)
    columns= right context index (k^r combinations)

Large tables are handled safely: the script can downsample the left/right
contexts to stay under a configurable maximum number of cells per panel.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import sys
from array import array
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

# Non-interactive backend for headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pylint: disable=wrong-import-position

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellarc.visualization import BG_COLOR, PALETTE  # noqa: E402
from cellarc.generation.serialization import deserialize_rule_table  # noqa: E402
from cellarc.generation.reconstruction import (  # noqa: E402
    infer_dataset_config,
    reconstruct_rule_table_payload,
)
from scripts.plots._episode_utils import (  # noqa: E402
    filter_records,
    load_jsonl,
    load_meta_lookup,
    merge_metadata,
    select_records,
)


def _np_dtype_for_typecode(typecode: str) -> np.dtype:
    if typecode == "B":
        return np.uint8
    if typecode == "H":
        return np.uint16
    if typecode == "I":
        return np.uint32
    # Fallback: the values should always fit in standard ints
    return np.int32


def _decode_payload_to_array(payload: Dict[str, Any]) -> Tuple[np.ndarray, int, int, int]:
    """Return (values_flat, k, r, center_index) from a serialized rule payload.

    Uses the project's deserializer for correctness and robustness; then views
    the contiguous value buffer as a NumPy array without copying.
    """
    table = deserialize_rule_table(payload)
    k = int(payload["alphabet_size"])  # alphabet size
    r = int(payload["radius"])  # radius
    center_index = int(payload.get("center_index", r))
    values = table.values_view()  # array(typecode)
    dtype = _np_dtype_for_typecode(table.typecode)
    flat = np.frombuffer(values, dtype=dtype)
    return flat, k, r, center_index


def _extract_rule_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Get a serialized rule_table payload from the record or reconstruct it."""
    payload = record.get("rule_table")
    if isinstance(payload, dict):
        return payload
    meta = record.get("meta") or {}
    # Some JSONL variants store rule_table under meta
    meta_payload = meta.get("rule_table") if isinstance(meta, dict) else None
    if isinstance(meta_payload, dict):
        return meta_payload
    config = infer_dataset_config(meta)
    if config is None:
        raise ValueError(
            "Episode is missing a rule_table and cannot be reconstructed from metadata."
        )
    return reconstruct_rule_table_payload(meta, config=config)


def _reshape_to_left_center_right(
    flat: np.ndarray, k: int, r: int, center_index: int
) -> np.ndarray:
    """Reshape flat values to (k^r, k, k^r) in lexicographic base-k order."""
    arity = 2 * r + 1
    if flat.size != k ** arity:
        raise ValueError(
            f"Unexpected rule table length {flat.size}; expected {k ** arity} for k={k}, r={r}."
        )
    # Dimensions [d0..d_{arity-1}] each of size k; lexicographic -> last axis changes fastest
    cube = flat.reshape((k,) * arity, order="C")
    # Merge left and right segments into single axes
    left_size = k ** center_index
    right_size = k ** (arity - center_index - 1)
    arr_lcr = cube.reshape((left_size, k, right_size), order="C")
    return arr_lcr


def _compute_panel_size(left: int, right: int, base: float = 2.6) -> Tuple[float, float]:
    """Heuristic: keep square-ish cells; clamp to reasonable inches."""
    if left <= 0 or right <= 0:
        return (base, base)
    aspect = right / max(1, left)
    height = max(2.2, min(4.5, base))
    width = max(2.2, min(6.0, height * aspect))
    return width, height


def _downsample_lcr(
    lcr: np.ndarray,
    *,
    max_cells_per_panel: int,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Stride-sample rows/cols if needed to keep panel size manageable.

    Returns (sampled_lcr, row_indices, col_indices) where indices are the
    chosen left/right positions.
    """
    left, k, right = lcr.shape
    area = left * right
    if area <= max_cells_per_panel:
        return lcr, list(range(left)), list(range(right))

    # Choose strides so left' * right' <= max_cells_per_panel while preserving aspect
    target = max(1, max_cells_per_panel)
    # Aim for left' ~= sqrt(target * left/right), right' ~= sqrt(target * right/left)
    left_prime = max(1, int(math.sqrt(target * left / max(1, right))))
    right_prime = max(1, int(math.sqrt(target * right / max(1, left))))
    left_prime = min(left, left_prime)
    right_prime = min(right, right_prime)
    stride_l = max(1, left // left_prime)
    stride_r = max(1, right // right_prime)
    rows = list(range(0, left, stride_l))[:left_prime]
    cols = list(range(0, right, stride_r))[:right_prime]
    sampled = lcr[np.ix_(rows, range(k), cols)]
    return sampled, rows, cols


def _layout_for_k(k: int) -> Tuple[int, int]:
    """Pick a compact subplot grid for the alphabet size."""
    if k <= 1:
        return 1, 1
    if k == 2:
        return 1, 2
    if k <= 4:
        return 2, 2
    # k in {5,6} -> 2x3
    return 2, 3


def _render_rule_figure(
    record: Dict[str, Any],
    output_path: Path,
    *,
    max_cells_per_panel: int,
    dpi: int,
) -> None:
    payload = _extract_rule_payload(record)
    flat, k, r, center_index = _decode_payload_to_array(payload)
    lcr = _reshape_to_left_center_right(flat, k, r, center_index)
    sampled, rows, cols = _downsample_lcr(lcr, max_cells_per_panel=max_cells_per_panel)

    left, _, right = sampled.shape
    nrows, ncols = _layout_for_k(k)
    panel_w, panel_h = _compute_panel_size(left, right)
    fig_w = ncols * panel_w + 0.6
    fig_h = nrows * panel_h + 0.8

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        dpi=dpi,
        facecolor=BG_COLOR,
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.08, wspace=0.15, hspace=0.18)

    # Metadata for title
    meta = record.get("meta") or {}
    fingerprint = (
        meta.get("fingerprint")
        or record.get("fingerprint")
        or record.get("id")
        or "record"
    )
    family = meta.get("family")
    lam_val = meta.get("lambda")
    lam_str = f"{float(lam_val):.3f}" if isinstance(lam_val, (int, float)) else "?"
    title = f"Rule table (k={k}, r={r}, family={family}, λ={lam_str}): {str(fingerprint)[:10]}"
    fig.suptitle(title, fontsize=11, y=0.98)

    # Draw each center slice
    for c in range(k):
        rr = c // ncols
        cc = c % ncols
        ax = axes[rr][cc]
        plane = sampled[:, c, :]
        im = ax.imshow(plane, interpolation="nearest", aspect="equal", cmap=PALETTE, vmin=0, vmax=max(1, k - 1))
        ax.set_title(f"center = {c}", fontsize=9)
        ax.set_xlabel("right context idx")
        ax.set_ylabel("left context idx")
        # Only show a few ticks for readability
        ax.set_xticks([0, max(0, right // 2), max(0, right - 1)])
        ax.set_yticks([0, max(0, left // 2), max(0, left - 1)])
    # Hide any unused axes if k < nrows*ncols
    for idx in range(k, nrows * ncols):
        rr = idx // ncols
        cc = idx % ncols
        axes[rr][cc].axis("off")

    # Shared colorbar
    cax = fig.add_axes([0.99, 0.12, 0.012, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("output state")
    cb.set_ticks(list(range(k)))

    # Note about downsampling (if any)
    full_left = k ** center_index
    full_right = k ** (2 * r + 1 - center_index - 1)
    if len(rows) != full_left or len(cols) != full_right:
        note = f"downsampled: left {len(rows)}/{full_left}, right {len(cols)}/{full_right}"
        fig.text(0.01, 0.01, note, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a JSONL split file (prefer *_100.jsonl for speed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the rule table figures to.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of episodes to sample (default: 10).",
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
        "--max-cells-per-panel",
        type=int,
        default=60000,
        help=(
            "Ceiling on left*right cells per center-slice panel (default: 60000). "
            "Larger tables are strided to fit within this bound."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output resolution (default: 220 dpi).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of generated figures.",
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
        print(f"[plot_rule_tables] Filtered {len(filtered)} / {len(records)} episodes.", flush=True)

    selected = list(select_records(filtered, args.count, rng))
    prefix = args.prefix or f"{args.input.stem}_rules"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for idx, record in enumerate(selected):
        meta = record.get("meta") or {}
        fingerprint = (
            meta.get("fingerprint")
            or record.get("fingerprint")
            or record.get("id")
            or f"record_{idx:02d}"
        )
        stub = f"{prefix}_{idx:02d}_{str(fingerprint)[:10]}"
        out_path = args.output_dir / f"{stub}.png"
        _render_rule_figure(
            record,
            out_path,
            max_cells_per_panel=max(1000, int(args.max_cells_per_panel)),
            dpi=args.dpi,
        )
        summary.append(
            {
                "file": str(out_path),
                "fingerprint": fingerprint,
                "split": meta.get("split") or record.get("split"),
                "family": meta.get("family"),
                "alphabet_size": meta.get("alphabet_size"),
                "lambda": meta.get("lambda"),
                "radius": meta.get("radius"),
                "steps": meta.get("steps"),
                "window": meta.get("window"),
            }
        )

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        payload = {"count": len(summary), "records": summary}
        with args.summary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
