#!/usr/bin/env python3
"""Plot a De Bruijn diagram for CA rules as a colored patch matrix.

For a 1D CA with window size W=2r+1 over alphabet size k, the De Bruijn graph
of order W-1 has N = k^(W-1) nodes, one for each length-(W-1) word. Each edge
corresponds to a length-W neighbourhood (an element of the rule table) and is
directed from the prefix (first W-1 symbols) to the suffix (last W-1 symbols).

This script visualizes the graph as an N×N matrix where the (u, v) cell is
coloured by the rule output for the unique edge u→v (if it exists). To avoid
excessive memory for large N, the node set is downsampled to a user-specified
maximum, and only edges between sampled nodes are shown. Missing edges are
rendered with the background colour.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Headless-safe
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


def _extract_rule_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = record.get("rule_table")
    if isinstance(payload, dict):
        return payload
    meta = record.get("meta") or {}
    meta_payload = meta.get("rule_table") if isinstance(meta, dict) else None
    if isinstance(meta_payload, dict):
        return meta_payload
    config = infer_dataset_config(meta)
    if config is None:
        raise ValueError(
            "Episode is missing a rule_table and cannot be reconstructed from metadata."
        )
    return reconstruct_rule_table_payload(meta, config=config)


def _int_to_base_k_tuple(x: int, k: int, length: int) -> Tuple[int, ...]:
    digits = [0] * length
    for i in range(length - 1, -1, -1):
        digits[i] = x % k
        x //= k
    return tuple(digits)


def _edge_destination(prefix: Tuple[int, ...], symbol: int) -> Tuple[int, ...]:
    return prefix[1:] + (symbol,)


def _build_sampled_adjacency(
    table_payload: Dict[str, Any],
    *,
    rng: random.Random,
    max_nodes: int,
) -> Tuple[np.ndarray, int, int, int, List[int]]:
    """Create a sampled adjacency matrix coloured by rule outputs.

    Returns (adj, k, r, arity, node_indices) where adj shape is (S, S) and
    node_indices are the integer-encoded nodes shown along each axis.
    """
    table = deserialize_rule_table(table_payload)
    k = int(table.alphabet_size)
    r = int(table.radius)
    arity = 2 * r + 1
    node_len = arity - 1  # W-1
    total_nodes = k ** node_len

    # Choose which nodes to display
    if total_nodes <= max_nodes:
        selected_nodes = list(range(total_nodes))
    else:
        selected_nodes = sorted(rng.sample(range(total_nodes), max_nodes))
    S = len(selected_nodes)
    node_index_to_pos = {idx: pos for pos, idx in enumerate(selected_nodes)}

    adj = np.full((S, S), -1, dtype=np.int32)

    for idx_int in selected_nodes:
        prefix = _int_to_base_k_tuple(idx_int, k=k, length=node_len)
        src_pos = node_index_to_pos[idx_int]
        for a in range(k):
            neighbourhood = prefix + (a,)
            value = table[neighbourhood]
            dst_tuple = _edge_destination(prefix, a)
            # Encode destination tuple back to integer index
            # base-k positional encoding
            dst_idx = 0
            for d in dst_tuple:
                dst_idx = dst_idx * k + int(d)
            dst_pos = node_index_to_pos.get(dst_idx)
            if dst_pos is not None:
                adj[src_pos, dst_pos] = int(value)

    return adj, k, r, arity, selected_nodes


def _render_adj_matrix(
    adj: np.ndarray,
    *,
    k: int,
    r: int,
    arity: int,
    total_nodes: int,
    node_count: int,
    output_path: Path,
    meta: Dict[str, Any],
    dpi: int,
) -> None:
    # Mask -1 entries to show background
    data = adj.astype(float)
    data[data < 0] = np.nan

    # Keep square-ish but scale to node_count
    size = max(3.0, min(12.0, 0.05 * node_count + 4.0))
    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi, facecolor=BG_COLOR)
    im = ax.imshow(
        data,
        interpolation="nearest",
        aspect="equal",
        cmap=PALETTE,
        vmin=0,
        vmax=max(1, k - 1),
    )
    ax.set_xlabel("suffix node index")
    ax.set_ylabel("prefix node index")
    ax.set_title(
        f"De Bruijn diagram (order W-1={arity-1}) • k={k}, r={r} • nodes {node_count}/{total_nodes}",
        fontsize=10,
    )
    # Light ticks for orientation (not all to avoid clutter)
    ticks = [0, max(0, node_count // 2), max(0, node_count - 1)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("rule output state")
    cbar.set_ticks(list(range(k)))

    # Footnote
    fingerprint = meta.get("fingerprint")
    family = meta.get("family")
    lam_val = meta.get("lambda")
    lam_str = f"{float(lam_val):.3f}" if isinstance(lam_val, (int, float)) else "?"
    info = f"family={family}, λ={lam_str}, fingerprint={str(fingerprint)[:10]}"
    fig.text(0.01, 0.01, info, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to JSONL split file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--count", type=int, default=10, help="Episodes to sample (default: 10).")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
    parser.add_argument("--prefix", type=str, default=None, help="Filename prefix (default: input stem).")
    parser.add_argument("--meta", type=Path, default=None, help="Optional path to metadata JSONL.")
    parser.add_argument("--split", dest="splits", nargs="+", default=None, help="Filter by split.")
    parser.add_argument("--family", dest="families", nargs="+", default=None, help="Filter by family.")
    parser.add_argument(
        "--alphabet-size",
        dest="alphabet_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Filter by alphabet size(s).",
    )
    parser.add_argument("--lambda-min", dest="lambda_min", type=float, default=None, help="Min λ.")
    parser.add_argument("--lambda-max", dest="lambda_max", type=float, default=None, help="Max λ.")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=128,
        help="Maximum number of nodes to display (default: 128).",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output resolution (default: 220 dpi).")
    parser.add_argument("--summary", type=Path, default=None, help="Optional JSON summary path.")
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
        print(f"[plot_debruijn_diagram] Filtered {len(filtered)} / {len(records)} episodes.")

    selected = list(select_records(filtered, args.count, rng))
    prefix = args.prefix or f"{args.input.stem}_debruijn"

    output_entries: List[Dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(selected):
        payload = _extract_rule_payload(record)
        k = int(payload["alphabet_size"])  # for total_nodes calc
        r = int(payload["radius"])  # for total_nodes calc
        arity = 2 * r + 1
        total_nodes = k ** (arity - 1)

        adj, k_eff, r_eff, arity_eff, node_indices = _build_sampled_adjacency(
            payload, rng=rng, max_nodes=max(4, int(args.max_nodes))
        )

        meta = record.get("meta") or {}
        fingerprint = (
            meta.get("fingerprint")
            or record.get("fingerprint")
            or record.get("id")
            or f"record_{idx:02d}"
        )
        stub = f"{prefix}_{idx:02d}_{str(fingerprint)[:10]}"
        out_path = args.output_dir / f"{stub}.png"
        _render_adj_matrix(
            adj,
            k=k_eff,
            r=r_eff,
            arity=arity_eff,
            total_nodes=total_nodes,
            node_count=adj.shape[0],
            output_path=out_path,
            meta=meta,
            dpi=args.dpi,
        )

        output_entries.append(
            {
                "file": str(out_path),
                "fingerprint": fingerprint,
                "split": meta.get("split") or record.get("split"),
                "family": meta.get("family"),
                "alphabet_size": k_eff,
                "radius": r_eff,
                "window": arity_eff,
                "nodes_shown": int(adj.shape[0]),
                "nodes_total": int(total_nodes),
            }
        )

    if args.summary:
        payload = {"count": len(output_entries), "records": output_entries}
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as handle:
            import json

            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()

