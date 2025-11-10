#!/usr/bin/env python3
"""Draw a classical De Bruijn graph for CA rules with coloured edges.

Nodes are all words of length L = W-1 over alphabet 0..k-1 (W = 2r+1). For a
prefix node u = x_0..x_{L-1} and symbol a, there is a directed edge to
v = x_1..x_{L-1}a corresponding to the neighbourhood x_0..x_{L-1}a. The edge
colour encodes the rule output for that neighbourhood.

To remain efficient, we sample up to --max-nodes nodes; only edges with both
endpoints in the sample are drawn.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402  # pylint: disable=wrong-import-position
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

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
    mp = meta.get("rule_table") if isinstance(meta, dict) else None
    if isinstance(mp, dict):
        return mp
    config = infer_dataset_config(meta)
    if config is None:
        raise ValueError("Missing rule_table and cannot reconstruct from metadata.")
    return reconstruct_rule_table_payload(meta, config=config)


def _int_to_base_k_tuple(x: int, k: int, length: int) -> Tuple[int, ...]:
    digits = [0] * length
    for i in range(length - 1, -1, -1):
        digits[i] = x % k
        x //= k
    return tuple(digits)


def _dest_index(prefix_idx: int, a: int, k: int, L: int) -> int:
    """Compute destination node index under base-k encoding.

    Treat nodes as length-L words encoded as integers in base-k. Transition is:
    dst = ((prefix_idx % k^{L-1}) * k) + a
    """
    if L <= 0:
        return a % k
    mod = k ** (L - 1)
    return ((prefix_idx % mod) * k) + (a % k)


def _palette_color_for_value(val: int, k: int):
    # Map discrete state in 0..k-1 to RGBA using the project palette
    if k <= 1:
        frac = 0.0
    else:
        frac = float(val) / float(k - 1)
    rgba = PALETTE(np.array([frac]))[0]
    return rgba


def _build_graph(
    payload: Dict[str, Any],
    *,
    rng: random.Random,
    max_nodes: int,
) -> Tuple[nx.DiGraph, int, int, int, List[int], List[Tuple[int, int]], List[tuple]]:
    table = deserialize_rule_table(payload)
    k = int(table.alphabet_size)
    r = int(table.radius)
    W = 2 * r + 1
    L = W - 1
    total_nodes = k ** L

    if total_nodes <= max_nodes:
        nodes = list(range(total_nodes))
    else:
        nodes = sorted(rng.sample(range(total_nodes), max_nodes))

    pos_map = {idx: i for i, idx in enumerate(nodes)}
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    edge_colors: List[tuple] = []
    edges: List[Tuple[int, int]] = []
    for u in nodes:
        for a in range(k):
            v = _dest_index(u, a, k, L)
            if v in pos_map:
                # Determine rule output value for neighbourhood represented by u+a
                word = _int_to_base_k_tuple(u, k, L) + (a,)
                val = table[word]
                edges.append((u, v))
                edge_colors.append(_palette_color_for_value(int(val), k))

    G.add_edges_from(edges)
    # Also keep the neighbourhood word for each edge in the same order as edges
    edge_words = [(_int_to_base_k_tuple(u, k, L) + ( (v % k), )) for (u, v) in []]  # placeholder not used
    # We will recompute words as needed during drawing to avoid ordering pitfalls
    return G, k, r, total_nodes, nodes, edges, edge_colors


def _circular_layout(nodes: List[int]) -> Dict[int, Tuple[float, float]]:
    n = len(nodes)
    if n == 0:
        return {}
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = {node: (float(np.cos(a)), float(np.sin(a))) for node, a in zip(nodes, angles)}
    return coords


def _draw_graph(
    G: nx.DiGraph,
    *,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    k: int,
    r: int,
    total_nodes: int,
    edge_colors: List[tuple],
    layout: str,
    node_size: int,
    edge_alpha: float,
    dpi: int,
    out_path: Path,
    meta: Dict[str, Any],
    show_node_patches: bool,
    node_patch_rows: int,
    node_patch_zoom: float,
    show_edge_patches: bool,
    edge_patch_rows: int,
    edge_patch_zoom: float,
    edge_patch_sample: Optional[int],
) -> None:
    if layout == "spring":
        pos = nx.spring_layout(G, seed=0, k=None)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = _circular_layout(nodes)

    fig_size = max(4.0, min(12.0, 0.05 * len(nodes) + 5.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi, facecolor=BG_COLOR)

    # Node circles removed per request; only edges + patch overlays are drawn.
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        edge_color=edge_colors,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=8,
        width=1.2,
        alpha=edge_alpha,
        ax=ax,
    )
    ax.set_axis_off()

    # Intentionally no title/metadata overlay — render graph only.

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Optionally overlay miniature node patches showing the node word (length L = W-1)
    if show_node_patches and len(nodes) > 0:
        W = 2 * r + 1
        L = max(0, W - 1)
        if L > 0:
            for node in nodes:
                p = pos[node]
                # Build a tiny colour strip: `node_patch_rows` x L
                word = _int_to_base_k_tuple(node, k, L)
                strip = np.tile(np.array(word, dtype=float), (max(1, node_patch_rows), 1))
                denom = float(max(1, k - 1))
                rgba = PALETTE(strip / denom)
                oi = OffsetImage(rgba, zoom=max(0.1, node_patch_zoom))
                ab = AnnotationBbox(oi, p, frameon=False, box_alignment=(0.5, 0.5), clip_on=False)
                ab.set_zorder(3)
                ax.add_artist(ab)

    # Optionally overlay edge patches showing the full neighbourhood (length W)
    if show_edge_patches and edges:
        W = 2 * r + 1
        L = max(0, W - 1)
        # Optionally subsample edges for clarity/performance
        edgelist = edges
        if edge_patch_sample is not None and edge_patch_sample > 0 and len(edgelist) > edge_patch_sample:
            rng = random.Random(0)
            edgelist = rng.sample(edgelist, edge_patch_sample)
        for (u, v) in edgelist:
            pu = pos[u]
            pv = pos[v]
            mid = ((pu[0] + pv[0]) * 0.5, (pu[1] + pv[1]) * 0.5)
            # Recover a from transition u->v
            # Given integer encoding, a is simply v % k when using the transition function
            a = v % k
            word = _int_to_base_k_tuple(u, k, L) + (a,)
            patch = np.tile(np.array(word, dtype=float), (max(1, edge_patch_rows), 1))
            rgba = PALETTE(patch / float(max(1, k - 1)))
            oi = OffsetImage(rgba, zoom=max(0.1, edge_patch_zoom))
            ab = AnnotationBbox(oi, mid, frameon=True, boxcoords="data", box_alignment=(0.5, 0.5), pad=0.1, clip_on=False)
            ab.set_zorder(4)
            ax.add_artist(ab)

    fig.savefig(out_path, dpi=dpi)
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
    parser.add_argument("--max-nodes", type=int, default=128, help="Maximum nodes to draw (default: 128).")
    parser.add_argument("--layout", type=str, default="circular", choices=["circular", "spring", "shell"], help="Graph layout.")
    parser.add_argument("--node-size", type=int, default=60, help="Node marker size (ignored when circles are hidden).")
    parser.add_argument("--show-node-patches", action="store_true", help="Overlay each node with a small coloured strip representing its (W-1)-digit word.")
    parser.add_argument("--node-patch-rows", type=int, default=3, help="Rows in the miniature node patch strip (visual thickness).")
    parser.add_argument("--node-patch-zoom", type=float, default=4.0, help="Zoom factor for node patch images (default increased ~5x).")
    parser.add_argument("--show-edge-patches", action="store_true", help="Overlay small patches at edge midpoints showing the full W-digit neighbourhood (prefix+(appended symbol)).")
    parser.add_argument("--edge-patch-rows", type=int, default=3, help="Rows in edge patch strip (visual thickness).")
    parser.add_argument("--edge-patch-zoom", type=float, default=1.2, help="Zoom factor for edge patches.")
    parser.add_argument("--edge-patch-sample", type=int, default=200, help="Max number of edges to annotate with patches (sampled). 0 disables sampling.")
    parser.add_argument("--edge-alpha", type=float, default=0.75, help="Edge alpha (0..1). Default: 0.75")
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
        print(f"[plot_debruijn_graph] Filtered {len(filtered)} / {len(records)} episodes.")

    selected = list(select_records(filtered, args.count, rng))
    prefix = args.prefix or f"{args.input.stem}_dbg"

    entries: List[Dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(selected):
        payload = _extract_rule_payload(record)
        G, k, r, total_nodes, nodes, edges, edge_colors = _build_graph(
            payload, rng=rng, max_nodes=max(4, int(args.max_nodes))
        )
        meta = record.get("meta") or {}
        fingerprint = meta.get("fingerprint") or record.get("fingerprint") or record.get("id") or f"record_{i:02d}"
        stub = f"{prefix}_{i:02d}_{str(fingerprint)[:10]}"
        out_path = args.output_dir / f"{stub}.png"

        _draw_graph(
            G,
            nodes=nodes,
            edges=edges,
            k=k,
            r=r,
            total_nodes=total_nodes,
            edge_colors=edge_colors,
            layout=args.layout,
            node_size=args.node_size,
            edge_alpha=args.edge_alpha,
            dpi=args.dpi,
            out_path=out_path,
            meta=meta,
            show_node_patches=args.show_node_patches,
            node_patch_rows=max(1, args.node_patch_rows),
            node_patch_zoom=args.node_patch_zoom,
            show_edge_patches=args.show_edge_patches,
            edge_patch_rows=max(1, args.edge_patch_rows),
            edge_patch_zoom=args.edge_patch_zoom,
            edge_patch_sample=(None if args.edge_patch_sample == 0 else int(args.edge_patch_sample)),
        )

        entries.append(
            {
                "file": str(out_path),
                "fingerprint": fingerprint,
                "split": meta.get("split") or record.get("split"),
                "family": meta.get("family"),
                "alphabet_size": k,
                "radius": r,
                "nodes_shown": len(nodes),
                "nodes_total": total_nodes,
            }
        )

    if args.summary:
        import json

        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as f:
            json.dump({"count": len(entries), "records": entries}, f, indent=2)


if __name__ == "__main__":
    main()
