"""Episode card visualisations for the Cell ARC dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from eval.common import EpisodeRecord
from cellarc.generation.cax_runner import AutomatonRunner
from cellarc.generation.serialization import deserialize_rule_table
from cellarc.utils import de_bruijn_cycle
from cellarc.visualization.ca_rollout_viz import _ensure_rule_table

from cellarc.visualization.palette import BG_COLOR, PALETTE


def runner_from_record(rec: Dict[str, Any], *, rng_seed: int = 0) -> AutomatonRunner:
    """Instantiate an AutomatonRunner for the serialized rule table."""
    if "rule_table" not in rec or not isinstance(rec.get("rule_table"), dict):
        entry = EpisodeRecord(record=rec, source=Path("<episode_cards>"))
        payload = _ensure_rule_table(entry)
    else:
        payload = rec["rule_table"]
    if not isinstance(payload, dict):
        raise ValueError("Episode record is missing a valid rule_table payload.")
    table = deserialize_rule_table(payload)
    alphabet_size = int(payload["alphabet_size"])
    radius = int(payload["radius"])
    return AutomatonRunner(
        alphabet_size=alphabet_size,
        radius=radius,
        table=table,
        rng_seed=rng_seed,
    )


def space_time_from_record(
    rec: Dict[str, Any],
    *,
    tau_max: Optional[int] = None,
    rng_seed: int = 0,
) -> np.ndarray:
    """Reconstruct the space–time diagram implied by a dataset record."""
    meta = rec["meta"]
    alphabet_size = int(meta["alphabet_size"])
    window = int(meta["window"])
    steps = int(meta["steps"])
    cycle = de_bruijn_cycle(alphabet_size, window)
    depth = tau_max if tau_max is not None else max(4, min(24, steps + 8))
    depth = max(0, depth)
    runner = runner_from_record(rec, rng_seed=rng_seed)
    history = runner.evolve(
        cycle,
        timesteps=depth + steps + 1,
        return_history=True,
    )
    return history


def show_episode_card(
    rec: Dict[str, Any],
    *,
    palette=None,
    tau_max: Optional[int] = None,
    rng_seed: int = 0,
    show_core: bool = True,
):
    """Render an ARC-style card with train I/O bands and an unrolled CA view."""
    space_time = space_time_from_record(rec, tau_max=tau_max, rng_seed=rng_seed)
    meta = rec["meta"]
    window = int(meta["window"])
    half = (window - 1) // 2
    spans = meta.get("train_spans", [])
    steps = int(meta["steps"])

    cmap = palette or PALETTE

    fig = plt.figure(figsize=(10, 4), facecolor=BG_COLOR)

    # Left panel: stacked train pairs plus the query/solution pair.
    ax_left = fig.add_subplot(1, 2, 1)
    ax_left.set_facecolor(BG_COLOR)
    tiles = []
    for pair in rec.get("train", []):
        inp = np.asarray(pair["input"], dtype=int)[None, :]
        out = np.asarray(pair["output"], dtype=int)[None, :]
        tiles.extend([inp, out])
        if show_core:
            gap_width = max(inp.shape[1], out.shape[1])
            gap = np.full((1, gap_width), -1, dtype=int)
            tiles.append(gap)
    query = rec.get("query")
    solution = rec.get("solution")
    if query is not None and solution is not None:
        q_arr = np.asarray(query, dtype=int)[None, :]
        s_arr = np.asarray(solution, dtype=int)[None, :]
        tiles.extend([q_arr, s_arr])
        if show_core:
            gap_width = max(q_arr.shape[1], s_arr.shape[1])
            tiles.append(np.full((1, gap_width), -1, dtype=int))
    if tiles:
        max_width = max(tile.shape[1] for tile in tiles)
        padded_tiles = [
            tile
            if tile.shape[1] == max_width
            else np.pad(
                tile,
                ((0, 0), (0, max_width - tile.shape[1])),
                mode="constant",
                constant_values=-1,
            )
            for tile in tiles
        ]
        # Pad rows so the stacked image renders even when I/O widths differ.
        stack = np.ma.masked_equal(np.concatenate(padded_tiles, axis=0), -1)
        ax_left.imshow(stack, aspect="auto", interpolation="nearest", cmap=cmap)
    ax_left.set_title("Train & Query I/O")
    ax_left.axis("off")

    # Right panel: space–time diagram with training spans highlighted.
    ax_right = fig.add_subplot(1, 2, 2)
    ax_right.set_facecolor(BG_COLOR)
    space_width = space_time.shape[1] if space_time.ndim > 1 else 0
    ax_right.imshow(
        space_time, aspect="auto", interpolation="nearest", cmap=cmap, zorder=0
    )
    ax_right.set_title("Unrolled CA (rows = time)")
    if space_width:
        ax_right.set_xlim(-0.5, space_width - 0.5)

    def _draw_outline(x: float, y: float, width: float, height: float, *, dashed: bool):
        shadow = plt.Rectangle(
            (x, y),
            width,
            height,
            fill=False,
            linewidth=2.4,
            edgecolor="black",
            alpha=0.6,
            zorder=3,
        )
        ax_right.add_patch(shadow)
        ax_right.add_patch(
            plt.Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                linewidth=1.2,
                edgecolor="white",
                linestyle="--" if dashed else "solid",
                zorder=4,
            )
        )

    def _draw_wrapped(start: int, width: int, tau: int, *, dashed: bool) -> None:
        if space_width <= 0 or width <= 0:
            return
        start_mod = start % space_width
        remaining = width
        segments = []
        first = min(remaining, space_width - start_mod)
        segments.append((start_mod, first))
        remaining -= first
        while remaining > 0:
            seg_width = min(remaining, space_width)
            segments.append((0, seg_width))
            remaining -= seg_width
        for x0, w0 in segments:
            _draw_outline(x0, tau, w0, 1, dashed=dashed)

    for span in spans:
        start = int(span.get("start", 0))
        length = int(span.get("length", 0))
        tau = int(span.get("time", 0))
        width = length + 2 * half
        _draw_wrapped(start - half, width, tau, dashed=False)
        _draw_wrapped(start - half, width, tau + steps, dashed=False)
    ax_right.set_xlabel("space")
    ax_right.set_ylabel("time")
    query_span = meta.get("query_span")
    if query_span:
        q_start = int(query_span.get("start", 0))
        q_len = int(query_span.get("length", 0))
        q_tau = int(query_span.get("time", 0))
        highlight_width = q_len + 2 * half
        _draw_wrapped(q_start - half, highlight_width, q_tau, dashed=True)
        _draw_wrapped(q_start - half, highlight_width, q_tau + steps, dashed=True)
    plt.tight_layout()
    return fig


__all__ = [
    "runner_from_record",
    "space_time_from_record",
    "show_episode_card",
]
