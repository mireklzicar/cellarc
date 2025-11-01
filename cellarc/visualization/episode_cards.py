"""Episode card visualisations for the Cell ARC dataset."""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from cellarc.generation.cax_runner import AutomatonRunner
from cellarc.generation.serialization import deserialize_rule_table
from cellarc.utils import de_bruijn_cycle

from cellarc.visualization.palette import PALETTE


def runner_from_record(rec: Dict[str, Any], *, rng_seed: int = 0) -> AutomatonRunner:
    """Instantiate an AutomatonRunner for the serialized rule table."""
    payload = rec["rule_table"]
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

    fig = plt.figure(figsize=(10, 4))

    # Left panel: stacked train pairs reminiscent of ARC tasks.
    ax_left = fig.add_subplot(1, 2, 1)
    tiles = []
    for pair in rec.get("train", []):
        inp = np.asarray(pair["input"], dtype=int)[None, :]
        out = np.asarray(pair["output"], dtype=int)[None, :]
        tiles.extend([inp, out])
        if show_core:
            gap = np.full((1, inp.shape[1]), -1, dtype=int)
            tiles.append(gap)
    if tiles:
        stack = np.concatenate(tiles, axis=0)
        ax_left.imshow(stack, aspect="auto", interpolation="nearest", cmap=cmap)
    ax_left.set_title("Train I/O (ARC-like)")
    ax_left.axis("off")

    # Right panel: space–time diagram with training spans highlighted.
    ax_right = fig.add_subplot(1, 2, 2)
    ax_right.imshow(space_time, aspect="auto", interpolation="nearest", cmap=cmap)
    ax_right.set_title("Unrolled CA (rows = time)")
    for span in spans:
        start = int(span.get("start", 0))
        length = int(span.get("length", 0))
        tau = int(span.get("time", 0))
        ax_right.add_patch(
            plt.Rectangle(
                (start - half, tau),
                length + 2 * half,
                1,
                fill=False,
                linewidth=1.0,
                edgecolor="white",
            )
        )
        ax_right.add_patch(
            plt.Rectangle(
                (start - half, tau + steps),
                length + 2 * half,
                1,
                fill=False,
                linewidth=1.0,
                edgecolor="white",
            )
        )
    ax_right.set_xlabel("space")
    ax_right.set_ylabel("time")
    plt.tight_layout()
    return fig


__all__ = [
    "runner_from_record",
    "space_time_from_record",
    "show_episode_card",
]
