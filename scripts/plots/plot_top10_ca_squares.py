#!/usr/bin/env python3
"""
Render CA square images for the top-10 easiest and hardest episodes listed in a CSV.

Inputs:
  - CSV with columns: difficulty,episode_id,split,... (see figures/episode_difficulty_top10.csv)
  - Dataset shards (JSONL) and companion metadata (JSONL) under artifacts/hf_cellarc/

Outputs:
  - Two directories with per-episode PNGs named by episode id:
      figures/episode_difficulty_ca_squares/easiest/
      figures/episode_difficulty_ca_squares/hardest/
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np

# Non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matplotlib.colors import ListedColormap  # noqa: E402
from cellarc.utils import de_bruijn_cycle  # noqa: E402
from cellarc.generation.reconstruction import (
    infer_dataset_config,
    reconstruct_rule_table_payload,
)
from cellarc.generation.serialization import deserialize_rule_table  # noqa: E402
from scripts.plots._episode_utils import load_jsonl, load_meta_lookup, merge_metadata  # noqa: E402


def _square_history(history: np.ndarray) -> np.ndarray:
    """Crop wide histories to square; do not tile narrow ones.

    - If width > height: center-crop to a square to remove extra edges.
    - If width <= height: keep original width (no tiling beyond actual CA).
    """
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
    # width <= height: do not tile, keep the true CA width
    return history


def _palette() -> tuple[str, ListedColormap]:
    """Local copy of the project palette to avoid importing visualization package."""
    cmap_hex = [
        "#252525",  # black
        "#0074D9",  # blue
        "#FF4136",  # red
        "#37D449",  # green
        "#FFDC00",  # yellow
        "#E6E6E6",  # grey
        "#F012BE",  # pink
        "#FF871E",  # orange
        "#54D2EB",  # light blue
        "#8D1D2C",  # brown
        "#FFFFFF",
    ]
    bg_color = "#EEEFF6"
    palette = ListedColormap(cmap_hex)
    palette.set_bad(color=bg_color)
    return bg_color, palette


def _read_top10_csv(path: Path) -> List[Tuple[str, str, str, float, float, float]]:
    """Backward-compatible: parse difficulty CSV if used.

    Returns rows as (difficulty, episode_id, split, de_bruijn, gpt5, transformer).
    """
    rows: List[Tuple[str, str, str, float, float, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            diff = str(rec.get("difficulty") or "").strip().lower()
            eid = str(rec.get("episode_id") or "").strip()
            split = str(rec.get("split") or "").strip()
            if not diff or not eid or not split:
                continue
            def _f(key: str) -> float:
                raw = rec.get(key)
                try:
                    return float(raw)
                except Exception:
                    return 0.0
            de_bruijn = _f("de_bruijn")
            gpt5 = _f("gpt-5-2025-08-07-high")
            transformer = _f("transformer_large_embedding")
            rows.append((diff, eid, split, de_bruijn, gpt5, transformer))
    return rows


def _read_accuracy_csv(path: Path) -> List[Tuple[str, str, float, float, float]]:
    """Parse the full per-task accuracy CSV.

    Returns rows as (episode_id, split, de_bruijn, gpt5, transformer).
    """
    rows: List[Tuple[str, str, float, float, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            eid = str(rec.get("episode_id") or "").strip()
            split = str(rec.get("split") or "").strip()
            if not eid or not split:
                continue
            def _f(key: str) -> float:
                raw = rec.get(key)
                try:
                    return float(raw)
                except Exception:
                    return 0.0
            de_bruijn = _f("de_bruijn")
            gpt5 = _f("gpt-5-2025-08-07-high")
            transformer = _f("transformer_large_embedding")
            rows.append((eid, split, de_bruijn, gpt5, transformer))
    return rows


def _load_split_records(
    base_dir: Path,
    meta_base_dir: Path,
    split: str,
) -> Dict[str, dict]:
    """Load records for a split and return a mapping id -> record (with metadata)."""
    data_path = base_dir / "data" / f"{split}.jsonl"
    meta_path = meta_base_dir / "data" / f"{split}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset split: {data_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata split: {meta_path}")

    records = load_jsonl(data_path)
    meta_lookup = load_meta_lookup(meta_path)
    merge_metadata(records, meta_lookup)
    out: Dict[str, dict] = {}
    for r in records:
        rid = str(r.get("id"))
        if rid:
            out[rid] = r
    return out


def _evolve_history_from_meta(record: dict) -> np.ndarray:
    """Reconstruct the rule table from metadata and evolve a rollout."""
    meta = record.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError("Episode is missing metadata; cannot reconstruct rule table.")

    config = infer_dataset_config(meta)
    payload = reconstruct_rule_table_payload(meta, config=config)
    table = deserialize_rule_table(payload)

    alphabet_size = int(meta["alphabet_size"])  # type: ignore[index]
    window = int(meta["window"])  # type: ignore[index]
    steps = int(meta["steps"])  # type: ignore[index]
    wrap = bool(meta.get("wrap", True))

    initial_state = de_bruijn_cycle(alphabet_size, window)
    depth = max(4, min(24, steps + 8))

    # Evolve for depth + steps + 1 steps (to match other viz defaults)
    history = _evolve_rule_table(initial_state, payload, steps=depth + steps + 1, wrap=wrap)
    return history


def _evolve_rule_table(
    initial_state: List[int],
    rule_table_payload: dict,
    *,
    steps: int,
    wrap: bool,
) -> np.ndarray:
    """Simulate the automaton defined by a serialized rule-table payload."""
    dense = deserialize_rule_table(rule_table_payload)
    values_array = np.asarray(dense.values_view(), dtype=np.int32)
    alphabet_size = int(rule_table_payload.get("alphabet_size"))
    radius = int(rule_table_payload.get("radius"))

    arity = 2 * radius + 1
    expected_length = alphabet_size ** arity
    if len(values_array) != expected_length:
        raise ValueError(
            f"Rule table length ({len(values_array)}) does not match alphabet_size**arity ({expected_length})."
        )

    state = np.asarray(initial_state, dtype=np.int32)
    width = state.size
    history = np.empty((steps + 1, width), dtype=np.int32)
    history[0] = state

    for step in range(steps):
        next_state = np.empty_like(state)
        for pos in range(width):
            code = 0
            for offset in range(-radius, radius + 1):
                idx = pos + offset
                if wrap:
                    idx %= width
                    digit = state[idx]
                else:
                    if 0 <= idx < width:
                        digit = state[idx]
                    else:
                        digit = 0
                code = code * alphabet_size + int(digit)
            next_state[pos] = values_array[code]
        history[step + 1] = next_state
        state = next_state
    return history


def _render_square(record: dict, output_path: Path, *, dpi: int = 240, size: float = 5.0) -> None:
    history = _evolve_history_from_meta(record)
    history = _square_history(history)
    bg_color, palette = _palette()
    # Slightly taller figure to fit subtitle
    fig, ax = plt.subplots(figsize=(size, size + 0.5), squeeze=True, dpi=dpi, facecolor=bg_color)
    ax.imshow(history, aspect="equal", interpolation="nearest", cmap=palette)
    ax.axis("off")
    # Leave space at the bottom for subtitle text
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _format_percent(x: float) -> str:
    try:
        return f"{int(round(100.0 * float(x)))}%"
    except Exception:
        return "0%"


def _render_square_with_subtitle(
    record: dict,
    output_path: Path,
    *,
    de_bruijn: float,
    gpt5: float,
    transformer: float,
    dpi: int = 240,
    size: float = 5.0,
) -> None:
    history = _evolve_history_from_meta(record)
    history = _square_history(history)
    bg_color, palette = _palette()

    # Compute figure width to match data aspect (no side margins/letterboxing)
    h, w = history.shape
    data_ratio = w / max(h, 1)
    bottom_frac = 0.28  # allocate more space for three lines of text
    fig_height = size + 0.9
    axes_height = fig_height * (1.0 - bottom_frac)
    fig_width = max(0.5, data_ratio * axes_height)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), squeeze=True, dpi=dpi, facecolor=bg_color)
    ax.imshow(history, aspect="equal", interpolation="nearest", cmap=palette)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=bottom_frac)

    # Multi-line subtitle: first GPT-5, then Transformer, then De Bruijn
    subtitle = (
        f"GPT-5: {_format_percent(gpt5)}\n"
        f"Transformer: {_format_percent(transformer)}\n"
        f"De Bruijn: {_format_percent(de_bruijn)}"
    )
    fig.text(0.5, bottom_frac / 2.0, subtitle, ha="center", va="center", fontsize=20, color="#222222")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    # Inputs
    csv_path = Path("/home/mirekl/Arc_Repositories/cellarc_baselines/outputs/analysis/de_bruijn_vs_gpt_per_task_accuracy.csv")
    dataset_base = REPO_ROOT / "artifacts/hf_cellarc/hf-cellarc_100k"
    meta_base = REPO_ROOT / "artifacts/hf_cellarc/hf-cellarc_100k_meta"

    # Outputs: just two dirs for the two splits
    out_base = REPO_ROOT / "figures/episode_split_ca_squares"
    out_base.mkdir(parents=True, exist_ok=True)

    rows = _read_accuracy_csv(csv_path)
    if not rows:
        raise SystemExit(f"No rows found in {csv_path}")

    # Group rows by split for efficient loading
    by_split: Dict[str, List[Tuple[str, str, float, float, float]]] = {}
    for eid, split, de_bruijn, gpt5, transformer in rows:
        by_split.setdefault(split, []).append((eid, split, de_bruijn, gpt5, transformer))

    # Load each split once and render the requested episodes
    for split, entries in by_split.items():
        id_to_record = _load_split_records(dataset_base, meta_base, split)
        split_out = out_base / split
        split_out.mkdir(parents=True, exist_ok=True)
        for eid, _, de_bruijn, gpt5, transformer in entries:
            rec = id_to_record.get(eid)
            if rec is None:
                print(f"[skip] Episode not found in {split}: {eid}", flush=True)
                continue
            # Compute mean accuracy across three models
            mean_acc = (float(de_bruijn) + float(gpt5) + float(transformer)) / 3.0
            prefix = str(int(round(mean_acc * 1000.0))).zfill(3)
            out_path = split_out / f"{prefix}_{eid}.png"
            _render_square_with_subtitle(
                rec,
                out_path,
                de_bruijn=de_bruijn,
                gpt5=gpt5,
                transformer=transformer,
            )
            print(f"[ok] Wrote {out_path}")


if __name__ == "__main__":
    main()
