#!/usr/bin/env python3
"""Animate cellular automata rollouts using rule tables from Cell ARC metadata."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise SystemExit(
        "Pillow is required for GIF export. Install it with `pip install pillow`."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))

from cell_arc import EpisodeDataset, random_palette_mapping
from eval.common import EpisodeRecord
from cellarc.generation.rules import (
    rule_table_cyclic_excitable,
    rule_table_linear_mod_k,
    rule_table_outer_inner_totalistic,
    rule_table_outer_totalistic,
    rule_table_permuted_totalistic,
    rule_table_random_lambda,
    rule_table_threshold,
    rule_table_totalistic,
)
from cellarc.generation.serialization import serialize_rule_table
from cellarc.generation.fingerprints import rule_fingerprint, induced_tstep_fingerprint
from cellarc.visualization.palette import BG_COLOR, CMAP_HEX
from cellarc.utils import choose_r_t_for_W


_DEFAULT_LAMBDA_RANGE = (0.2, 0.7)


_POOL_V1_CONFIG = {
    "k_range": (2, 6),
    "max_radius": 3,
    "max_steps": 5,
    "query_within_coverage": False,
    "unroll_tau_max": 32,
}


def _infer_dataset_config(meta: Dict[str, object]) -> Optional[Dict[str, object]]:
    dataset_version = meta.get("dataset_version")
    if dataset_version == "pool_v1":
        return _POOL_V1_CONFIG
    return None


def _reconstruct_rule_table_payload(meta: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    episode_seed = meta.get("episode_seed")
    if episode_seed is None:
        raise ValueError("Episode metadata missing 'episode_seed'; cannot reconstruct rule table.")

    k_lo, k_hi = config["k_range"]
    max_radius = config["max_radius"]
    max_steps = config["max_steps"]
    lambda_for_random = config.get("lambda_for_random", _DEFAULT_LAMBDA_RANGE)

    alphabet_size = int(meta.get("alphabet_size"))
    radius = int(meta.get("radius"))
    steps = int(meta.get("steps"))
    window = int(meta.get("window"))
    family = meta.get("family")
    probe_fp_expected = meta.get("probe_fingerprint")

    episode_rng = random.Random(int(episode_seed))

    alphabet_choices = list(range(int(k_lo), int(k_hi) + 1))
    weights = [1.0 / (idx + 1) for idx in range(len(alphabet_choices))]
    k_draw = episode_rng.choices(alphabet_choices, weights=weights, k=1)[0]
    if int(k_draw) != alphabet_size:
        raise ValueError(
            f"alphabet_size mismatch during reconstruction (expected {alphabet_size}, drew {k_draw}). "
            "Update reconstruction configuration."
        )

    r_draw, t_draw = choose_r_t_for_W(
        window,
        max_radius=max_radius,
        max_steps=max_steps,
        strategy="uniform",
        rng=episode_rng,
    )
    if int(r_draw) != radius or int(t_draw) != steps:
        raise ValueError(
            "radius/steps mismatch during reconstruction; adjust reconstruction caps."
        )

    episode_rng.random()  # consume the draw used for family selection

    def _unwrap(result):
        if isinstance(result, tuple) and len(result) == 4:
            table, lam_actual, qstate, extra = result
        else:
            table, lam_actual, qstate = result
            extra = {}
        return table, lam_actual, qstate, extra

    if family == "random":
        lam_target = episode_rng.uniform(*lambda_for_random)
        table, lam_actual, qstate = rule_table_random_lambda(
            alphabet_size, radius, episode_rng, lambda_val=lam_target, quiescent_state=0
        )
    elif family == "totalistic":
        table, lam_actual, qstate = rule_table_totalistic(alphabet_size, radius, episode_rng)
    elif family == "outer_totalistic":
        lam_target = episode_rng.uniform(*lambda_for_random)
        table, lam_actual, qstate, _ = _unwrap(
            rule_table_outer_totalistic(
                alphabet_size, radius, episode_rng, lambda_val=lam_target, qstate=0
            )
        )
    elif family == "outer_inner_totalistic":
        lam_target = episode_rng.uniform(*lambda_for_random)
        table, lam_actual, qstate, _ = _unwrap(
            rule_table_outer_inner_totalistic(
                alphabet_size, radius, episode_rng, lambda_val=lam_target, qstate=0
            )
        )
    elif family == "threshold":
        table, lam_actual, qstate, _ = _unwrap(
            rule_table_threshold(alphabet_size, radius, episode_rng, qstate=0)
        )
    elif family == "linear_mod_k":
        sparsity_choice = episode_rng.choice([1, 2, 3])
        table, lam_actual, qstate, _ = _unwrap(
            rule_table_linear_mod_k(
                alphabet_size, radius, episode_rng, sparsity=sparsity_choice, bias_prob=0.3
            )
        )
    elif family == "cyclic_excitable":
        trig = 1 if alphabet_size >= 3 else 1
        min_trig = 1 if radius == 1 else episode_rng.choice([1, 2])
        table, lam_actual, qstate, _ = _unwrap(
            rule_table_cyclic_excitable(
                alphabet_size, radius, episode_rng, trigger_state=trig, min_triggers=min_trig
            )
        )
    elif family == "permuted_totalistic":
        table, lam_actual, qstate, _ = _unwrap(
            rule_table_permuted_totalistic(alphabet_size, radius, episode_rng)
        )
    else:
        raise ValueError(f"Unknown family '{family}' encountered; cannot reconstruct rule table.")

    fp_actual = rule_fingerprint(table, alphabet_size, radius)
    if probe_fp_expected and fp_actual != probe_fp_expected:
        raise ValueError("Reconstructed rule table fingerprint does not match metadata.")

    tstep_fp_expected = meta.get("fingerprint")
    if tstep_fp_expected:
        tstep_fp_actual = induced_tstep_fingerprint(table, alphabet_size, radius, steps)
        if tstep_fp_actual != tstep_fp_expected:
            raise ValueError("Reconstructed CA t-step fingerprint does not match metadata.")

    payload = serialize_rule_table(
        table,
        alphabet_size=alphabet_size,
        radius=radius,
        quiescent_state=qstate,
    )
    return payload


def _ensure_rule_table(entry: EpisodeRecord) -> Dict[str, object]:
    record = entry.record
    rule_table = record.get("rule_table")
    if isinstance(rule_table, dict):
        return rule_table

    meta = record.get("meta", {}) or {}
    config = _infer_dataset_config(meta)
    if config is None:
        raise ValueError(
            "Selected episode does not contain a rule_table payload and no reconstruction rules are known."
        )

    payload = _reconstruct_rule_table_payload(meta, config)
    record["rule_table"] = payload
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Dataset JSONL or manifest inputs.",
    )
    parser.add_argument(
        "--fingerprint",
        type=str,
        help="Fingerprint to visualise (optional).",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Episode index to visualise (0-based).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of episodes to render when sampling sequentially (default: 3).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of rollout steps to simulate (default: 10).",
    )
    parser.add_argument(
        "--initial",
        choices=("query", "train"),
        default="query",
        help="Choose the starting sequence: use the query (default) or first train input.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Playback speed for the GIF animation (frames per second, default: 6).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=12,
        help="Pixel scaling factor applied to GIF frames (default: 12).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used when shuffling episodes without explicit index (default: 12345).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle episodes before sampling (loads all selected inputs into memory).",
    )
    parser.add_argument(
        "--square",
        action="store_true",
        help="Override steps so the rollout height matches the initial width (square PNGs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/test"),
        help="Directory to write GIF and PNG files (default: plots/test).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply the random palette mapping used by the data loader before rendering.",
    )
    return parser.parse_args(argv)


def iter_records_from_input(path: Path) -> Iterable[EpisodeRecord]:
    """Yield EpisodeRecord objects from a JSONL shard or manifest."""

    if not path.exists():
        raise FileNotFoundError(f"Input source does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        dataset = EpisodeDataset(paths=[path])
        manifest_path = None
    elif path.suffix.lower() == ".json" and path.name.endswith("_manifest.json"):
        dataset = EpisodeDataset(manifest=path)
        manifest_path = path
    else:
        raise ValueError(
            f"Unsupported input type for {path}. Expected *.jsonl or *_manifest.json"
        )

    for record in dataset:
        yield EpisodeRecord(record=record, source=path, manifest=manifest_path)


def iter_dataset_records(inputs: Sequence[str]) -> Iterable[EpisodeRecord]:
    """Iterate over all records from the provided inputs in order."""

    for raw in inputs:
        path = Path(raw)
        yield from iter_records_from_input(path)


def iter_sampled_records(args: argparse.Namespace) -> Iterable[EpisodeRecord]:
    """Yield the selected dataset entries based on CLI arguments."""

    if args.fingerprint or args.index is not None:
        fingerprint = args.fingerprint
        target_index = args.index

        for idx, entry in enumerate(iter_dataset_records(args.inputs)):
            if fingerprint:
                meta = entry.record.get("meta", {}) or {}
                candidates = [
                    entry.record.get("fingerprint"),
                    meta.get("fingerprint"),
                    meta.get("probe_fingerprint"),
                ]
                if any(str(fp) == fingerprint for fp in candidates if fp):
                    yield entry
                    return
            if target_index is not None and idx == target_index:
                yield entry
                return

        raise ValueError("Episode not found with the provided filters.")

    records_iter = list(iter_dataset_records(args.inputs)) if args.shuffle else iter_dataset_records(args.inputs)
    if args.shuffle:
        if not records_iter:
            raise ValueError("No records were found in the provided inputs.")
        rng = random.Random(args.seed)
        rng.shuffle(records_iter)

    count = max(1, args.count)
    produced = 0

    for entry in records_iter:
        yield entry
        produced += 1
        if produced >= count:
            break

def pick_initial_state(record: dict, preference: str) -> Tuple[List[int], str]:
    """Select an initial sequence from the record."""

    if preference == "query":
        query = record.get("query")
        if isinstance(query, list) and query:
            return [int(v) for v in query], "query"

    train = record.get("train") or []
    for idx, pair in enumerate(train):
        if isinstance(pair, dict) and isinstance(pair.get("input"), list) and pair["input"]:
            return [int(v) for v in pair["input"]], f"train{idx}"

    query = record.get("query")
    if isinstance(query, list) and query:
        return [int(v) for v in query], "query"

    raise ValueError("Unable to determine an initial state (missing query/train inputs).")


def evolve_rule_table(
    initial_state: Sequence[int],
    rule_table: dict,
    *,
    steps: int,
    wrap: bool,
) -> np.ndarray:
    """Simulate the cellular automaton defined by ``rule_table``."""

    values = rule_table.get("values")
    alphabet_size = int(rule_table.get("alphabet_size"))
    radius = int(rule_table.get("radius"))

    if values is None:
        raise ValueError("Rule table is missing the 'values' field.")

    arity = 2 * radius + 1
    expected_length = alphabet_size ** arity
    if len(values) != expected_length:
        raise ValueError(
            f"Rule table length ({len(values)}) does not match alphabet_size**arity ({expected_length})."
        )

    state = np.asarray(initial_state, dtype=np.int32)
    width = state.size
    history = np.empty((steps + 1, width), dtype=np.int32)
    history[0] = state

    table = np.asarray(values, dtype=np.int32)

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
            next_state[pos] = table[code]
        history[step + 1] = next_state
        state = next_state

    return history


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Convert #RRGGBB colour strings to 8-bit RGB tuples."""

    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


def apply_palette_mapping(history: np.ndarray, mapping: Optional[Dict[int, int]]) -> np.ndarray:
    """Apply a palette remapping to the rollout for visualisation."""

    if not mapping:
        return history

    max_history = int(history.max()) if history.size else 0
    max_key = max(mapping.keys(), default=0)
    max_value = max(mapping.values(), default=0)
    size = max(max_history, max_key, max_value) + 1

    lut = np.arange(size, dtype=np.int32)
    for key, value in mapping.items():
        lut[key] = value

    return lut[history]


def save_rollout_png(history: np.ndarray, path: Path, *, scale: int) -> None:
    """Render the full rollout as a static PNG with no extra annotations."""

    if scale < 1:
        raise ValueError("scale must be >= 1.")

    rgb_palette = np.array([hex_to_rgb(code) for code in CMAP_HEX], dtype=np.uint8)
    image = rgb_palette[history]
    if scale > 1:
        image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def save_rollout_gif(
    history: np.ndarray,
    path: Path,
    *,
    fps: float,
    scale: int,
) -> None:
    """Animate the rollout, revealing one timestep at a time."""

    if fps <= 0:
        raise ValueError("fps must be positive.")
    if scale < 1:
        raise ValueError("scale must be >= 1.")

    step_count, width = history.shape
    rgb_palette = np.array([hex_to_rgb(code) for code in CMAP_HEX], dtype=np.uint8)
    bg_rgb = np.array(hex_to_rgb(BG_COLOR), dtype=np.uint8)

    frames: List[Image.Image] = []
    for frame_idx in range(step_count):
        frame = np.tile(bg_rgb, (step_count, width, 1))
        visible = history[: frame_idx + 1]
        colourised = rgb_palette[visible]
        frame[: frame_idx + 1] = colourised
        upscaled = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
        frames.append(Image.fromarray(upscaled, mode="RGB"))

    duration = int(max(20, 1000.0 / fps))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration,
    )


def render_episode(entry: EpisodeRecord, args: argparse.Namespace, ordinal: int) -> None:
    record = entry.record
    rule_table = _ensure_rule_table(entry)

    meta = record.get("meta", {}) or {}
    palette_mapping: Optional[Dict[int, int]] = None
    if args.augment:
        alphabet_size = int(
            meta.get("alphabet_size")
            or rule_table.get("alphabet_size")
            or len(CMAP_HEX)
        )
        palette = tuple(range(len(CMAP_HEX)))
        rng = random.Random()
        seed_value: Optional[int]
        if args.seed is not None:
            seed_value = args.seed + ordinal
        else:
            seed_value = None
        rng.seed(seed_value)
        palette_mapping = random_palette_mapping(alphabet_size, palette=palette, rng=rng)

    wrap = bool(meta.get("wrap", True))
    fingerprint = (
        record.get("fingerprint")
        or meta.get("fingerprint")
        or f"idx{entry.source.name}_{ordinal:03d}"
    )

    initial_state, origin = pick_initial_state(record, args.initial)

    steps = args.steps
    if args.square:
        width = len(initial_state)
        steps = max(width - 1, 0)

    history = evolve_rule_table(
        initial_state,
        rule_table,
        steps=steps,
        wrap=wrap,
    )

    display_history = apply_palette_mapping(history, palette_mapping)

    if history.min() < 0:
        raise ValueError("Encountered negative cell state values in rollout.")
    max_value = int(display_history.max())
    if max_value >= len(CMAP_HEX):
        raise ValueError(
            f"Cell state {max_value} exceeds the available palette size ({len(CMAP_HEX)} colours)."
        )

    base_name = f"{ordinal:03d}_{fingerprint[:12]}"
    gif_path = args.output_dir / f"{base_name}.gif"
    png_path = args.output_dir / f"{base_name}.png"

    save_rollout_png(display_history, png_path, scale=args.scale)
    save_rollout_gif(display_history, gif_path, fps=args.fps, scale=args.scale)

    print(f"Wrote {gif_path} and {png_path} (start={origin}, steps={steps})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for ordinal, entry in enumerate(iter_sampled_records(args), start=1):
        render_episode(entry, args, ordinal)
        rendered += 1

    if rendered == 0:
        raise ValueError("No episodes matched the supplied filters.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
