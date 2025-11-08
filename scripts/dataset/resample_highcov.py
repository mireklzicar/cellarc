#!/usr/bin/env python3
"""
Resample enriched CA episodes so that query-window coverage improves.

The script rebuilds the training examples for each episode by sampling from the
reconstructed evolution until the query-window coverage reaches the desired
threshold (within a fixed attempt budget). Episodes that fail to reach the
target are skipped, preventing the query sequence from leaking into the training
set while still favouring high-coverage supervision.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is absent
    tqdm = None  # type: ignore

from cellarc.generation.cax_runner import AutomatonRunner
from cellarc.generation.helpers import ring_slice
from cellarc.generation.serialization import deserialize_rule_table
from cellarc.utils import de_bruijn_cycle


class EpisodeContext:
    """Container for reconstructed CA evolution tracks."""

    __slots__ = ("history", "steps", "half_window")

    def __init__(self, *, history: Sequence[Sequence[int]], steps: int, half_window: int) -> None:
        self.history = history
        self.steps = steps
        self.half_window = half_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/deprecated/pool_downsampled/downsampled_enriched.jsonl"),
        help="Enriched JSONL shard to resample (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/processing/resampled_highcov/downsampled_enriched.jsonl"),
        help="Destination for the resampled JSONL file (default: %(default)s).",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=Path("artifacts/processing/resampled_highcov/downsampled_enriched_meta.jsonl"),
        help="Companion meta JSONL file mirroring the input conventions (default: %(default)s).",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.95,
        help="Desired minimum query-window coverage (weighted) (default: %(default)s).",
    )
    parser.add_argument(
        "--unroll-tau-max",
        type=int,
        default=16,
        help="Maximum unrolled history depth to reconstruct for hybrid/unrolled episodes (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=200,
        help="Maximum candidate segments to draw while attempting to reach the coverage target (default: %(default)s).",
    )
    return parser.parse_args()


def iter_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - guardrail
                raise ValueError(f"Failed to parse {path} line {line_no}: {exc}") from exc


def count_records(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def build_episode_context(record: dict, *, tau_cap: int) -> EpisodeContext:
    meta = record["meta"]
    alphabet_size = int(meta["alphabet_size"])
    radius = int(meta["radius"])
    steps = int(meta["steps"])
    window = int(meta["window"])
    half_window = int(meta.get("train_context", (window - 1) // 2))

    rule_payload = record.get("rule_table")
    if rule_payload is None:
        raise ValueError("Episode is missing an embedded rule_table payload.")

    rule_table = deserialize_rule_table(rule_payload)
    runner = AutomatonRunner(
        alphabet_size=alphabet_size,
        radius=radius,
        table=rule_table,
        rng_seed=int(meta.get("episode_seed", 0)),
    )

    cycle = de_bruijn_cycle(alphabet_size, window)
    construction = meta.get("construction", "cycle")
    if construction == "cycle":
        history = runner.evolve(cycle, timesteps=steps + 1, return_history=True)
    else:
        tau_cap = max(0, tau_cap)
        total_steps = tau_cap + steps + 1
        history = runner.evolve(cycle, timesteps=total_steps, return_history=True)

    history_lists = [[int(v) for v in row] for row in history]
    return EpisodeContext(history=history_lists, steps=steps, half_window=half_window)


def _observed_windows(train_inputs: Sequence[Sequence[int]], window: int) -> int:
    if window <= 0:
        return 0
    half = window // 2
    seen: set[tuple[int, ...]] = set()
    for seq in train_inputs:
        if len(seq) < window:
            continue
        for idx in range(half, len(seq) - half):
            win = tuple(int(seq[idx - half + j]) for j in range(window))
            seen.add(win)
    return len(seen)


def _centered_window(seq: Sequence[int], idx: int, width: int, wrap: bool) -> tuple[int, ...]:
    half = width // 2
    n = len(seq)
    if n == 0 or width <= 0:
        return ()
    if wrap:
        return tuple(int(seq[(idx - half + j) % n]) for j in range(width))
    window: List[int] = []
    for j in range(idx - half, idx + half + 1):
        if 0 <= j < n:
            window.append(int(seq[j]))
        else:
            window.append(0)
    return tuple(window)


def _collect_train_windows(train_pairs: Sequence[dict], width: int) -> Dict[tuple[int, ...], int]:
    counts: Dict[tuple[int, ...], int] = {}
    if width <= 0 or width % 2 == 0:
        return counts
    for pair in train_pairs:
        seq = pair.get("input", [])
        n = len(seq)
        if n < width:
            continue
        half = width // 2
        for idx in range(half, n - half):
            win = tuple(int(seq[idx - half + j]) for j in range(width))
            counts[win] = counts.get(win, 0) + 1
    return counts


def _collect_query_windows(query: Sequence[int], width: int, wrap: bool) -> Dict[tuple[int, ...], int]:
    counts: Dict[tuple[int, ...], int] = {}
    if width <= 0 or width % 2 == 0:
        return counts
    n = len(query)
    if n == 0:
        return counts
    for idx in range(n):
        win = _centered_window(query, idx, width, wrap=wrap)
        if not win:
            continue
        counts[win] = counts.get(win, 0) + 1
    return counts


def resample_episode(
    record: dict,
    *,
    coverage_target: float,
    unroll_tau_max: int,
    max_attempts: int,
) -> Optional[dict]:
    coverage_target = max(0.0, min(1.0, coverage_target))
    max_attempts = max(1, int(max_attempts))

    meta = dict(record["meta"])
    coverage = dict(meta.get("coverage", {}))

    length = int(coverage.get("cycle_length", meta.get("windows_total")))
    if length <= 0:
        raise ValueError("Episode reports non-positive cycle length; cannot resample.")

    orig_train = record.get("train") or []
    train_examples = len(orig_train)
    if train_examples <= 0:
        raise ValueError("Episode does not contain any training examples to resample.")

    window = int(meta["window"])
    half_window = int(meta.get("train_context", (window - 1) // 2))
    wrap = bool(meta.get("wrap", True))

    query_seq = [int(v) for v in record.get("query") or []]
    n_query = len(query_seq)

    query_core_len = n_query - 2 * half_window
    if query_core_len <= 0:
        query_core_len = window
    query_core_len = max(1, query_core_len)

    query_span_meta = meta.get("query_span") or {}
    query_time = int(meta.get("query_time", 0))
    query_start = int(query_span_meta.get("start", 0))

    query_counts = _collect_query_windows(query_seq, window, wrap=wrap)
    total_query_windows = sum(query_counts.values())

    orig_spans = meta.get("train_spans") or []
    orig_core_lengths_meta = meta.get("train_core_lengths") or []

    core_length_pool: List[int] = []
    for span in orig_spans:
        if isinstance(span, dict):
            span_len = int(span.get("length", 0))
            if span_len > 0:
                core_length_pool.append(span_len)
    if not core_length_pool:
        for value in orig_core_lengths_meta:
            value_int = int(value)
            if value_int > 0:
                core_length_pool.append(value_int)
    if not core_length_pool:
        core_length_pool = [query_core_len]

    tau_candidates = [int(span.get("time", 0)) for span in orig_spans if isinstance(span, dict)]
    tau_candidates = [tau for tau in tau_candidates if tau != query_time] or tau_candidates or [query_time]

    tau_cap = min(unroll_tau_max, 256 - int(meta["steps"]))
    tau_cap = max(tau_cap, max([query_time] + tau_candidates, default=0))
    ctx = build_episode_context(record, tau_cap=tau_cap)

    def to_int_list(seq: Sequence[int]) -> List[int]:
        return [int(v) for v in seq]

    def slice_pair(start: int, tau: int, span_len: int) -> tuple[List[int], List[int]]:
        if meta.get("construction") == "cycle":
            source = ctx.history[0]
            target = ctx.history[ctx.steps]
        else:
            tau_clamped = min(max(0, tau), len(ctx.history) - ctx.steps - 1)
            source = ctx.history[tau_clamped]
            target = ctx.history[tau_clamped + ctx.steps]
        span_width = span_len + 2 * ctx.half_window
        x = ring_slice(source, start - ctx.half_window, span_width)
        y = ring_slice(target, start - ctx.half_window, span_width)
        return to_int_list(x), to_int_list(y)

    def normalize_span(span: Optional[dict], *, default_length: int) -> Dict[str, int]:
        if isinstance(span, dict):
            return {
                "start": int(span.get("start", 0)),
                "length": int(span.get("length", default_length)),
                "time": int(span.get("time", 0)),
            }
        return {"start": query_start, "length": int(default_length), "time": query_time}

    fingerprint = meta.get("fingerprint")
    rng_seed = int(meta.get("episode_seed", 0))
    if isinstance(fingerprint, str):
        try:
            seed_extra = int(fingerprint[:16], 16)
        except ValueError:
            seed_extra = 0
        rng_seed ^= seed_extra
    rng = random.Random(rng_seed)

    seen_segments: set[Tuple[int, int, int]] = set()

    def choose_core_len() -> int:
        value = core_length_pool[rng.randrange(len(core_length_pool))]
        value = max(1, value)
        return min(length, value)

    def sample_segment(core_len: int, *, allow_duplicates: bool) -> Optional[Tuple[Dict[str, List[int]], Dict[str, int]]]:
        for _ in range(16):
            start = rng.randrange(max(1, length))
            tau = rng.choice(tau_candidates)
            key = (start % max(1, length), tau, core_len)
            if not allow_duplicates and key in seen_segments:
                continue
            x, y = slice_pair(start, tau, core_len)
            if not allow_duplicates:
                seen_segments.add(key)
            return (
                {"input": x, "output": y},
                {"start": int(start % max(1, length)), "length": int(core_len), "time": int(tau)},
            )
        return None

    new_train: List[Dict[str, List[int]]] = []
    new_spans: List[Dict[str, int]] = []
    core_lengths: List[int] = []

    coverage_needed = coverage_target > 0.0 and total_query_windows > 0 and train_examples > 0
    covered_weighted = 0.0
    covered_windows: set[Tuple[int, ...]] = set()

    attempts = 0
    while (
        coverage_needed
        and attempts < max_attempts
        and len(new_train) < train_examples
    ):
        attempts += 1
        core_len = choose_core_len()
        sampled = sample_segment(core_len, allow_duplicates=False)
        if sampled is None:
            continue
        pair, span = sampled
        if pair["input"] == query_seq:
            continue
        candidate_counts = _collect_train_windows([pair], window)
        newly_covering = [
            win for win in candidate_counts.keys() if win in query_counts and win not in covered_windows
        ]
        if not newly_covering:
            continue
        new_train.append(pair)
        new_spans.append(span)
        core_lengths.append(int(span["length"]))
        for win in newly_covering:
            covered_windows.add(win)
            covered_weighted += query_counts[win]
        current_cov = covered_weighted / total_query_windows if total_query_windows else 1.0
        if current_cov >= coverage_target:
            break

    weighted_cov = 1.0 if total_query_windows == 0 else covered_weighted / total_query_windows
    if coverage_needed and weighted_cov + 1e-9 < coverage_target:
        return None

    remaining_slots = max(0, train_examples - len(new_train))
    if remaining_slots > 0:
        for idx, pair in enumerate(orig_train):
            if remaining_slots == 0:
                break
            input_seq = [int(v) for v in (pair.get("input") or [])]
            if input_seq == query_seq:
                continue
            output_seq = [int(v) for v in (pair.get("output") or [])]
            span_meta = orig_spans[idx] if idx < len(orig_spans) else None
            default_length = (
                int(orig_core_lengths_meta[idx])
                if idx < len(orig_core_lengths_meta) and int(orig_core_lengths_meta[idx]) > 0
                else max(1, len(input_seq) - 2 * half_window)
            )
            normalized_span = normalize_span(span_meta, default_length=default_length)
            new_train.append({"input": input_seq, "output": output_seq})
            new_spans.append(normalized_span)
            core_lengths.append(int(normalized_span["length"]))
            remaining_slots -= 1

    if remaining_slots > 0:
        fallback_limit = max(remaining_slots * 20, 64)
        fallback_attempts = 0
        while remaining_slots > 0 and fallback_attempts < fallback_limit:
            fallback_attempts += 1
            core_len = choose_core_len()
            sampled = sample_segment(core_len, allow_duplicates=True)
            if sampled is None:
                continue
            pair, span = sampled
            if pair["input"] == query_seq:
                continue
            new_train.append(pair)
            new_spans.append(span)
            core_lengths.append(int(span["length"]))
            remaining_slots -= 1
        if remaining_slots > 0:
            return None

    if len(new_train) != train_examples:
        return None

    train_inputs = [pair["input"] for pair in new_train]
    observed_windows = _observed_windows(train_inputs, window)
    windows_total = max(1, meta.get("windows_total", 1))
    observed_fraction = observed_windows / windows_total

    train_counts = _collect_train_windows(new_train, window)
    if total_query_windows > 0:
        covered_weighted = sum(cnt for win, cnt in query_counts.items() if train_counts.get(win, 0) > 0)
        weighted_cov = covered_weighted / total_query_windows
        avg_depth = sum(train_counts.get(win, 0) for win in query_counts) / max(1, len(query_counts))
        unique_cov = len(set(query_counts) & set(train_counts)) / max(1, len(set(query_counts)))
    else:
        weighted_cov = 1.0
        avg_depth = math.nan
        unique_cov = math.nan

    if weighted_cov + 1e-9 < coverage_target and total_query_windows > 0:
        return None

    coverage.update(
        {
            "fraction": float(min(1.0, sum(core_lengths) / max(1, length))),
            "windows": int(min(length, sum(core_lengths))),
            "segments": int(len(new_train)),
            "mode": "resampled_sampling",
            "query_within_coverage": bool(weighted_cov >= coverage_target or total_query_windows == 0),
            "observed_windows": int(observed_windows),
            "observed_fraction": float(observed_fraction),
        }
    )

    meta["train_core_lengths"] = [int(val) for val in core_lengths]
    meta["train_spans"] = new_spans
    meta["coverage"] = coverage
    meta["query_window_coverage_weighted"] = float(weighted_cov)
    meta["query_window_coverage_unique"] = None if math.isnan(unique_cov) else float(unique_cov)
    meta["query_window_avg_depth"] = None if math.isnan(avg_depth) else float(avg_depth)

    record = {
        **record,
        "train": new_train,
        "meta": meta,
    }
    return record


def ensure_outdir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    if path.is_dir():
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input JSONL shard not found at {args.input}")

    ensure_outdir(args.output, overwrite=args.overwrite)
    ensure_outdir(args.output_meta, overwrite=args.overwrite)

    processed = 0
    kept = 0
    improved = 0
    skipped = 0
    total = count_records(args.input)
    progress = tqdm(total=total, desc="Resampling", unit="episode") if tqdm is not None else None

    with args.output.open("w", encoding="utf-8") as data_out, args.output_meta.open(
        "w", encoding="utf-8"
    ) as meta_out:
        for record in iter_records(args.input):
            processed += 1
            before = record["meta"].get("query_window_coverage_weighted", 0.0)
            updated = resample_episode(
                record,
                coverage_target=args.target_coverage,
                unroll_tau_max=args.unroll_tau_max,
                max_attempts=args.max_attempts,
            )
            if updated is None:
                skipped += 1
                if progress is not None:
                    progress.update(1)
                continue

            after = updated["meta"].get("query_window_coverage_weighted", 0.0)
            if after > before + 1e-9:
                improved += 1
            kept += 1

            data_out.write(json.dumps(updated, separators=(",", ":")))
            data_out.write("\n")
            meta_out.write(
                json.dumps(
                    {
                        "fingerprint": updated["meta"]["fingerprint"],
                        "meta": updated["meta"],
                    },
                    separators=(",", ":"),
                )
            )
            meta_out.write("\n")

            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    message = (
        f"Resampled {kept} of {processed} episodes "
        f"(query coverage improved on {improved}, skipped {skipped}, "
        f"target ≥ {args.target_coverage:.2f})."
    )
    print(message)


if __name__ == "__main__":  # pragma: no cover - CLI hook
    main()
