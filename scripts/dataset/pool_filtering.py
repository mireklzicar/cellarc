#!/usr/bin/env python3
"""
Filter and downsample CA pool episodes to produce a balanced subset.

Steps:
  1. Deduplicate episodes by fingerprint (falling back to canonical JSON),
     require the solution to be novel relative to the query and train outputs,
     and enforce a maximum flattened length.
  2. Stratified downsample the filtered set to a target size by sampling
     approximately equally from a 2D grid defined by lambda and observed coverage
     fraction histogram bins.
  3. Emit the selected episodes and their metadata into a new directory so that
     downstream tooling (e.g., pool_stats.py) can operate on the result.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


DEFAULT_TARGET = 110_000
DEFAULT_MAX_FLATTENED = 256
DEFAULT_BINS = 100


@dataclass
class EpisodeRecord:
    fingerprint: str
    episode: Dict
    lambda_value: float
    observed_fraction: float
    coverage_fraction: float
    lambda_bin_index: int
    coverage_bin_index: int
    meta_record: Dict


def iter_episode_files(pool_dir: Path) -> Iterable[Path]:
    for path in sorted(pool_dir.glob("*.jsonl")):
        if path.name.endswith("_meta.jsonl"):
            continue
        yield path


def iter_meta_files(pool_dir: Path) -> Iterable[Path]:
    for path in sorted(pool_dir.glob("*_meta.jsonl")):
        yield path


def load_json(path: Path, line_no: int, raw: str) -> Dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON in {path}:{line_no}") from exc


def episode_key(episode: Dict) -> Tuple[str, str]:
    meta = episode.get("meta")
    if isinstance(meta, dict) and "fingerprint" in meta:
        return ("fingerprint", meta["fingerprint"])
    canonical = json.dumps(episode, sort_keys=True, separators=(",", ":"))
    return ("json", canonical)


def has_novel_solution(episode: Dict) -> bool:
    solution = episode.get("solution")
    if solution is None:
        return False

    query = episode.get("query")
    if query is not None and solution == query:
        return False

    train = episode.get("train") or []
    for example in train:
        if isinstance(example, dict) and solution == example.get("output"):
            return False

    return True


def has_identity_pair(episode: Dict) -> bool:
    """Return True if any train pair has identical input and output."""
    for example in episode.get("train") or []:
        if not isinstance(example, dict):
            continue
        if example.get("input") == example.get("output"):
            return True
    return False


def has_duplicate_pairs(episode: Dict) -> bool:
    """Return True if any train I/O pair appears more than once."""
    seen = set()
    for example in episode.get("train") or []:
        if not isinstance(example, dict):
            continue
        inp = example.get("input")
        out = example.get("output")
        inp_key = tuple(inp) if isinstance(inp, list) else tuple()
        out_key = tuple(out) if isinstance(out, list) else tuple()
        key = (inp_key, out_key)
        if key in seen:
            return True
        seen.add(key)
    return False


def episode_sequences(episode: Dict) -> Iterator[List]:
    train = episode.get("train") or []
    for example in train:
        if not isinstance(example, dict):
            continue
        for field in ("input", "output"):
            seq = example.get(field)
            if isinstance(seq, list):
                yield seq

    for field in ("query", "solution"):
        seq = episode.get(field)
        if isinstance(seq, list):
            yield seq


def flattened_length(obj: Sequence) -> int:
    total = 0
    stack = [obj]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        else:
            total += 1
    return total


def build_meta_index(pool_dir: Path) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for path in iter_meta_files(pool_dir):
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                raw = line.strip()
                if not raw:
                    continue
                record = load_json(path, line_no, raw)
                fingerprint = record.get("fingerprint")
                if not isinstance(fingerprint, str) or fingerprint in index:
                    continue
                meta = record.get("meta") or {}
                coverage = meta.get("coverage") or {}
                entry = {
                    "record": record,
                    "fraction": coverage.get("fraction"),
                    "observed_fraction": coverage.get("observed_fraction"),
                }
                index[fingerprint] = entry
    return index


def summarise(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "mean": float(mean(values)),
        "max": float(max(values)),
    }


def collect_filtered(
    pool_dir: Path,
    max_flattened: int,
    meta_index: Dict[str, Dict],
) -> Tuple[List[EpisodeRecord], Dict[str, int]]:
    seen_keys = set()
    filtered: List[EpisodeRecord] = []

    stats = {
        "total": 0,
        "unique": 0,
        "novel": 0,
        "passed_length": 0,
        "train_identity_filtered": 0,
        "train_duplicate_filtered": 0,
        "missing_lambda": 0,
        "missing_observed_fraction": 0,
        "missing_fingerprint": 0,
    }

    for path in iter_episode_files(pool_dir):
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                raw = line.strip()
                if not raw:
                    continue
                stats["total"] += 1
                episode = load_json(path, line_no, raw)

                key = episode_key(episode)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                stats["unique"] += 1

                if not has_novel_solution(episode):
                    continue
                stats["novel"] += 1

                if has_identity_pair(episode):
                    stats["train_identity_filtered"] += 1
                    continue

                if has_duplicate_pairs(episode):
                    stats["train_duplicate_filtered"] += 1
                    continue

                sequences = list(episode_sequences(episode))
                flattened_total = sum(flattened_length(seq) for seq in sequences)
                if flattened_total > max_flattened:
                    continue
                stats["passed_length"] += 1

                meta = episode.get("meta") or {}
                lambda_value = meta.get("lambda")
                if not isinstance(lambda_value, (int, float)):
                    stats["missing_lambda"] += 1
                    continue
                lambda_value = float(lambda_value)

                fingerprint = meta.get("fingerprint")
                if not isinstance(fingerprint, str):
                    stats["missing_fingerprint"] += 1
                    continue

                entry = meta_index.get(fingerprint)
                observed_fraction = None
                coverage_fraction = None

                if entry:
                    frac = entry.get("fraction")
                    if isinstance(frac, (int, float)):
                        coverage_fraction = float(frac)
                    obs = entry.get("observed_fraction")
                    if isinstance(obs, (int, float)):
                        observed_fraction = float(obs)

                # Fallback to episode meta if available.
                coverage_meta = meta.get("coverage") or {}
                if coverage_fraction is None:
                    frac = coverage_meta.get("fraction")
                    if isinstance(frac, (int, float)):
                        coverage_fraction = float(frac)
                if observed_fraction is None:
                    obs = coverage_meta.get("observed_fraction")
                    if isinstance(obs, (int, float)):
                        observed_fraction = float(obs)

                if observed_fraction is None:
                    stats["missing_observed_fraction"] += 1
                    continue

                meta_record = entry["record"] if entry and "record" in entry else {
                    "fingerprint": fingerprint,
                    "meta": meta,
                }

                filtered.append(
                    EpisodeRecord(
                        fingerprint=fingerprint or "",
                        episode=episode,
                        lambda_value=lambda_value,
                        observed_fraction=observed_fraction,
                        coverage_fraction=coverage_fraction if coverage_fraction is not None else observed_fraction,
                        lambda_bin_index=-1,
                        coverage_bin_index=-1,
                        meta_record=meta_record,
                    )
                )

    return filtered, stats


def assign_bins(
    episodes: List[EpisodeRecord],
    lambda_bins: int,
    coverage_bins: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if not episodes:
        return (0.0, 0.0), (0.0, 0.0)

    lambda_values = [ep.lambda_value for ep in episodes]
    coverage_values = [ep.observed_fraction for ep in episodes]

    lambda_min = min(lambda_values)
    lambda_max = max(lambda_values)
    coverage_min = min(coverage_values)
    coverage_max = max(coverage_values)

    def bin_index(value: float, lo: float, hi: float, bins: int) -> int:
        if bins <= 1 or lo == hi:
            return 0
        width = (hi - lo) / bins
        if width <= 0:
            return 0
        idx = int((value - lo) / width)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        return idx

    for ep in episodes:
        ep.lambda_bin_index = bin_index(ep.lambda_value, lambda_min, lambda_max, lambda_bins)
        ep.coverage_bin_index = bin_index(ep.observed_fraction, coverage_min, coverage_max, coverage_bins)

    return (lambda_min, lambda_max), (coverage_min, coverage_max)


def stratified_downsample(
    episodes: List[EpisodeRecord],
    target: int,
    rng: random.Random,
) -> List[EpisodeRecord]:
    if target <= 0 or len(episodes) <= target:
        return episodes

    buckets: Dict[Tuple[int, int], List[EpisodeRecord]] = {}
    for ep in episodes:
        buckets.setdefault((ep.lambda_bin_index, ep.coverage_bin_index), []).append(ep)

    nonempty_keys = [key for key, items in buckets.items() if items]
    total_available = sum(len(buckets[key]) for key in nonempty_keys)
    if target >= total_available:
        return [ep for key in nonempty_keys for ep in buckets[key]]

    rng.shuffle(nonempty_keys)
    for key in nonempty_keys:
        rng.shuffle(buckets[key])

    bucket_count = len(nonempty_keys)
    base_take = target // bucket_count
    selected: List[EpisodeRecord] = []
    leftovers: Dict[Tuple[int, int], List[EpisodeRecord]] = {}

    for key in nonempty_keys:
        items = buckets[key]
        take = min(base_take, len(items))
        selected.extend(items[:take])
        leftovers[key] = items[take:]

    remaining = target - len(selected)
    if remaining <= 0:
        return selected[:target]

    keys_cycle = [key for key in nonempty_keys if leftovers[key]]
    idx = 0
    while remaining > 0 and keys_cycle:
        key = keys_cycle[idx % len(keys_cycle)]
        pool = leftovers[key]
        if not pool:
            keys_cycle.pop(idx % len(keys_cycle))
            if not keys_cycle:
                break
            idx = idx % len(keys_cycle)
            continue

        selected.append(pool.pop())
        remaining -= 1

        if not pool:
            keys_cycle.pop(idx % len(keys_cycle))
            if keys_cycle:
                idx = idx % len(keys_cycle)
        else:
            idx = (idx + 1) % len(keys_cycle)

    if remaining > 0:
        # Fallback: gather any remaining episodes regardless of bucket ordering.
        rest = [ep for pool in leftovers.values() for ep in pool]
        if rest:
            rng.shuffle(rest)
            selected.extend(rest[:remaining])

    return selected[:target]


def write_output(
    outdir: Path,
    selected: List[EpisodeRecord],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    episodes_path = outdir / "downsampled.jsonl"
    meta_path = outdir / "downsampled_meta.jsonl"
    fingerprint_path = outdir / "downsampled_fingerprints.txt"

    with episodes_path.open("w", encoding="utf-8") as efh:
        for ep in selected:
            json.dump(ep.episode, efh)
            efh.write("\n")

    with meta_path.open("w", encoding="utf-8") as mfh:
        for ep in selected:
            json.dump(ep.meta_record, mfh)
            mfh.write("\n")

    with fingerprint_path.open("w", encoding="utf-8") as fph:
        for ep in selected:
            fph.write(f"{ep.fingerprint}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter and downsample a CA pool to diversify lambda/coverage."
    )
    parser.add_argument(
        "pool_dir",
        nargs="?",
        default="artifacts/pool",
        type=Path,
        help="Directory containing pool shard JSONL files (default: artifacts/pool)",
    )
    parser.add_argument(
        "--outdir",
        default=Path("artifacts/pool_downsampled"),
        type=Path,
        help="Directory to write the downsampled pool (default: artifacts/pool_downsampled)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"Target number of episodes after downsampling (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--max-flattened-length",
        type=int,
        default=DEFAULT_MAX_FLATTENED,
        help=f"Maximum flattened length for filtered episodes (default: {DEFAULT_MAX_FLATTENED})",
    )
    parser.add_argument(
        "--lambda-bins",
        type=int,
        default=DEFAULT_BINS,
        help=f"Number of bins for lambda stratification (default: {DEFAULT_BINS})",
    )
    parser.add_argument(
        "--coverage-bins",
        type=int,
        default=DEFAULT_BINS,
        help=f"Number of bins for observed coverage stratification (default: {DEFAULT_BINS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for downsampling (default: 12345)",
    )
    args = parser.parse_args()

    pool_dir: Path = args.pool_dir
    if not pool_dir.exists():
        raise SystemExit(f"Pool directory {pool_dir} does not exist")
    if not pool_dir.is_dir():
        raise SystemExit(f"Pool path {pool_dir} is not a directory")

    meta_index = build_meta_index(pool_dir)
    episodes, stats = collect_filtered(pool_dir, args.max_flattened_length, meta_index)

    print("=== Filtering Summary ===")
    print(f"Total serialized episodes: {stats['total']}")
    print(f"Unique episodes: {stats['unique']}")
    print(f"Novel episodes: {stats['novel']}")
    print(f"Filtered (train input == output): {stats['train_identity_filtered']}")
    print(f"Filtered (duplicate train pairs): {stats['train_duplicate_filtered']}")
    print(f"Novel episodes passing length <= {args.max_flattened_length}: {stats['passed_length']}")
    print(f"Skipped (missing lambda): {stats['missing_lambda']}")
    print(f"Skipped (missing observed fraction): {stats['missing_observed_fraction']}")
    print(f"Skipped (missing fingerprint): {stats['missing_fingerprint']}")
    print(f"Filtered episodes retained: {len(episodes)}")

    if not episodes:
        raise SystemExit("No episodes left after filtering; aborting.")

    lambda_range, coverage_range = assign_bins(
        episodes, args.lambda_bins, args.coverage_bins
    )

    print("\nLambda range: {:.6f} – {:.6f}".format(*lambda_range))
    print(
        "Observed coverage fraction range: {:.6f} – {:.6f}".format(*coverage_range)
    )

    rng = random.Random(args.seed)
    selected = stratified_downsample(episodes, args.target, rng)
    print(f"\nSelected episodes: {len(selected)} (target {args.target})")

    lambda_summary = summarise([ep.lambda_value for ep in selected])
    observed_summary = summarise([ep.observed_fraction for ep in selected])
    print("\nLambda summary (selected):", lambda_summary)
    print("Observed coverage summary (selected):", observed_summary)

    write_output(args.outdir, selected)
    print(f"\nWrote downsampled pool to {args.outdir}")


if __name__ == "__main__":
    main()
