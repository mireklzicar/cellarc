#!/usr/bin/env python3
"""
Compute detailed statistics over a pool of CA episodes stored as JSON Lines.

For every non-meta *.jsonl file in the pool directory we:
  a) count total serialized episodes,
  b) deduplicate by meta.fingerprint (falling back to canonical JSON) to count uniques,
  c) filter to unique episodes whose solution differs from the query and every train
     output and whose flattened length is <= max_flattened, then report aggregate
     statistics over that filtered subset:
       • list-length statistics (min/median/mean/max),
       • flattened episode length statistics,
       • lambda_bin proportions,
       • lambda histogram (100 bins),
       • coverage fraction / observed fraction summaries and histograms.

Results are printed and also written to artifacts/pool_stats (or a custom outdir).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


DEFAULT_MAX_FLATTENED = 256


def iter_episode_files(pool_dir: Path) -> Iterable[Path]:
    """Yield episode JSONL files, skipping the *_meta.jsonl companions."""
    for path in sorted(pool_dir.glob("*.jsonl")):
        if path.name.endswith("_meta.jsonl"):
            continue
        yield path


def iter_meta_files(pool_dir: Path) -> Iterable[Path]:
    """Yield *_meta.jsonl files associated with the pool."""
    for path in sorted(pool_dir.glob("*_meta.jsonl")):
        yield path


def load_episode(path: Path, line_no: int, raw: str) -> Dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON in {path}:{line_no}") from exc


def episode_key(episode: Dict) -> Tuple[str, str]:
    """
    Produce a stable deduplication key.

    Prefer the provided meta.fingerprint when present; otherwise fall back to a
    canonical JSON string of the episode payload.
    """
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


def episode_sequences(episode: Dict) -> Iterator[List]:
    """Yield all relevant list-valued sequences from the episode."""
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
    """Count scalar items in (possibly nested) list structures."""
    total = 0
    stack = [obj]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        else:
            total += 1
    return total


def summarise_lengths(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": int(min(values)),
        "median": float(median(values)),
        "mean": float(mean(values)),
        "max": int(max(values)),
    }


def summarise_floats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "median": float(median(values)),
        "mean": float(mean(values)),
        "max": float(max(values)),
    }


def compute_histogram(values: List[float], bins: int = 100) -> List[Dict[str, float]]:
    if not values:
        return []

    lo = min(values)
    hi = max(values)
    if lo == hi:
        return [{"bin_start": float(lo), "bin_end": float(hi), "count": len(values)}]

    width = (hi - lo) / bins
    # Guard against zero width due to numerical issues.
    if width == 0:
        return [{"bin_start": float(lo), "bin_end": float(hi), "count": len(values)}]

    counts = [0] * bins
    for val in values:
        idx = int((val - lo) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    histogram = []
    for i, count in enumerate(counts):
        start = lo + i * width
        end = start + width
        histogram.append(
            {"bin_start": float(start), "bin_end": float(end), "count": int(count)}
        )
    return histogram


def build_meta_index(pool_dir: Path) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    for path in iter_meta_files(pool_dir):
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse JSON in {path}:{line_no}") from exc
                fingerprint = record.get("fingerprint")
                if not isinstance(fingerprint, str) or fingerprint in index:
                    continue
                meta = record.get("meta") or {}
                coverage = meta.get("coverage") or {}
                fraction = coverage.get("fraction")
                observed_fraction = coverage.get("observed_fraction")
                entry: Dict[str, object] = {"record": record}
                if isinstance(fraction, (int, float)):
                    entry["fraction"] = float(fraction)
                if isinstance(observed_fraction, (int, float)):
                    entry["observed_fraction"] = float(observed_fraction)
                index[fingerprint] = entry
    return index


def collect_stats(pool_dir: Path, max_flattened: int) -> Dict:
    meta_index = build_meta_index(pool_dir)

    total = 0
    seen_keys = set()
    novel_count = 0
    novel_candidates = 0

    list_lengths: List[int] = []
    flattened_lengths: List[int] = []
    lambda_values: List[float] = []
    lambda_bins: Counter[str] = Counter()
    coverage_fractions: List[float] = []
    coverage_observed_fractions: List[float] = []
    fingerprints: List[str] = []

    for path in iter_episode_files(pool_dir):
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                raw = line.strip()
                if not raw:
                    continue

                episode = load_episode(path, line_no, raw)
                total += 1

                key = episode_key(episode)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                if not has_novel_solution(episode):
                    continue

                sequences = list(episode_sequences(episode))
                flattened_total = sum(flattened_length(seq) for seq in sequences)

                novel_candidates += 1
                if flattened_total > max_flattened:
                    continue

                novel_count += 1

                list_lengths.extend(len(seq) for seq in sequences)
                flattened_lengths.append(flattened_total)

                meta = episode.get("meta") or {}
                lambda_val = meta.get("lambda")
                if isinstance(lambda_val, (int, float)):
                    lambda_values.append(float(lambda_val))

                lambda_bin = meta.get("lambda_bin") or "<missing>"
                lambda_bins[str(lambda_bin)] += 1

                fingerprint = meta.get("fingerprint")
                meta_cov = None
                if isinstance(fingerprint, str):
                    fingerprints.append(fingerprint)
                    meta_cov = meta_index.get(fingerprint)

                coverage = meta.get("coverage") or {}
                fraction = None
                observed_fraction = None
                if meta_cov:
                    fraction = meta_cov.get("fraction")
                    observed_fraction = meta_cov.get("observed_fraction")
                if fraction is None:
                    raw_fraction = coverage.get("fraction")
                    if isinstance(raw_fraction, (int, float)):
                        fraction = float(raw_fraction)
                if observed_fraction is None:
                    raw_observed = coverage.get("observed_fraction")
                    if isinstance(raw_observed, (int, float)):
                        observed_fraction = float(raw_observed)

                if fraction is not None:
                    coverage_fractions.append(float(fraction))
                if observed_fraction is not None:
                    coverage_observed_fractions.append(float(observed_fraction))

    unique = len(seen_keys)

    return {
        "total": total,
        "unique": unique,
        "novel_unique_candidates": novel_candidates,
        "novel_unique": novel_count,
        "list_lengths": list_lengths,
        "flattened_lengths": flattened_lengths,
        "lambda_values": lambda_values,
        "lambda_bins": lambda_bins,
        "coverage_fractions": coverage_fractions,
        "coverage_observed_fractions": coverage_observed_fractions,
        "fingerprints": fingerprints,
    }


def write_outputs(outdir: Path, summary: Dict, fingerprints: List[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    fingerprints_path = outdir / "novel_fingerprints.txt"
    with fingerprints_path.open("w", encoding="utf-8") as fh:
        for fp in fingerprints:
            fh.write(f"{fp}\n")


def plot_histogram(
    values: List[float],
    outdir: Path,
    filename: str,
    title: str,
    *,
    bins: int = 100,
    xlabel: str = "value",
) -> None:
    if not values:
        print(f"No values available for {title.lower()}; skipping histogram plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting; histogram image skipped.")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="#3c78d8", edgecolor="black", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path = outdir / filename
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved histogram plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise CA pool statistics from JSONL shards."
    )
    parser.add_argument(
        "pool_dir",
        nargs="?",
        default="artifacts/pool",
        type=Path,
        help="Directory containing shard *.jsonl files (default: artifacts/pool)",
    )
    parser.add_argument(
        "--outdir",
        default=Path("artifacts/pool_stats"),
        type=Path,
        help="Directory to write detailed statistics (default: artifacts/pool_stats)",
    )
    parser.add_argument(
        "--max-flattened-length",
        type=int,
        default=DEFAULT_MAX_FLATTENED,
        help=f"Maximum flattened length for filtered episodes (default: {DEFAULT_MAX_FLATTENED})",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save a lambda histogram plot (requires matplotlib)",
    )
    args = parser.parse_args()

    pool_dir: Path = args.pool_dir
    if not pool_dir.exists():
        raise SystemExit(f"Pool directory {pool_dir} does not exist")
    if not pool_dir.is_dir():
        raise SystemExit(f"Pool path {pool_dir} is not a directory")

    stats = collect_stats(pool_dir, args.max_flattened_length)

    list_stats = summarise_lengths(stats["list_lengths"])
    flattened_stats = summarise_lengths(stats["flattened_lengths"])

    novel_count = stats["novel_unique"]
    lambda_bin_stats = {
        bin_name: {
            "count": count,
            "proportion": (count / novel_count) if novel_count else 0.0,
        }
        for bin_name, count in sorted(stats["lambda_bins"].items())
    }
    lambda_hist = compute_histogram(stats["lambda_values"], bins=100)
    coverage_fraction_stats = summarise_floats(stats["coverage_fractions"])
    coverage_fraction_hist = compute_histogram(stats["coverage_fractions"], bins=100)
    coverage_observed_stats = summarise_floats(stats["coverage_observed_fractions"])
    coverage_observed_hist = compute_histogram(
        stats["coverage_observed_fractions"], bins=100
    )

    summary = {
        "totals": {
            "total_serialized": stats["total"],
            "unique_episodes": stats["unique"],
            "unique_novel_episodes_candidates": stats["novel_unique_candidates"],
            "unique_novel_episodes": stats["novel_unique"],
        },
        "filters": {
            "novel_solution": True,
            "max_flattened_length": args.max_flattened_length,
        },
        "list_length_stats": list_stats,
        "flattened_length_stats": flattened_stats,
        "lambda_bin_stats": lambda_bin_stats,
        "lambda_histogram": lambda_hist,
        "coverage_fraction_stats": coverage_fraction_stats,
        "coverage_fraction_histogram": coverage_fraction_hist,
        "coverage_observed_fraction_stats": coverage_observed_stats,
        "coverage_observed_fraction_histogram": coverage_observed_hist,
    }

    write_outputs(args.outdir, summary, stats["fingerprints"])

    if args.plot:
        plot_histogram(
            stats["lambda_values"],
            args.outdir,
            "lambda_histogram.png",
            "Lambda Distribution (novel unique episodes)",
            bins=100,
            xlabel="lambda",
        )
        plot_histogram(
            stats["coverage_fractions"],
            args.outdir,
            "coverage_fraction_histogram.png",
            "Coverage Fraction Distribution (novel unique episodes)",
            bins=100,
            xlabel="coverage fraction",
        )
        plot_histogram(
            stats["coverage_observed_fractions"],
            args.outdir,
            "coverage_observed_fraction_histogram.png",
            "Observed Coverage Fraction Distribution (novel unique episodes)",
            bins=100,
            xlabel="observed fraction",
        )

    print(f"Total episodes: {stats['total']}")
    print(f"Unique episodes: {stats['unique']}")
    print(
        "Unique episodes with novel solution before length filter: "
        f"{stats['novel_unique_candidates']}"
    )
    print(
        "Unique episodes with novel solution and flattened length "
        f"<= {args.max_flattened_length}: {stats['novel_unique']}"
    )

    print("\nList length stats (novel unique episodes):")
    for key, value in summary["list_length_stats"].items():
        print(f"  {key}: {value}")

    print("\nFlattened episode length stats (novel unique episodes):")
    for key, value in summary["flattened_length_stats"].items():
        print(f"  {key}: {value}")

    print("\nLambda bin proportions (novel unique episodes):")
    for bin_name, data in summary["lambda_bin_stats"].items():
        pct = data["proportion"] * 100.0
        print(f"  {bin_name}: count={data['count']} ({pct:.2f}%)")

    print("\nCoverage fraction stats (novel unique episodes):")
    for key, value in summary["coverage_fraction_stats"].items():
        print(f"  {key}: {value}")

    print("\nObserved coverage fraction stats (novel unique episodes):")
    for key, value in summary["coverage_observed_fraction_stats"].items():
        print(f"  {key}: {value}")

    print(f"\nWrote detailed outputs to {args.outdir}")


if __name__ == "__main__":
    main()
