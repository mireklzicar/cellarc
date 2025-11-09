#!/usr/bin/env python3
"""Generate a LaTeX table with basic dataset properties.

Reads:
 - analysis/stats/data_files.json (split sizes)
 - analysis/stats/dataset_stats.json (global distributions, medians)
 - artifacts/dataset_stats/dataset_metadata_summary.csv (optional; not required)

Writes:
 - BASIC_DATASE_PROPERTIES.tex in the repo root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_FILES_JSON = ROOT / "analysis" / "stats" / "data_files.json"
DATASET_STATS_JSON = ROOT / "analysis" / "stats" / "dataset_stats.json"
SUMMARY_CSV = ROOT / "artifacts" / "dataset_stats" / "dataset_metadata_summary.csv"
OUTPUT_TEX = ROOT / "BASIC_DATASE_PROPERTIES.tex"


def _format_family_label(name: str) -> str:
    if not name:
        return "Unknown"
    cleaned = str(name).strip().replace("_", " ")
    tokens = cleaned.split()
    result: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i].lower()
        next_token = tokens[i + 1].lower() if i + 1 < len(tokens) else None
        if token == "mod" and next_token in {"k", "(k)"}:
            result.append("mod(k)")
            i += 2
            continue
        if token.endswith("(k)"):
            result.append(token)
        elif token in {"ca", "io"}:
            result.append(token.upper())
        else:
            result.append(token.capitalize())
        i += 1
    return " ".join(result)


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _range_from_counts(items: Iterable[Dict[str, int]]) -> Tuple[int, int]:
    values = [int(it["value"]) for it in items]
    return (min(values), max(values)) if values else (0, 0)


def _percent(x: float) -> str:
    return f"{x:.2f}\\%"


def read_sources() -> Dict[str, object]:
    data_files = _read_json(DATA_FILES_JSON)
    stats = _read_json(DATASET_STATS_JSON)
    global_stats = stats.get("global", {})

    # Split sizes
    split_sizes = {name: int(info.get("jsonl", {}).get("records", 0)) for name, info in data_files.items()}
    total_episodes = int(global_stats.get("num_episodes", sum(split_sizes.values())))

    # Ranges
    alphabet_min, alphabet_max = _range_from_counts(global_stats.get("alphabet_sizes", []))
    radius_min, radius_max = _range_from_counts(global_stats.get("radii", []))
    steps_min, steps_max = _range_from_counts(global_stats.get("steps", []))
    window_min, window_max = _range_from_counts(global_stats.get("window_sizes", []))

    # Vocabulary (k^W) coarse bounds from min/max
    vocab_min = (alphabet_min ** window_min) if alphabet_min and window_min else 0
    vocab_max = (alphabet_max ** window_max) if alphabet_max and window_max else 0

    # Medians
    sample_len_median = global_stats.get("sample_length", {}).get("median")
    episode_len_median = global_stats.get("episode_total_length", {}).get("median")

    # Training examples per episode (constant 5 expected)
    train_examples = None
    for item in global_stats.get("train_example_counts", []):
        train_examples = int(item.get("value", 0))
        break

    # Families (overall counts and percentages)
    family_counts = global_stats.get("families", [])
    families_sorted = sorted(family_counts, key=lambda d: int(d.get("count", 0)), reverse=True)
    fam_rows: List[Tuple[str, float]] = []
    denom = float(total_episodes) if total_episodes else 0.0
    for item in families_sorted:
        name = _format_family_label(str(item.get("value")))
        cnt = int(item.get("count", 0))
        pct = (cnt / denom * 100.0) if denom else 0.0
        fam_rows.append((name, pct))

    return {
        "split_sizes": split_sizes,
        "total": total_episodes,
        "alphabet_range": (alphabet_min, alphabet_max),
        "radius_range": (radius_min, radius_max),
        "steps_range": (steps_min, steps_max),
        "window_range": (window_min, window_max),
        "vocab_range": (vocab_min, vocab_max),
        "sample_len_median": sample_len_median,
        "episode_len_median": episode_len_median,
        "train_examples": train_examples,
        "families": fam_rows,
    }


def write_latex(payload: Dict[str, object]) -> None:
    (k_min, k_max) = payload["alphabet_range"]  # type: ignore
    (r_min, r_max) = payload["radius_range"]  # type: ignore
    (t_min, t_max) = payload["steps_range"]  # type: ignore
    (w_min, w_max) = payload["window_range"]  # type: ignore
    (vmin, vmax) = payload["vocab_range"]  # type: ignore
    split_sizes: Dict[str, int] = payload["split_sizes"]  # type: ignore
    total = payload["total"]  # type: ignore
    sample_med = payload["sample_len_median"]  # type: ignore
    episode_med = payload["episode_len_median"]  # type: ignore
    train_examples = payload["train_examples"]  # type: ignore
    families: List[Tuple[str, float]] = payload["families"]  # type: ignore

    # Order splits for readability
    pretty_split_names = {
        "train": "Train",
        "val": "Val",
        "test_interpolation": "Test (interp)",
        "test_extrapolation": "Test (extra)",
    }
    split_order = ["train", "val", "test_interpolation", "test_extrapolation"]

    lines: List[str] = []
    lines.append("% Auto-generated by export_basic_dataset_properties_tex.py")
    lines.append("% Basic dataset properties for CellARC-100k")
    def row(s: str) -> str:
        return s + " \\\\"

    lines.append("\\begin{tabular}{llr}")
    lines.append(row("\\toprule"))
    lines.append(row("\\textbf{Category} & \\textbf{Metric} & \\textbf{Value}"))
    lines.append(row("\\midrule"))
    # Rule-space
    lines.append(row("\\textbf{Rule-space} & Alphabet size $k$ (range) & %d--%d" % (k_min, k_max)))
    lines.append(row(" & Window size $W$ (range) & %d--%d" % (w_min, w_max)))
    lines.append(row(" & Radius $r$ (range) & %d--%d" % (r_min, r_max)))
    lines.append(row(" & Steps $t$ (range) & %d--%d" % (t_min, t_max)))
    if vmin and vmax:
        lines.append(row(" & Neighborhood vocab $k^W$ (range) & %s--%s" % (f"{vmin:,}", f"{vmax:,}")))
    lines.append(row("\\midrule"))
    # Episode sizes
    if train_examples is not None:
        lines.append(row("\\textbf{Episode size} & Training examples / episode & %s" % (f"{train_examples:,}")))
    if sample_med is not None:
        lines.append(row(" & Median sample length & %s" % (f"{int(sample_med):,}")))
    if episode_med is not None:
        lines.append(row(" & Median flattened episode length & %s" % (f"{int(episode_med):,}")))
    lines.append(row("\\midrule"))
    # Splits
    lines.append(row("\\textbf{Splits} & Total episodes & %s" % (f"{int(total):,}")))
    for key in split_order:
        if key in split_sizes:
            lines.append(row(" & %s & %s" % (pretty_split_names.get(key, key.replace("_", " ")).replace(" ", "~"), f"{split_sizes[key]:,}")))
    # Families
    lines.append(row("\\midrule"))
    lines.append(row("\\textbf{Families} & (overall share) & "))
    for name, pct in families:
        # escape % in LaTeX
        lines.append(row(" & %s & %s" % (name.replace(" ", "~"), _percent(pct))))
    lines.append(row("\\bottomrule"))
    lines.append("\\end{tabular}")

    OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = read_sources()
    write_latex(payload)
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
