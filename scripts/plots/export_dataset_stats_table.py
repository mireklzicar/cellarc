#!/usr/bin/env python3
"""Export CellARC dataset metadata in a long/CSV format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from cellarc import download_benchmark

OUTPUT_PATH = Path("artifacts") / "dataset_stats" / "dataset_metadata_long.csv"
SUMMARY_PATH = Path("artifacts") / "dataset_stats" / "dataset_metadata_summary.csv"
DEFAULT_CACHE_HOME = Path(os.getenv("CELLARC_HOME", Path.home() / ".cache" / "cellarc"))
DEFAULT_DATASET_ROOT = DEFAULT_CACHE_HOME / "hf-cellarc_100k_meta" / "data"
SPLIT_ORDER = ["train", "val", "test_interpolation", "test_extrapolation"]

SPLIT_FILES: Dict[str, str] = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test_interpolation": "test_interpolation.jsonl",
    "test_extrapolation": "test_extrapolation.jsonl",
}

SECTION_COLUMN_LABELS: Dict[str, Dict[str, str]] = {
    "family": {"family": "family"},
    "complexity": {
        "lambda": "lambda",
        "lambda_bin": "lambda_bin",
        "avg_cell_entropy": "avg_cell_entropy",
        "query_window_coverage_weighted": "query_window_coverage_weighted",
    },
    "properties": {
        "alphabet_size": "alphabet_size",
        "radius": "radius",
        "steps": "steps",
        "window_resolved": "window",
        "flattened_episode_length": "flattened_episode_length",
    },
}

COLUMN_TO_SECTION = {
    column: section
    for section, mapping in SECTION_COLUMN_LABELS.items()
    for column in mapping
}
COLUMN_TO_LABEL = {
    column: label
    for mapping in SECTION_COLUMN_LABELS.values()
    for column, label in mapping.items()
}
CATEGORICAL_SUMMARY_COLUMNS = ["family", "lambda_bin", "alphabet_size"]
NUMERIC_MEDIAN_COLUMNS = [
    "lambda",
    "avg_cell_entropy",
    "query_window_coverage_weighted",
    "radius",
    "steps",
    "window_resolved",
    "flattened_episode_length",
]
SECTION_ORDER = ["family", "complexity", "properties"]


def resolve_dataset_root(explicit: Optional[Path]) -> Path:
    """Return the dataset directory, downloading from the Hub when needed."""

    if explicit:
        return explicit
    default_root = DEFAULT_DATASET_ROOT
    if default_root.exists():
        return default_root
    repo_path = download_benchmark(name="cellarc_100k", include_metadata=True)
    resolved = repo_path / "data"
    if not resolved.exists():
        raise FileNotFoundError(
            "Unable to locate dataset JSONL files. Specify --dataset-root or ensure the "
            "downloaded snapshot contains a 'data' directory."
        )
    return resolved


def _load_split(path: Path, split: str) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record.get("meta", {})
            coverage = meta.get("coverage", {}) or {}
            query = record.get("query") or []
            train_pairs = record.get("train") or []
            window = meta.get("window")
            radius = meta.get("radius")
            steps = meta.get("steps")
            auto_window = 2 * radius * steps + 1 if radius is not None and steps is not None else None
            flattened_len = sum(
                len(pair.get("input", [])) + len(pair.get("output", [])) for pair in train_pairs
            ) + len(query)
            yield {
                "id": record.get("id") or meta.get("fingerprint"),
                "split": split,
                "alphabet_size": meta.get("alphabet_size"),
                "radius": radius,
                "steps": steps,
                "window": window,
                "window_formula": auto_window,
                "lambda": meta.get("lambda"),
                "lambda_bin": meta.get("lambda_bin"),
                "avg_cell_entropy": meta.get("avg_cell_entropy"),
                "query_window_coverage_weighted": meta.get("query_window_coverage_weighted"),
                "family": meta.get("family"),
                "query_length": len(query),
                "coverage_windows": meta.get("coverage_windows"),
                "coverage_fraction": coverage.get("fraction"),
                "coverage_observed_fraction": coverage.get("observed_fraction"),
                "flattened_episode_length": flattened_len,
            }


def load_dataset(dataset_root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for split, filename in SPLIT_FILES.items():
        split_path = dataset_root / filename
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        rows.extend(_load_split(split_path, split))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No records loaded from dataset; check paths.")
    return df


def _format_family_label(name: object) -> object:
    if not isinstance(name, str):
        return name
    cleaned = name.strip().replace("_", " ")
    tokens = cleaned.split()
    if not tokens:
        return "Unknown"
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


def _format_lambda_bin_label(name: object) -> object:
    if not isinstance(name, str):
        return name
    return str(name).replace("_", " ").title()


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["window_resolved"] = working["window"].fillna(working["window_formula"])
    return working


def build_long_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = _prepare_dataframe(df)
    long_frames: List[pd.DataFrame] = []
    id_vars = ["id", "split"]
    for section, column_map in SECTION_COLUMN_LABELS.items():
        columns = list(column_map.keys())
        missing = [col for col in columns if col not in working.columns]
        for column in missing:
            working[column] = pd.NA
        subset = working[id_vars + columns]
        melted = subset.melt(
            id_vars=id_vars,
            value_vars=columns,
            var_name="metric",
            value_name="value",
        ).dropna(subset=["value"])
        if melted.empty:
            continue
        melted["section"] = section
        melted["metric"] = melted["metric"].map(column_map)
        long_frames.append(melted)

    if not long_frames:
        raise RuntimeError("Unable to build long dataframe; no relevant metadata found.")

    result = pd.concat(long_frames, ignore_index=True)
    family_mask = result["section"].eq("family") & result["metric"].eq("family")
    result.loc[family_mask, "value"] = result.loc[family_mask, "value"].apply(_format_family_label)
    result = result.sort_values(["split", "section", "metric", "id"]).reset_index(drop=True)
    return result


def build_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = _prepare_dataframe(df)
    category_formatters: Dict[str, Callable[[object], object]] = {
        "family": _format_family_label,
        "lambda_bin": _format_lambda_bin_label,
        "alphabet_size": lambda value: value,
    }
    summary_rows: List[dict] = []
    split_sequence = [split for split in SPLIT_ORDER if split in working["split"].unique()]
    split_sequence.append("all")

    for split in split_sequence:
        subset = working if split == "all" else working.loc[working["split"] == split]
        if subset.empty:
            continue
        for column in CATEGORICAL_SUMMARY_COLUMNS:
            if column not in subset.columns:
                continue
            values = subset[column].dropna()
            if values.empty:
                continue
            counts = values.value_counts()
            total = counts.sum()
            formatter = category_formatters.get(column, lambda value: value)
            for category, count in counts.items():
                pct = (float(count) / float(total)) * 100.0 if total else 0.0
                summary_rows.append(
                    {
                        "split": split,
                        "section": COLUMN_TO_SECTION.get(column, ""),
                        "metric": COLUMN_TO_LABEL.get(column, column),
                        "category": formatter(category),
                        "statistic": "percentage",
                        "value": pct,
                    }
                )
        for column in NUMERIC_MEDIAN_COLUMNS:
            if column not in subset.columns:
                continue
            values = subset[column].dropna()
            if values.empty:
                continue
            summary_rows.append(
                {
                    "split": split,
                    "section": COLUMN_TO_SECTION.get(column, ""),
                    "metric": COLUMN_TO_LABEL.get(column, column),
                    "category": "median",
                    "statistic": "median",
                    "value": float(values.median()),
                }
            )
        summary_rows.append(
            {
                "split": split,
                "section": "properties",
                "metric": "num_samples",
                "category": "episodes",
                "statistic": "count",
                "value": int(len(subset)),
            }
        )

    if not summary_rows:
        raise RuntimeError("Unable to build summary dataframe; no metadata available.")

    summary_df = pd.DataFrame(summary_rows)
    split_categories = split_sequence
    summary_df["split"] = pd.Categorical(summary_df["split"], categories=split_categories, ordered=True)
    summary_df["section"] = pd.Categorical(summary_df["section"], categories=SECTION_ORDER, ordered=True)
    summary_df["category_label"] = summary_df["category"].astype(str)
    summary_df = summary_df.sort_values(
        ["split", "section", "metric", "statistic", "category_label"]
    ).drop(columns="category_label")
    summary_df["split"] = summary_df["split"].astype(str)
    summary_df["section"] = summary_df["section"].astype(str)
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Directory containing split JSONL files. Defaults to the Hugging Face cache "
        "(~/.cache/cellarc/hf-cellarc_100k_meta/data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"CSV file to write. Defaults to {OUTPUT_PATH.as_posix()}.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=SUMMARY_PATH,
        help=f"Summary CSV file to write. Defaults to {SUMMARY_PATH.as_posix()}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    df = load_dataset(dataset_root)
    long_df = build_long_dataframe(df)
    summary_df = build_summary_dataframe(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.output, index=False)
    summary_df.to_csv(args.summary_output, index=False)
    print(f"Wrote {len(long_df):,} rows to {args.output}")
    print(f"Wrote {len(summary_df):,} rows to {args.summary_output}")


if __name__ == "__main__":
    main()
