#!/usr/bin/env python3
"""Generate dataset distribution plots for CellARC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import CategoricalDtype


DATASET_ROOT = Path("artifacts/datasets/cellarc_100k_meta/data")
OUTPUT_DIR = Path("artifacts/paper_plots/dataset_stats")

SPLIT_FILES: Dict[str, str] = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test_interpolation": "test_interpolation.jsonl",
    "test_extrapolation": "test_extrapolation.jsonl",
}

SPLIT_ORDER = ["train", "val", "test_interpolation", "test_extrapolation"]
SPLIT_COLORS = {
    "train": "#4C72B0",
    "val": "#55A868",
    "test_interpolation": "#C44E52",
    "test_extrapolation": "#8172B2",
}


def _load_split(path: Path, split: str) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record.get("meta", {})
            coverage = meta.get("coverage", {}) or {}
            morphology = meta.get("morphology", {}) or {}
            query = record.get("query") or []
            window = meta.get("window")
            radius = meta.get("radius")
            steps = meta.get("steps")
            auto_window = 2 * radius * steps + 1 if radius is not None and steps is not None else None
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
                "coverage_fraction": coverage.get("fraction"),
                "coverage_observed_fraction": coverage.get("observed_fraction"),
                "coverage_windows": meta.get("coverage_windows"),
                "train_context": meta.get("train_context"),
                "family": meta.get("family"),
                "query_window_coverage_weighted": meta.get("query_window_coverage_weighted"),
                "query_window_coverage_unique": meta.get("query_window_coverage_unique"),
                "query_window_avg_depth": meta.get("query_window_avg_depth"),
                "query_length": len(query),
                "coverage_windows_per_query_len": (
                    meta.get("coverage_windows") / len(query) if query else np.nan
                ),
                "derrida_like": morphology.get("derrida_like"),
                "absorbing": morphology.get("absorbing"),
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
    if "lambda_bin" in df.columns:
        lambda_bins = ["ordered", "critical", "chaotic"]
        present_bins = [b for b in lambda_bins if b in set(df["lambda_bin"].dropna())]
        if present_bins:
            df["lambda_bin"] = pd.Categorical(df["lambda_bin"], categories=present_bins, ordered=True)
    return df


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-colorblind")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def plot_rule_space_histograms(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=False)
    hist_specs = [
        ("alphabet_size", "Alphabet size $k$", 1),
        ("radius", "Radius $r$", 1),
        ("steps", "Steps $t$", 1),
    ]
    for ax, (column, label, step) in zip(axes, hist_specs):
        data = df[column].dropna()
        if data.empty:
            continue
        bins = np.arange(data.min() - 0.5, data.max() + 1.5, step)
        ax.hist(data, bins=bins, color="#4C72B0", edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_xticks(sorted(data.unique()))
    fig.suptitle("Rule-space coverage across CellARC splits")
    output_path = output_dir / "rule_space_histograms.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_radius_step_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    grouped = (
        df.groupby(["radius", "steps"])
        .size()
        .reset_index(name="count")
        .dropna(subset=["radius", "steps"])
    )
    if grouped.empty:
        raise RuntimeError("No radius/steps combinations to plot.")
    grouped["window"] = 2 * grouped["radius"] * grouped["steps"] + 1
    sizes = 400 * grouped["count"] / grouped["count"].max()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sc = ax.scatter(
        grouped["radius"],
        grouped["steps"],
        s=sizes,
        c=grouped["window"],
        cmap="viridis",
        alpha=0.8,
        linewidths=0.5,
        edgecolor="black",
    )
    for _, row in grouped.iterrows():
        ax.text(
            row["radius"] + 0.03,
            row["steps"] + 0.05,
            f"W={int(row['window'])}",
            fontsize=8,
            color="black",
        )
    ax.set_xlabel("Radius $r$")
    ax.set_ylabel("Steps $t$")
    ax.set_xticks(sorted(df["radius"].dropna().unique()))
    ax.set_yticks(sorted(df["steps"].dropna().unique()))
    ax.set_title("Radius vs. steps with window size $W = 2rt + 1$")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Window size $W$")
    output_path = output_dir / "radius_vs_steps.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_lambda_coverage_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    data = df[["lambda_bin", "coverage_observed_fraction"]].dropna()
    if data.empty:
        raise RuntimeError("No data for lambda/coverage heatmap.")
    dtype = data["lambda_bin"].dtype
    if isinstance(dtype, CategoricalDtype):
        lambda_bins = list(df["lambda_bin"].cat.categories)
    else:
        lambda_bins = sorted(data["lambda_bin"].unique())
    coverage_min, coverage_max = data["coverage_observed_fraction"].min(), data["coverage_observed_fraction"].max()
    coverage_bins = np.linspace(coverage_min, coverage_max, 11)
    coverage_labels = [f"{coverage_bins[i]:.2f}–{coverage_bins[i + 1]:.2f}" for i in range(len(coverage_bins) - 1)]
    binned = data.assign(
        coverage_bin=pd.cut(
            data["coverage_observed_fraction"],
            bins=coverage_bins,
            labels=coverage_labels,
            include_lowest=True,
        )
    )
    pivot = (
        binned.pivot_table(
            index="lambda_bin",
            columns="coverage_bin",
            aggfunc="size",
            fill_value=0,
            observed=False,
        )
        .reindex(lambda_bins)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Observed coverage fraction bins")
    ax.set_ylabel("Langton λ bin")
    ax.set_title("Coverage vs. Langton λ across splits")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = int(pivot.iloc[i, j])
            if value > 0:
                ax.text(j, i, str(value), ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Episode count")
    output_path = output_dir / "lambda_vs_coverage_heatmap.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_query_window_coverage_histograms(df: pd.DataFrame, output_dir: Path) -> Path:
    column = "query_window_coverage_weighted"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    bin_edges = np.linspace(0.0, 1.0, 101)
    max_height = 0.0
    for ax, split in zip(axes.flat, SPLIT_ORDER):
        series = df.loc[df["split"] == split, column].dropna()
        label = split.replace("_", " ")
        if series.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(label)
            continue
        counts, _, _ = ax.hist(
            series,
            bins=bin_edges,
            color=SPLIT_COLORS.get(split, "#4C72B0"),
            edgecolor="white",
        )
        max_height = max(max_height, counts.max() if counts.size else 0.0)
        ax.set_title(label)
        ax.axvline(series.mean(), color="black", linestyle="--", linewidth=1)
    for ax in axes[1, :]:
        ax.set_xlabel("Query window coverage (weighted)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Episodes")
    if max_height > 0:
        for ax in axes.flat:
            ax.set_ylim(0, max_height * 1.05)
    for ax in axes.flat:
        ax.set_xlim(0, 1)
    fig.suptitle("Query window coverage across splits")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / "query_window_coverage_histograms.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_split_boxplots(df: pd.DataFrame, output_dir: Path) -> Path:
    metrics = [
        ("query_window_coverage_weighted", "Query window coverage (weighted)"),
        ("lambda", "Langton λ"),
        ("avg_cell_entropy", "Average cell entropy"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 9), sharex=False)
    for ax, (column, label) in zip(axes, metrics):
        data_to_plot = []
        split_labels = []
        color_patches = []
        for split in SPLIT_ORDER:
            split_series = df.loc[df["split"] == split, column].dropna()
            if split_series.empty:
                continue
            data_to_plot.append(split_series)
            split_labels.append(split.replace("_", " "))
            color_patches.append(SPLIT_COLORS.get(split, "#4C72B0"))
        if not data_to_plot:
            continue
        box = ax.boxplot(
            data_to_plot,
            tick_labels=split_labels,
            patch_artist=True,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"linewidth": 1},
            whiskerprops={"linewidth": 1},
            capprops={"linewidth": 1},
        )
        for patch, color in zip(box["boxes"], color_patches):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("Split")
    fig.suptitle("Distribution shift across splits (boxplots)")
    output_path = output_dir / "split_metric_histograms.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_split_sizes(df: pd.DataFrame, output_dir: Path) -> Path:
    counts = (
        df.groupby("split")
        .size()
        .reindex(SPLIT_ORDER)
        .fillna(0)
        .astype(int)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.str.replace("_", " "), counts.values, color="#4C72B0")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + counts.values.max() * 0.01, f"{value:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Episodes")
    ax.set_title("Episodes per split")
    output_path = output_dir / "split_sizes.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_family_mix(df: pd.DataFrame, output_dir: Path) -> Path:
    data = df.dropna(subset=["family", "split"])
    if data.empty:
        raise RuntimeError("No family metadata available.")
    pivot = (
        data.pivot_table(index="split", columns="family", aggfunc="size", fill_value=0)
        .reindex(SPLIT_ORDER)
        .fillna(0)
    )
    totals = pivot.sum(axis=1)
    proportions = pivot.div(totals, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    families = pivot.columns.tolist()
    cmap = plt.get_cmap("tab20")
    bottom = np.zeros(len(pivot))
    for idx, family in enumerate(families):
        values = proportions[family].values
        ax.bar(
            pivot.index.str.replace("_", " "),
            values,
            bottom=bottom,
            label=family,
            color=cmap(idx % cmap.N),
            edgecolor="white",
        )
        bottom += values
    ax.set_ylabel("Fraction of episodes")
    ax.set_title("Family mix per split")
    ax.set_ylim(0, 1)
    ax.legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left")
    output_path = output_dir / "family_mix_per_split.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_plots(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plotters = [
        plot_rule_space_histograms,
        plot_radius_step_scatter,
        plot_lambda_coverage_heatmap,
        plot_query_window_coverage_histograms,
        plot_split_boxplots,
        plot_split_sizes,
        plot_family_mix,
    ]
    outputs: List[Path] = []
    for plotter in plotters:
        outputs.append(plotter(df, output_dir))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Path to directory containing split JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write plot images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_plot_style()
    df = load_dataset(args.dataset_root)
    outputs = generate_plots(df, args.output_dir)
    print("Wrote plots:")
    for path in outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()
