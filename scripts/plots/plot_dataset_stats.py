#!/usr/bin/env python3
"""Generate dataset distribution plots for CellARC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cellarc import download_benchmark
OUTPUT_DIR = Path("figures") / "dataset_stats"
FAMILY_MIX_JSON = OUTPUT_DIR / "family_mix_per_split.json"
DEFAULT_CACHE_HOME = Path(os.getenv("CELLARC_HOME", Path.home() / ".cache" / "cellarc"))
DEFAULT_DATASET_ROOT = DEFAULT_CACHE_HOME / "dada" #"hf-cellarc_100k_meta" / "data"

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

TAU = np.sqrt(2 * np.pi)
LAMBDA_BIN_ORDER = ["chaotic", "edge", "ordered"]
PER_SPLIT_BAR_WIDTH = 0.8 / 3  # narrower bars for per-split stacks
PER_SPLIT_FONT_SCALE = 2.0
PER_SPLIT_BAR_TEXT_SIZE = int(7 * PER_SPLIT_FONT_SCALE)
PER_SPLIT_CATEGORY_SPACING = 0.33  # tighter distance between stacked split bars
JOINT_BAR_WIDTH = 0.55  # wider bars for the joint figure where we have more horizontal space


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
        unique_bins = [b for b in df["lambda_bin"].dropna().unique()]
        ordered_bins = [b for b in LAMBDA_BIN_ORDER if b in unique_bins]
        ordered_bins.extend([b for b in unique_bins if b not in ordered_bins])
        if ordered_bins:
            df["lambda_bin"] = pd.Categorical(df["lambda_bin"], categories=ordered_bins, ordered=True)
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


def _format_lambda_label(name: str) -> str:
    if not name:
        return "Unknown"
    return str(name).replace("_", " ").title()


def _format_split_label(name: str, *, long: bool = False) -> str:
    if not name:
        return ""
    normalized = str(name).lower()
    if normalized == "test_interpolation":
        suffix = "interpolation" if long else "interp."
        return f"test\n({suffix})"
    if normalized == "test_extrapolation":
        suffix = "extrapolation" if long else "extra."
        return f"test\n({suffix})"
    return str(name).replace("_", " ")


def _gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(grid)
    std = np.std(values, ddof=1)
    if std <= 0:
        std = 1e-3
    bandwidth = 1.06 * std * (values.size ** (-1 / 5))
    bandwidth = max(bandwidth, 1e-3)
    diffs = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * diffs**2).sum(axis=1)
    return density / (values.size * bandwidth * TAU)


def plot_split_kde_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    metrics = [
        ("query_window_coverage_weighted", "Query window coverage (weighted)", (0.0, 1.0)),
        ("lambda", "Langton λ", None),
        ("avg_cell_entropy", "Average cell entropy", None),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8.5, 9), sharex=False)
    for ax, (column, label, fixed_range) in zip(axes, metrics):
        combined = df[column].dropna()
        if combined.empty:
            ax.set_visible(False)
            continue
        if fixed_range:
            xmin, xmax = fixed_range
        else:
            xmin, xmax = combined.min(), combined.max()
        margin = (xmax - xmin) * 0.08 if xmax > xmin else 0.05
        grid = np.linspace(xmin - margin, xmax + margin, 256)
        plotted = False
        for split in SPLIT_ORDER:
            values = df.loc[df["split"] == split, column].dropna().to_numpy()
            if values.size == 0:
                continue
            density = _gaussian_kde(values, grid)
            color = SPLIT_COLORS.get(split, "#4C72B0")
            label_text = split.replace("_", " ")
            ax.plot(grid, density, label=label_text, color=color, linewidth=2)
            ax.fill_between(grid, 0, density, color=color, alpha=0.15)
            plotted = True
        ax.set_ylabel("Density")
        ax.set_title(label)
        if fixed_range:
            ax.set_xlim(*fixed_range)
        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    axes[-1].set_xlabel("Value")
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    fig.suptitle("Distribution shift across splits (KDEs)")
    output_path = output_dir / "split_metric_kdes.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

def plot_family_and_lambda_mix_joint(df: pd.DataFrame, output_dir: Path) -> Path:
    fam_df = df.dropna(subset=["family", "split"])
    fam_pivot = (
        fam_df.pivot_table(index="split", columns="family", aggfunc="size", fill_value=0, observed=False)
        .reindex(SPLIT_ORDER)
        .fillna(0)
    )
    fam_totals = fam_pivot.sum(axis=1).replace(0, np.nan)
    fam_prop = fam_pivot.div(fam_totals, axis=0).fillna(0)
    fam_labels = {c: _format_family_label(c) for c in fam_pivot.columns}

    lam_df = df.dropna(subset=["lambda_bin", "split"])
    lam_pivot = (
        lam_df.pivot_table(index="split", columns="lambda_bin", aggfunc="size", fill_value=0, observed=False)
        .reindex(SPLIT_ORDER)
        .fillna(0)
    )
    present = list(lam_pivot.columns)
    lam_cols = [x for x in LAMBDA_BIN_ORDER if x in present]
    lam_cols.extend([x for x in present if x not in lam_cols])
    lam_pivot = lam_pivot[lam_cols]
    lam_totals = lam_pivot.sum(axis=1).replace(0, np.nan)
    lam_prop = lam_pivot.div(lam_totals, axis=0).fillna(0)
    lam_labels = {c: _format_lambda_label(c) for c in lam_cols}

    if fam_prop.empty or lam_prop.empty:
        raise RuntimeError("Need both family and lambda metadata to build the joint chart.")

    output_path = output_dir / "family_and_lambda_mix_joint.png"
    base_font = float(plt.rcParams.get("font.size", 10.0))
    rc_overrides = {"font.size": base_font * PER_SPLIT_FONT_SCALE}

    n_splits = len(SPLIT_ORDER)
    split_positions = np.arange(n_splits, dtype=float)
    split_tick_labels = [_format_split_label(name, long=True) for name in SPLIT_ORDER]

    def _legend_cols(labels: List[str]) -> int:
        if len(labels) > 12:
            return 3
        if len(labels) > 5:
            return 2
        return 1

    def _draw_stack(ax: plt.Axes, proportions: pd.DataFrame, formatted: Dict[str, str], cmap, legend_title: str) -> None:
        bottom = np.zeros(n_splits)
        handles: List[object] = []
        labels: List[str] = []
        for idx, column in enumerate(proportions.columns):
            values = proportions[column].reindex(SPLIT_ORDER).values
            rects = ax.bar(
                split_positions,
                values,
                width=JOINT_BAR_WIDTH,
                bottom=bottom,
                color=cmap(idx % cmap.N),
                edgecolor="white",
                label=formatted.get(column, column),
            )
            for split_idx, rect in enumerate(rects):
                height = rect.get_height()
                if height <= 0 or height < 0.035:
                    continue
                text_y = bottom[split_idx] + height / 2
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    text_y,
                    f"{height*100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=PER_SPLIT_BAR_TEXT_SIZE,
                    color="white",
                )
            bottom += values
            if rects:
                handles.append(rects[0])
                labels.append(formatted.get(column, column))
        ax.set_xticks(split_positions)
        ax.set_xticklabels(split_tick_labels)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.6, n_splits - 0.4)
        ax.tick_params(axis="x", labelrotation=0)
        legend_labels = [label.replace(" ", "\n") for label in labels]
        ax.legend(
            handles,
            legend_labels,
            title=legend_title,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            frameon=False,
            borderaxespad=0.0,
            ncol=_legend_cols(legend_labels),
        )

    with plt.rc_context(rc_overrides):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
        cmap_fam = plt.get_cmap("tab20")
        cmap_lam = plt.get_cmap("Set2")

        _draw_stack(axes[0], fam_prop, fam_labels, cmap_fam, "Family")
        axes[0].set_ylabel("Fraction of episodes")
        axes[0].set_title("Family mix per split")

        _draw_stack(axes[1], lam_prop, lam_labels, cmap_lam, "λ bin")
        axes[1].set_title("λ-bin mix per split")

        fig.suptitle("Family & λ-bin composition per split")
        fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.2, wspace=0.25)
        fig.savefig(output_path)
        plt.close(fig)

    return output_path


def plot_entropy_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    column = "avg_cell_entropy"
    split_data: List[np.ndarray] = []
    tick_labels: List[str] = []
    colors: List[str] = []
    for split in SPLIT_ORDER:
        values = df.loc[df["split"] == split, column].dropna().to_numpy()
        if values.size == 0:
            continue
        split_data.append(values)
        tick_labels.append(_format_split_label(split))
        colors.append(SPLIT_COLORS.get(split, "#4C72B0"))
    if not split_data:
        raise RuntimeError("No cell entropy data available for boxplot.")
    output_path = output_dir / "cell_entropy_boxplot.png"
    base_font = float(plt.rcParams.get("font.size", 10.0))
    rc_overrides = {"font.size": base_font * 1.4}
    with plt.rc_context(rc_overrides):
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        boxprops = {"linewidth": 1.2}
        medianprops = {"linewidth": 1.6, "color": "#222222"}
        bp = ax.boxplot(
            split_data,
            tick_labels=tick_labels,
            patch_artist=True,
            widths=0.45,
            showfliers=False,
            boxprops=boxprops,
            medianprops=medianprops,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.55)
        for whisker in bp["whiskers"]:
            whisker.set_linewidth(1.0)
        for cap in bp["caps"]:
            cap.set_linewidth(1.0)
        ax.set_ylabel("Average cell entropy")
        ax.set_title("Cell entropy distribution per split")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    return output_path


def plot_lambda_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    column = "lambda"
    split_data: List[np.ndarray] = []
    tick_labels: List[str] = []
    colors: List[str] = []
    for split in SPLIT_ORDER:
        values = df.loc[df["split"] == split, column].dropna().to_numpy()
        if values.size == 0:
            continue
        split_data.append(values)
        tick_labels.append(_format_split_label(split))
        colors.append(SPLIT_COLORS.get(split, "#4C72B0"))
    if not split_data:
        raise RuntimeError("No Langton λ data available for boxplot.")
    output_path = output_dir / "lambda_boxplot.png"
    base_font = float(plt.rcParams.get("font.size", 10.0))
    rc_overrides = {"font.size": base_font * 1.4}
    with plt.rc_context(rc_overrides):
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        boxprops = {"linewidth": 1.2}
        medianprops = {"linewidth": 1.6, "color": "#222222"}
        bp = ax.boxplot(
            split_data,
            tick_labels=tick_labels,
            patch_artist=True,
            widths=0.45,
            showfliers=False,
            boxprops=boxprops,
            medianprops=medianprops,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.55)
        for whisker in bp["whiskers"]:
            whisker.set_linewidth(1.0)
        for cap in bp["caps"]:
            cap.set_linewidth(1.0)
        ax.set_ylabel("Langton λ")
        ax.set_title("Langton λ distribution per split")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    return output_path


def plot_query_window_coverage_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    column = "query_window_coverage_weighted"
    split_data: List[np.ndarray] = []
    tick_labels: List[str] = []
    colors: List[str] = []
    for split in SPLIT_ORDER:
        values = df.loc[df["split"] == split, column].dropna().to_numpy()
        if values.size == 0:
            continue
        split_data.append(values)
        tick_labels.append(_format_split_label(split))
        colors.append(SPLIT_COLORS.get(split, "#4C72B0"))
    if not split_data:
        raise RuntimeError("No query window coverage data available for boxplot.")
    output_path = output_dir / "query_window_coverage_boxplot.png"
    base_font = float(plt.rcParams.get("font.size", 10.0))
    rc_overrides = {"font.size": base_font * 1.4}
    with plt.rc_context(rc_overrides):
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        boxprops = {"linewidth": 1.2}
        medianprops = {"linewidth": 1.6, "color": "#222222"}
        bp = ax.boxplot(
            split_data,
            tick_labels=tick_labels,
            patch_artist=True,
            widths=0.45,
            showfliers=False,
            boxprops=boxprops,
            medianprops=medianprops,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.55)
        for whisker in bp["whiskers"]:
            whisker.set_linewidth(1.0)
        for cap in bp["caps"]:
            cap.set_linewidth(1.0)
        ax.set_ylabel("Query window coverage (weighted)")
        coverage_max = max(float(values.max()) for values in split_data)
        upper = min(1.0, coverage_max + 0.03)
        ax.set_ylim(0.5, upper)
        ax.set_title("Query window coverage distribution per split")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    return output_path


def plot_entropy_violin(df: pd.DataFrame, output_dir: Path) -> Path:
    """Violin plot of average cell entropy per split.

    Uses split colors and larger font scale similar to the family mix figure.
    """
    column = "avg_cell_entropy"
    split_data: List[np.ndarray] = []
    tick_labels: List[str] = []
    colors: List[str] = []
    for split in SPLIT_ORDER:
        values = df.loc[df["split"] == split, column].dropna().to_numpy()
        if values.size == 0:
            continue
        split_data.append(values)
        tick_labels.append(_format_split_label(split))
        colors.append(SPLIT_COLORS.get(split, "#4C72B0"))
    if not split_data:
        raise RuntimeError("No cell entropy data available for violin plot.")

    output_path = output_dir / "cell_entropy_violin.png"

    base_font = float(plt.rcParams.get("font.size", 10.0))
    rc_overrides = {"font.size": base_font * PER_SPLIT_FONT_SCALE}
    with plt.rc_context(rc_overrides):
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        vp = ax.violinplot(
            split_data,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.6,
        )
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)

        medians = [np.median(values) for values in split_data]
        positions = np.arange(1, len(split_data) + 1)
        ax.plot(positions, medians, color="#222222", linewidth=1.2, marker="o", markersize=4, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Average cell entropy")
        # Dynamic y-limits with a small margin
        all_values = np.concatenate(split_data)
        ymin, ymax = float(np.min(all_values)), float(np.max(all_values))
        if np.isfinite(ymin) and np.isfinite(ymax):
            rng = max(ymax - ymin, 1e-6)
            margin = rng * 0.05
            ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_title("Cell entropy per split (violin)")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    return output_path


def plot_query_window_coverage_violin(df: pd.DataFrame, output_dir: Path) -> Path:
    """Violin plot of query_window_coverage_weighted per split.

    Styled with larger fonts similar to the family mix figure, and colored
    by split using SPLIT_COLORS for visual consistency across plots.
    """
    column = "query_window_coverage_weighted"
    split_data: List[np.ndarray] = []
    tick_labels: List[str] = []
    colors: List[str] = []
    for split in SPLIT_ORDER:
        values = df.loc[df["split"] == split, column].dropna().to_numpy()
        if values.size == 0:
            continue
        split_data.append(values)
        tick_labels.append(_format_split_label(split))
        colors.append(SPLIT_COLORS.get(split, "#4C72B0"))
    if not split_data:
        raise RuntimeError("No query window coverage data available for violin plot.")

    output_path = output_dir / "query_window_coverage_violin.png"

    # Match the larger font scale used in the family mix figure
    base_font = float(plt.rcParams.get("font.size", 10.0))
    rc_overrides = {"font.size": base_font * PER_SPLIT_FONT_SCALE}
    with plt.rc_context(rc_overrides):
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        vp = ax.violinplot(
            split_data,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.6,
        )
        # Color each violin according to the split color
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)

        # Overlay medians as points and a thin line for clarity
        medians = [np.median(values) for values in split_data]
        positions = np.arange(1, len(split_data) + 1)
        ax.plot(positions, medians, color="#222222", linewidth=1.2, marker="o", markersize=4, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Query window coverage (weighted)")
        coverage_max = max(float(values.max()) for values in split_data)
        upper = min(1.0, coverage_max + 0.03)
        ax.set_ylim(0.5, upper)
        ax.set_title("Query window coverage per split (violin)")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    return output_path


def plot_lambda_entropy_coverage_violin_joint(df: pd.DataFrame, output_dir: Path) -> Path:
    """Joint violin plot with λ, entropy, and query coverage side-by-side.

    - Uses split colors (`SPLIT_COLORS`) for each violin body
    - Larger font scale similar to the family mix figure
    - Medians overlaid as small points and thin line
    - No y-axis labels to keep the layout clean, titles per panel
    """

    def _collect(column: str):
        data: List[np.ndarray] = []
        ticks: List[str] = []
        cols: List[str] = []
        for split in SPLIT_ORDER:
            values = df.loc[df["split"] == split, column].dropna().to_numpy()
            if values.size == 0:
                continue
            data.append(values)
            ticks.append(_format_split_label(split))
            cols.append(SPLIT_COLORS.get(split, "#4C72B0"))
        if not data:
            raise RuntimeError(f"No data for metric: {column}")
        return data, ticks, cols

    cov_data, cov_ticks, cov_colors = _collect("query_window_coverage_weighted")
    ent_data, ent_ticks, ent_colors = _collect("avg_cell_entropy")
    lam_data, lam_ticks, lam_colors = _collect("lambda")

    output_path = output_dir / "lambda_entropy_coverage_violin_joint.png"

    base_font = float(plt.rcParams.get("font.size", 10.0))
    rc_overrides = {"font.size": base_font * PER_SPLIT_FONT_SCALE}
    with plt.rc_context(rc_overrides):
        fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.4), sharex=False)

        def _draw(
            ax: plt.Axes,
            split_data: List[np.ndarray],
            tick_labels: List[str],
            colors: List[str],
            title: str,
            ylim: Optional[tuple] = None,
        ):
            vp = ax.violinplot(
                split_data,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.6,
            )
            for body, color in zip(vp["bodies"], colors):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.55)
            medians = [np.median(values) for values in split_data]
            positions = np.arange(1, len(split_data) + 1)
            ax.plot(positions, medians, color="#222222", linewidth=1.2, marker="o", markersize=4, zorder=3)
            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels)
            ax.set_ylabel("")  # intentionally blank for clean layout
            if ylim is not None:
                ax.set_ylim(*ylim)
            else:
                all_values = np.concatenate(split_data)
                ymin, ymax = float(np.min(all_values)), float(np.max(all_values))
                if np.isfinite(ymin) and np.isfinite(ymax):
                    rng = max(ymax - ymin, 1e-6)
                    margin = rng * 0.05
                    ax.set_ylim(ymin - margin, ymax + margin)
            # Slightly raise the subplot titles for better spacing
            ax.set_title(title, pad=22)
            ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)

        # λ panel
        _draw(axes[0], lam_data, lam_ticks, lam_colors, title="Langton λ")

        # Entropy panel
        _draw(axes[1], ent_data, ent_ticks, ent_colors, title="Average cell entropy")

        # Coverage panel with constrained y-range similar to dedicated plot
        cov_max = max(float(values.max()) for values in cov_data)
        cov_upper = min(1.0, cov_max + 0.03)
        _draw(axes[2], cov_data, cov_ticks, cov_colors, title="Query window coverage (weighted)", ylim=(0.5, cov_upper))

        fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.18, wspace=0.28)
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


def plot_family_mix(df: pd.DataFrame, output_dir: Path, *, summary_path: Optional[Path] = None) -> Path:
    data = df.dropna(subset=["family", "split"])
    if data.empty:
        raise RuntimeError("No family metadata available.")
    pivot = (
        data.pivot_table(index="split", columns="family", aggfunc="size", fill_value=0, observed=False)
        .reindex(SPLIT_ORDER)
        .fillna(0)
    )
    totals = pivot.sum(axis=1)
    proportions = pivot.div(totals, axis=0)
    formatted = {fam: _format_family_label(fam) for fam in pivot.columns}
    families = list(pivot.columns)
    output_path = output_dir / "family_mix_per_split.png"
    base_font = float(plt.rcParams.get("font.size", 10.0))
    rc_overrides = {"font.size": base_font * PER_SPLIT_FONT_SCALE}
    with plt.rc_context(rc_overrides):
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        cmap = plt.get_cmap("tab20")
        bottom = np.zeros(len(pivot))
        handles: List[object] = []
        labels: List[str] = []
        split_positions = np.arange(len(pivot)) * PER_SPLIT_CATEGORY_SPACING
        tick_labels = [_format_split_label(name) for name in pivot.index]
        for idx, family in enumerate(families):
            values = proportions[family].values
            rects = ax.bar(
                split_positions,
                values,
                bottom=bottom,
                width=PER_SPLIT_BAR_WIDTH,
                label=formatted.get(family, family),
                color=cmap(idx % cmap.N),
                edgecolor="white",
            )
            for rect_idx, rect in enumerate(rects):
                height = rect.get_height()
                if height <= 0 or height < 0.035:
                    continue
                text_y = bottom[rect_idx] + height / 2
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    text_y,
                    f"{height*100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=PER_SPLIT_BAR_TEXT_SIZE,
                    color="white",
                )
            bottom += values
            if rects:
                handles.append(rects[0])
                labels.append(formatted.get(family, family))
        ax.set_ylabel("Fraction of episodes")
        ax.set_ylim(0, 1)
        if len(split_positions) > 0:
            margin = PER_SPLIT_BAR_WIDTH * 0.7
            ax.set_xlim(split_positions[0] - margin, split_positions[-1] + margin)
        ax.set_xticks(split_positions)
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis="x", labelrotation=0)
        legend_labels = [label.replace(" ", "\n") for label in labels]
        ax.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.12, right=0.78, top=0.95, bottom=0.15)
        fig.savefig(output_path)
        plt.close(fig)
    if summary_path:
        formatted_proportions = proportions.rename(columns=formatted)
        mean_per_family = {
            family: float(value)
            for family, value in formatted_proportions.mean(axis=0).to_dict().items()
        }
        per_split = {
            split: {family: float(val) for family, val in values.items()}
            for split, values in formatted_proportions.to_dict(orient="index").items()
        }
        counts_per_split = {
            split: {formatted.get(family, family): int(val) for family, val in values.items()}
            for split, values in pivot.to_dict(orient="index").items()
        }
        mean_percentage = {family: round(value * 100, 2) for family, value in mean_per_family.items()}
        per_split_percentage = {
            split: {family: round(val * 100, 2) for family, val in values.items()}
            for split, values in per_split.items()
        }
        payload = {
            "per_split_fraction": per_split,
            "per_split_percentage": per_split_percentage,
            "mean_fraction": mean_per_family,
            "mean_percentage": mean_percentage,
            "per_split_counts": counts_per_split,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return output_path


def plot_lambda_mix(df: pd.DataFrame, output_dir: Path, *, summary_path: Optional[Path] = None) -> Path:
    data = df.dropna(subset=["lambda_bin", "split"])
    if data.empty:
        raise RuntimeError("No lambda bin metadata available.")
    pivot = (
        data.pivot_table(index="split", columns="lambda_bin", aggfunc="size", fill_value=0, observed=False)
        .reindex(SPLIT_ORDER)
        .fillna(0)
    )
    if pivot.empty:
        raise RuntimeError("No lambda bin counts available.")
    present = list(pivot.columns)
    ordered_cols = [name for name in LAMBDA_BIN_ORDER if name in present]
    ordered_cols.extend([name for name in present if name not in ordered_cols])
    pivot = pivot[ordered_cols]
    totals = pivot.sum(axis=1)
    proportions = pivot.div(totals.replace(0, np.nan), axis=0).fillna(0)
    formatted = {name: _format_lambda_label(name) for name in ordered_cols}
    output_path = output_dir / "lambda_mix_per_split.png"
    base_font = float(plt.rcParams.get("font.size", 10.0))
    rc_overrides = {"font.size": base_font * PER_SPLIT_FONT_SCALE}
    with plt.rc_context(rc_overrides):
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        cmap = plt.get_cmap("Set2")
        bottom = np.zeros(len(pivot))
        handles: List[object] = []
        labels: List[str] = []
        split_positions = np.arange(len(pivot)) * PER_SPLIT_CATEGORY_SPACING
        tick_labels = [_format_split_label(name) for name in pivot.index]
        for idx, name in enumerate(ordered_cols):
            values = proportions[name].values
            rects = ax.bar(
                split_positions,
                values,
                bottom=bottom,
                width=PER_SPLIT_BAR_WIDTH,
                label=formatted.get(name, name),
                color=cmap(idx % cmap.N),
                edgecolor="white",
            )
            for rect_idx, rect in enumerate(rects):
                height = rect.get_height()
                if height <= 0 or height < 0.035:
                    continue
                text_y = bottom[rect_idx] + height / 2
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    text_y,
                    f"{height*100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=PER_SPLIT_BAR_TEXT_SIZE,
                    color="white",
                )
            bottom += values
            if rects:
                handles.append(rects[0])
                labels.append(formatted.get(name, name))
        ax.set_ylabel("Fraction of episodes")
        ax.set_ylim(0, 1)
        if len(split_positions) > 0:
            margin = PER_SPLIT_BAR_WIDTH * 0.7
            ax.set_xlim(split_positions[0] - margin, split_positions[-1] + margin)
        ax.set_xticks(split_positions)
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis="x", labelrotation=0)
        legend_labels = [label.replace(" ", "\n") for label in labels]
        ax.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.12, right=0.78, top=0.95, bottom=0.15)
        fig.savefig(output_path)
        plt.close(fig)
    if summary_path:
        formatted_proportions = proportions.rename(columns=formatted)
        mean_per_bin = {
            name: float(value)
            for name, value in formatted_proportions.mean(axis=0).to_dict().items()
        }
        per_split = {
            split: {name: float(val) for name, val in values.items()}
            for split, values in formatted_proportions.to_dict(orient="index").items()
        }
        counts_per_split = {
            split: {formatted.get(name, name): int(val) for name, val in values.items()}
            for split, values in pivot.to_dict(orient="index").items()
        }
        mean_percentage = {name: round(value * 100, 2) for name, value in mean_per_bin.items()}
        per_split_percentage = {
            split: {name: round(val * 100, 2) for name, val in values.items()}
            for split, values in per_split.items()
        }
        payload = {
            "per_split_fraction": per_split,
            "per_split_percentage": per_split_percentage,
            "mean_fraction": mean_per_bin,
            "mean_percentage": mean_percentage,
            "per_split_counts": counts_per_split,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return output_path


def plot_family_pies(df: pd.DataFrame, output_dir: Path) -> Path:
    data = df.dropna(subset=["family", "split"])
    if data.empty:
        raise RuntimeError("No family metadata available.")
    train_subset = data.loc[data["split"] == "train", "family"]
    if train_subset.empty:
        raise RuntimeError("Training split is missing family annotations.")
    counts = train_subset.value_counts()
    top = counts.head(12)
    if len(counts) > len(top):
        other = counts.sum() - top.sum()
        top["Other"] = other
    labels = [_format_family_label(name) for name in top.index]
    values = top.values
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
    )
    total = values.sum()
    legend_labels = [
        f"{label} — {value / total:.1%}" if total else label
        for label, value in zip(labels, values)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Family",
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        frameon=False,
    )
    ax.set_title("Cellular Automata Family Composition")
    ax.set_aspect("equal")
    fig.tight_layout()
    output_path = output_dir / "family_pies.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_plots(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plotters = [
        plot_rule_space_histograms,
        plot_split_kde_distributions,
        plot_family_and_lambda_mix_joint,
        plot_split_sizes,
        lambda df_, out: plot_family_mix(df_, out, summary_path=out / "family_mix_per_split.json"),
        lambda df_, out: plot_lambda_mix(df_, out, summary_path=out / "lambda_mix_per_split.json"),
        plot_entropy_boxplot,
        plot_entropy_violin,
        plot_lambda_boxplot,
        plot_query_window_coverage_boxplot,
        plot_query_window_coverage_violin,
        plot_lambda_entropy_coverage_violin_joint,
        plot_family_pies,
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
        default=None,
        help="Directory containing split JSONL files. Defaults to the Hugging Face cache "
        "(~/.cache/cellarc/hf-cellarc_100k_meta/data).",
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
    dataset_root = resolve_dataset_root(args.dataset_root)
    df = load_dataset(dataset_root)
    outputs = generate_plots(df, args.output_dir)
    print("Wrote plots:")
    for path in outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()
