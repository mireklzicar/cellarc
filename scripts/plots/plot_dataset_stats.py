#!/usr/bin/env python3
"""Generate dataset distribution plots for CellARC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
DATASET_ROOT = Path("artifacts/datasets/cellarc_100k_meta/data")
OUTPUT_DIR = Path("figures") / "dataset_stats"
FAMILY_MIX_JSON = OUTPUT_DIR / "family_mix_per_split.json"

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
    fig = plt.figure(figsize=(8, 5.4))
    grid = fig.add_gridspec(2, 1, height_ratios=(0.7, 4), hspace=0.02)
    legend_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    legend_ax.axis("off")
    families = pivot.columns.tolist()
    formatted = {fam: _format_family_label(fam) for fam in families}
    cmap = plt.get_cmap("tab20")
    bottom = np.zeros(len(pivot))
    handles: List[object] = []
    labels: List[str] = []
    for idx, family in enumerate(families):
        values = proportions[family].values
        rects = ax.bar(
            pivot.index.str.replace("_", " "),
            values,
            bottom=bottom,
            label=formatted.get(family, family),
            color=cmap(idx % cmap.N),
            edgecolor="white",
        )
        for rect_idx, rect in enumerate(rects):
            height = rect.get_height()
            if height <= 0:
                continue
            if height < 0.035:
                continue
            text_y = bottom[rect_idx] + height / 2
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                text_y,
                f"{height*100:.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
        )
        bottom += values
        if rects:
            handles.append(rects[0])
            labels.append(formatted.get(family, family))
    ax.set_ylabel("Fraction of episodes")
    ax.set_ylim(0, 1)
    legend_cols = min(4, max(1, len(handles)))
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=legend_cols,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08, hspace=0.02)
    output_path = output_dir / "family_mix_per_split.png"
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


def plot_lambda_mix(df: pd.DataFrame, output_dir: Path) -> Path:
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
    fig = plt.figure(figsize=(8, 5.4))
    grid = fig.add_gridspec(2, 1, height_ratios=(0.7, 4), hspace=0.02)
    legend_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    legend_ax.axis("off")
    cmap = plt.get_cmap("Set2")
    bottom = np.zeros(len(pivot))
    handles: List[object] = []
    labels: List[str] = []
    formatted = {name: _format_lambda_label(name) for name in ordered_cols}
    for idx, name in enumerate(ordered_cols):
        values = proportions[name].values
        rects = ax.bar(
            pivot.index.str.replace("_", " "),
            values,
            bottom=bottom,
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
                fontsize=7,
                color="white",
            )
        bottom += values
        if rects:
            handles.append(rects[0])
            labels.append(formatted.get(name, name))
    ax.set_ylabel("Fraction of episodes")
    ax.set_ylim(0, 1)
    legend_cols = min(4, max(1, len(handles)))
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=legend_cols,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08, hspace=0.02)
    output_path = output_dir / "lambda_mix_per_split.png"
    fig.savefig(output_path)
    plt.close(fig)
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
        plot_split_sizes,
        lambda df_, out: plot_family_mix(df_, out, summary_path=out / "family_mix_per_split.json"),
        plot_lambda_mix,
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
