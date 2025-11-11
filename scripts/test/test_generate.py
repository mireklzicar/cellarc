#!/usr/bin/env python3
"""Helper script to sample hybrid CA episodes and render an unrolled visual."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

_RUNNING_UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST")
_GENERATION_EXTRA_MESSAGE = (
    "cellarc generation helpers require the `cax` dependency, which only ships "
    "wheels for Python 3.11+. Install with `pip install cellarc[all]` from a "
    "Python 3.11+ environment."
)

try:
    import cax  # type: ignore  # noqa: F401
except ModuleNotFoundError as exc:
    if _RUNNING_UNDER_PYTEST:
        import pytest

        pytest.skip(_GENERATION_EXTRA_MESSAGE, allow_module_level=True)
    raise ModuleNotFoundError(_GENERATION_EXTRA_MESSAGE) from exc

from cellarc.generation import sample_task
from cellarc.visualization import show_episode_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of CA episodes to generate (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20240317,
        help="RNG seed for reproducibility (default: 20240317).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("artifacts/hybrid_episodes.jsonl"),
        help="Destination JSONL for generated episodes (default: artifacts/hybrid_episodes.jsonl).",
    )
    parser.add_argument(
        "--train-examples",
        type=int,
        default=5,
        help="Training pair count per episode passed to sample_task (default: 4).",
    )
    parser.add_argument(
        "--construction",
        choices=["cycle", "unrolled", "hybrid"],
        default="hybrid",
        help="Construction mode for sampling (default: hybrid).",
    )
    parser.add_argument(
        "--tau-max",
        type=int,
        default=32,
        help="Maximum tau depth when unrolling episodes (default: 32).",
    )
    parser.add_argument(
        "--plot-index",
        type=int,
        default=0,
        help="Index of the sampled episode to visualise (default: 0).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("artifacts/hybrid_episode0.png"),
        help="Output path for the rendered episode card (default: artifacts/hybrid_episode0.png).",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Render a visual for every sampled episode using numbered filenames.",
    )
    parser.add_argument(
        "--plot-tau-max",
        type=int,
        default=48,
        help="Temporal depth used when rendering the episode card (default: 48).",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip rendering the episode card even if plot-index is valid.",
    )
    return parser.parse_args()


def generate_episodes(
    args: argparse.Namespace, *, collect_records: bool = True
) -> List[Tuple[int, dict]]:
    rng = random.Random(args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records_to_plot: List[Tuple[int, dict]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as sink:
        for idx in range(args.count):
            record = sample_task(
                rng,
                train_examples=args.train_examples,
                construction=args.construction,
                unroll_tau_max=args.tau_max,
            )
            sink.write(json.dumps(record) + "\n")
            if not collect_records:
                continue
            if args.plot_all:
                records_to_plot.append((idx, record))
            elif idx == args.plot_index:
                records_to_plot.append((idx, record))

    print(f"Wrote {args.count} episodes to {args.output_jsonl}")
    if collect_records and not args.plot_all and not records_to_plot:
        raise ValueError(
            f"plot-index {args.plot_index} is out of range for count {args.count}"
        )
    return records_to_plot


def render_episode(
    record: dict, args: argparse.Namespace, *, episode_index: Optional[int] = None
) -> None:
    fig = show_episode_card(record, tau_max=args.plot_tau_max)
    suffix = args.plot_path.suffix or ".png"
    if args.plot_all and episode_index is not None:
        target_path = args.plot_path.with_name(
            f"{args.plot_path.stem}_{episode_index:04d}{suffix}"
        )
    else:
        target_path = args.plot_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved episode visual to {target_path}")


def main() -> None:
    args = parse_args()
    records = generate_episodes(args, collect_records=not args.skip_plot)
    if args.skip_plot:
        return
    for idx, record in records:
        render_episode(record, args, episode_index=idx)


if __name__ == "__main__":
    main()
