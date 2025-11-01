#!/usr/bin/env python3
"""Helper script to sample hybrid CA episodes and render an unrolled visual."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

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
        default=4,
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


def generate_episodes(args: argparse.Namespace) -> Optional[dict]:
    rng = random.Random(args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    target_record: Optional[dict] = None

    with args.output_jsonl.open("w", encoding="utf-8") as sink:
        for idx in range(args.count):
            record = sample_task(
                rng,
                train_examples=args.train_examples,
                construction=args.construction,
                unroll_tau_max=args.tau_max,
            )
            if idx == args.plot_index:
                target_record = record
            sink.write(json.dumps(record) + "\n")

    print(f"Wrote {args.count} episodes to {args.output_jsonl}")
    return target_record


def render_episode(record: dict, args: argparse.Namespace) -> None:
    fig = show_episode_card(record, tau_max=args.plot_tau_max)
    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved episode visual to {args.plot_path}")


def main() -> None:
    args = parse_args()
    record = generate_episodes(args)
    if args.skip_plot:
        return
    if record is None:
        raise ValueError(
            f"plot-index {args.plot_index} is out of range for count {args.count}"
        )
    render_episode(record, args)


if __name__ == "__main__":
    main()
