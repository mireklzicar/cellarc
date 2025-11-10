#!/usr/bin/env python3
"""End-to-end processing pipeline for CA pool generation and splitting."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import os
from pathlib import Path

CACHE_HOME = Path(os.getenv("CELLARC_HOME", Path.home() / ".cache" / "cellarc"))
EXPORT_ROOT = CACHE_HOME / "exports"


def run(cmd, *, cwd: Path | None = None) -> None:
    printable = " ".join(cmd)
    print(f"[pipeline] Running: {printable}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError as exc:
        print(f"[pipeline] Command failed: {printable}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc


def prompt_overwrite(path: Path) -> bool:
    reply = input(f"{path} already exists. Overwrite? [y/N] ").strip().lower()
    return reply in {"y", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-make-pool", action="store_true", help="Skip pool generation step.")
    parser.add_argument("--processing-root", type=Path, default=Path("artifacts/processing"))
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    root = args.processing_root
    root.mkdir(parents=True, exist_ok=True)

    pool_dir = Path("artifacts/pool")
    pool_stats_dir = root / "pool_stats"
    downsampled_dir = root / "pool_downsampled"
    enriched_path = downsampled_dir / "downsampled_enriched.jsonl"

    # Step 1: make_pool
    if not args.skip_make_pool:
        if pool_dir.exists():
            if not prompt_overwrite(pool_dir):
                print("[pipeline] Reusing existing pool directory.")
            else:
                print("[pipeline] Removing existing pool directory.")
                shutil.rmtree(pool_dir)
        run(
            [
                sys.executable,
                "scripts/make_pool.py",
                "--outdir",
                str(pool_dir),
                "--per-shard",
                "3000",
                "--shards",
                "3",
                "--family-bonus",
                "8",
                "--train-examples",
                "5",
                "--avg-train-len",
                "64",
                "--constructions",
                "cycle",
                "unrolled",
                "hybrid",
                "--coverage-modes",
                "chunked",
                "uniform",
                "--coverage-min",
                "0.95",
                "--coverage-max",
                "1.0",
                "--query-within-coverage",
                "--max-attempts-per-item",
                "20",
            ]
        )

    # Step 2: stats on raw pool
    pool_stats_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "scripts/pool_stats.py",
            str(pool_dir),
            "--outdir",
            str(pool_stats_dir),
        ]
    )

    # Step 3: filtering/downsampling
    if downsampled_dir.exists():
        shutil.rmtree(downsampled_dir)
    run(
        [
            sys.executable,
            "scripts/pool_filtering.py",
            str(pool_dir),
            "--outdir",
            str(downsampled_dir),
        ]
    )

    # Step 4: sanity check
    run(
        [
            sys.executable,
            "scripts/pool_sanity_check.py",
            "--input",
            str(downsampled_dir / "downsampled.jsonl"),
            "--meta",
            str(downsampled_dir / "downsampled_meta.jsonl"),
        ]
    )

    # Step 5: stats on downsampled pool
    run(
        [
            sys.executable,
            "scripts/pool_stats.py",
            str(downsampled_dir),
            "--outdir",
            str(downsampled_dir / "stats"),
        ]
    )

    # Step 6: enrichment
    downsampled_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "scripts/enrich_downsampled.py",
            "--input",
            str(downsampled_dir / "downsampled.jsonl"),
            "--input-meta",
            str(downsampled_dir / "downsampled_meta.jsonl"),
            "--output",
            str(enriched_path),
            "--output-meta",
            str(downsampled_dir / "downsampled_enriched_meta.jsonl"),
        ]
    )

    # Step 6b: sanity check enriched shard
    run(
        [
            sys.executable,
            "scripts/pool_sanity_check.py",
            "--input",
            str(enriched_path),
            "--meta",
            str(downsampled_dir / "downsampled_enriched_meta.jsonl"),
        ]
    )

    # Step 7: split
    run(
        [
            sys.executable,
            "scripts/split_pool.py",
            "--input",
            str(enriched_path),
            "--output-dir",
            str(downsampled_dir / "splits"),
            "--train-count",
            "9000",
            "--val-count",
            "500",
            "--test-interp-count",
            "500",
            "--test-extra-count",
            "1000",
        ]
    )

    # Step 8: build HF-friendly dataset packages
    run(
        [
            sys.executable,
            "scripts/build_hf_dataset.py",
            "--source-dir",
            str(downsampled_dir / "splits"),
            "--target-root",
            str(EXPORT_ROOT),
            "--dataset-name",
            "cellarc_highcov",
            "--extended-suffix",
            "_meta",
            "--overwrite",
        ]
    )

    print("[pipeline] Processing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
