#!/usr/bin/env python3
"""
Generate a large, lightly-constrained pool of CA episodes in multiple shards,
mixing families, constructions, and coverage modes. Produces paired *.jsonl and *_meta.jsonl files.

Requires: this repo's package `cellarc` importable; tqdm optional.
"""

from __future__ import annotations
import argparse, math, random
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from cellarc.generation.dataset import generate_dataset_jsonl

# Nearly uniform family mix (we’ll also do single-family shards to guarantee coverage)
UNIFORM_FAMILIES: Dict[str, float] = {
    "random": 1, "totalistic": 1, "outer_totalistic": 1, "outer_inner_totalistic": 1,
    "threshold": 1, "linear_mod_k": 1
} 
# avoiding "cyclic_excitable": 1, "permuted_totalistic": 1 for redundancy 

def single_family(name: str) -> Dict[str, float]:
    return {name: 1.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/pool"),
                    help="Where to write shards (default: artifacts/pool)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--per-shard", type=int, default=10000,
                    help="episodes per shard")
    ap.add_argument("--shards", type=int, default=9,
                    help="how many mixed shards with near-uniform families")
    ap.add_argument("--family-bonus", type=int, default=8,
                    help="extra single-family shards to guarantee each family is well-represented")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--max-radius", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--train-examples", type=int, default=5)
    ap.add_argument("--avg-train-len", type=int, default=48)
    ap.add_argument("--balance-by", choices=["lambda","entropy","all"], default="all")
    ap.add_argument("--unique-by", choices=["tstep","rule"], default="tstep")
    ap.add_argument("--constructions", nargs="+",
                    default=["cycle","unrolled","hybrid"])
    ap.add_argument("--coverage-modes", nargs="+",
                    default=["chunked","uniform"])
    ap.add_argument("--sample-timeout", type=float, default=None,
                    help="Maximum seconds allowed per sampled episode (default: no limit)")
    ap.add_argument("--max-attempts-per-item", type=int, default=200,
                    help="Attempt budget multiplier per episode before aborting")
    ap.add_argument("--coverage-min", type=float, default=0.9,
                    help="Lower bound for sampling coverage fractions (default: 0.9).")
    ap.add_argument("--coverage-max", type=float, default=1.0,
                    help="Upper bound for sampling coverage fractions (default: 1.0).")
    ap.add_argument(
        "--allow-query-outside-coverage",
        dest="query_within_coverage",
        action="store_false",
        help="Permit query windows to fall outside the coverage partitions.",
    )
    ap.add_argument(
        "--query-within-coverage",
        dest="query_within_coverage",
        action="store_true",
        help="Force query windows to be sampled from covered regions (default).",
    )
    ap.set_defaults(query_within_coverage=True)
    ap.add_argument(
        "--include-rule-table",
        action="store_true",
        help="Serialize rule tables alongside each episode (default: skip for smaller shards)",
    )
    ap.add_argument("--no-complexity", action="store_true",
                    help="Skip average lambda/entropy computations for faster sampling")
    ap.add_argument("--no-morphology", action="store_true",
                    help="Skip morphology annotations for faster sampling")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    seen = set()

    coverage_min = min(args.coverage_min, args.coverage_max)
    coverage_max = max(args.coverage_min, args.coverage_max)
    if not (0.0 < coverage_min <= 1.0) or not (0.0 < coverage_max <= 1.0):
        raise ValueError("coverage bounds must lie in (0, 1].")
    if coverage_min > coverage_max:
        coverage_min, coverage_max = coverage_max, coverage_min

    def coverage_sampler(local_rng: random.Random) -> float:
        if math.isclose(coverage_min, coverage_max):
            return float(coverage_max)
        return float(local_rng.uniform(coverage_min, coverage_max))

    def run_shard(tag: str,
                  *,
                  family_mix: Optional[Dict[str,float]] = None,
                  construction: str = "cycle",
                  coverage_mode: str = "chunked",
                  idx: int = 0):
        base = f"{tag}_{construction}_{coverage_mode}_{idx:02d}"
        core = args.outdir / f"{base}.jsonl"
        meta = args.outdir / f"{base}_meta.jsonl"
        print(f"[make_pool] writing {core.name} / {meta.name}")
        stats = generate_dataset_jsonl(
            core,
            count=args.per_shard,
            seed=args.seed + hash(base) % (1<<30),
            meta_path=meta,
            k_range=(args.k_min, args.k_max),
            max_radius=args.max_radius,
            max_steps=args.max_steps,
            train_examples=args.train_examples,
            target_avg_train_len=args.avg_train_len,
            family_mix=family_mix,
            unique_by=args.unique_by,
            balance_by=args.balance_by,
            coverage_fraction=coverage_sampler,          # callable!
            coverage_mode=coverage_mode,
            cap_lambda=None,
            cap_entropy=None,
            compute_complexity=not args.no_complexity,
            annotate_morphology=not args.no_morphology,
            query_within_coverage=args.query_within_coverage,
            construction=construction,
            unroll_tau_max=16,
            seen_fingerprints=seen,
            dataset_version="pool_v1",
            show_progress=True,
            progress_desc=base,
            max_attempts_per_item=args.max_attempts_per_item,
            sample_timeout=args.sample_timeout,
            include_rule_table=args.include_rule_table,
        )
        print(f"[make_pool] {base}: wrote {stats['written']} episodes")

    # Mixed shards
    total_mixed = args.shards * len(args.constructions) * len(args.coverage_modes)
    with tqdm(total=total_mixed, desc="Mixed shards", unit="shard") as pbar:
        for i in range(args.shards):
            for cons in args.constructions:
                for covm in args.coverage_modes:
                    run_shard("mixed", family_mix=UNIFORM_FAMILIES,
                              construction=cons, coverage_mode=covm, idx=i)
                    pbar.update(1)

    # Single-family shards (one per family to guarantee depth)
    fams = list(UNIFORM_FAMILIES.keys())
    total_family = len(fams) * len(args.constructions)
    with tqdm(total=total_family, desc="Family shards", unit="shard") as pbar:
        for j, fam in enumerate(fams):
            for cons in args.constructions:
                run_shard(f"fam_{fam}", family_mix=single_family(fam),
                          construction=cons, coverage_mode="chunked", idx=j)
                pbar.update(1)

if __name__ == "__main__":
    main()
