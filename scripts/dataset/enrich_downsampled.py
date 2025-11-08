#!/usr/bin/env python3
"""
Reconstruct rule tables and annotate complexity/morphology for downsampled pools.

Reads the `downsampled.jsonl` and `downsampled_meta.jsonl` files, reconstructs the
rule table payloads using the original episode seed, computes average cell entropy,
mutual information, and morphology descriptors, and writes enriched JSONL shards.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

from cellarc.generation.cax_runner import AutomatonRunner
from cellarc.generation.metrics import average_cell_entropy, average_mutual_information
from cellarc.generation.morphology import quick_morphology_features
from cellarc.generation.sampling import entropy_bin
from cellarc.generation.serialization import deserialize_rule_table
from cellarc.generation.reconstruction import (
    infer_dataset_config,
    reconstruct_rule_table_payload,
)

COMPLEXITY_ROLLOUT: Tuple[int, int] = (30, 256)


def _load_meta(path: Path) -> Dict[str, Dict[str, object]]:
    meta_map: Dict[str, Dict[str, object]] = {}
    with path.open() as fh:
        for line in fh:
            entry = json.loads(line)
            meta_map[str(entry["fingerprint"])] = entry["meta"]
    return meta_map


def enrich_downsampled(
    records_path: Path,
    meta_path: Path,
    out_records: Path,
    out_meta: Path,
) -> None:
    meta_by_fp = _load_meta(meta_path)
    width, horizon = COMPLEXITY_ROLLOUT

    written = 0
    with records_path.open() as records_fh, out_records.open("w") as rec_out, out_meta.open("w") as meta_out:
        for line in records_fh:
            record = json.loads(line)
            meta_stub = record.get("meta") or {}
            fingerprint = str(meta_stub.get("fingerprint"))
            if fingerprint not in meta_by_fp:
                raise KeyError(f"Fingerprint {fingerprint} missing from meta shard.")

            meta_full = meta_by_fp[fingerprint]
            config = infer_dataset_config(meta_full)
            if config is None:
                raise ValueError(f"No reconstruction config available for dataset_version={meta_full.get('dataset_version')!r}.")

            payload = meta_full.get("rule_table")
            if not isinstance(payload, dict):
                payload = reconstruct_rule_table_payload(meta_full, config=config)

            dense = deserialize_rule_table(payload)
            k = int(meta_full["alphabet_size"])
            r = int(meta_full["radius"])
            t = int(meta_full["steps"])
            episode_seed = int(meta_full["episode_seed"])

            episode_rng = random.Random(episode_seed)
            random_init = [episode_rng.randrange(k) for _ in range(width)]
            runner = AutomatonRunner(
                alphabet_size=k,
                radius=r,
                table=dense,
                rng_seed=episode_rng.randrange(1 << 30),
            )
            history = runner.evolve(random_init, timesteps=horizon, return_history=True)

            avg_entropy = float(average_cell_entropy(history))
            ami_1 = float(average_mutual_information(history, temporal_distance=1))
            morph_rng = random.Random(episode_seed)
            morphology = quick_morphology_features(
                dense,
                k,
                r,
                t,
                width=width,
                horizon=horizon,
                rng=morph_rng,
            )

            meta_full.update({
                "schema_version": "1.0.2",
                "avg_cell_entropy": avg_entropy,
                "entropy_bin": entropy_bin(avg_entropy),
                "avg_mutual_information_d1": ami_1,
                "morphology": morphology,
                "rule_table": payload,
            })

            enriched_record = {
                "train": record["train"],
                "query": record["query"],
                "solution": record["solution"],
                "meta": meta_full,
                "rule_table": meta_full["rule_table"],
            }
            rec_out.write(json.dumps(enriched_record, separators=(",", ":")) + "\n")

            meta_out.write(
                json.dumps(
                    {"fingerprint": fingerprint, "meta": meta_full},
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1

    print(f"Enriched {written} records → {out_records.name}, {out_meta.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich downsampled pool with rule tables and metrics.")
    ap.add_argument("--input", type=Path, default=Path("artifacts/pool_downsampled/downsampled.jsonl"))
    ap.add_argument("--input-meta", type=Path, default=Path("artifacts/pool_downsampled/downsampled_meta.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/pool_downsampled/downsampled_enriched.jsonl"))
    ap.add_argument("--output-meta", type=Path, default=Path("artifacts/pool_downsampled/downsampled_enriched_meta.jsonl"))
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enrich_downsampled(args.input, args.input_meta, args.output, args.output_meta)


if __name__ == "__main__":
    main()
