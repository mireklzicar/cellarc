# CellARc Dataset Creation Pipeline

This document is the authoritative record for reproducing the datasets that ship on
the Hugging Face Hub (`mireklzicar/cellarc_100k(_meta)`). Local exports are staged
under `${CELLARC_HOME:-~/.cache/cellarc}/exports`, which mirrors the Hub layout
and can be synced with `huggingface_hub`. All commands are meant to run from the
repository root with Python 3.11 and the dependencies listed in
`requirements.txt`.

The current release contains **103 000** enriched cellular automaton (CA)
episodes (train 100 000, validation 1 000, test_interpolation 1 000,
test_extrapolation 1 000). You can verify the final artefacts via:

- `${CELLARC_HOME:-~/.cache/cellarc}/exports/cellarc_100k_meta/data_files.json`
  – split sizes and byte counts.
- `${CELLARC_HOME:-~/.cache/cellarc}/exports/cellarc_100k_meta/dataset_stats.json`
  – aggregate metrics. Notably, `test_extrapolation` has the **lowest**
  `coverage_fraction` (approx. 0.196), the **highest** Langton lambda (approx.
  0.582) and entropy (approx. 1.30), matching the intended extrapolation regime.

---

## Command Checklist

```
# 0. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1-6. Pool generation, filtering, enrichment (can be run via processing_pipeline.py)
python scripts/make_pool.py ...
python scripts/pool_stats.py ...
python scripts/pool_filtering.py ...
python scripts/pool_sanity_check.py ...
python scripts/pool_stats.py ...
python scripts/enrich_downsampled.py ...
python scripts/pool_sanity_check.py ...

# 7. Coverage-focused resampling
python scripts/resample_highcov.py ...

# 8. Coverage/lambda-aware splitting
python scripts/split_pool.py ...

# 9. Packaging
python scripts/build_hf_dataset.py ...                             # writes ${CELLARC_HOME:-~/.cache/cellarc}/exports/*
python scripts/build_hf_dataset.py --target-root artifacts/hf_cellarc ...
``` 

The remainder of this document expands every step, explains why it exists, and records the exact arguments used when producing the published dataset.

---

## 0. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONHASHSEED=0     # guards deterministic fingerprints
```

---

## 1. Raw pool generation -- `scripts/make_pool.py`

Command used for the 2024 release:

```bash
python scripts/make_pool.py \
  --outdir artifacts/pool \
  --seed 12345 \
  --per-shard 3000 \
  --shards 3 \
  --train-examples 5 \
  --avg-train-len 64 \
  --constructions cycle unrolled hybrid \
  --coverage-modes chunked uniform \
  --coverage-min 0.05 \
  --coverage-max 0.95 \
  --k-min 2 --k-max 6 \
  --max-radius 3 \
  --max-steps 4 \
  --max-attempts-per-item 200
```

This produces 36 mixed shards plus 18 single-family shards (approx. 5.1 x 10^5 episodes). Coverage fractions are sampled uniformly within `[0.05, 0.95]`, which ultimately yields the 0.07-0.94 coverage span reported in `dataset_stats.json`.

---

## 2. Raw pool diagnostics -- `scripts/pool_stats.py`

```bash
python scripts/pool_stats.py artifacts/pool --outdir artifacts/processing/pool_stats
```

The summary ensures we actually generated the intended family mix and coverage histogram before committing compute to filtering.

---

## 3. Filtering & downsampling -- `scripts/pool_filtering.py`

```bash
python scripts/pool_filtering.py artifacts/pool \
  --outdir artifacts/processing/pool_downsampled \
  --target 110000 \
  --max-flattened-length 256 \
  --bins 100
```

Filtering enforces:
- fingerprint-level deduplication;
- novel solution relative to query/train outputs;
- flattened supervision length <= 256;
- balanced sampling across a 2-D histogram of `(lambda_bin, observed_coverage_fraction)`.

The output (`downsampled*.jsonl`) contains ~110 k high-quality episodes that still cover sparse regimes.

---

## 4. Sanity check -- `scripts/pool_sanity_check.py`

```bash
python scripts/pool_sanity_check.py \
  --input artifacts/processing/pool_downsampled/downsampled.jsonl \
  --meta artifacts/processing/pool_downsampled/downsampled_meta.jsonl
```

Guards against duplicate fingerprints/probe fingerprints and filters pathological absorbing rules (lambda < 0.02 **and** entropy < 0.02) before the expensive enrichment step.

---

## 5. Downsampled diagnostics -- `scripts/pool_stats.py`

```bash
python scripts/pool_stats.py artifacts/processing/pool_downsampled \
  --outdir artifacts/processing/pool_downsampled/stats
```

Useful for plotting the post-filter histogram and comparing against the raw pool. These statistics informed the write-up in `PLOTS.md`.

---

## 6. Metadata enrichment -- `scripts/enrich_downsampled.py`

```bash
python scripts/enrich_downsampled.py \
  --input artifacts/processing/pool_downsampled/downsampled.jsonl \
  --input-meta artifacts/processing/pool_downsampled/downsampled_meta.jsonl \
  --output artifacts/processing/pool_downsampled/downsampled_enriched.jsonl \
  --output-meta artifacts/processing/pool_downsampled/downsampled_enriched_meta.jsonl
```

Each episode is replayed (width 30, horizon 256) to compute:
- Langton lambda & entropy bins (promoted to schema `1.0.2`);
- mutual information and morphology descriptors;
- inline rule tables for reproducibility.

Run `pool_sanity_check.py` again on the enriched pair to confirm the reconstruction step did not introduce anomalies.

> **Automation tip:** Steps 1-6 above are scripted inside `python scripts/processing_pipeline.py --processing-root artifacts/processing --seed 12345`. We still keep the manual commands here because `processing_pipeline` intentionally stops before coverage resampling so that we can tweak those parameters independently.

---

## 7. Coverage-aware resampling -- `scripts/resample_highcov.py`

Goal: rewrite training spans so that the train windows cover the query windows with weighted coverage >= 0.5 while avoiding leakage (query windows never copied verbatim into train).

```bash
python scripts/resample_highcov.py \
  --input artifacts/processing/pool_downsampled/downsampled_enriched.jsonl \
  --output artifacts/processing/resampled_highcov/downsampled_enriched.jsonl \
  --output-meta artifacts/processing/resampled_highcov/downsampled_enriched_meta.jsonl \
  --target-coverage 0.5 \
  --max-attempts 200 \
  --unroll-tau-max 16 \
  --overwrite
```

Roughly 95 k-100 k episodes pass this constraint. Episodes that cannot hit the coverage target are skipped, which is why we keep the original enriched shard around as a fallback.

---

## 8. Coverage/lambda-driven splitting -- `scripts/split_pool.py`

This stage combines two objectives:
1. Keep only episodes that still satisfy the resampling constraint (coverage threshold >= 0.5).
2. Assign the **lowest coverage / highest lambda / highest entropy** episodes to `test_extrapolation` (see `dataset_stats.json['splits']`).

```bash
python scripts/split_pool.py \
  --input artifacts/processing/resampled_highcov/downsampled_enriched.jsonl \
  --output-dir artifacts/processing/resampled_highcov/splits \
  --train-count 100000 \
  --val-count 1000 \
  --test-interp-count 1000 \
  --test-extra-count 1000 \
  --coverage-threshold 0.5 \
  --seed 12345
```

- Episodes below the threshold are discarded.
- Remaining episodes are ranked by the average of three normalized scores: increasing coverage rank (lower is better), decreasing lambda rank, decreasing entropy rank.
- The lowest-ranked 1 000 episodes become `test_extrapolation`. The rest are shuffled (seed 12345) before slicing into train/val/test_interpolation.

Any leftovers beyond the requested split sizes are written to `unused.jsonl` for auditing.

---

## 9. Packaging -- `scripts/build_hf_dataset.py`

We run the exporter twice: once for the local
`${CELLARC_HOME:-~/.cache/cellarc}/exports/*` tree (used internally) and once
for the staging tree we sync with the Hugging Face Hub.

```bash
# Local artefacts used throughout the repo
python scripts/build_hf_dataset.py \
  --source-dir artifacts/processing/resampled_highcov/splits \
  --target-root "${CELLARC_HOME:-~/.cache/cellarc}/exports" \
  --dataset-name cellarc_100k \
  --extended-suffix _meta \
  --chunk-size 1000 \
  --parquet-compression snappy \
  --overwrite

# Mirror for HF uploads (`huggingface-cli upload` picks files from artifacts/hf_cellarc/*)
python scripts/build_hf_dataset.py \
  --source-dir artifacts/processing/resampled_highcov/splits \
  --target-root artifacts/hf_cellarc \
  --dataset-name hf-cellarc_100k \
  --extended-suffix _meta \
  --chunk-size 1000 \
  --parquet-compression snappy \
  --overwrite
```

Each invocation emits **two** directories:
- `.../cellarc_100k` - lightweight supervision-only JSONL/Parquet (ids + train/query/solution).
- `.../cellarc_100k_meta` - full metadata JSONL with the same Parquet views.

---

## 10. Verification

After packaging, double-check:

1. `${CELLARC_HOME:-~/.cache/cellarc}/exports/cellarc_100k_meta/data_files.json`
   - train: 100 000 records, JSONL size approx. 288 MB.
   - val/test_interpolation/test_extrapolation: 1 000 records each.

2. `${CELLARC_HOME:-~/.cache/cellarc}/exports/cellarc_100k_meta/dataset_stats.json`
   - Global lambda mean approx. 0.563 (min 0.0156, max 1.0).
   - Coverage fraction mean approx. 0.373 (min 0.069, max 0.938).
   - Train sample length mean approx. 11.83 with exactly five training examples per episode.
   - `test_extrapolation` block reports coverage mean approx. 0.196, lambda mean approx. 0.582, entropy mean approx. 1.303, confirming the intended distribution shift.

3. Randomly open `data/*.jsonl` and ensure each record includes `meta.schema_version == "1.0.2"` and `rule_table.format_version == "1.0.2"`.

---

## Optional: processing pipeline wrapper

`processing_pipeline.py` automates steps 1-6 (pool generation through enrichment) and can be rerun whenever we want to refresh the pool:

```bash
python scripts/processing_pipeline.py --processing-root artifacts/processing --seed 12345
```

We intentionally stop the pipeline before resampling/splitting so we can tune `resample_highcov.py` / `split_pool.py` without editing the orchestrator.

---

## Scripts outside the core pipeline

To reduce maintenance surface area:

| Script | Status | Notes |
|--------|--------|-------|
| `scripts/resplit_dataset.py` | Optional | Handy when we need to reshuffle existing packaged splits, but **not** part of the canonical build. Keep but mark as maintenance tooling. |
| `scripts/plot_dataset_stats.py`, `scripts/plot_episode_cards.py` | Ancillary | Purely for visualization; safe to move under a `tools/` or `deprecated/` namespace. |
| `scripts/evaluate_solver.py`, `scripts/solver_analysis.py` | Solver-only | Unrelated to dataset creation; consider relocating next to evaluation notebooks. |

Everything else listed above is required for reproduction. Removing or renaming the ancillary scripts will prevent future confusion about what must be run to rebuild `cellarc_100k`.

---

By following the commands and verification steps in this document you can recreate
the export tree under `${CELLARC_HOME:-~/.cache/cellarc}/exports` and the mirror
in `artifacts/hf_cellarc`, matching the statistics recorded alongside the
release.
