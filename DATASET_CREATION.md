# CellARc Dataset Creation Pipeline

This document captures the end-to-end process used to generate the CellARc cellular automata (CA) datasets that ship in `artifacts/datasets`. It is intended as the authoritative reference when writing the accompanying paper or reproducing the corpora.

The canonical run invokes the orchestrator:

```bash
python scripts/processing_pipeline.py --processing-root artifacts/processing --seed 12345
```

Unless stated otherwise, all commands use Python 3.11, run from the repository root, and inherit the default arguments hard-coded in the scripts.

## High-level flow
- **Pool generation (`make_pool.py`)** – sample ~5.1e5 lightly constrained CA episodes across constructions, families, and coverage regimes.
- **Raw diagnostics (`pool_stats.py`)** – profile the unfiltered pool for QA.
- **Filtering & downsampling (`pool_filtering.py`)** – deduplicate, remove pathological episodes, and stratify down to 110 k balanced samples.
- **Sanity checks (`pool_sanity_check.py`)** – assert fingerprint/probe uniqueness and exclude degenerate regimes.
- **Enrichment (`enrich_downsampled.py`)** – reconstruct rule tables and compute complexity/morphology metrics for the curated set.
- **Splitting (`split_pool.py`)** – carve train/val/test_interpolation/test_extrapolation splits with controlled distribution shift.
- **Packaging (`build_hf_dataset.py`)** – export Hugging Face-compatible releases: `cellarc_100k` (lightweight) and `cellarc_100k_meta` (full metadata).

Intermediate artefacts live under `artifacts/processing`, while the final datasets are in `artifacts/datasets`.

---

## 1. Raw pool generation – `scripts/make_pool.py`

### Configuration used in the pipeline

| parameter | value | rationale |
|-----------|-------|-----------|
| `--outdir` | `artifacts/pool` | Writable stash for raw shards. |
| `--per-shard` | 10 000 | Produces manageable shard sizes for downstream parallelism. |
| `--shards` | 6 | Mixed shards per construction/coverage combo (see below). |
| `--family-bonus` | 8 | Ensures depth for each CA family via single-family shards. |
| `--k-min`, `--k-max` | 2, 6 | Alphabet sizes 2–6 to span binary through hexary rules. |
| `--max-radius` | 3 | Covers Moore neighborhoods up to radius 3. |
| `--max-steps` | 4 | Allows up to four unrolled steps during sampling. |
| `--train-examples` | 5 | Fixed five-shot supervision. |
| `--avg-train-len` | 64 | Nudges sampling toward wider windows than the downstream cap (len ≤ 21 after trimming). |
| `--constructions` | `cycle`, `unrolled`, `hybrid` | Mix of curriculum styles. |
| `--coverage-modes` | `chunked`, `uniform` | Alternates between contiguous tilings and uniform draws. |
| `--sample-timeout` | 0.1 s | Abort expensive samplers quickly. |
| `--max-attempts-per-item` | 20 | Controls stochastic retries per episode. |
| `--no-complexity`, `--no-morphology` | set | Skip heavy metrics here; they are recomputed post-filtering. |
| `--seed` | 12345 | Global RNG root to make shards reproducible. |

Other defaults: `k_range=(2,6)`, `unique_by="tstep"`, `balance_by="all"`, `include_rule_table=False`. Duplicate avoidance shares a `seen_fingerprints` set across shards so each sampled fingerprint is unique globally.

### Coverage sampling

`make_pool` draws per-episode coverage fractions via:

```
cov_sampler(u) = {
  U(0.05, 0.25) if u < 0.50          # emphasis on low coverage (hard regime)
  U(0.25, 0.60) if 0.50 ≤ u < 0.85   # mid coverage
  U(0.80, 0.90) otherwise            # thin tail of nearly complete coverage
}
```

This skews the raw pool toward sparse observations (≈50 %), while still reserving capacity for mid/high coverage to support interpolation tests.

### Shard layout

- **Mixed shards**: `6 (seeds) × 3 (constructions) × 2 (coverage_modes) = 36` shards, each 10 000 episodes, with the nearly uniform family mixture:
  ```
  {random, totalistic, outer_totalistic, outer_inner_totalistic, threshold, linear_mod_k}
  ```
- **Family shards**: `6 families × 3 constructions = 18` shards, each single-family, totalling 54 shards.

The canonical run produced 510 354 serialized episodes. Deduplication takes place later; here we simply write paired `*.jsonl` and `*_meta.jsonl` files.

### Output

Raw shards accumulate under `artifacts/pool/`. Each JSONL line stores the supervision payload (`train`, `query`, `solution`, minimal `meta`), and the companion meta file records the full sampling metadata (coverage stats, fingerprints, seeds, etc.).

---

## 2. Raw pool diagnostics – `scripts/pool_stats.py`

We profile the unfiltered shards to confirm coverage of the intended regimes. The pipeline stores the report in `artifacts/processing/pool_stats/summary.json`.

Key figures (after enforcing novel solutions and `flattened_length ≤ 256` just for measurement):

- **Total serialized**: 510 354 episodes (all unique thanks to generation-time guarding).
- **Novel & length-compliant**: 329 034 episodes (64.3 % of the raw pool).
- **Flattened footprint**: median 120 cells, mean 126.4, max 252.
- **Langton λ bins**: chaotic 60.5 %, edge 37.5 %, ordered 2.0 %.
- **Coverage**: observed fractions span ≈0.02–0.96 with heavy mass in 0.05–0.4.

The same directory carries `novel_fingerprints.txt` with the fingerprints that pass the novelty filter, enabling downstream reproducibility checks.

---

## 3. Filtering and downsampling – `scripts/pool_filtering.py`

### Filters applied
1. **Deduplication** by fingerprint, falling back to a canonical JSON hash.
2. **Novel solution** – the held-out solution must differ from the query and every train output.
3. **No identity pairs** – removes degenerate train examples with identical input/output.
4. **No duplicate train pairs** – ensures support set diversity.
5. **Flattened length ≤ 256** – caps total scalar elements across train/query/solution.
6. **Complete metadata** – requires `meta.lambda`, `meta.coverage.observed_fraction`, and `meta.fingerprint`.

Episodes failing any criterion are discarded before downsampling.

### Stratified sampling

- **Target**: 110 000 episodes (provides slack for later splits).
- **Strata**: 2D grid over Langton λ (100 bins) × observed coverage fraction (100 bins).
- **Sampler**: even quota per populated bin, then round-robin fill, seeded with `--seed 12345`.

Because the filtered pool is much larger than the target (329 034 > 110 000), stratification preserves broad support across λ and coverage regimes without over-representing dense clusters.

### Outputs

`artifacts/processing/pool_downsampled` contains:

- `downsampled.jsonl` / `_meta.jsonl`: 110 000 selected episodes and their provenance.
- `downsampled_fingerprints.txt`: the curated fingerprint list.
- `stats/summary.json`: diagnostics mirroring those gathered on the raw pool.

Downsampled stats:

- Flattened length median 156 (mean 141.9).
- Langton λ bins: chaotic 59.2 %, edge 36.9 %, ordered 3.9 %.
- Coverage fraction mean 0.374 (min 0.069, max 0.938).

---

## 4. Sanity checks – `scripts/pool_sanity_check.py`

Before enrichment, we assert integrity of the curated shard:

- **Uniqueness** – no duplicate `fingerprint` or `probe_fingerprint`.
- **Pathology guard** – rejects episodes that are simultaneously absorbing with λ < 0.02 and entropy < 0.02, or those with fewer than 128 observed windows and coverage < 1e-4.

The check reads both the records and the meta stream; the November 2024 run passed with zero violations.

---

## 5. Metadata enrichment – `scripts/enrich_downsampled.py`

Purpose: restore heavy metadata that was skipped during pool generation to make downstream analysis possible.

Workflow per episode:
1. Infer the original sampling configuration (`infer_dataset_config`) from the meta stub.
2. Reconstruct or reuse the rule table payload (`reconstruct_rule_table_payload`).
3. Deserialize the dense lookup table and re-run the CA with:
   - **Width** 30, **horizon** 256 (see `COMPLEXITY_ROLLOUT`).
   - Deterministic RNG seeded from the episode seed.
4. Compute metrics:
   - Average cell entropy (`avg_cell_entropy`) and entropy bin assignment.
   - Average mutual information at temporal lag 1 (`avg_mutual_information`).
   - Morphology descriptors (`quick_morphology_features`), including absorbing/periodic flags.
5. Write enriched records to `downsampled_enriched.jsonl` and full meta to `downsampled_enriched_meta.jsonl`.

This stage also upgrades the schema version to `1.0.2`, inlines the base64 rule table so the later dataset packages retain full provenance, and emits coverage/morphology descriptors needed for downstream analysis.

---

## 6. Post-enrichment sanity check

We rerun `pool_sanity_check.py` on the enriched shard to ensure that reconstructing rule tables did not introduce duplicates or revive pathological episodes. The same criteria as §4 apply; the enriched run passed.

---

## 7. Coverage-driven resplitting – `scripts/resplit_dataset.py`

To build the 1.0.2 release we collapse the previously balanced splits, recompute key coverage statistics, and carve out a new evaluation regime:

1. Load every enriched episode from the extended dataset.
2. For each episode compute:
   - `query_window_coverage_weighted`
   - `query_window_coverage_unique`
   - `query_window_avg_depth`
   - `coverage_windows` (promoted from the nested coverage block)
   - `ncd_train_query_solution`
3. Sort the merged pool by `query_window_coverage_weighted` (lowest first) and take the first 1 000 episodes as **test_extrapolation**.
4. Shuffle the remaining episodes with RNG seed 12345 and slice:
   - `train`: 100 000
   - `val`: 1 000
   - `test_interpolation`: 1 000

The script outputs refreshed JSONL splits under `artifacts/processing/resplit_splits/`, ready for packaging.

---

## 8. Hugging Face packaging – `scripts/build_hf_dataset.py`

The final step consumes the coverage-aware splits and transforms them into two dataset packages under `artifacts/datasets/`:

- `cellarc_100k/` – lightweight JSONL (id/train/query/solution) plus Parquet mirrors.
- `cellarc_100k_meta/` – identical Parquet files, but JSONL retains full metadata + rule tables.

The exporter buffers 1 000 episodes per Parquet chunk, writes compressed (`snappy`) columns with the schema declared in `features.json`, and collects aggregate statistics in `dataset_stats.json`.

### Final split sizes

| split | episodes | notes |
|-------|----------|-------|
| `train` | 100 000 | Balanced across constructions and families. |
| `val` | 1 000 | Held-out for model selection. |
| `test_interpolation` | 1 000 | Matches train coverage regime. |
| `test_extrapolation` | 1 000 | Low-coverage, high-λ slice for distribution shift evaluation. |

Total episodes packaged: 103 000.

### Global statistics (from `dataset_stats.json`)

- **Alphabet sizes**: counts `[14 334×k=2, 17 548×3, 28 317×4, 25 094×5, 17 707×6]`.
- **Radii**: `[76 310×r=1, 12 423×2, 14 267×3]`.
- **Langton λ**: mean 0.563, min 0.0156, max 1.0.
- **Average cell entropy**: mean 1.197 bits, span 0–2.58.
- **Coverage fraction**: mean 0.373, min 0.069, max 0.938.
- **Episode footprint**: flattened length mean 141.9, median 156, max 252.
- **Train sample length**: mean 11.83 cells (min 5, max 21) with exactly 5 training exemplars per episode.

Per-split statistics are recorded in the same JSON for traceability.

---

## Reproduction checklist

1. **Environment** – install `requirements.txt` (PyArrow, datasets, tqdm, JAX optional).
2. **Pool generation** – `python scripts/make_pool.py` with the parameter block above (already encapsulated by the pipeline).
3. **Diagnostics** – `python scripts/pool_stats.py artifacts/pool --outdir artifacts/processing/pool_stats`.
4. **Filtering** – `python scripts/pool_filtering.py artifacts/pool --outdir artifacts/processing/pool_downsampled`.
5. **Sanity check** – `python scripts/pool_sanity_check.py --input artifacts/processing/pool_downsampled/downsampled.jsonl --meta artifacts/processing/pool_downsampled/downsampled_meta.jsonl`.
6. **Enrichment** – `python scripts/enrich_downsampled.py --input ... --output ...`.
7. **Sanity check (enriched)** – rerun `pool_sanity_check.py` on the enriched pair.
8. **Splitting** – `python scripts/split_pool.py --input artifacts/processing/pool_downsampled/downsampled_enriched.jsonl --output-dir artifacts/processing/pool_downsampled/splits`.
9. **Packaging** – `python scripts/build_hf_dataset.py --overwrite`.

Running `scripts/processing_pipeline.py` performs steps 2–8 automatically (step 1 is optional via `--skip-make-pool`; step 9 is separate because it targets the Hugging Face layout).

---

## Parameter rationale & notes

- **Family mix**: the combination of mixed and single-family shards guarantees depth per family while keeping the global distribution close to uniform. This proved necessary for threshold and linear_mod_k rules, which are rarer under unconstrained sampling.
- **Coverage emphasis**: low coverage episodes are over-sampled when constructing the pool so that the stratified downsampler can still find enough sparse cases after all quality filters. The interpolation split nevertheless mirrors the broader 0.07–0.94 coverage range.
- **Attempt/time budgets**: limiting per-episode sampling to 20 retries or 0.1 s avoids pathological rules that would otherwise stall generation; failed attempts simply contribute to the larger pool size.
- **Deferred complexity metrics**: computing entropy/morphology during pool generation would slow down shard writing by ~5×. Post-filter enrichment keeps the raw pool light-weight while ensuring the final dataset retains the rich metadata required for analysis.
- **Test extrapolation slice**: sorting by coverage then λ concentrates the evaluation shift on under-observed, high-chaoticity regimes—matching the narrative goal of testing extrapolation.

This pipeline, along with the recorded seeds and fingerprints, supports exact regeneration of the published datasets.
