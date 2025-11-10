# cellarc

This repository contains dataset generation pipeline for CellARC:
- Model training and baselines: https://github.com/mireklzicar/cellarc 
- Website: https://cellarc.mireklzicar.com/

![](assets/ca_demo.png)

## Installation

```bash
pip install cellarc
```

For generation and simulation features (JAX/CAX-based rule runners, automatic
dataset synthesis, Hugging Face integration) install the full extra:

```bash
pip install cellarc[all]
```

Dataset snapshots are fetched directly from the Hugging Face Hub and cached in
`~/.cache/cellarc` (override with the `CELLARC_HOME` environment variable).
There are no repository fallbacks; if a download fails, the loader raises an
error so the issue can be fixed explicitly.

## Working with datasets

```python
from cellarc import EpisodeDataset, EpisodeDataLoader

# Load the supervision-only split shipped in ``mireklzicar/cellarc_100k``.
train = EpisodeDataset.from_huggingface("train", include_metadata=False)

# Iterate over metadata-enriched episodes (``mireklzicar/cellarc_100k_meta``).
val = EpisodeDataset.from_huggingface("val", include_metadata=True)

print(len(train), len(val))

# Batch episodes with optional augmentation.
loader = EpisodeDataLoader(
    val,
    batch_size=8,
    shuffle=True,
    seed=1234,
)

first_batch = next(iter(loader))
print(first_batch[0]["meta"]["fingerprint"])
```

The available remote splits are `train`, `val`, `test_interpolation`, and
`test_extrapolation`. Each split is stored as `data/<split>.jsonl` (the default
loader) and `data/<split>.parquet`; set `fmt="parquet"` when using
`datasets`/`pyarrow` for faster IO.

### Quick 100-episode subsets

For faster iteration, the dataset repositories provide fixed 100-episode
subsets for every split: `train_100`, `val_100`, `test_interpolation_100`, and
`test_extrapolation_100`. You can access them via the same loader API:

```python
from cellarc import EpisodeDataset

# Load the 100-episode training subset (with metadata merged in).
train_small = EpisodeDataset.from_huggingface("train_100", include_metadata=True)
print(len(train_small))  # -> 100

# Iterate or batch as usual…
for episode in train_small:
    print(episode["id"])  # first few IDs
    break
```

### Refreshing the cache

Force-refresh a snapshot when you need a clean copy:

```bash
python - <<'PY'
from cellarc import download_benchmark
download_benchmark(name="cellarc_100k", include_metadata=True, force_download=True)
PY
```

## Optional generation stack

With the `all` extra installed you gain access to the sampling and simulation
utilities:

```python
import random
from pathlib import Path

from cellarc import generate_dataset_jsonl, sample_task

task = sample_task(rng=random.Random(0))
generate_dataset_jsonl(Path("episodes.jsonl"), count=128, include_rule_table=True)
```

These helpers depend on `jax`, `flax`, and `cax`. If the import fails, install
the extra or vendor the required frameworks manually.

## Further reading

- Dataset cards live at https://huggingface.co/datasets/mireklzicar/cellarc_100k
  and https://huggingface.co/datasets/mireklzicar/cellarc_100k_meta.
- Solver experiments: `SOLVER_RESULTS.md`.
