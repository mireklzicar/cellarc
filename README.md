# cellarc

Utilities for generating, analysing, and visualising few-shot cellular automata
episodes. The package ships with light-weight dataset tooling by default and
keeps the heavier simulation stack behind an optional extra.

## Installation

```bash
pip install cellarc
```

For generation and simulation features (JAX/CAX-based rule runners, automatic
dataset synthesis, Hugging Face integration) install the full extra:

```bash
pip install cellarc[all]
```

The wheel looks for local mirrors inside `artifacts/hf_cellarc` and
`artifacts/datasets` before downloading from the Hub. Set `CELLARC_HOME` to
override the cache location.

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

- Dataset cards: `artifacts/datasets/cellarc_100k/README.md` and
  `artifacts/datasets/cellarc_100k_meta/README.md`.
- Solver experiments: `SOLVER_RESULTS.md`.
