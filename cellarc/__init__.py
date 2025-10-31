"""
cell_arc package: tools for generating and solving few-shot cellular automata tasks.
"""

from .data import (
    EpisodeDataLoader,
    EpisodeDataset,
    available_remote_datasets,
    available_remote_splits,
    augment_episode,
    download_benchmark,
    load_manifest,
    random_palette_mapping,
)
from .generation import generate_dataset_jsonl, sample_task_cellpylib
from .signatures import batch_signatures, compute_signature, signatures_as_rows
from .solver import LearnedLocalMap, learn_from_record, learn_local_map_from_pairs
from .utils import de_bruijn_cycle, choose_r_t_for_W, window_size

__version__ = "0.1.0"

__all__ = [
    "EpisodeDataLoader",
    "EpisodeDataset",
    "available_remote_datasets",
    "available_remote_splits",
    "augment_episode",
    "download_benchmark",
    "load_manifest",
    "random_palette_mapping",
    "generate_dataset_jsonl",
    "sample_task_cellpylib",
    "LearnedLocalMap",
    "learn_from_record",
    "learn_local_map_from_pairs",
    "de_bruijn_cycle",
    "choose_r_t_for_W",
    "window_size",
    "compute_signature",
    "batch_signatures",
    "signatures_as_rows",
]
