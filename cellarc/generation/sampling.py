"""Episode sampling logic for the cellular automata ARC benchmark."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from .cax_runner import AutomatonRunner

from ..utils import choose_r_t_for_W, de_bruijn_cycle
from .constants import SCHEMA_VERSION
from .fingerprints import induced_tstep_fingerprint, rule_fingerprint
from .helpers import ring_slice
from .metrics import average_cell_entropy, average_mutual_information
from .morphology import quick_morphology_features
from .rules import (
    rule_table_cyclic_excitable,
    rule_table_linear_mod_k,
    rule_table_outer_inner_totalistic,
    rule_table_outer_totalistic,
    rule_table_permuted_totalistic,
    rule_table_random_lambda,
    rule_table_threshold,
    rule_table_totalistic,
)
from .serialization import serialize_rule_table


def lambda_bin(lam: float) -> str:
    return "ordered" if lam < 0.20 else "edge" if lam < 0.50 else "chaotic"


def entropy_bin(H: float) -> str:
    return "low" if H < 0.25 else "mid" if H < 0.6 else "high"


def sample_task_cellpylib(
    rng: random.Random,
    *,
    k_range=(2, 6),
    max_radius=3,
    max_steps=5,
    train_examples=4,
    target_avg_train_len=48,
    family_mix: Optional[Dict[str, float]] = None,
    lambda_for_random: Tuple[float, float] = (0.2, 0.7),
    unique_by: str = "tstep",
    complexity_rollout=(30, 256),
    coverage_fraction: float = 1.0,
    coverage_mode: str = "chunked",
    compute_complexity: bool = True,
    annotate_morphology: bool = True,
    query_within_coverage: bool = False,
    schema_version: str = SCHEMA_VERSION,
    dataset_version: Optional[str] = None,
):
    """Generate a single training episode with metadata and fingerprints."""
    episode_seed = rng.randrange(1 << 62)
    episode_rng = random.Random(episode_seed)

    k_lo, k_hi = k_range
    alphabet_choices = list(range(k_lo, k_hi + 1))
    weights = [1.0 / (idx + 1) for idx in range(len(alphabet_choices))]
    k = episode_rng.choices(alphabet_choices, weights=weights, k=1)[0]

    total_budget = max(1, train_examples * target_avg_train_len)
    W_by_budget = int(math.floor(math.log(max(1, total_budget), k))) if k >= 2 else 1
    if W_by_budget % 2 == 0:
        W_by_budget -= 1
    W_by_budget = max(W_by_budget, 3)
    W_cap = 2 * max_radius * max_steps + 1
    W = min(W_by_budget, W_cap)

    while True:
        try:
            r, t = choose_r_t_for_W(
                W,
                max_radius=max_radius,
                max_steps=max_steps,
                strategy="uniform",
                rng=episode_rng,
            )
            break
        except ValueError:
            W -= 2
            if W < 3:
                raise RuntimeError(
                    "Failed to find feasible (r,t) for the given budget/caps."
                )

    fam_mix = family_mix or {
        "random": 0.35,
        "totalistic": 0.15,
        "outer_totalistic": 0.15,
        "outer_inner_totalistic": 0.10,
        "threshold": 0.10,
        "linear_mod_k": 0.10,
        "cyclic_excitable": 0.05,
        "permuted_totalistic": 0.0,
    }
    items = [(name, w) for name, w in fam_mix.items() if w > 0]
    total_w = sum(w for _, w in items)
    u = episode_rng.random() * total_w
    acc = 0.0
    family = items[-1][0]
    for name, weight in items:
        acc += weight
        if u <= acc:
            family = name
            break

    def _unwrap(result):
        if isinstance(result, tuple) and len(result) == 4:
            table, lam_actual, qstate, extra = result
        else:
            table, lam_actual, qstate = result
            extra = {}
        return table, lam_actual, qstate, extra

    family_params: Dict[str, object] = {}

    if family == "random":
        lam_target = episode_rng.uniform(*lambda_for_random)
        table, lam_actual, qstate = rule_table_random_lambda(
            k, r, episode_rng, lambda_val=lam_target, quiescent_state=0
        )
        family_params = {"lambda_target": float(lam_target), "quiescent_state": 0}
    elif family == "totalistic":
        table, lam_actual, qstate = rule_table_totalistic(k, r, episode_rng)
    elif family == "outer_totalistic":
        lam_target = episode_rng.uniform(*lambda_for_random)
        table, lam_actual, qstate, extra = _unwrap(
            rule_table_outer_totalistic(
                k, r, episode_rng, lambda_val=lam_target, qstate=0
            )
        )
        family_params = extra
    elif family == "outer_inner_totalistic":
        lam_target = episode_rng.uniform(*lambda_for_random)
        table, lam_actual, qstate, extra = _unwrap(
            rule_table_outer_inner_totalistic(
                k, r, episode_rng, lambda_val=lam_target, qstate=0
            )
        )
        family_params = extra
    elif family == "threshold":
        table, lam_actual, qstate, extra = _unwrap(
            rule_table_threshold(k, r, episode_rng, qstate=0)
        )
        family_params = extra
    elif family == "linear_mod_k":
        sparsity_choice = episode_rng.choice([1, 2, 3])
        table, lam_actual, qstate, extra = _unwrap(
            rule_table_linear_mod_k(
                k, r, episode_rng, sparsity=sparsity_choice, bias_prob=0.3
            )
        )
        family_params = extra
    elif family == "cyclic_excitable":
        trig = 1 if k >= 3 else 1
        min_trig = 1 if r == 1 else episode_rng.choice([1, 2])
        table, lam_actual, qstate, extra = _unwrap(
            rule_table_cyclic_excitable(
                k, r, episode_rng, trigger_state=trig, min_triggers=min_trig
            )
        )
        family_params = extra
    elif family == "permuted_totalistic":
        table, lam_actual, qstate, extra = _unwrap(
            rule_table_permuted_totalistic(k, r, episode_rng)
        )
        family_params = extra
    else:
        raise ValueError(f"Unknown family: {family}")

    cycle = de_bruijn_cycle(k, W)
    length = len(cycle)
    half = (W - 1) // 2

    runner = AutomatonRunner(
        alphabet_size=k,
        radius=r,
        table=table,
        rng_seed=episode_rng.randrange(1 << 30),
    )
    full_out = runner.evolve(cycle, timesteps=t + 1).tolist()

    episode_count = max(1, train_examples)
    if query_within_coverage:
        base = length // episode_count
        rem = length % episode_count
        lengths = [base + (1 if i < rem else 0) for i in range(episode_count)]
        offset = episode_rng.randrange(length)
        starts: List[int] = []
        acc = offset
        for seg_len in lengths:
            starts.append(acc % length)
            acc += seg_len
        windows_revealed = length
        coverage_fraction_effective = 1.0
        coverage_mode_effective = "full_cycle_partition"
    else:
        if not (0 < coverage_fraction <= 1.0):
            raise ValueError("coverage_fraction must lie in (0, 1].")
        target_windows = int(round(coverage_fraction * length))
        target_windows = max(W, episode_count, target_windows)
        target_windows = min(length, target_windows)
        base = target_windows // episode_count
        rem = target_windows % episode_count
        lengths = [base + (1 if i < rem else 0) for i in range(episode_count)]

        if coverage_mode == "uniform":
            starts = [int((i * length) / episode_count) % length for i in range(episode_count)]
            jitter = max(1, length // max(4 * episode_count, 1))
            starts = [(s + episode_rng.randrange(0, jitter)) % length for s in starts]
        elif coverage_mode == "chunked":
            starts = [episode_rng.randrange(0, length) for _ in range(episode_count)]
        else:
            raise ValueError("coverage_mode must be 'chunked' or 'uniform'")

        windows_revealed = sum(lengths)
        coverage_fraction_effective = float(min(1.0, windows_revealed / length))
        coverage_mode_effective = coverage_mode

    train_pairs: List[Tuple[List[int], List[int]]] = []
    for start, seg_len in zip(starts, lengths):
        x = ring_slice(cycle, start - half, seg_len + 2 * half)
        y = ring_slice(full_out, start - half, seg_len + 2 * half)
        train_pairs.append((x, y))

    avg_core = sum(lengths) // max(1, len(lengths))
    q_len = max(avg_core + W, avg_core + episode_rng.randint(W, 2 * W))
    query = [episode_rng.randrange(k) for _ in range(q_len)]
    solution = runner.evolve(query, timesteps=t + 1).tolist()

    width, horizon = complexity_rollout
    if compute_complexity:
        random_init = [episode_rng.randrange(k) for _ in range(width)]
        ca_roll = runner.evolve(random_init, timesteps=horizon, return_history=True)
        avg_cell_entropy = float(average_cell_entropy(ca_roll))
        ami_1 = float(average_mutual_information(ca_roll, temporal_distance=1))
    else:
        avg_cell_entropy = None
        ami_1 = None

    morphology = (
        quick_morphology_features(
            table,
            k,
            r,
            t,
            width=width,
            horizon=horizon,
            rng=episode_rng,
        )
        if annotate_morphology
        else None
    )

    if unique_by == "tstep":
        fp = induced_tstep_fingerprint(table, k, r, t)
    elif unique_by == "rule":
        fp = rule_fingerprint(table, k, r)
    else:
        raise ValueError("unique_by must be 'rule' or 'tstep'")

    probe_fp = rule_fingerprint(table, k, r)

    rule_table_payload = serialize_rule_table(
        table, alphabet_size=k, radius=r, quiescent_state=qstate
    )

    record = {
        "train": [{"input": x, "output": y} for x, y in train_pairs],
        "query": query,
        "solution": solution,
        "meta": {
            "schema_version": schema_version,
            "dataset_version": dataset_version,
            "alphabet_size": k,
            "radius": r,
            "steps": t,
            "window": W,
            "windows_total": k ** W,
            "train_context": half,
            "train_core_lengths": lengths,
            "family": family,
            "family_params": family_params,
            "lambda": float(lam_actual),
            "lambda_bin": lambda_bin(lam_actual),
            "avg_cell_entropy": avg_cell_entropy,
            "entropy_bin": entropy_bin(avg_cell_entropy)
            if avg_cell_entropy is not None
            else None,
            "avg_mutual_information_d1": ami_1,
            "fingerprint": fp,
            "probe_fingerprint": probe_fp,
            "unique_by": unique_by,
            "wrap": True,
            "episode_seed": int(episode_seed),
            "coverage": {
                "scheme": "de_bruijn_subcover",
                "fraction": coverage_fraction_effective,
                "windows": int(windows_revealed),
                "segments": int(len(lengths)),
                "mode": coverage_mode_effective,
                "query_within_coverage": bool(query_within_coverage),
                "cycle_length": int(length),
            },
            "morphology": morphology,
        },
        "rule_table": rule_table_payload,
    }
    return record


__all__ = ["entropy_bin", "lambda_bin", "sample_task_cellpylib"]
