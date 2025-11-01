"""Families of cellular automata rule generators."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .helpers import enumerate_neighborhoods, neighborhood_index


def rule_table_random_lambda(
    k: int,
    r: int,
    rng: random.Random,
    lambda_val: float = 0.5,
    quiescent_state: int = 0,
) -> Tuple[Dict[Tuple[int, ...], int], float, int]:
    """Langton-style random table over k symbols with target λ."""
    neighborhoods = list(enumerate_neighborhoods(k, r))
    non_q = [s for s in range(k) if s != quiescent_state]
    table: Dict[Tuple[int, ...], int] = {}
    non_q_count = 0
    for nb in neighborhoods:
        if rng.random() < lambda_val:
            val = rng.choice(non_q)
            non_q_count += 1
        else:
            val = quiescent_state
        table[nb] = val
    lam_actual = non_q_count / len(neighborhoods) if neighborhoods else 0.0
    return table, lam_actual, quiescent_state


def rule_table_totalistic(
    k: int,
    r: int,
    rng: random.Random,
) -> Tuple[Dict[Tuple[int, ...], int], float, int]:
    """Totalistic: output depends only on sum of neighborhood entries."""
    arity = 2 * r + 1
    max_sum = (k - 1) * arity
    sum_lookup = [rng.randrange(k) for _ in range(max_sum + 1)]
    table = {nb: sum_lookup[sum(nb)] for nb in enumerate_neighborhoods(k, r)}
    lam = sum(1 for v in table.values() if v != 0) / len(table) if table else 0.0
    return table, lam, 0


def rule_table_outer_totalistic(
    k: int,
    r: int,
    rng: random.Random,
    lambda_val: float = 0.5,
    qstate: int = 0,
) -> Tuple[Dict[Tuple[int, ...], int], float, int, Dict[str, object]]:
    """Outer-totalistic: depends on total neighborhood sum with separate center."""
    neighborhoods, _, by_center_total = neighborhood_index(k, r)
    non_q = [s for s in range(k) if s != qstate]
    table: Dict[Tuple[int, ...], int] = {}
    non_q_count = 0
    for (_, total_sum), combos in by_center_total.items():
        if rng.random() < lambda_val:
            val = rng.choice(non_q)
        else:
            val = qstate
        if val != qstate:
            non_q_count += len(combos)
        for nb in combos:
            table[nb] = val
    lam = non_q_count / len(neighborhoods) if neighborhoods else 0.0
    return table, lam, qstate, {
        "lambda_target": float(lambda_val),
        "quiescent_state": int(qstate),
    }


def rule_table_outer_inner_totalistic(
    k: int,
    r: int,
    rng: random.Random,
    lambda_val: float = 0.5,
    qstate: int = 0,
) -> Tuple[Dict[Tuple[int, ...], int], float, int, Dict[str, object]]:
    """Outer-inner totalistic: center tracked separately, depends on neighbors sum."""
    neighborhoods, by_center_outer, _ = neighborhood_index(k, r)
    non_q = [s for s in range(k) if s != qstate]
    table: Dict[Tuple[int, ...], int] = {}
    non_q_count = 0
    for (_, outer_sum), combos in by_center_outer.items():
        if rng.random() < lambda_val:
            val = rng.choice(non_q)
        else:
            val = qstate
        if val != qstate:
            non_q_count += len(combos)
        for nb in combos:
            table[nb] = val
    lam = non_q_count / len(neighborhoods) if neighborhoods else 0.0
    return table, lam, qstate, {
        "lambda_target": float(lambda_val),
        "quiescent_state": int(qstate),
    }


def rule_table_threshold(
    k: int,
    r: int,
    rng: random.Random,
    qstate: int = 0,
) -> Tuple[Dict[Tuple[int, ...], int], float, int, Dict[str, object]]:
    """Piecewise threshold family with per-state randomised behaviour."""
    neighborhoods, by_center_outer, _ = neighborhood_index(k, r)
    theta = {c: float(rng.uniform(0.25 * (k - 1), 0.75 * (k - 1))) for c in range(k)}
    pairs: Dict[int, Tuple[int, int]] = {}
    pair_modes: Dict[int, str] = {}
    for c in range(k):
        if rng.random() < 0.5:
            pairs[c] = (qstate, c)
            pair_modes[c] = "majority_like"
        else:
            lo = rng.randrange(k)
            hi = rng.randrange(k)
            pairs[c] = (lo, hi)
            pair_modes[c] = "random_pairs"

    table: Dict[Tuple[int, ...], int] = {}
    non_q = 0
    total = len(neighborhoods)
    for (center, outer_sum), combos in by_center_outer.items():
        lo, hi = pairs[center]
        avg = outer_sum / max(1, 2 * r)
        val = hi if avg >= theta[center] else lo
        if val != qstate:
            non_q += len(combos)
        for nb in combos:
            table[nb] = val
    lam = non_q / total if total else 0.0
    return table, lam, qstate, {
        "theta": theta,
        "pair_modes": pair_modes,
        "theta_range": [0.25 * (k - 1), 0.75 * (k - 1)],
    }


def rule_table_linear_mod_k(
    k: int,
    r: int,
    rng: random.Random,
    sparsity: int = 2,
    bias_prob: float = 0.3,
) -> Tuple[Dict[Tuple[int, ...], int], float, int, Dict[str, object]]:
    """Affine linear rule modulo k with sparse coefficients."""
    neighborhoods, _, _ = neighborhood_index(k, r)
    width = 2 * r + 1
    idxs = list(range(width))
    active_count = min(max(1, sparsity), width)
    active = rng.sample(idxs, k=active_count)
    alpha = [0] * width
    for idx in active:
        alpha[idx] = rng.randrange(k)
    if all(alpha[idx] == 0 for idx in range(width) if idx != r):
        alpha[r] = (alpha[r] + 1) % k
    bias = rng.randrange(k) if rng.random() < bias_prob else 0
    table: Dict[Tuple[int, ...], int] = {}
    non_q = 0
    for nb in neighborhoods:
        val = (sum(alpha[j] * nb[j] for j in range(width)) + bias) % k
        table[nb] = val
        if val != 0:
            non_q += 1
    lam = non_q / len(neighborhoods) if neighborhoods else 0.0
    return table, lam, 0, {
        "sparsity": int(active_count),
        "active_indices": active,
        "coefficients": alpha,
        "bias": int(bias),
    }


def rule_table_cyclic_excitable(
    k: int,
    r: int,
    rng: random.Random,
    trigger_state: Optional[int] = None,
    min_triggers: int = 1,
) -> Tuple[Dict[Tuple[int, ...], int], float, int, Dict[str, int]]:
    """Simple cyclic excitable rule family."""
    neighborhoods, _, _ = neighborhood_index(k, r)
    trigger_state = trigger_state if trigger_state is not None else 1
    table: Dict[Tuple[int, ...], int] = {}
    non_q = 0
    for nb in neighborhoods:
        center = nb[r]
        neighbours = nb[:r] + nb[r + 1 :]
        triggers = sum(
            1 for v in neighbours if (v == trigger_state if k >= 3 else v != 0)
        )
        if center == 0 and triggers >= min_triggers:
            val = 1 % k
        elif center != 0:
            val = (center + 1) % k
        else:
            val = 0
        table[nb] = val
        if val != 0:
            non_q += 1
    lam = non_q / len(neighborhoods) if neighborhoods else 0.0
    return table, lam, 0, {
        "trigger_state": int(trigger_state),
        "min_triggers": int(min_triggers),
    }


def rule_table_permuted_totalistic(
    k: int,
    r: int,
    rng: random.Random,
) -> Tuple[Dict[Tuple[int, ...], int], float, int, Dict[str, List[int]]]:
    """Totalistic outputs followed by a random permutation of states."""
    neighborhoods, _, _ = neighborhood_index(k, r)
    arity = 2 * r + 1
    max_sum = (k - 1) * arity
    sum_lookup = [rng.randrange(k) for _ in range(max_sum + 1)]
    perm = list(range(k))
    rng.shuffle(perm)
    table: Dict[Tuple[int, ...], int] = {}
    non_q = 0
    for nb in neighborhoods:
        val = perm[sum_lookup[sum(nb)]]
        table[nb] = val
        if val != 0:
            non_q += 1
    lam = non_q / len(neighborhoods) if neighborhoods else 0.0
    return table, lam, 0, {
        "permutation": perm,
    }


__all__ = [
    "rule_table_random_lambda",
    "rule_table_totalistic",
    "rule_table_outer_totalistic",
    "rule_table_outer_inner_totalistic",
    "rule_table_threshold",
    "rule_table_linear_mod_k",
    "rule_table_cyclic_excitable",
    "rule_table_permuted_totalistic",
]
