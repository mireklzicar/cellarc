import json, math, hashlib, random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from functools import lru_cache
from collections import Counter

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is absent
    tqdm = None  # type: ignore


SCHEMA_VERSION = "1.0.0"

import numpy as np
import cellpylib as cpl

# bring in your helpers
from .utils import de_bruijn_cycle, choose_r_t_for_W
from .solver import learn_local_map_from_pairs  # optional (for sanity checks)

# ---------- helpers ----------

def as_init(seq):  # 1D -> CellPyLib (2D) initial condition
    return np.array([seq], dtype=int)

def enumerate_neighborhoods(k: int, r: int):
    arity = 2*r + 1
    for idx in range(k**arity):
        x = idx
        digits = []
        for _ in range(arity):
            digits.append(x % k)
            x //= k
        yield tuple(reversed(digits))


def serialize_rule_table(
    table: Dict[Tuple[int, ...], int],
    *,
    alphabet_size: int,
    radius: int,
    quiescent_state: int,
) -> Dict[str, Union[int, List[int], str]]:
    """Serialize a local rule table into a compact lexicographic list."""
    arity = 2 * radius + 1
    values = [int(table[nb]) for nb in enumerate_neighborhoods(alphabet_size, radius)]
    return {
        "format_version": "1.0",
        "alphabet_size": int(alphabet_size),
        "radius": int(radius),
        "arity": int(arity),
        "center_index": int(radius),
        "ordering": "lexicographic_base_k",
        "quiescent_state": int(quiescent_state),
        "values": values,
    }


@lru_cache(maxsize=128)
def _neighborhood_index(k: int, r: int):
    neighborhoods = list(enumerate_neighborhoods(k, r))
    by_center_outer_sum: Dict[Tuple[int, int], List[Tuple[int, ...]]] = {}
    by_center_total_sum: Dict[Tuple[int, int], List[Tuple[int, ...]]] = {}
    for nb in neighborhoods:
        center = nb[r]
        total = sum(nb)
        outer = total - center
        by_center_outer_sum.setdefault((center, outer), []).append(nb)
        by_center_total_sum.setdefault((center, total), []).append(nb)
    return neighborhoods, by_center_outer_sum, by_center_total_sum

def rule_table_random_lambda(k: int, r: int, rng: random.Random,
                             lambda_val: float = 0.5, quiescent_state: int = 0):
    """Langton-style random table over k symbols with target λ."""
    arity = 2*r + 1
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
    lam_actual = non_q_count / len(neighborhoods)
    return table, lam_actual, quiescent_state

def rule_table_totalistic(k: int, r: int, rng: random.Random):
    """Totalistic: output depends only on sum of neighborhood entries."""
    arity = 2*r + 1
    max_sum = (k - 1) * arity
    # random output for each possible sum
    sum_lookup = [rng.randrange(k) for _ in range(max_sum + 1)]
    table = {nb: sum_lookup[sum(nb)] for nb in enumerate_neighborhoods(k, r)}
    # λ relative to quiescent 0
    lam = sum(1 for v in table.values() if v != 0) / len(table) if table else 0.0
    return table, lam, 0


def rule_table_outer_totalistic(
    k: int,
    r: int,
    rng: random.Random,
    lambda_val: float = 0.5,
    qstate: int = 0,
):
    """Outer-totalistic: center tracked separately, depends on total neighborhood sum."""
    neighborhoods, _, by_center_total = _neighborhood_index(k, r)
    non_q = [s for s in range(k) if s != qstate]
    table: Dict[Tuple[int, ...], int] = {}
    non_q_count = 0
    for (center, total_sum), combos in by_center_total.items():
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
):
    """Outer-inner totalistic: center tracked separately, depends on neighbors-only sum."""
    neighborhoods, by_center_outer, _ = _neighborhood_index(k, r)
    non_q = [s for s in range(k) if s != qstate]
    table: Dict[Tuple[int, ...], int] = {}
    non_q_count = 0
    for (center, outer_sum), combos in by_center_outer.items():
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
):
    neighborhoods, by_center_outer, _ = _neighborhood_index(k, r)
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
):
    neighborhoods, _, _ = _neighborhood_index(k, r)
    W = 2 * r + 1
    idxs = list(range(W))
    active_count = min(max(1, sparsity), W)
    active = rng.sample(idxs, k=active_count)
    alpha = [0] * W
    for idx in active:
        alpha[idx] = rng.randrange(k)
    if all(alpha[idx] == 0 for idx in range(W) if idx != r):
        alpha[r] = (alpha[r] + 1) % k
    bias = rng.randrange(k) if rng.random() < bias_prob else 0
    table: Dict[Tuple[int, ...], int] = {}
    non_q = 0
    for nb in neighborhoods:
        val = (sum(alpha[j] * nb[j] for j in range(W)) + bias) % k
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
):
    neighborhoods, _, _ = _neighborhood_index(k, r)
    trigger_state = trigger_state if trigger_state is not None else 1
    table: Dict[Tuple[int, ...], int] = {}
    non_q = 0
    for nb in neighborhoods:
        center = nb[r]
        neighbours = nb[:r] + nb[r + 1 :]
        triggers = sum(1 for v in neighbours if (v == trigger_state if k >= 3 else v != 0))
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
):
    neighborhoods, _, _ = _neighborhood_index(k, r)
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

def apply_rule_from_table(table: Dict[Tuple[int, ...], int]):
    return lambda n, c, t: table[tuple(int(x) for x in n)]

def induced_tstep_fingerprint(table: Dict[Tuple[int, ...], int], k: int, r: int, t: int) -> str:
    """Strong uniqueness: hash the induced t-step map over all k**W windows."""
    W = 2*r*t + 1
    cycle = de_bruijn_cycle(k, W)                  # length L = k**W
    half = (W - 1)//2
    out = cpl.evolve(as_init(cycle), timesteps=t+1,
                     apply_rule=apply_rule_from_table(table), r=r, memoize=True)[-1].tolist()
    L = len(cycle)
    # build mapping (window -> output)
    mapping = {}
    for i in range(L):
        win = tuple(cycle[(i - half + j) % L] for j in range(W))
        mapping[win] = out[i]
    h = hashlib.sha256()
    h.update(f'k={k};r={r};t={t};W={W};'.encode())
    for win in sorted(mapping.keys()):
        h.update(bytes(win))
        h.update(bytes([mapping[win]]))
    return h.hexdigest()

def rule_fingerprint(table: Dict[Tuple[int, ...], int], k: int, r: int) -> str:
    """Cheaper uniqueness: hash the one-step local rule table."""
    h = hashlib.sha256()
    h.update(f'k={k};r={r};'.encode())
    for nb in sorted(table.keys()):
        h.update(bytes(nb))
        h.update(bytes([table[nb]]))
    return h.hexdigest()

def ring_slice(seq: List[int], start: int, count: int) -> List[int]:
    n = len(seq)
    return [seq[(start + j) % n] for j in range(count)]

def lambda_bin(lam: float) -> str:
    return ("ordered" if lam < 0.20 else
            "edge"    if lam < 0.50 else
            "chaotic")

def entropy_bin(H: float) -> str:
    # tweak thresholds to taste
    return ("low" if H < 0.25 else "mid" if H < 0.6 else "high")


def quick_morphology_features(
    table: Dict[Tuple[int, ...], int],
    k: int,
    r: int,
    t: int,
    *,
    width: int = 30,
    horizon: int = 256,
    rng: Optional[random.Random] = None,
) -> Dict[str, Union[bool, float, int]]:
    """
    Lightweight CA morphology summary for annotation: absorbing behaviour,
    density stats, temporal period estimate, spatial correlation length,
    and a Derrida-like sensitivity proxy.
    """
    rng = rng or random
    apply = apply_rule_from_table(table)

    # roll-out from a random initial row
    seed = rng.randrange(1 << 30)
    rng_np = np.random.default_rng(seed)
    init = np.array([rng_np.integers(low=0, high=k, size=width, dtype=int)], dtype=int)
    steps = max(t + 1, horizon)
    ca = cpl.evolve(init, timesteps=steps, apply_rule=apply, r=r, memoize=True)
    A = np.asarray(ca, dtype=int)
    if A.size == 0:
        return {
            "absorbing": True,
            "density_mean": 0.0,
            "density_var": 0.0,
            "period_estimate": 1,
            "spatial_corr_length": 0,
            "derrida_like": 0.0,
        }

    last = A[-1]
    tail = A[-min(10, len(A)) :]
    absorbing = bool(np.all(tail == tail[-1]))

    dens = A.mean(axis=1).astype(float)
    density_mean = float(dens.mean())
    density_var = float(dens.var())

    # temporal autocorrelation of the last frame
    max_lag = min(128, len(A) - 1)
    period_estimate = 1
    if max_lag >= 1:
        base = A[-1].astype(float)
        base_c = base - base.mean()
        base_norm = float(np.dot(base_c, base_c))
        cors: List[float] = []
        for lag in range(1, max_lag + 1):
            other = A[-1 - lag].astype(float)
            other_c = other - other.mean()
            denom = math.sqrt(
                float(np.dot(base_c, base_c)) * float(np.dot(other_c, other_c))
            )
            if denom <= 0:
                cors.append(0.0)
            else:
                cors.append(float(np.clip(np.dot(base_c, other_c) / denom, -1.0, 1.0)))
        if cors:
            period_estimate = int(1 + int(np.argmax(cors)))

    # spatial correlation length from final slice
    last_centered = last.astype(float) - float(last.mean())
    if np.allclose(last_centered, 0):
        spatial_corr_length = 0
    else:
        ac = np.correlate(last_centered, last_centered, mode="full")
        ac = ac[len(ac) // 2 :]
        max_ac = ac[0] if ac[0] != 0 else 1.0
        ac = ac / max_ac
        below = np.where(ac < 1 / math.e)[0]
        spatial_corr_length = int(below[0]) if below.size else int(len(ac))

    # Derrida-like: flip ~1% of cells and track divergence over short rollout
    perturb_steps = 50
    last_row = last.copy()
    perturbed = last_row.copy()
    flip_count = max(1, len(perturbed) // 100)
    flip_indices = rng_np.choice(len(perturbed), size=flip_count, replace=False)
    perturbed[flip_indices] = (perturbed[flip_indices] + 1) % k
    evo1 = np.asarray(
        cpl.evolve(
            np.array([last_row], dtype=int),
            timesteps=perturb_steps,
            apply_rule=apply,
            r=r,
            memoize=True,
        ),
        dtype=int,
    )
    evo2 = np.asarray(
        cpl.evolve(
            np.array([perturbed], dtype=int),
            timesteps=perturb_steps,
            apply_rule=apply,
            r=r,
            memoize=True,
        ),
        dtype=int,
    )
    diffs = [float(np.mean(frame1 != frame2)) for frame1, frame2 in zip(evo1, evo2)]
    if len(diffs) <= 1:
        derrida_like = 0.0
    else:
        derrida_like = float((diffs[-1] - diffs[0]) / max(1, len(diffs) - 1))

    return {
        "absorbing": absorbing,
        "density_mean": density_mean,
        "density_var": density_var,
        "period_estimate": period_estimate,
        "spatial_corr_length": spatial_corr_length,
        "derrida_like": derrida_like,
    }

# ---------- single-task sampler ----------

def sample_task_cellpylib(
    rng: random.Random,
    *,
    k_range=(2, 6),
    max_radius=3,
    max_steps=5,
    train_examples=4,
    target_avg_train_len=48,
    family_mix: Optional[Dict[str, float]] = None,   # {'random': 0.6, 'totalistic': 0.4}
    lambda_for_random: Tuple[float, float] = (0.2, 0.7),  # sampled uniformly in this interval
    unique_by: str = "tstep",                        # 'rule' or 'tstep'
    complexity_rollout=(30, 256),                    # (width, horizon) for complexity metrics
    coverage_fraction: float = 1.0,                  # fraction of k**W windows exposed in train
    coverage_mode: str = "chunked",                  # 'chunked' or 'uniform'
    compute_complexity: bool = True,
    annotate_morphology: bool = True,
    query_within_coverage: bool = False,
    schema_version: str = SCHEMA_VERSION,
    dataset_version: Optional[str] = None,
):
    episode_seed = rng.randrange(1 << 62)
    episode_rng = random.Random(episode_seed)

    # choose k with a light bias toward smaller alphabets (keeps k**W sane)
    k_lo, k_hi = k_range
    alphabet_choices = list(range(k_lo, k_hi + 1))
    weights = [1.0 / (idx + 1) for idx in range(len(alphabet_choices))]
    k = episode_rng.choices(alphabet_choices, weights=weights, k=1)[0]

    # budget W from token budget ~ train_examples * target_avg_train_len
    total_budget = max(1, train_examples * target_avg_train_len)
    W_by_budget = int(math.floor(math.log(max(1, total_budget), k))) if k >= 2 else 1
    if W_by_budget % 2 == 0: W_by_budget -= 1
    W_by_budget = max(W_by_budget, 3)
    # cap by steps/radius feasibility with r<=max_radius, t<=max_steps
    W_cap = 2*max_radius*max_steps + 1
    W = min(W_by_budget, W_cap)

    # choose a feasible (r, t) pair for this W
    while True:
        try:
            r, t = choose_r_t_for_W(W, max_radius=max_radius, max_steps=max_steps, strategy="uniform", rng=episode_rng)
            break
        except ValueError:
            W -= 2
            if W < 3:
                raise RuntimeError("Failed to find feasible (r,t) for the given budget/caps.")

    # pick a family
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
    for name, w in items:
        acc += w
        if u <= acc:
            family = name; break

    # build rule table
    def _unwrap(result):
        if isinstance(result, tuple) and len(result) == 4:
            table, lam_actual, qstate, extra = result
        else:
            table, lam_actual, qstate = result
            extra = {}
        return table, lam_actual, qstate, extra

    family_params: Dict[str, Union[int, float, str, List[int], Dict[str, float], Dict[str, str]]] = {}

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
            rule_table_outer_totalistic(k, r, episode_rng, lambda_val=lam_target, qstate=0)
        )
        family_params = extra
    elif family == "outer_inner_totalistic":
        lam_target = episode_rng.uniform(*lambda_for_random)
        table, lam_actual, qstate, extra = _unwrap(
            rule_table_outer_inner_totalistic(k, r, episode_rng, lambda_val=lam_target, qstate=0)
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

    # de Bruijn coverage → few-shot train pairs
    cycle = de_bruijn_cycle(k, W)
    L = len(cycle)
    half = (W - 1)//2

    full_out = cpl.evolve(as_init(cycle), timesteps=t+1,
                          apply_rule=apply_rule_from_table(table), r=r, memoize=True)[-1].tolist()

    S = max(1, train_examples)
    if query_within_coverage:
        base = L // S
        rem = L % S
        lengths = [base + (1 if i < rem else 0) for i in range(S)]
        offset = episode_rng.randrange(L)
        starts: List[int] = []
        acc = offset
        for seg_len in lengths:
            starts.append(acc % L)
            acc += seg_len
        windows_revealed = L
        coverage_fraction_effective = 1.0
        coverage_mode_effective = "full_cycle_partition"
    else:
        if not (0 < coverage_fraction <= 1.0):
            raise ValueError("coverage_fraction must lie in (0, 1].")
        target_windows = int(round(coverage_fraction * L))
        target_windows = max(W, S, target_windows)
        target_windows = min(L, target_windows)
        base = target_windows // S
        rem = target_windows % S
        lengths = [base + (1 if i < rem else 0) for i in range(S)]

        if coverage_mode == "uniform":
            starts = [int((i * L) / S) % L for i in range(S)]
            jitter = max(1, L // max(4 * S, 1))
            starts = [(s + episode_rng.randrange(0, jitter)) % L for s in starts]
        elif coverage_mode == "chunked":
            starts = [episode_rng.randrange(0, L) for _ in range(S)]
        else:
            raise ValueError("coverage_mode must be 'chunked' or 'uniform'")

        windows_revealed = sum(lengths)
        coverage_fraction_effective = float(min(1.0, windows_revealed / L))
        coverage_mode_effective = coverage_mode

    train_pairs: List[Tuple[List[int], List[int]]] = []
    for s, seg_len in zip(starts, lengths):
        x = ring_slice(cycle,    s - half, seg_len + 2*half)
        y = ring_slice(full_out, s - half, seg_len + 2*half)
        train_pairs.append((x, y))

    # query/solution
    avg_core = sum(lengths)//max(1, len(lengths))
    q_len = max(avg_core + W, avg_core + episode_rng.randint(W, 2*W))
    query = [episode_rng.randrange(k) for _ in range(q_len)]
    solution = cpl.evolve(as_init(query), timesteps=t+1,
                          apply_rule=apply_rule_from_table(table), r=r, memoize=True)[-1].tolist()

    # complexity metrics (CellPyLib only)
    width, horizon = complexity_rollout
    if compute_complexity:
        ca_roll = cpl.evolve(cpl.init_random(width, k=k), timesteps=horizon,
                             apply_rule=apply_rule_from_table(table), r=r, memoize=True)
        avg_cell_entropy = float(cpl.average_cell_entropy(ca_roll))
        ami_1 = float(cpl.average_mutual_information(ca_roll, temporal_distance=1))
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

    # uniqueness fingerprint
    if unique_by == "tstep":
        fp = induced_tstep_fingerprint(table, k, r, t)
    elif unique_by == "rule":
        fp = rule_fingerprint(table, k, r)
    else:
        raise ValueError("unique_by must be 'rule' or 'tstep'")

    probe_fp = rule_fingerprint(table, k, r)

    rule_table_payload = serialize_rule_table(
        table,
        alphabet_size=k,
        radius=r,
        quiescent_state=qstate,
    )

    record = {
        "train": [{"input": x, "output": y} for x, y in train_pairs],
        "query": query,
        "solution": solution,
        "meta": {
            "schema_version": schema_version,
            "dataset_version": dataset_version,
            "alphabet_size": k, "radius": r, "steps": t, "window": W, "windows_total": k**W,
            "train_context": half, "train_core_lengths": lengths, "family": family,
            "family_params": family_params,
            "lambda": float(lam_actual), "lambda_bin": lambda_bin(lam_actual),
            "avg_cell_entropy": avg_cell_entropy,
            "entropy_bin": entropy_bin(avg_cell_entropy) if avg_cell_entropy is not None else None,
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
                "cycle_length": int(L),
            },
            "morphology": morphology,
        },
        "rule_table": rule_table_payload,
    }
    return record

# ---------- dataset driver with balancing & uniqueness ----------

def generate_dataset_jsonl(
    path: Path,
    *,
    count: int,
    seed: Optional[int] = None,
    meta_path: Optional[Path] = None,
    k_range=(2, 6),
    max_radius=3,
    max_steps=5,
    train_examples=4,
    target_avg_train_len=48,
    family_mix=None,
    unique_by="tstep",
    balance_by: str = "lambda",   # 'none' | 'lambda' | 'entropy'
    max_attempts_per_item: int = 200,
    coverage_fraction: Union[
        float,
        Sequence[float],
        Callable[[random.Random], float],
    ] = 1.0,
    coverage_mode: str = "chunked",
    cap_lambda: Optional[float] = None,
    cap_entropy: Optional[float] = None,
    compute_complexity: bool = True,
    annotate_morphology: bool = True,
    query_within_coverage: bool = False,
    seen_fingerprints: Optional[Set[str]] = None,
    schema_version: str = SCHEMA_VERSION,
    dataset_version: str = "dev",
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
):
    rng = random.Random(seed)
    shared_seen = seen_fingerprints if seen_fingerprints is not None else set()
    produced = 0
    attempts = 0
    accepted_fps: List[str] = []
    accepted_probe_fps: List[str] = []
    lambda_sum = 0.0
    lambda_min = float("inf")
    lambda_max = float("-inf")
    coverage_fraction_sum = 0.0
    coverage_fraction_min = float("inf")
    coverage_fraction_max = float("-inf")
    coverage_windows_sum = 0.0
    coverage_windows_min = float("inf")
    coverage_windows_max = float("-inf")
    family_counts: Counter[str] = Counter()
    coverage_mode_counts: Counter[str] = Counter()
    train_length_sum = 0.0
    train_length_min = float("inf")
    train_length_max = float("-inf")
    train_length_count = 0
    train_length_hist: Counter[int] = Counter()
    query_length_sum = 0.0
    query_length_min = float("inf")
    query_length_max = float("-inf")
    query_length_count = 0
    query_length_hist: Counter[int] = Counter()

    progress = None
    if show_progress and tqdm is not None and count > 0:
        desc = progress_desc or path.stem
        progress = tqdm(total=count, desc=desc, unit="episode")

    meta_path = meta_path or path.with_name(f"{path.stem}_meta.jsonl")

    if balance_by == "entropy" and not compute_complexity:
        raise ValueError("balance_by='entropy' requires compute_complexity=True.")
    if cap_entropy is not None and not compute_complexity:
        raise ValueError("cap_entropy requires compute_complexity=True.")

    # bins for balancing
    if balance_by == "lambda":
        bins = ["ordered", "edge", "chaotic"]
    elif balance_by == "entropy":
        bins = ["low", "mid", "high"]
    else:
        bins = ["all"]

    per_bin_cap = math.ceil(count / len(bins))
    bin_counts = {b: 0 for b in bins}

    def sample_coverage_fraction() -> float:
        spec = coverage_fraction
        if callable(spec):
            val = float(spec(rng))
        elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            if len(spec) == 0:
                raise ValueError("coverage_fraction sequence must be non-empty.")
            if len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
                lo, hi = float(spec[0]), float(spec[1])
                if lo > hi:
                    lo, hi = hi, lo
                val = float(lo) if math.isclose(lo, hi) else float(rng.uniform(lo, hi))
            else:
                val = float(rng.choice(list(spec)))
        else:
            val = float(spec)
        if not (0 < val <= 1.0):
            raise ValueError("coverage_fraction values must lie in (0, 1].")
        return val

    try:
        with path.open("w", encoding="utf-8") as f_core, meta_path.open("w", encoding="utf-8") as f_meta:
            while produced < count:
                attempts += 1
                if attempts > count * max_attempts_per_item:
                    raise RuntimeError("Attempt budget exceeded; relax caps or balancing.")

                rec = sample_task_cellpylib(
                    rng,
                    k_range=k_range,
                    max_radius=max_radius,
                    max_steps=max_steps,
                    train_examples=train_examples,
                    target_avg_train_len=target_avg_train_len,
                    family_mix=family_mix,
                    unique_by=unique_by,
                    coverage_fraction=sample_coverage_fraction(),
                    coverage_mode=coverage_mode,
                    compute_complexity=compute_complexity,
                    annotate_morphology=annotate_morphology,
                    query_within_coverage=query_within_coverage,
                    schema_version=schema_version,
                    dataset_version=dataset_version,
                )

                if cap_lambda is not None and rec["meta"]["lambda"] > cap_lambda:
                    continue
                if cap_entropy is not None and rec["meta"]["avg_cell_entropy"] > cap_entropy:
                    continue

                # uniqueness
                fp = rec["meta"]["fingerprint"]
                if fp in shared_seen:
                    continue
                probe_fp = rec["meta"]["probe_fingerprint"]

                # balancing
                if balance_by == "lambda":
                    bin_key = rec["meta"]["lambda_bin"]
                elif balance_by == "entropy":
                    bin_key = rec["meta"]["entropy_bin"]
                else:
                    bin_key = "all"

                if bin_counts[bin_key] >= per_bin_cap:
                    continue

                # accept
                shared_seen.add(fp)
                accepted_fps.append(fp)
                accepted_probe_fps.append(probe_fp)

                full_meta = dict(rec["meta"])
                minimal_meta = {
                    key: full_meta[key]
                    for key in ("fingerprint", "lambda", "lambda_bin")
                    if key in full_meta
                }

                core_record = {
                    "train": rec["train"],
                    "query": rec["query"],
                    "solution": rec["solution"],
                    "meta": minimal_meta,
                }

                extras_payload: Dict[str, object] = {
                    "fingerprint": fp,
                    "meta": full_meta,
                }
                for key, value in rec.items():
                    if key not in {"train", "query", "solution", "meta"}:
                        extras_payload[key] = value

                f_core.write(json.dumps(core_record))
                f_core.write("\n")
                f_meta.write(json.dumps(extras_payload))
                f_meta.write("\n")

                produced += 1
                bin_counts[bin_key] += 1
                if progress is not None:
                    progress.update(1)

                for pair in rec["train"]:
                    seq_len = len(pair["input"])
                    train_length_sum += seq_len
                    train_length_min = min(train_length_min, seq_len)
                    train_length_max = max(train_length_max, seq_len)
                    train_length_count += 1
                    train_length_hist[seq_len] += 1

                query_len = len(rec["query"])
                query_length_sum += query_len
                query_length_min = min(query_length_min, query_len)
                query_length_max = max(query_length_max, query_len)
                query_length_count += 1
                query_length_hist[query_len] += 1

                lam_val = float(rec["meta"]["lambda"])
                lambda_sum += lam_val
                lambda_min = min(lambda_min, lam_val)
                lambda_max = max(lambda_max, lam_val)

                coverage_info = rec["meta"].get("coverage", {})
                cov_frac = float(coverage_info.get("fraction", 0.0))
                cov_windows = float(coverage_info.get("windows", 0))
                coverage_fraction_sum += cov_frac
                coverage_fraction_min = min(coverage_fraction_min, cov_frac)
                coverage_fraction_max = max(coverage_fraction_max, cov_frac)
                coverage_windows_sum += cov_windows
                coverage_windows_min = min(coverage_windows_min, cov_windows)
                coverage_windows_max = max(coverage_windows_max, cov_windows)

                family_counts[rec["meta"].get("family", "unknown")] += 1
                coverage_mode_counts[str(coverage_info.get("mode", "unknown"))] += 1
    finally:
        if progress is not None:
            progress.close()

    return {
        "written": produced,
        "bins": bin_counts,
        "path": str(path),
        "meta_path": str(meta_path),
        "fingerprints": accepted_fps,
        "probe_fingerprints": accepted_probe_fps,
        "stats": {
            "count": produced,
            "lambda": {
                "sum": lambda_sum,
                "min": None if produced == 0 else lambda_min,
                "max": None if produced == 0 else lambda_max,
            },
            "coverage_fraction": {
                "sum": coverage_fraction_sum,
                "min": None if produced == 0 else coverage_fraction_min,
                "max": None if produced == 0 else coverage_fraction_max,
            },
            "coverage_windows": {
                "sum": coverage_windows_sum,
                "min": None if produced == 0 else coverage_windows_min,
                "max": None if produced == 0 else coverage_windows_max,
            },
            "train_sequence_length": {
                "sum": train_length_sum,
                "min": None if train_length_count == 0 else train_length_min,
                "max": None if train_length_count == 0 else train_length_max,
                "count": train_length_count,
                "hist": dict(train_length_hist),
            },
            "query_length": {
                "sum": query_length_sum,
                "min": None if query_length_count == 0 else query_length_min,
                "max": None if query_length_count == 0 else query_length_max,
                "count": query_length_count,
                "hist": dict(query_length_hist),
            },
            "families": dict(family_counts),
            "coverage_modes": dict(coverage_mode_counts),
        },
    }
