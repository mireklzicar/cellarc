"""Fingerprints and reusable evaluation helpers for cellular automata rules."""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

from ..utils import de_bruijn_cycle
from .cax_runner import evolve_rule_table


def apply_rule_from_table(table: Dict[Tuple[int, ...], int]):
    """Return a callable that applies the provided rule table."""
    return lambda n, c, t: table[tuple(int(x) for x in n)]


def induced_tstep_fingerprint(table: Dict[Tuple[int, ...], int], k: int, r: int, t: int) -> str:
    """Hash the induced t-step map over all windows to enforce uniqueness."""
    width = 2 * r * t + 1
    cycle = de_bruijn_cycle(k, width)
    half = (width - 1) // 2
    evolved = evolve_rule_table(
        table,
        cycle,
        timesteps=t + 1,
        alphabet_size=k,
        radius=r,
    ).tolist()
    length = len(cycle)
    mapping = {}
    for i in range(length):
        window = tuple(cycle[(i - half + j) % length] for j in range(width))
        mapping[window] = evolved[i]
    h = hashlib.sha256()
    h.update(f"k={k};r={r};t={t};W={width};".encode())
    for window in sorted(mapping.keys()):
        h.update(bytes(window))
        h.update(bytes([mapping[window]]))
    return h.hexdigest()


def rule_fingerprint(table: Dict[Tuple[int, ...], int], k: int, r: int) -> str:
    """Hash the one-step local rule table."""
    h = hashlib.sha256()
    h.update(f"k={k};r={r};".encode())
    for nb in sorted(table.keys()):
        h.update(bytes(nb))
        h.update(bytes([table[nb]]))
    return h.hexdigest()


__all__ = ["apply_rule_from_table", "induced_tstep_fingerprint", "rule_fingerprint"]
