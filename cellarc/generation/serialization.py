"""Serialization utilities for cellular automata rules."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Union

from .helpers import enumerate_neighborhoods


def serialize_rule_table(
    table: Dict[Tuple[int, ...], int],
    *,
    alphabet_size: int,
    radius: int,
    quiescent_state: int,
) -> Dict[str, Union[int, List[int], str]]:
    """Serialize a local rule table into a compact lexicographic representation."""
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


def deserialize_rule_table(payload: Dict[str, Union[int, List[int], str]]) -> Dict[Tuple[int, ...], int]:
    """Invert ``serialize_rule_table`` and reconstruct the rule mapping."""
    k = int(payload["alphabet_size"])
    r = int(payload["radius"])
    values: Iterable[int] = payload["values"]  # type: ignore[assignment]
    table: Dict[Tuple[int, ...], int] = {}
    for neighborhood, value in zip(enumerate_neighborhoods(k, r), values):
        table[neighborhood] = int(value)
    return table


__all__ = ["serialize_rule_table", "deserialize_rule_table"]
