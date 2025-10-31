from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

Symbol = int
Word = List[Symbol]
Window = Tuple[Symbol, ...]


def _centered_window(seq: Word, i: int, W: int, wrap: bool) -> Window:
    """Return the length-W window centered at i (half = (W-1)//2)."""
    n = len(seq)
    half = W // 2
    if wrap:
        return tuple(seq[(i - half + j) % n] for j in range(W))
    out: List[Symbol] = []
    for j in range(i - half, i + half + 1):
        if 0 <= j < n:
            out.append(seq[j])
        else:
            out.append(0)
    return tuple(out)


@dataclass
class LearnedLocalMap:
    W: int
    wrap: bool
    mapping: Dict[Window, Symbol]
    alphabet_size: Optional[int] = None
    windows_total: Optional[int] = None

    @property
    def coverage_complete(self) -> bool:
        if self.alphabet_size is None or self.windows_total is None:
            return False
        return len(self.mapping) == self.windows_total == (self.alphabet_size ** self.W)

    def predict(self, query: Word) -> Word:
        n = len(query)
        out: List[Symbol] = [0] * n
        missing: List[Window] = []
        for i in range(n):
            win = _centered_window(query, i, self.W, self.wrap)
            try:
                out[i] = self.mapping[win]
            except KeyError:
                missing.append(win)
        if missing:
            ex = missing[0]
            raise KeyError(
                f"Missing {len(missing)} of {self.alphabet_size ** self.W if self.alphabet_size else '??'} "
                f"windows; e.g. first unseen window={ex}"
            )
        return out

    def predict_with_backoff(self, query: Word, policy: str = "majority", default: int = 0) -> Word:
        """
        Predict while handling unseen windows via a simple backoff policy:
          - 'majority': emit the most frequent symbol observed in training.
          - 'default':  always emit the supplied default symbol.
        """
        if policy not in {"majority", "default"}:
            raise ValueError("policy must be 'majority' or 'default'")

        if policy == "majority" and self.mapping:
            from collections import Counter

            mode_symbol = Counter(self.mapping.values()).most_common(1)[0][0]
        else:
            mode_symbol = default

        n = len(query)
        out: List[Symbol] = [mode_symbol] * n
        for i in range(n):
            win = _centered_window(query, i, self.W, self.wrap)
            out[i] = self.mapping.get(win, mode_symbol)
        return out


def learn_local_map_from_pairs(
    train_pairs: Iterable[Tuple[Word, Word]],
    W: int,
    wrap: bool = True,
    alphabet_size: Optional[int] = None,
    windows_total: Optional[int] = None,
) -> LearnedLocalMap:
    """
    Learn F: (length-W window) -> output, using only *interior* indices so we never
    depend on per-segment wrap. With context-added segments, this recovers all k**W windows.
    """
    table: Dict[Window, Symbol] = {}
    half = W // 2
    for x, y in train_pairs:
        if len(x) != len(y):
            raise ValueError("Input/output lengths in a training pair must match.")
        n = len(x)
        for i in range(half, n - half):
            win = _centered_window(x, i, W, wrap=True)
            val = y[i]
            prev = table.get(win)
            if prev is None:
                table[win] = val
            elif prev != val:
                raise ValueError(f"Inconsistent mapping for window {win}: saw {prev} then {val}.")
    return LearnedLocalMap(
        W=W,
        wrap=wrap,
        mapping=table,
        alphabet_size=alphabet_size,
        windows_total=windows_total,
    )


def _infer_wrap_from_record(rec: dict) -> bool:
    try:
        meta = rec.get("meta", {})
        if "wrap" in meta:
            return bool(meta["wrap"])
    except Exception:
        pass
    try:
        return bool(rec["program"]["ca"]["wrap"])
    except Exception:
        return True


def learn_from_record(rec: dict) -> LearnedLocalMap:
    meta = rec.get("meta", {})
    W = int(meta["window"])
    k = int(meta.get("alphabet_size", 0)) or None
    windows_total = int(meta.get("windows_total", 0)) or None

    wrap = _infer_wrap_from_record(rec)

    train_pairs = [(pair["input"], pair["output"]) for pair in rec["train"]]
    return learn_local_map_from_pairs(
        train_pairs=train_pairs,
        W=W,
        wrap=wrap,
        alphabet_size=k,
        windows_total=windows_total,
    )
