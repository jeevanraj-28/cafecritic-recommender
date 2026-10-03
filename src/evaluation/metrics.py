# src/evaluation/metrics.py
"""Small, dependency-free evaluation helpers."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error."""
    pairs = list(zip(y_true, y_pred))
    return math.sqrt(sum((t - p) ** 2 for t, p in pairs) / len(pairs))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error."""
    pairs = list(zip(y_true, y_pred))
    return sum(abs(t - p) for t, p in pairs) / len(pairs)


def precision_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the top-k recommendations that are relevant."""
    relevant = set(relevant)
    top = list(recommended)[:k]
    return sum(item in relevant for item in top) / k if k else 0.0


def recall_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the relevant items that appear in the top-k recommendations."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return sum(item in relevant for item in list(recommended)[:k]) / len(relevant)
