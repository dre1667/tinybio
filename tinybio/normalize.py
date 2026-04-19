"""Per-cell and per-gene normalization for scRNA-seq count matrices.

All functions take and return ``tinygrad.Tensor`` so the whole pipeline
can stay on the GPU once the input matrix is transferred.
"""
from __future__ import annotations

from tinygrad import Tensor

_INF = float("inf")


def cpm(X: Tensor, target_sum: float = 1e6) -> Tensor:
    """Normalize each row (cell) to ``target_sum`` total counts."""
    cell_total = X.sum(axis=1, keepdim=True).clip(1e-12, _INF)
    return X / cell_total * target_sum


def log1p(X: Tensor) -> Tensor:
    """Element-wise ``log(1 + X)``."""
    return (X + 1.0).log()


def scale(X: Tensor, eps: float = 1e-12) -> Tensor:
    """Per-feature (column) z-score: ``(X - mean) / std``."""
    mu = X.mean(axis=0, keepdim=True)
    sigma = X.std(axis=0, keepdim=True)
    return (X - mu) / sigma.clip(eps, _INF)
