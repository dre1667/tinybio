"""Unit tests for tinybio.hvg."""
from __future__ import annotations

import numpy as np
import pytest
from tinygrad import Tensor

from tinybio.hvg import select_highly_variable


def test_returns_top_n_by_variance():
    # Column variances in increasing order: col k has variance ~k.
    rng = np.random.default_rng(0)
    n = 200
    X_np = np.stack(
        [rng.standard_normal(n).astype(np.float32) * (k + 1) for k in range(10)],
        axis=1,
    )
    idx = select_highly_variable(Tensor(X_np), n_top=3)
    assert idx.tolist() == [7, 8, 9]


def test_returns_sorted_ascending():
    rng = np.random.default_rng(1)
    X = Tensor(rng.standard_normal((100, 20)).astype(np.float32))
    idx = select_highly_variable(X, n_top=5)
    assert np.all(np.diff(idx) > 0)


def test_handles_n_top_ge_n_genes():
    X = Tensor(np.ones((10, 5), dtype=np.float32))
    idx = select_highly_variable(X, n_top=100)
    assert idx.tolist() == list(range(5))


@pytest.mark.parametrize("n_top", [1, 5, 50])
def test_returns_requested_count(n_top):
    rng = np.random.default_rng(2)
    X = Tensor(rng.standard_normal((30, 100)).astype(np.float32))
    idx = select_highly_variable(X, n_top=n_top)
    assert idx.shape == (n_top,)
    assert idx.dtype == np.int64
