"""Unit tests for tinybio.normalize."""
from __future__ import annotations

import numpy as np
import pytest
from tinygrad import Tensor

from tinybio.normalize import cpm, log1p, scale


def _np(t: Tensor) -> np.ndarray:
    return t.realize().numpy()


def test_cpm_row_sums_to_target():
    X = Tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 2.0]])
    out = _np(cpm(X, target_sum=1.0))
    np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0], rtol=1e-6)


def test_cpm_default_target_is_million():
    X = Tensor([[1.0, 2.0, 3.0]])
    out = _np(cpm(X))
    np.testing.assert_allclose(out.sum(axis=1), [1e6], rtol=1e-6)


def test_cpm_handles_zero_row_without_nan():
    X = Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = _np(cpm(X, target_sum=3.0))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[1], [1.0, 1.0, 1.0], rtol=1e-6)


def test_log1p_matches_numpy():
    x = np.array([[0.0, 1.0, 2.718], [10.0, 1e-3, 100.0]], dtype=np.float32)
    out = _np(log1p(Tensor(x)))
    # fp32 log1p differs from numpy's by a few ulp; atol is looser than rtol
    # near zero where log1p(x) ~= x.
    np.testing.assert_allclose(out, np.log1p(x), atol=1e-6, rtol=1e-4)


def test_scale_produces_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((50, 10)).astype(np.float32) * 3.0 + 2.0
    out = _np(scale(Tensor(X_np)))
    np.testing.assert_allclose(out.mean(axis=0), np.zeros(10), atol=1e-5)
    # tinygrad's .std() is unbiased (ddof=1); numpy's default is ddof=0, so
    # the output's numpy-ddof-0 std is sqrt((N-1)/N) ~= 0.9899 here by design.
    np.testing.assert_allclose(out.std(axis=0, ddof=1), np.ones(10), atol=1e-3)


def test_scale_handles_zero_variance_column():
    # A column that's constant should not produce NaN/inf.
    X = Tensor([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])
    out = _np(scale(X))
    assert np.isfinite(out).all()
    # Constant column becomes (0 - 1) / eps — bounded but not tested; we just
    # care no NaN/inf escapes.


@pytest.mark.parametrize("n,d", [(5, 3), (100, 20)])
def test_pipeline_shape_preserved(n, d):
    rng = np.random.default_rng(7)
    X = Tensor(rng.integers(0, 20, size=(n, d)).astype(np.float32))
    Y = scale(log1p(cpm(X)))
    assert Y.shape == (n, d)
