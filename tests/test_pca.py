"""Unit tests for tinybio.pca (randomized truncated SVD)."""
from __future__ import annotations

import numpy as np
import pytest
from tinygrad import Tensor

from tinybio.pca import pca, randomized_svd


def _abs_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.linalg.norm(A, axis=0, keepdims=True).clip(1e-12)
    B = B / np.linalg.norm(B, axis=0, keepdims=True).clip(1e-12)
    return np.abs((A * B).sum(axis=0))


def _synthetic_low_rank(n: int, d: int, rank: int = 30, noise: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, rank)).astype(np.float32))
    V, _ = np.linalg.qr(rng.standard_normal((d, rank)).astype(np.float32))
    s = np.linspace(50.0, 5.0, rank, dtype=np.float32)
    X = (U * s) @ V.T + noise * rng.standard_normal((n, d)).astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)
    return X.astype(np.float32)


def test_singular_values_match_numpy_on_low_rank():
    X_np = _synthetic_low_rank(400, 120, rank=30)
    _, S_t, _ = randomized_svd(Tensor(X_np), n_components=10, n_iter=7, seed=42)
    S_ref = np.linalg.svd(X_np, full_matrices=False)[1][:10]
    np.testing.assert_allclose(S_t, S_ref, rtol=1e-5)


def test_top_singular_vectors_match_numpy_on_low_rank():
    X_np = _synthetic_low_rank(400, 120, rank=30)
    U_t, _, _ = randomized_svd(Tensor(X_np), n_components=10, n_iter=7, seed=42)
    U_ref = np.linalg.svd(X_np, full_matrices=False)[0][:, :10]
    cos = _abs_cos(U_t, U_ref)
    assert cos.min() > 0.999


def test_returned_shapes():
    X_np = _synthetic_low_rank(200, 80, rank=20)
    U, S, Vt = randomized_svd(Tensor(X_np), n_components=7)
    assert U.shape == (200, 7)
    assert S.shape == (7,)
    assert Vt.shape == (7, 80)


def test_singular_values_descending():
    X_np = _synthetic_low_rank(300, 100, rank=25)
    _, S, _ = randomized_svd(Tensor(X_np), n_components=15)
    assert np.all(np.diff(S) <= 0)


def test_pca_embedding_shape_and_equals_U_times_S():
    X_np = _synthetic_low_rank(150, 60, rank=15)
    U, S, _ = randomized_svd(Tensor(X_np), n_components=8)
    emb, sv = pca(Tensor(X_np), n_components=8)
    assert emb.shape == (150, 8)
    np.testing.assert_allclose(sv, S, rtol=1e-5)
    np.testing.assert_allclose(emb, U * S, rtol=1e-5)


def test_rejects_oversample_too_large():
    X = Tensor(np.ones((20, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        randomized_svd(X, n_components=5, oversample=50)


def test_seed_reproducibility():
    X = Tensor(_synthetic_low_rank(200, 80, rank=20))
    U1, S1, _ = randomized_svd(X, n_components=6, seed=123)
    U2, S2, _ = randomized_svd(X, n_components=6, seed=123)
    np.testing.assert_allclose(U1, U2, rtol=1e-5)
    np.testing.assert_allclose(S1, S2, rtol=1e-5)
