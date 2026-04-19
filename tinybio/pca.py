"""Randomized truncated SVD for PCA.

tinygrad exposes ``Tensor.svd()``, but its one-sided Jacobi implementation
runs ``num * log2(num) * 2 + 2`` sweeps per call (~44k sweeps for a
2000-wide matrix, each sweep launching tens of small kernels). Over
Thunderbolt that is ~10^6 kernel launches and effectively unusable on
realistic single-cell data. See BUGS.md.

We use randomized SVD (Halko, Martinsson & Tropp 2011). Each PCA call
issues a constant number of large matmuls on the GPU, thin QRs on the
CPU, and one small SVD on the CPU at the end. The inner A-operator
application ``X @ (X.T @ Q)`` is captured with ``TinyJit`` so the two
matmuls dispatch as one graph, amortizing Thunderbolt launch latency
that otherwise dominates wall time for this workload.
"""
from __future__ import annotations

import numpy as np
from tinygrad import Tensor, TinyJit


def _thin_qr_cpu(Y: Tensor) -> Tensor:
    """Economy QR via numpy on CPU. Fast for (rows, l) with l small."""
    Y_np = Y.realize().numpy()
    Q_np, _ = np.linalg.qr(Y_np, mode="reduced")
    return Tensor(Q_np.astype(np.float32))


_APPLY_A_CACHE: dict[tuple, "TinyJit"] = {}


def _apply_A(X: Tensor, Q: Tensor) -> Tensor:
    """One-sided power step: returns ``X @ (X.T @ Q)``.

    Each unique ``(X.shape, Q.shape)`` pair is captured as its own JIT
    graph and cached, so the two matmuls fuse into a single dispatch on
    repeat calls. This is the load-bearing optimization for small-to-
    medium inputs where per-kernel launch latency dominates.
    """
    key = (tuple(X.shape), tuple(Q.shape))
    fn = _APPLY_A_CACHE.get(key)
    if fn is None:
        fn = TinyJit(lambda X_, Q_: (X_ @ (X_.T @ Q_)).realize())
        _APPLY_A_CACHE[key] = fn
    return fn(X, Q)


def randomized_svd(
    X: Tensor,
    n_components: int,
    n_iter: int = 10,
    oversample: int = 30,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-``n_components`` singular triples of ``X`` via randomized SVD.

    Parameters
    ----------
    X : tinygrad.Tensor
        ``(m, n)`` dense matrix, already on the target device.
    n_components : int
        Number of singular triples to return.
    n_iter : int
        Number of one-sided power iterations. Higher → more accurate
        but slower. Default 10 reaches top-10 cosine > 0.999 vs scanpy on
        scRNA-seq data in local testing; sklearn's ``randomized_svd``
        default is 7, which is a bit too lax for slow-decay spectra
        like PBMC3k.
    oversample : int
        Extra probe columns beyond ``n_components`` for stability. Higher
        values better resolve clusters of close singular values.
    seed : int | None
        Seed for the random probe. ``None`` uses numpy's global RNG.

    Returns
    -------
    (U, S, Vt) : (np.ndarray, np.ndarray, np.ndarray)
        ``U``  shape ``(m, k)`` — left singular vectors.
        ``S``  shape ``(k,)``  — singular values (descending).
        ``Vt`` shape ``(k, n)`` — right singular vectors as rows.
    """
    m, n = int(X.shape[0]), int(X.shape[1])
    k = n_components
    l = k + oversample
    if l >= min(m, n):
        raise ValueError(
            f"n_components + oversample ({l}) must be < min(m, n) ({min(m, n)}); "
            f"use numpy.linalg.svd directly for such small matrices."
        )

    rng = np.random.default_rng(seed)
    Omega = Tensor(rng.standard_normal(size=(n, l)).astype(np.float32))

    Y = (X @ Omega).realize()
    for _ in range(n_iter):
        Q = _thin_qr_cpu(Y)
        Y = _apply_A(X, Q)
    Q = _thin_qr_cpu(Y)

    B = (Q.T @ X).realize().numpy()
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)

    Ub_top = Ub[:, :k].astype(np.float32)
    U = (Q @ Tensor(Ub_top)).realize().numpy()
    return U, S[:k], Vt[:k]


def pca(
    X: Tensor,
    n_components: int = 50,
    n_iter: int = 10,
    oversample: int = 30,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA embedding via randomized truncated SVD.

    ``X`` must already be per-feature centered (and typically scaled to
    unit variance). Returns ``(embedding, singular_values)`` where
    ``embedding = U @ diag(S)`` has shape ``(m, n_components)``.
    """
    U, S, _ = randomized_svd(
        X, n_components=n_components, n_iter=n_iter, oversample=oversample, seed=seed
    )
    return U * S, S
