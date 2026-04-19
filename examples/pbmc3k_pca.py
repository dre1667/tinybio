#!/usr/bin/env python3
"""M1 gate prototype: does tinygrad GPU PCA on AMD match / beat scanpy CPU PCA?

Pipeline: PBMC3k (2700 cells x 32738 genes) -> CPM+log1p -> top-2000 HVGs
-> per-gene z-score -> top-50 PCs via tinybio randomized truncated SVD on
AMD (see tinybio/pca.py), and via scanpy.pp.pca on CPU. Compare cosine
similarity per component (sign-flip safe) and benchmark wall time
(median of N_RUNS warm runs each).

Run:
    DEV=AMD python3 examples/pbmc3k_pca.py             # simple cold
    DEV=AMD JITBEAM=2 python3 examples/pbmc3k_pca.py   # with autotune
"""
from __future__ import annotations

import time

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.utils.extmath import randomized_svd as sk_randomized_svd
from tinygrad import Tensor

from tinybio import hvg, normalize
from tinybio.pca import pca as tb_pca


N_COMPONENTS = 50
N_TOP_HVG = 2000
N_RUNS = 5


def load_and_prep() -> np.ndarray:
    """PBMC3k -> CPM -> log1p -> top HVG by log-variance -> per-gene z-score.

    All per-cell and per-gene normalization runs through the tinybio modules
    on the GPU; HVG selection round-trips only the (n_genes,) variance vector
    to CPU for the argpartition.
    """
    adata = sc.datasets.pbmc3k()
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = Tensor(X.astype(np.float32))

    X = normalize.log1p(normalize.cpm(X))
    idx = hvg.select_highly_variable(X, n_top=N_TOP_HVG)
    X = X[:, idx.tolist()]
    X = normalize.scale(X)
    return X.realize().numpy()


def tinygrad_pca(X_np: np.ndarray, n_components: int = N_COMPONENTS):
    """GPU PCA via randomized truncated SVD. Returns (embedding, singular_values)."""
    X = Tensor(X_np).realize()
    return tb_pca(X, n_components=n_components)


def scanpy_pca(X_np: np.ndarray, n_components: int = N_COMPONENTS):
    """CPU PCA via scanpy.pp.pca. Returns (embedding, singular_values)."""
    adata = ad.AnnData(X=X_np.copy())
    sc.pp.pca(adata, n_comps=n_components)
    emb = np.asarray(adata.obsm["X_pca"])
    var = np.asarray(adata.uns["pca"]["variance"])
    s = np.sqrt(np.clip(var, 0.0, None) * (X_np.shape[0] - 1))
    return emb, s


def sklearn_pca(X_np: np.ndarray, n_components: int = N_COMPONENTS):
    """CPU randomized SVD — same algorithm as tinybio, apples-to-apples."""
    U, S, _ = sk_randomized_svd(X_np, n_components=n_components, n_iter=10, random_state=0)
    return U * S, S


def abs_cosine_per_component(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.linalg.norm(A, axis=0, keepdims=True).clip(1e-12)
    B = B / np.linalg.norm(B, axis=0, keepdims=True).clip(1e-12)
    return np.abs((A * B).sum(axis=0))


def time_once(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return time.perf_counter() - t0, out


def main() -> None:
    print(f"Loading PBMC3k + preprocessing (CPM, log1p, top-{N_TOP_HVG} HVGs, z-score)...", flush=True)
    X = load_and_prep()
    print(f"  matrix: {X.shape[0]} cells x {X.shape[1]} genes, dtype={X.dtype}")

    print("\nWarming tinygrad kernels (cold run; includes JIT + any autotune)...", flush=True)
    t_cold, (emb_tg, s_tg) = time_once(tinygrad_pca, X)
    print(f"  cold run: {t_cold*1000:.1f} ms")

    print("\nscanpy reference PCA (for numerical verification)...", flush=True)
    _, (emb_sc, s_sc) = time_once(scanpy_pca, X)

    cos = abs_cosine_per_component(emb_tg, emb_sc)
    print("\nPer-component cosine similarity (abs, sign-flip safe):")
    print(f"  top-10 min / mean: {cos[:10].min():.6f} / {cos[:10].mean():.6f}")
    print(f"  top-50 min / mean: {cos[:50].min():.6f} / {cos[:50].mean():.6f}")
    print(f"  per-PC cos[:10]:   {np.array2string(cos[:10], precision=4, suppress_small=True)}")

    print("\nSingular values, top 5:")
    print(f"  tinygrad: {np.array2string(s_tg[:5], precision=3)}")
    print(f"  scanpy  : {np.array2string(s_sc[:5], precision=3)}")

    print(f"\nBenchmark: median of {N_RUNS} warm runs each", flush=True)
    def med(fn):
        return float(np.median([time_once(fn, X)[0] for _ in range(N_RUNS)]))
    tg_med = med(tinygrad_pca)
    sk_med = med(sklearn_pca)
    sc_med = med(scanpy_pca)

    print(f"\n{'='*78}")
    print(f"PBMC3k n={X.shape[0]:,}  top-50 PCA")
    print(f"{'='*78}")
    print(f"  tinybio (AMD eGPU):            {tg_med*1000:>10,.0f} ms")
    print(f"  sklearn randomized_svd (CPU):  {sk_med*1000:>10,.0f} ms   → GPU {sk_med/tg_med:>5.2f}x faster (same algorithm)")
    print(f"  scanpy (CPU, arpack):          {sc_med*1000:>10,.0f} ms   → GPU {sc_med/tg_med:>5.2f}x faster (real-user default)")

    top10_min = float(cos[:10].min())
    print(f"\nTop-10 min/mean cosine vs scanpy: {cos[:10].min():.6f} / {cos[:10].mean():.6f}")
    print(f"  numerical: {'PASS' if top10_min >= 0.999 else 'FAIL'} (target >= 0.999)")


if __name__ == "__main__":
    main()
