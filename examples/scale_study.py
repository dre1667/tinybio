#!/usr/bin/env python3
"""Scale study: does tinybio GPU PCA cross over scanpy CPU PCA as N grows?

Synthetic z-scored rank-50-plus-noise matrices of shape (N, 2000) at a
log-scale of N. Top-50 PCs via tinybio randomized truncated SVD (AMD
eGPU) vs scanpy.pp.pca (CPU arpack). End-to-end wall time, median of 3
warm runs each (after one cold warmup).

Synthetic is honest here: PCA wall time is a function of matrix shape,
not biological content. Real PBMC68k download for the follow-up.

Run:
    DEV=AMD python3 examples/scale_study.py
"""
from __future__ import annotations

import time

import anndata as ad
import numpy as np
import scanpy as sc
from tinygrad import Tensor

from tinybio.pca import pca as tb_pca


N_COMPONENTS = 50
N_FEATURES = 2000
N_SIZES = [3_000, 10_000, 30_000, 100_000]
N_RUNS = 3
EFFECTIVE_RANK = 50


def synth_scaled(n: int, d: int, seed: int = 0) -> np.ndarray:
    """(n, d) z-scored float32 matrix with ~low-rank + noise structure."""
    rng = np.random.default_rng(seed)
    r = EFFECTIVE_RANK
    Ur, _ = np.linalg.qr(rng.standard_normal((n, r)).astype(np.float32))
    Vr, _ = np.linalg.qr(rng.standard_normal((d, r)).astype(np.float32))
    s = np.linspace(50.0, 5.0, r, dtype=np.float32)
    X = Ur @ np.diag(s) @ Vr.T + 0.1 * rng.standard_normal((n, d)).astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True)
    return ((X - mu) / np.clip(sig, 1e-12, None)).astype(np.float32)


def tinygrad_pca_end_to_end(X_np: np.ndarray):
    X = Tensor(X_np).realize()
    return tb_pca(X, n_components=N_COMPONENTS)


def scanpy_pca(X_np: np.ndarray):
    adata = ad.AnnData(X=X_np.copy())
    sc.pp.pca(adata, n_comps=N_COMPONENTS)
    return np.asarray(adata.obsm["X_pca"])


def median_time(fn, X, n_runs: int) -> float:
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> None:
    print(f"Scale study: top-{N_COMPONENTS} PCA, (N, {N_FEATURES}), median of {N_RUNS} warm runs")
    print(f"{'N':>8}  {'tinybio (ms)':>14}  {'scanpy (ms)':>14}  {'speedup':>10}")
    print(f"{'-'*8}  {'-'*14}  {'-'*14}  {'-'*10}")
    for n in N_SIZES:
        X = synth_scaled(n, N_FEATURES)
        # Warm-up (first call: JIT compile / arpack init).
        tinygrad_pca_end_to_end(X)
        scanpy_pca(X)
        tg = median_time(tinygrad_pca_end_to_end, X, N_RUNS)
        sc_t = median_time(scanpy_pca, X, N_RUNS)
        spdup = sc_t / tg if tg > 0 else float("inf")
        marker = "GPU win" if spdup >= 1.0 else "CPU win"
        print(f"{n:>8d}  {tg*1000:>14.1f}  {sc_t*1000:>14.1f}  {spdup:>9.2f}x  {marker}")


if __name__ == "__main__":
    main()
