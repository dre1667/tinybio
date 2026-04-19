#!/usr/bin/env python3
"""Quick profile: where does time go in tinybio PCA at TMS scale?

Isolates the three phases (apply_A matmul pair, CPU QR round-trip,
final SVD+emit) on a (245389, 2000) warm-JIT tensor so the output
tells us which dominates the 7.6 s wall time seen in
examples/tabula_muris_senis_pca.py.
"""
from __future__ import annotations

import time
import numpy as np
import anndata as ad
import scanpy as sc
from pathlib import Path
from tinygrad import Tensor

from tinybio.pca import _apply_A, _thin_qr_cpu, randomized_svd

H5AD = Path("data/tms/droplet-raw.h5ad")
N_TOP_HVG = 2000
N_COMPS = 50
OVERSAMPLE = 30
N_ITER = 10


def load():
    adata = ad.read_h5ad(H5AD)
    if adata.raw is not None:
        adata = ad.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.raw.var)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_HVG, flavor="seurat", subset=True)
    sc.pp.scale(adata, max_value=None)
    return np.ascontiguousarray(adata.X).astype(np.float32)


def main():
    X_np = load()
    m, n = X_np.shape
    l = N_COMPS + OVERSAMPLE
    print(f"Matrix: {m} x {n}  (l = {l})")

    print("\nTransfer X to GPU...", flush=True)
    t = time.perf_counter(); X = Tensor(X_np).realize(); print(f"  {(time.perf_counter()-t)*1000:.1f} ms")

    rng = np.random.default_rng(0)
    # apply_A(X, Q) computes X @ (X.T @ Q); Q must be (m, l).
    Y_seed = (X @ Tensor(rng.standard_normal((n, l)).astype(np.float32))).realize()
    Q_valid = _thin_qr_cpu(Y_seed)

    # Warm apply_A JIT (3 calls: non-JIT, capture, JIT-run)
    print("\nWarm up apply_A JIT (3 calls)...", flush=True)
    for i in range(3):
        t = time.perf_counter(); Y = _apply_A(X, Q_valid); _ = Y.realize().numpy().shape; print(f"  warmup {i+1}: {(time.perf_counter()-t)*1000:.1f} ms")

    print("\nTime apply_A in isolation (10 calls, JIT warmed):")
    ts = []
    for i in range(10):
        t = time.perf_counter()
        out = _apply_A(X, Q_valid)
        _ = out.realize().numpy().shape  # force full completion including transfer-back
        ts.append(time.perf_counter() - t)
    ts_np = np.array(ts) * 1000
    print(f"  per-call ms: {ts_np.round(1).tolist()}")
    print(f"  median: {np.median(ts_np):.1f} ms  min: {ts_np.min():.1f} ms  sum(10): {ts_np.sum():.0f} ms")

    print("\nTime apply_A without the .numpy() sync (just .realize()):")
    ts = []
    for i in range(10):
        t = time.perf_counter()
        out = _apply_A(X, Q_valid).realize()
        ts.append(time.perf_counter() - t)
    # Final sync to ensure GPU finished everything
    _ = out.numpy().shape
    ts_np = np.array(ts) * 1000
    print(f"  per-call ms: {ts_np.round(1).tolist()}")
    print(f"  median: {np.median(ts_np):.1f} ms  min: {ts_np.min():.1f} ms  sum(10): {ts_np.sum():.0f} ms")

    print("\nTime CPU thin QR (11 calls on fresh Y each time):")
    ts = []
    for i in range(11):
        Y_big = _apply_A(X, Q_valid).realize()
        t = time.perf_counter()
        Q_back = _thin_qr_cpu(Y_big)
        Q_back.realize()  # force transfer back
        ts.append(time.perf_counter() - t)
    ts_np = np.array(ts) * 1000
    print(f"  per-call ms: {ts_np.round(1).tolist()}")
    print(f"  median: {np.median(ts_np):.1f} ms  min: {ts_np.min():.1f} ms  sum(11): {ts_np.sum():.0f} ms")

    print("\nFull randomized_svd (warm):")
    for i in range(3):
        t = time.perf_counter()
        U, S, Vt = randomized_svd(X, n_components=N_COMPS, n_iter=N_ITER)
        print(f"  run {i+1}: {(time.perf_counter()-t)*1000:.1f} ms")


if __name__ == "__main__":
    main()
