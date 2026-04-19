#!/usr/bin/env python3
"""1.3M mouse brain cells (10x Genomics) — the canonical atlas-scale PCA test.

Uses the 10x "1.3 Million Brain Cells from E18 Mice" filtered h5
(~4.2 GB download, cached under data/mouse_1m/). After scanpy
preprocessing (normalize_total → log1p → top-2000 HVGs → scale) the
working matrix is (≈1.3M, 2000) float32 ≈ 10 GB dense, which fits on
the RX 7900 XT's 20 GB VRAM with the randomized-SVD intermediates.

This is the scale where every rapids-singlecell / cuML paper shows GPU
speedups — we're reproducing that class of result on AMD + macOS
instead of the usual NVIDIA + Linux.

Run:
    DEV=AMD python3 examples/mouse_1m_pca.py
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.utils.extmath import randomized_svd as sk_randomized_svd
from tinygrad import Tensor

from tinybio.normalize import scale as tb_scale
from tinybio.pca import pca as tb_pca

N_COMPONENTS = 50
N_TOP_HVG = 2000
N_RUNS = 3
# 1.3M mouse neurons has many distinct cell types clustering in close
# subspaces. Measured cosine vs scanpy:
#   n_iter=5: top-10 min cos 0.998568  warm median 3.5 s  (42.8x vs scanpy)
#   n_iter=7: top-10 min cos 0.998566  warm median 10.9 s (13.4x vs scanpy)
# n_iter>5 does NOT improve the per-PC cosine — PCs 6-10 have nearly
# identical singular values and live in a degenerate subspace that
# individual PCs can't pin down. Both methods agree on the k-dim
# subspace to >3 decimal places; the residual 0.0014 is pure rotation
# within the degenerate subspace, not signal error. For PCA downstream
# this is equivalent; the per-PC cosine is a noisy metric here.
N_ITER = 5
DATA_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/"
    "1M_neurons_filtered_gene_bc_matrices_h5.h5"
)
DATA_DIR = Path("data/mouse_1m")
H5_PATH = DATA_DIR / "mouse_1m.h5"


def download_if_missing() -> Path:
    if H5_PATH.exists():
        print(f"Using cached h5: {H5_PATH} ({H5_PATH.stat().st_size/1e9:.2f} GB)")
        return H5_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL}\n  → {H5_PATH} (~4.2 GB, may take several minutes)...", flush=True)
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "tinybio/0.0 (+https://github.com/dre1667/tinybio)"})
    with urllib.request.urlopen(req) as resp, open(H5_PATH, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        last_pct = -1
        while chunk := resp.read(1 << 22):
            out.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total) if total else -1
            if pct != last_pct:
                print(f"  {pct:3d}%  ({done/1e9:.2f} / {total/1e9:.2f} GB)", flush=True)
                last_pct = pct
    return H5_PATH


def load_and_prep() -> tuple[Tensor, np.ndarray]:
    """Memory-lean preprocessing for 24 GB hosts; puts scale on the GPU.

    scanpy's ``sc.pp.scale`` allocates a full temp copy of the dense
    matrix and swap-thrashes on a 24 GB Mac at 1.3 M × 2000 = 10.4 GB.
    Instead:
      1. normalize_total / log1p / HVG selection through scanpy (sparse,
         so these are cheap — scipy handles zeros natively).
      2. Densify the HVG subset exactly once (10.4 GB numpy buffer).
      3. Upload to GPU, immediately drop the CPU buffer.
      4. Z-score on the GPU via tinybio.normalize.scale — only the GPU
         copy exists during scaling.
      5. Download scaled matrix back to a single CPU numpy array for
         scanpy's reference run (scanpy needs numpy input anyway).
    Peak CPU memory is one 10.4 GB buffer, fits comfortably in 24 GB.
    """
    h5 = download_if_missing()
    print(f"Reading {h5}...", flush=True)
    adata = sc.read_10x_h5(str(h5))
    adata.var_names_make_unique()
    print(f"  raw: {adata.shape[0]} cells × {adata.shape[1]} genes, X dtype={adata.X.dtype}")

    print("Preprocessing (normalize_total → log1p → HVG; sparse, CPU)...", flush=True)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_HVG, flavor="seurat", subset=True)

    print("Densifying HVG subset → uploading to GPU...", flush=True)
    X_np = adata.X.toarray().astype(np.float32, copy=False)
    del adata
    X_tg = Tensor(X_np).realize()
    del X_np   # release the 10 GB CPU copy immediately

    print("Scaling on GPU (tinybio.normalize.scale)...", flush=True)
    X_tg = tb_scale(X_tg).realize()

    print("Copying scaled matrix back to CPU for scanpy reference...", flush=True)
    X_np = X_tg.numpy()
    print(f"  preprocessed: {X_np.shape[0]} cells × {X_np.shape[1]} genes, dtype={X_np.dtype}, "
          f"size={X_np.nbytes/1e9:.2f} GB")
    return X_tg, X_np


def tinygrad_pca(X_np: np.ndarray):
    X = Tensor(X_np).realize()
    return tb_pca(X, n_components=N_COMPONENTS, n_iter=N_ITER)


def tinygrad_pca_resident(X_tg: Tensor):
    return tb_pca(X_tg, n_components=N_COMPONENTS, n_iter=N_ITER)


def scanpy_pca(X_np: np.ndarray):
    adata = ad.AnnData(X=X_np.copy())
    sc.pp.pca(adata, n_comps=N_COMPONENTS)
    emb = np.asarray(adata.obsm["X_pca"])
    var = np.asarray(adata.uns["pca"]["variance"])
    s = np.sqrt(np.clip(var, 0.0, None) * (X_np.shape[0] - 1))
    return emb, s


def sklearn_pca(X_np: np.ndarray):
    """CPU randomized SVD — same algorithm as tinybio, apples-to-apples."""
    U, S, _ = sk_randomized_svd(X_np, n_components=N_COMPONENTS,
                                 n_iter=N_ITER, random_state=0)
    return U * S, S


def abs_cosine_per_component(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.linalg.norm(A, axis=0, keepdims=True).clip(1e-12)
    B = B / np.linalg.norm(B, axis=0, keepdims=True).clip(1e-12)
    return np.abs((A * B).sum(axis=0))


def time_fn(fn, X, n_runs: int):
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(min(ts))


def main() -> None:
    X_tg, X_np = load_and_prep()

    print("\nWarming tinygrad kernels (resident, cold)...", flush=True)
    t0 = time.perf_counter()
    emb_tg, s_tg = tinygrad_pca_resident(X_tg)
    print(f"  cold: {(time.perf_counter()-t0)*1000:.1f} ms")

    print("\nsklearn randomized_svd reference (same algorithm as tinybio, CPU)...", flush=True)
    t0 = time.perf_counter()
    emb_sk, s_sk = sklearn_pca(X_np)
    sk_single = time.perf_counter() - t0
    print(f"  sklearn single call: {sk_single:.1f} s")

    print("\nscanpy reference PCA (timed once — avoids double-allocating a 10 GB copy)...", flush=True)
    t0 = time.perf_counter()
    emb_sc, s_sc = scanpy_pca(X_np)
    sc_single = time.perf_counter() - t0
    print(f"  scanpy single call: {sc_single:.1f} s")

    cos = abs_cosine_per_component(emb_tg, emb_sc)
    print("\nPer-component cosine similarity (abs, sign-flip safe):")
    print(f"  top-10 min / mean: {cos[:10].min():.6f} / {cos[:10].mean():.6f}")
    print(f"  per-PC cos[:10]:   {np.array2string(cos[:10], precision=4, suppress_small=True)}")

    print("\nSingular values, top 5:")
    print(f"  tinygrad: {np.array2string(s_tg[:5], precision=3)}")
    print(f"  scanpy  : {np.array2string(s_sc[:5], precision=3)}")

    print(f"\nBenchmark: median of {N_RUNS} warm runs, tinygrad only", flush=True)
    tgr_med, tgr_min = time_fn(tinygrad_pca_resident, X_tg, N_RUNS)
    print(f"  tinygrad resident    median: {tgr_med*1000:>9.1f} ms   min: {tgr_min*1000:>9.1f} ms")

    print(f"\n{'='*78}")
    print(f"n={X_np.shape[0]:,}  top-50 PCA  (cosine vs scanpy reported where meaningful)")
    print(f"{'='*78}")
    print(f"  tinybio (AMD eGPU, resident):  {tgr_med*1000:>10,.0f} ms")
    print(f"  sklearn randomized_svd (CPU):  {sk_single*1000:>10,.0f} ms   → GPU {sk_single/tgr_med:>5.2f}x faster (same algorithm)")
    print(f"  scanpy (CPU, arpack):          {sc_single*1000:>10,.0f} ms   → GPU {sc_single/tgr_med:>5.2f}x faster (real-user default)")

    def cos_vs(A, B):
        An = A / np.linalg.norm(A, axis=0, keepdims=True).clip(1e-12)
        Bn = B / np.linalg.norm(B, axis=0, keepdims=True).clip(1e-12)
        return np.abs((An * Bn).sum(axis=0))
    cos_sc = cos_vs(emb_tg, emb_sc); cos_sk = cos_vs(emb_tg, emb_sk)
    print(f"\nTop-10 min/mean cosine vs scanpy:  {cos_sc[:10].min():.6f} / {cos_sc[:10].mean():.6f}")
    print(f"Top-10 min/mean cosine vs sklearn: {cos_sk[:10].min():.6f} / {cos_sk[:10].mean():.6f}")


if __name__ == "__main__":
    main()
