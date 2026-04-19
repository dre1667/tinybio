#!/usr/bin/env python3
"""M3-ish scale-up: PBMC68k (68,579 cells × 32,738 genes) PCA on AMD eGPU.

Downloads the 10x Genomics "Fresh 68k PBMCs (Donor A)" filtered matrix
(~124 MB tarball, cached under data/), runs scanpy's standard preprocessing
(normalize_total → log1p → top-2000 HVGs → scale) once, then compares
top-50 PCA via tinybio randomized truncated SVD (AMD eGPU) against
scanpy.pp.pca (CPU arpack) on the same preprocessed matrix.

Run:
    DEV=AMD python3 examples/pbmc68k_pca.py
"""
from __future__ import annotations

import tarfile
import time
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from tinygrad import Tensor

from sklearn.utils.extmath import randomized_svd as sk_randomized_svd
from tinybio.normalize import scale as tb_scale
from tinybio.pca import pca as tb_pca

N_COMPONENTS = 50
N_TOP_HVG = 2000
N_RUNS = 3
DATA_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
    "fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
)
DATA_DIR = Path("data/pbmc68k")
MATRIX_SUBDIR = DATA_DIR / "filtered_matrices_mex/hg19"


def download_if_missing() -> Path:
    if MATRIX_SUBDIR.exists():
        print(f"Using cached data: {MATRIX_SUBDIR}")
        return MATRIX_SUBDIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "pbmc68k.tar.gz"
    if not tar_path.exists():
        print(f"Downloading {DATA_URL} → {tar_path} (~124 MB)...", flush=True)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "tinybio/0.0 (+https://github.com/dre1667/tinybio)"})
        with urllib.request.urlopen(req) as resp, open(tar_path, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    print(f"Extracting {tar_path} into {DATA_DIR}...", flush=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(DATA_DIR)
    return MATRIX_SUBDIR


def load_and_prep() -> np.ndarray:
    mtx_dir = download_if_missing()
    print(f"Reading 10x matrix from {mtx_dir}...", flush=True)
    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)
    print(f"  raw: {adata.shape[0]} cells × {adata.shape[1]} genes, dtype={adata.X.dtype}")

    print("Preprocessing (normalize_total → log1p → HVG; sparse, CPU)...", flush=True)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_HVG, flavor="seurat", subset=True)
    print("Densifying HVG subset → GPU scale...", flush=True)
    X_np = adata.X.toarray().astype(np.float32, copy=False)
    del adata
    X_tg = tb_scale(Tensor(X_np).realize()).realize()
    X = X_tg.numpy()
    print(f"  preprocessed: {X.shape[0]} cells × {X.shape[1]} genes, dtype={X.dtype}, "
          f"size={X.nbytes/1e6:.1f} MB")
    return X


# Measured on PBMC68k: n_iter=7 hits top-10 cos 0.999953 vs n_iter=10's
# 0.999960 — both well above the 0.999 target — but at ~35% less compute.
# Real scRNA-seq spectra converge faster than the worst-case default.
N_ITER = 7


def tinygrad_pca(X_np: np.ndarray):
    X = Tensor(X_np).realize()
    return tb_pca(X, n_components=N_COMPONENTS, n_iter=N_ITER)


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


def time_fn(fn, X, n_runs: int) -> tuple[float, float]:
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(min(ts))


def main() -> None:
    X = load_and_prep()

    print("\nWarming tinygrad kernels (cold run)...", flush=True)
    t0 = time.perf_counter()
    emb_tg, s_tg = tinygrad_pca(X)
    print(f"  cold: {(time.perf_counter()-t0)*1000:.1f} ms")

    print("\nscanpy reference PCA (for numerical verification)...", flush=True)
    t0 = time.perf_counter()
    emb_sc, s_sc = scanpy_pca(X)
    t_sc_first = time.perf_counter() - t0
    print(f"  first: {t_sc_first*1000:.1f} ms")

    cos = abs_cosine_per_component(emb_tg, emb_sc)
    print("\nPer-component cosine similarity (abs, sign-flip safe):")
    print(f"  top-10 min / mean: {cos[:10].min():.6f} / {cos[:10].mean():.6f}")
    print(f"  top-50 min / mean: {cos[:50].min():.6f} / {cos[:50].mean():.6f}")
    print(f"  per-PC cos[:10]:   {np.array2string(cos[:10], precision=4, suppress_small=True)}")

    print("\nSingular values, top 5:")
    print(f"  tinygrad: {np.array2string(s_tg[:5], precision=3)}")
    print(f"  scanpy  : {np.array2string(s_sc[:5], precision=3)}")

    print(f"\nBenchmark: median of {N_RUNS} warm runs each", flush=True)
    tg_med, tg_min = time_fn(tinygrad_pca, X, N_RUNS)
    sc_med, sc_min = time_fn(scanpy_pca, X, N_RUNS)
    sk_med, sk_min = time_fn(sklearn_pca, X, N_RUNS)
    print(f"  tinybio  median: {tg_med*1000:>8.1f} ms")
    print(f"  sklearn  median: {sk_med*1000:>8.1f} ms")
    print(f"  scanpy   median: {sc_med*1000:>8.1f} ms")

    print(f"\n{'='*78}")
    print(f"PBMC68k n={X.shape[0]:,}  top-50 PCA")
    print(f"{'='*78}")
    print(f"  tinybio (AMD eGPU):            {tg_med*1000:>10,.0f} ms")
    print(f"  sklearn randomized_svd (CPU):  {sk_med*1000:>10,.0f} ms   → GPU {sk_med/tg_med:>5.2f}x faster (same algorithm)")
    print(f"  scanpy (CPU, arpack):          {sc_med*1000:>10,.0f} ms   → GPU {sc_med/tg_med:>5.2f}x faster (real-user default)")

    top10_min = float(cos[:10].min())
    print(f"\nTop-10 min/mean cosine vs scanpy: {cos[:10].min():.6f} / {cos[:10].mean():.6f}")
    print(f"  numerical: {'PASS' if top10_min >= 0.999 else 'FAIL'} (target >= 0.999)")


if __name__ == "__main__":
    main()
