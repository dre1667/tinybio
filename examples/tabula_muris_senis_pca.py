#!/usr/bin/env python3
"""Atlas-scale PCA benchmark on Tabula Muris Senis droplet (~245k cells).

Downloads the combined raw-counts h5ad from the CZ Biohub S3 bucket
(~4 GB, cached under data/tms/), runs scanpy's standard preprocessing
pipeline (normalize_total → log1p → top-2000 HVGs → scale) once on all
~245k mouse cells across ~20 tissues, then compares top-50 PCA:
  - tinybio randomized truncated SVD on the AMD eGPU
  - scanpy.pp.pca on CPU (arpack)

This is the "atlas scale" comparison — the regime the paper's headline
speedup story lives in.

Run:
    DEV=AMD python3 examples/tabula_muris_senis_pca.py
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
DATA_URL = (
    "https://czb-tabula-muris-senis.s3.us-west-2.amazonaws.com/"
    "Data-objects/tabula-muris-senis-droplet-official-raw-obj.h5ad"
)
DATA_DIR = Path("data/tms")
H5AD_PATH = DATA_DIR / "droplet-raw.h5ad"


def download_if_missing() -> Path:
    if H5AD_PATH.exists():
        print(f"Using cached h5ad: {H5AD_PATH} ({H5AD_PATH.stat().st_size/1e9:.2f} GB)")
        return H5AD_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL}\n  → {H5AD_PATH} (~4 GB, may take several minutes)...", flush=True)
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "tinybio/0.0 (+https://github.com/dre1667/tinybio)"})
    with urllib.request.urlopen(req) as resp, open(H5AD_PATH, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        last_pct = -1
        while chunk := resp.read(1 << 22):  # 4 MB chunks
            out.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total) if total else -1
            if pct != last_pct:
                print(f"  {pct:3d}%  ({done/1e9:.2f} / {total/1e9:.2f} GB)", flush=True)
                last_pct = pct
    return H5AD_PATH


def load_and_prep() -> np.ndarray:
    h5ad = download_if_missing()
    print(f"Reading {h5ad}...", flush=True)
    adata = ad.read_h5ad(h5ad)
    print(f"  raw: {adata.shape[0]} cells × {adata.shape[1]} genes, X dtype={adata.X.dtype}")

    # If the raw .X is already log/normalized (processed), revert to raw if present.
    if "raw" in dir(adata) and adata.raw is not None:
        print("  using adata.raw for counts")
        adata = ad.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.raw.var)

    # Sparse CPU for CPM/log1p/HVG (these stay sparse — tinygrad doesn't
    # support sparse, and densifying the raw 245k x 20k matrix ~20 GB over
    # TB4 would cost more than the scipy sparse ops themselves).
    print("Preprocessing (normalize_total → log1p → HVG; sparse, CPU)...", flush=True)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_HVG, flavor="seurat", subset=True)
    # Dense HVG subset (~2 GB) is worth transferring to GPU — z-score is
    # embarrassingly parallel over rows and columns, and scanpy's sc.pp.scale
    # double-allocates on CPU (the reason the 1M case swap-thrashes).
    print("Densifying HVG subset → GPU scale...", flush=True)
    X_np = adata.X.toarray().astype(np.float32, copy=False)
    del adata
    X_tg = tb_scale(Tensor(X_np).realize()).realize()
    X = X_tg.numpy()
    print(f"  preprocessed: {X.shape[0]} cells × {X.shape[1]} genes, dtype={X.dtype}, "
          f"size={X.nbytes/1e9:.2f} GB")
    return X


# Atlas-scale scRNA-seq spectra decay sharply — power iteration converges in
# far fewer steps than on small heterogeneous datasets. Measured on TMS:
#   n_iter=4: top-10 cos 0.999935, wall 946 ms
#   n_iter=10 (default): top-10 cos 0.999936, wall 1966 ms
# Using n_iter=5 gives identical accuracy with ~2x less compute.
N_ITER = 5


def tinygrad_pca(X_np: np.ndarray):
    """End-to-end: fresh numpy → Tensor on GPU → PCA → numpy."""
    X = Tensor(X_np).realize()
    return tb_pca(X, n_components=N_COMPONENTS, n_iter=N_ITER)


def tinygrad_pca_resident(X_tg: Tensor):
    """Compute-only: X already on GPU (data resident across multiple ops)."""
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


def time_fn(fn, X, n_runs: int) -> tuple[float, float]:
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(min(ts))


def main() -> None:
    X = load_and_prep()

    print("\nWarming tinygrad kernels (cold run includes JIT capture + 1.96 GB transfer)...", flush=True)
    t0 = time.perf_counter()
    emb_tg, s_tg = tinygrad_pca(X)
    print(f"  cold: {(time.perf_counter()-t0)*1000:.1f} ms")

    print("\nscanpy reference PCA (for numerical verification)...", flush=True)
    t0 = time.perf_counter()
    emb_sc, s_sc = scanpy_pca(X)
    print(f"  first: {(time.perf_counter()-t0)*1000:.1f} ms")

    cos = abs_cosine_per_component(emb_tg, emb_sc)
    print("\nPer-component cosine similarity (abs, sign-flip safe):")
    print(f"  top-10 min / mean: {cos[:10].min():.6f} / {cos[:10].mean():.6f}")
    print(f"  per-PC cos[:10]:   {np.array2string(cos[:10], precision=4, suppress_small=True)}")

    print("\nSingular values, top 5:")
    print(f"  tinygrad: {np.array2string(s_tg[:5], precision=3)}")
    print(f"  scanpy  : {np.array2string(s_sc[:5], precision=3)}")

    print(f"\nBenchmark: median of {N_RUNS} warm runs each", flush=True)
    tg_med, tg_min = time_fn(tinygrad_pca, X, N_RUNS)
    sc_med, sc_min = time_fn(scanpy_pca, X, N_RUNS)
    print(f"  tinygrad end-to-end  median: {tg_med*1000:>9.1f} ms   min: {tg_min*1000:>9.1f} ms")
    print(f"  scanpy                median: {sc_med*1000:>9.1f} ms   min: {sc_min*1000:>9.1f} ms")
    speedup = sc_med / tg_med
    if speedup >= 1.0:
        print(f"  tinygrad end-to-end is {speedup:.2f}x faster than scanpy")
    else:
        print(f"  tinygrad end-to-end is {1/speedup:.2f}x slower than scanpy")

    print("\nCompute-only (X pre-transferred to GPU, no per-call TB4 copy):", flush=True)
    X_tg = Tensor(X).realize()
    _ = tinygrad_pca_resident(X_tg)  # warm JIT for this path
    tgr_med, tgr_min = time_fn(tinygrad_pca_resident, X_tg, N_RUNS)
    print(f"  tinygrad resident    median: {tgr_med*1000:>9.1f} ms   min: {tgr_min*1000:>9.1f} ms")

    # sklearn randomized_svd (same algorithm as tinybio, but on CPU).
    print(f"\nsklearn randomized_svd reference ({N_RUNS} runs):", flush=True)
    sk_med, sk_min = time_fn(sklearn_pca, X, N_RUNS)
    print(f"  sklearn median: {sk_med*1000:>9.1f} ms   min: {sk_min*1000:>9.1f} ms")

    print(f"\n{'='*78}")
    print(f"TMS droplet n={X.shape[0]:,}  top-50 PCA")
    print(f"{'='*78}")
    print(f"  tinybio (AMD eGPU, resident):  {tgr_med*1000:>10,.0f} ms")
    print(f"  sklearn randomized_svd (CPU):  {sk_med*1000:>10,.0f} ms   → GPU {sk_med/tgr_med:>5.2f}x faster (same algorithm)")
    print(f"  scanpy (CPU, arpack):          {sc_med*1000:>10,.0f} ms   → GPU {sc_med/tgr_med:>5.2f}x faster (real-user default)")

    top10_min = float(cos[:10].min())
    print(f"\nTop-10 min/mean cosine vs scanpy: {cos[:10].min():.6f} / {cos[:10].mean():.6f}")
    print(f"  numerical: {'PASS' if top10_min >= 0.999 else 'FAIL'} (target >= 0.999)")


if __name__ == "__main__":
    main()
