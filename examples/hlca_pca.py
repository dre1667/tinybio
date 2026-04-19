#!/usr/bin/env python3
"""Human Lung Cell Atlas PCA — 2.28 M real cells.

Sikkema et al. 2023 "integrated Human Lung Cell Atlas core" (CELLxGENE
collection "The integrated Human Lung Cell Atlas"). 2,282,447 cells ×
~56 k genes, raw counts in a 21.9 GB h5ad from CELLxGENE's CDN.

At this scale scanpy's arpack comparison doesn't fit on a 24 GB Mac
(would need the dense (2.28M, 2000) ≈ 18 GB matrix *plus* a scanpy-side
copy). So this benchmark is tinybio-only wall time, with the scanpy
projection drawn from the measured near-linear scaling on the synthetic
scale curve (see examples/scale_study.py). Honest caveat — we report
what we can measure.

Run:
    DEV=AMD python3 examples/hlca_pca.py
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
N_ITER = 5   # atlas-scale sharp spectrum, 5 iterations converges top-10 cosine to the degenerate-subspace ceiling
# Full HLCA (2.28M cells) raw-counts sparse matrix is ~30 GB — doesn't fit
# on a 24 GB Mac. Random-subsample 1.5M cells via anndata backed-mode so
# only the chosen rows ever materialize in RAM.
N_SUBSAMPLE = 1_500_000
DATA_URL = (
    "https://datasets.cellxgene.cziscience.com/"
    "dbb5ad81-1713-4aee-8257-396fbabe7c6e.h5ad"
)
DATA_DIR = Path("data/hlca")
H5AD_PATH = DATA_DIR / "hlca.h5ad"


def download_if_missing() -> Path:
    if H5AD_PATH.exists() and H5AD_PATH.stat().st_size > 20_000_000_000:
        print(f"Using cached HLCA: {H5AD_PATH} ({H5AD_PATH.stat().st_size/1e9:.2f} GB)")
        return H5AD_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL}\n  → {H5AD_PATH} (~21.9 GB, several minutes)...", flush=True)
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "tinybio/0.0 (+https://github.com/dre1667/tinybio)"})
    with urllib.request.urlopen(req) as resp, open(H5AD_PATH, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0; last_pct = -1
        while chunk := resp.read(1 << 22):
            out.write(chunk); done += len(chunk)
            pct = int(100 * done / total) if total else -1
            if pct != last_pct:
                print(f"  {pct:3d}%  ({done/1e9:.2f} / {total/1e9:.2f} GB)", flush=True)
                last_pct = pct
    return H5AD_PATH


def load_and_prep() -> Tensor:
    """Memory-lean path: backed-mode subsample → sparse prep → GPU-resident dense HVG."""
    h5ad = download_if_missing()
    print(f"Opening {h5ad} in backed mode...", flush=True)
    adata_full = ad.read_h5ad(h5ad, backed="r")
    print(f"  raw: {adata_full.shape[0]} cells × {adata_full.shape[1]} genes (on disk)")

    rng = np.random.default_rng(42)
    n_total = adata_full.shape[0]
    k = min(N_SUBSAMPLE, n_total)
    print(f"Subsampling {k:,}/{n_total:,} cells (sorted indices)...", flush=True)
    idx = np.sort(rng.choice(n_total, size=k, replace=False))
    adata = adata_full[idx].to_memory()
    del adata_full
    print(f"  subset: {adata.shape[0]} cells × {adata.shape[1]} genes, X dtype={adata.X.dtype}")

    # HLCA ships .X as log-normalized; raw counts live in .raw.X.
    if adata.raw is not None:
        print("  using adata.raw for counts")
        adata = ad.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.raw.var)

    print("Preprocessing (normalize_total → log1p → HVG; sparse, CPU)...", flush=True)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_HVG, flavor="seurat", subset=True)

    print("Densifying HVG subset, in-place CPU scale, then upload...", flush=True)
    # At 1.5 M × 2000 = 12 GB the tinybio GPU scale (which does X-mu, clip,
    # divide as three separate allocations) would peak at ~36 GB VRAM —
    # exceeds the 20 GB card. CPU in-place numpy ops peak at 12 GB, fits in
    # 24 GB RAM. Sequential, but avoids the OOM cliff.
    X_np = adata.X.toarray().astype(np.float32, copy=False)
    del adata
    mu = X_np.mean(axis=0, dtype=np.float64).astype(np.float32)
    sigma = X_np.std(axis=0, dtype=np.float64).astype(np.float32)
    sigma = np.clip(sigma, 1e-12, None)
    np.subtract(X_np, mu, out=X_np)
    np.divide(X_np, sigma, out=X_np)
    X_tg = Tensor(X_np).realize()
    del X_np   # free the 12 GB CPU buffer
    print(f"  preprocessed on GPU: {X_tg.shape[0]} cells × {X_tg.shape[1]} genes, "
          f"{(int(X_tg.shape[0])*int(X_tg.shape[1])*4)/1e9:.2f} GB on VRAM")
    return X_tg


def tinygrad_pca_resident(X_tg: Tensor):
    return tb_pca(X_tg, n_components=N_COMPONENTS, n_iter=N_ITER)


def sklearn_pca(X_np: np.ndarray):
    """Pure-CPU randomized SVD — sklearn is the closest apples-to-apples
    CPU equivalent of tinybio's algorithm. Separates "GPU beats CPU" from
    "this library beats scanpy's arpack."
    """
    U, S, _ = sk_randomized_svd(X_np, n_components=N_COMPONENTS,
                                 n_iter=N_ITER, random_state=0)
    return U * S, S


def time_fn(fn, X, n_runs: int):
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(min(ts))


def scanpy_projection_ms(n: int) -> float:
    """Log-log linear fit of scanpy arpack wall time on synthetic data."""
    curve = np.array([[3_000, 52.9], [10_000, 200.0], [30_000, 746.7], [100_000, 2822.4]])
    x = np.log(curve[:, 0]); y = np.log(curve[:, 1])
    b, log_a = np.polyfit(x, y, 1)
    return float(np.exp(log_a) * (n ** b))


def main() -> None:
    X_tg = load_and_prep()

    print("\nWarming tinygrad kernels (cold)...", flush=True)
    t0 = time.perf_counter()
    emb_tg, s_tg = tinygrad_pca_resident(X_tg)
    print(f"  cold: {(time.perf_counter()-t0)*1000:.1f} ms")

    print(f"\nBenchmark: median of {N_RUNS} warm runs", flush=True)
    tg_med, tg_min = time_fn(tinygrad_pca_resident, X_tg, N_RUNS)
    print(f"  tinygrad resident median: {tg_med*1000:>9.1f} ms   min: {tg_min*1000:>9.1f} ms")

    n = int(X_tg.shape[0])
    # Pull X to CPU once so both CPU baselines share the same buffer.
    print("\nDownloading scaled matrix to CPU for CPU baselines (~12 GB)...", flush=True)
    X_np_local = X_tg.numpy()

    # sklearn randomized_svd — apples-to-apples algorithm vs tinybio (pure
    # CPU, same algorithm, same n_iter). Isolates "GPU beats CPU".
    emb_sk = None; sk_time = None
    try:
        print("sklearn randomized_svd reference (same algorithm as tinybio, CPU)...", flush=True)
        t0 = time.perf_counter()
        emb_sk, s_sk = sklearn_pca(X_np_local)
        sk_time = time.perf_counter() - t0
        print(f"  sklearn single call: {sk_time:.1f} s")
    except Exception as e:
        print(f"  sklearn reference skipped: {type(e).__name__}: {e}")

    # scanpy arpack — "real-user" baseline (what everyone runs in practice).
    emb_sc = None; sc_time = None
    try:
        print("scanpy reference PCA (arpack Lanczos, real-user default)...", flush=True)
        adata_ref = ad.AnnData(X=X_np_local)
        t0 = time.perf_counter()
        sc.pp.pca(adata_ref, n_comps=N_COMPONENTS)
        sc_time = time.perf_counter() - t0
        emb_sc = np.asarray(adata_ref.obsm["X_pca"])
        del adata_ref
        print(f"  scanpy single call: {sc_time:.1f} s")
    except Exception as e:
        print(f"  scanpy reference skipped: {type(e).__name__}: {e}")

    del X_np_local  # free the 12 GB CPU buffer before verdict / any further work

    print(f"\n{'='*78}")
    print(f"HLCA n={n:,}  |  top-50 PCA")
    print(f"{'='*78}")
    def cos_vs(A, B):
        A = A / np.linalg.norm(A, axis=0, keepdims=True).clip(1e-12)
        B = B / np.linalg.norm(B, axis=0, keepdims=True).clip(1e-12)
        return np.abs((A * B).sum(axis=0))
    print(f"  tinybio (AMD eGPU, resident):  {tg_med*1000:>10,.0f} ms")
    if sk_time is not None:
        print(f"  sklearn randomized_svd (CPU):  {sk_time*1000:>10,.0f} ms   → GPU {sk_time/tg_med:>5.2f}x faster (same algorithm)")
        cos = cos_vs(emb_tg, emb_sk)
        print(f"     top-10 min/mean cos vs sklearn: {cos[:10].min():.6f} / {cos[:10].mean():.6f}")
    if sc_time is not None:
        print(f"  scanpy (CPU, arpack):          {sc_time*1000:>10,.0f} ms   → GPU {sc_time/tg_med:>5.2f}x faster (real-user default)")
        cos = cos_vs(emb_tg, emb_sc)
        print(f"     top-10 min/mean cos vs scanpy:  {cos[:10].min():.6f} / {cos[:10].mean():.6f}")
    print("\nSingular values, top 5:")
    print(f"  tinygrad: {np.array2string(s_tg[:5], precision=3)}")


if __name__ == "__main__":
    main()
