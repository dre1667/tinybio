# tinybio

GPU-accelerated single-cell RNA-seq preprocessing on [tinygrad](https://github.com/tinygrad/tinygrad), targeting **AMD eGPU on macOS** — the niche that every other GPU bioinformatics tool skips (most assume CUDA or Linux ROCm).

> Status: **pre-alpha**. Milestone 1 (PCA prototype on PBMC3k) passes numerically; speed story scales up at M3 (PBMC68k). See [ROADMAP.md](./ROADMAP.md).

## Why

`scanpy`'s PCA runs on CPU. `rapids-singlecell` needs NVIDIA. On an AMD eGPU + macOS — increasingly common for researchers who want GPU compute without a Linux workstation — there is no GPU-accelerated path. `tinybio` fills that gap using tinygrad's AMD backend.

## Benchmarks (AMD RX 7900 XT via Thunderbolt 4, macOS 26.3, M4 Pro host)

Top-50 PCA, numerical agreement against scanpy's arpack reference (abs cosine similarity per component, sign-flip safe). Warm wall times are median of 3 runs.

### Real scRNA-seq datasets

| Dataset | Shape after HVG | tinybio (AMD) | scanpy (CPU arpack) | speedup | top-10 min cos |
|---|---|---:|---:|---:|---:|
| **PBMC3k**  | 2,700 × 2,000   |   56 ms | 88 ms   | **1.57×** | 0.9997 |
| **PBMC68k** | 68,579 × 2,000  |  564 ms | 1653 ms | **2.93×** | 1.0000 |
| **Tabula Muris Senis** droplet, all tissues | 245,389 × 2,000 | 2660 ms end-to-end / 1972 ms resident | 3580 ms | **1.35×** / **1.81×** | 0.9999 |

"End-to-end" transfers X from numpy to GPU each call (what you'd measure feeding scanpy input into tinybio in isolation). "Resident" measures PCA only, with X already on the GPU — what you'd get if preprocessing also ran through tinybio so data never leaves the device. Both are reported honestly.

### Synthetic scale-up

(N cells × 2,000 features, rank-50-plus-noise + z-score, top-50 PCA)

| N | tinybio (ms) | scanpy (ms) | speedup |
|---:|---:|---:|---:|
|     3,000 |    57 |    53 |    0.93× (tied) |
|    10,000 |    88 |   200 | **2.28×** |
|    30,000 |   196 |   747 | **3.81×** |
|   100,000 |   491 |  2822 | **5.75×** |

**Reading.** Numerical correctness is solid everywhere (top-10 cos ≥ 0.999 on all three real datasets, top-5 singular values match scanpy to 3 dp). tinybio is faster than scanpy's CPU arpack on every benchmark past PBMC3k, and the gap widens with scale. The synthetic curve shows clean super-linear speedup: **5.75× at 100k cells**. On real data the ratio is a bit tighter because scanpy's Lanczos converges faster on sharply-decaying real scRNA-seq spectra — but even Tabula Muris Senis at 245k cells shows a solid GPU win.

### What made it fast

1. **Randomized truncated SVD (Halko 2011)**, not tinygrad's built-in `Tensor.svd()`. The built-in Jacobi sweep issues ~10⁶ kernel launches on a 2000-wide matrix — unusable on Thunderbolt. See [BUGS.md](./BUGS.md).
2. **`@TinyJit` on the power step** `X @ (X.T @ Q)` with a shape-keyed cache — fuses the two matmuls into one captured graph so dispatch latency is paid once per call instead of per matmul.
3. **GPU-resident orthonormalization** via eigendecomposition of the tiny `(l, l)` Gram matrix — transfers only ~25 kB per step instead of the full `(rows, l)` probe. At 245k cells this alone turned the PCA from ~7.6 s to ~1.97 s (see [tinybio/pca.py:`_chol_qr`](./tinybio/pca.py)).
4. `DEV=AMD JITBEAM=2` is a **no-op** here — the workload is launch/memory-bandwidth bound, not compute-bound, so per-kernel autotune doesn't help.

### Running the benchmarks

```bash
DEV=AMD python3 examples/pbmc3k_pca.py                   # 2700 cells
DEV=AMD python3 examples/pbmc68k_pca.py                  # 68k cells (first run: ~124 MB download)
DEV=AMD python3 examples/tabula_muris_senis_pca.py       # 245k cells (first run: ~4 GB download)
DEV=AMD python3 examples/scale_study.py                  # synthetic N-sweep
```

## Install (planned)

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install git+https://github.com/tinygrad/tinygrad.git
pip install tinybio
```

## Quickstart (planned)

```python
import scanpy as sc
import tinybio as tb
from tinygrad import Tensor

adata = sc.datasets.pbmc3k()
X = preprocess(adata)              # CPM, log1p, HVG, z-score (your pipeline)
pcs, singular_values = tb.pca.pca(Tensor(X), n_components=50)
```

## License

MIT. See [LICENSE](./LICENSE).
