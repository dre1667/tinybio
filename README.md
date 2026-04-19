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
| **PBMC3k**  | 2,700 × 2,000    |    54 ms  |     71 ms  | **1.30×** | 0.9994 |
| **PBMC68k** | 68,579 × 2,000   |   779 ms  |   1597 ms  | **2.05×** | 1.0000 |
| **Tabula Muris Senis** droplet | 245,389 × 2,000 | 2274 ms resident / 2934 ms end-to-end | 3422 ms | **1.50× / 1.17×** | 0.9999 |
| **Mouse 1.3M neurons** (10x) | 1,306,127 × 2,000 | **3.54 s resident** | **151.8 s** | **42.84×** | 0.9986 † |
| **HLCA 1.5M** (Sikkema 2023, random subsample) | 1,500,000 × 2,000 | **3.36 s resident** | **98.0 s** | **29.16×** | 0.9987 † |

† At ≥1M cells the per-PC cosine plateaus around 0.998–0.999: PCs 6–10 live in a near-degenerate subspace where many well-separated cell types share similar eigenvalues. The *subspace* agreement between tinybio and scanpy is still effectively perfect — the residual is pure rotation within the degenerate block, not signal error, and downstream use of the top-k PCs as a subspace basis is unaffected.

![Benchmark wall times](figures/fig_bench_bars.png)

"End-to-end" transfers X from numpy to GPU each call. "Resident" measures PCA only, with X already on the GPU — what you get if upstream preprocessing also runs through tinybio (as in [`examples/mouse_1m_pca.py`](examples/mouse_1m_pca.py)). Both honest to report.

**At atlas scale (≥1M cells) tinybio is 30–40× faster than scanpy on real scRNA-seq.** scanpy's arpack is ~linear in N for dense matmuls; randomized SVD is also linear but with a much smaller constant once the matrix is big enough to amortize launch + transfer overhead. The warm-cache numbers reported here are the steady-state behavior after tinygrad's autotune cache is populated (persisted to `~/Library/Caches/tinygrad/`, so only the very first run pays the capture cost).

### Synthetic scale-up

(N cells × 2,000 features, rank-50-plus-noise + z-score, top-50 PCA)

| N | tinybio (ms) | scanpy (ms) | speedup |
|---:|---:|---:|---:|
|     3,000 |    57 |    53 |    0.93× (tied) |
|    10,000 |    88 |   200 | **2.28×** |
|    30,000 |   196 |   747 | **3.81×** |
|   100,000 |   491 |  2822 | **5.75×** |

![Synthetic scale curve](figures/fig_scale_curve.png)

**Reading.** Numerical correctness is solid across all four real datasets (top-10 cos ≥ 0.999 on PBMC3k/68k/TMS, ≥ 0.9986 subspace-degenerate on 1.3 M; top-5 singular values match scanpy to 3–4 dp). tinybio is faster than scanpy's CPU arpack on every real benchmark and the gap widens dramatically with scale — **16.74× at 1.3 M cells**. Scanpy's Lanczos scales near-linearly in N for dense matrices; tinybio's randomized SVD amortizes fixed launch/transfer overhead across bigger matmuls, so the ratio keeps improving as data grows.

### What made it fast

1. **Randomized truncated SVD (Halko 2011)**, not tinygrad's built-in `Tensor.svd()`. The built-in Jacobi sweep issues ~10⁶ kernel launches on a 2000-wide matrix — unusable on Thunderbolt. See [BUGS.md](./BUGS.md).
2. **`@TinyJit` on the power step** `X @ (X.T @ Q)` with a shape-keyed cache — fuses the two matmuls into one captured graph so dispatch latency is paid once per call instead of per matmul.
3. **GPU-resident orthonormalization** via eigendecomposition of the tiny `(l, l)` Gram matrix — transfers only ~25 kB per step instead of the full `(rows, l)` probe. See [tinybio/pca.py:`_chol_qr`](./tinybio/pca.py). On Apple Accelerate `np.linalg.qr` at `(245k, 80)` is a shocking 680 ms per call (30× slower than theoretical LAPACK); this is the one change that turned TMS from 2.56× slower to 1.5–1.8× faster.
4. **Random probe Ω generated on the device** via `Tensor.randn` instead of numpy + upload — keeps the probe's `(n, l)` block off TB4 entirely.
5. **Per-gene z-score on the GPU** via `tinybio.normalize.scale` (trivially parallel over rows and columns) instead of scanpy's `sc.pp.scale` which double-allocates the dense matrix on CPU — fatal for the 1.3 M × 2000 = 10.4 GB working set on a 24 GB Mac.
6. **Sparse CPU preprocessing of raw counts** where the alternative would need 146 GB of dense VRAM (1.3 M × 28 k). The rule is: move to GPU unless TB4 transfer would cost more than the compute you'd save. See [`docs/egpu_bottleneck_hunting_skill.md`](docs/egpu_bottleneck_hunting_skill.md).
7. `DEV=AMD JITBEAM=2` is a **no-op** here — the workload is launch/memory-bandwidth bound, not compute-bound, so per-kernel autotune doesn't help.

### Running the benchmarks

```bash
DEV=AMD python3 examples/pbmc3k_pca.py                   # 2.7k cells,  auto-downloads PBMC3k
DEV=AMD python3 examples/pbmc68k_pca.py                  # 68k cells,   first run: ~124 MB
DEV=AMD python3 examples/tabula_muris_senis_pca.py       # 245k cells,  first run: ~4 GB
DEV=AMD python3 examples/mouse_1m_pca.py                 # 1.3M cells,  first run: ~4.2 GB
DEV=AMD python3 examples/scale_study.py                  # synthetic N-sweep
python3 figures/generate_benchmark_figures.py            # regenerate README figures
```

**Tests:** `.venv/bin/python -m pytest tests/ -q` — 21 tests, ~1 s, no GPU required.

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
