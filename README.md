# tinybio

GPU-accelerated single-cell RNA-seq preprocessing on [tinygrad](https://github.com/tinygrad/tinygrad), targeting **AMD eGPU on macOS** — the niche that every other GPU bioinformatics tool skips (most assume CUDA or Linux ROCm).

> Status: **pre-alpha**. Milestone 1 (PCA prototype on PBMC3k) passes numerically; speed story scales up at M3 (PBMC68k). See [ROADMAP.md](./ROADMAP.md).

## Why

`scanpy`'s PCA runs on CPU. `rapids-singlecell` needs NVIDIA. On an AMD eGPU + macOS — increasingly common for researchers who want GPU compute without a Linux workstation — there is no GPU-accelerated path. `tinybio` fills that gap using tinygrad's AMD backend.

## Benchmarks (AMD RX 7900 XT via Thunderbolt 4, macOS 26.3, M4 Pro host)

Top-50 PCA, numerical agreement against scanpy's arpack reference (abs cosine similarity per component, sign-flip safe). Warm wall times are median of 3 runs.

| Dataset | Shape after HVG | tinybio (AMD) | scanpy (CPU arpack) | ratio | top-10 min cos |
|---|---|---:|---:|---:|---:|
| **PBMC3k**  | 2,700 × 2,000  |   90 ms | 87 ms   | 0.96× (tied) | 0.9994 |
| **PBMC68k** | 68,579 × 2,000 | 1703 ms | 1685 ms | 0.99× (tied) | 1.0000 |

Synthetic scale-up across N cells at d=2000 features (rank-50-plus-noise + z-score), top-50 PCA:

| N | tinybio (ms) | scanpy (ms) | ratio |
|---:|---:|---:|---:|
|     3,000 |    96 |    53 | 0.55× |
|    10,000 |   265 |   205 | 0.77× |
|    30,000 |   654 |   680 | **1.04×** (GPU win) |
|   100,000 |  2237 |  3170 | **1.42×** (GPU win) |

**Reading.** Numerical correctness is solid everywhere (top-10 cos ≥ 0.999 on both real datasets, top-5 singular values match scanpy to 3 dp). On real scRNA-seq data, tinybio is neck-and-neck with scanpy's arpack on both PBMC3k and PBMC68k. On synthetic dense matrices with a typical single-cell spectrum shape, tinybio pulls ahead from ~30k cells and reaches **1.42× at 100k**.

The workload is **launch-bound + transfer-bound**, not compute-bound on this eGPU. `DEV=AMD JITBEAM=2` is a no-op (157 vs 160 ms on PBMC3k, see [BUGS.md](./BUGS.md)) because per-kernel autotune can't help dispatch latency. The one-sided power step `X @ (X.T @ Q)` is wrapped in `@TinyJit` with a shape-keyed cache (see [tinybio/pca.py](./tinybio/pca.py)) so the two matmuls fuse to a single dispatch; that alone moved PBMC3k from 160 → 90 ms (1.8× speedup).

**Running benchmarks:**
```bash
DEV=AMD python3 examples/pbmc3k_pca.py       # small
DEV=AMD python3 examples/pbmc68k_pca.py      # real 68k (first run downloads ~124 MB)
DEV=AMD python3 examples/scale_study.py      # synthetic scale curve
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
