# tinybio

GPU-accelerated single-cell RNA-seq preprocessing on [tinygrad](https://github.com/tinygrad/tinygrad), targeting **AMD eGPU on macOS** — the niche that every other GPU bioinformatics tool skips (most assume CUDA or Linux ROCm).

> Status: **pre-alpha**. Milestone 1 (PCA prototype on PBMC3k) passes numerically; speed story scales up at M3 (PBMC68k). See [ROADMAP.md](./ROADMAP.md).

## Why

`scanpy`'s PCA runs on CPU. `rapids-singlecell` needs NVIDIA. On an AMD eGPU + macOS — increasingly common for researchers who want GPU compute without a Linux workstation — there is no GPU-accelerated path. `tinybio` fills that gap using tinygrad's AMD backend.

## M1 gate results (PBMC3k, AMD RX 7900 XT via Thunderbolt 4, macOS 26.3)

PBMC3k (2700 cells × 32738 genes) → CPM → log1p → top-2000 HVGs → z-score → top-50 PCs.

|                           | tinybio (AMD eGPU) | scanpy (CPU, arpack) |
|---------------------------|---------------------:|---------------------:|
| **warm wall time** (ms)   |                  160 |                   84 |
| **cold wall time** (ms)   |                 1460 |                    — |
| **top-10 min cosine**     |        0.9994        |      1.0 (ref)       |
| **top-10 mean cosine**    |        0.9999        |      1.0 (ref)       |
| **top-5 singular values** | 472.67, 353.31, 223.32, 204.96, 138.31 | same to 3 dp |

**Reading:** numerically correct against scanpy's arpack reference. At PBMC3k scale (2700 cells, 2000 genes after HVG) the CPU path is 1.9× faster — this is the expected regime where the scatter-rasterization benchmark (see [docs/rasterize_benchmarks.md](./docs/rasterize_benchmarks.md)) also shows the GPU losing to CPU, since Thunderbolt + kernel-launch overhead dominates small problems. The headline speedup story is at PBMC68k / atlas scale (Milestone 3).

`DEV=AMD JITBEAM=2` gives the same timing as plain `DEV=AMD` on this workload (157 ms vs 160 ms warm, 1400 ms vs 1460 ms cold) — we are launch-bound, not compute-bound, so per-kernel autotune doesn't help. The `TinyJit`-captured-graph path (fusing the 14 matmul launches into one) is the obvious next optimization; deferred to M2.

Benchmark with `DEV=AMD python3 examples/pbmc3k_pca.py`.

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
