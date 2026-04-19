# tinybio

GPU-accelerated single-cell RNA-seq preprocessing on [tinygrad](https://github.com/tinygrad/tinygrad), targeting **AMD eGPU on macOS** — the niche that every other GPU bioinformatics tool skips (most assume CUDA or Linux ROCm).

> Status: **pre-alpha**. Milestone 1 (PCA prototype on PBMC3k) passes numerically; speed story scales up at M3 (PBMC68k). See [ROADMAP.md](./ROADMAP.md).

## Why

`scanpy`'s PCA runs on CPU. `rapids-singlecell` needs NVIDIA. On an AMD eGPU + macOS — increasingly common for researchers who want GPU compute without a Linux workstation — there is no GPU-accelerated path. `tinybio` fills that gap using tinygrad's AMD backend.

## Benchmarks (AMD RX 7900 XT via Thunderbolt 4, macOS 26.3, M4 Pro host)

Top-50 PCA, numerical agreement against scanpy's arpack reference (abs cosine similarity per component, sign-flip safe). Warm wall times are median of 3 runs.

| Dataset | Shape after HVG | tinybio (AMD) | scanpy (CPU arpack) | ratio | top-10 min cos |
|---|---|---:|---:|---:|---:|
| **PBMC3k**  | 2,700 × 2,000  | 160 ms  | 84 ms   | 0.53× (CPU win) | 0.9994 |
| **PBMC68k** | 68,579 × 2,000 | 1758 ms | 1565 ms | 0.89× (CPU win) | 1.0000 |

Synthetic scale-up across N cells at d=2000 features (rank-50-plus-noise + z-score), top-50 PCA:

| N | tinybio (ms) | scanpy (ms) | ratio |
|---:|---:|---:|---:|
|     3,000 |   162 |    53 | 0.32× |
|    10,000 |   331 |   197 | 0.60× |
|    30,000 |   774 |   699 | 0.90× |
|   100,000 |  2290 |  2724 | **1.19×** (GPU win) |

**Reading.** Numerical correctness is solid everywhere (top-10 cos ≥ 0.999 on both real datasets). But the speed story is nuanced — tinybio crosses over CPU scanpy somewhere around 50–100k cells on synthetic dense data, and loses modestly on real PBMC68k. The workload is **launch-bound + round-trip-bound**, not compute-bound: `DEV=AMD JITBEAM=2` gives the same timing as plain `DEV=AMD` (157 vs 160 ms on PBMC3k), confirming per-kernel autotune does nothing here. The real lever is:

- **`@TinyJit` graph capture** to fuse the ~14 matmul launches per PCA into one captured graph (eliminates dispatch overhead).
- **LU-based power-iteration normalization** in place of numpy QR round-trips (sklearn uses this; removes ~21 CPU↔GPU hops).

Both are in scope for M2. Running benchmarks: `DEV=AMD python3 examples/pbmc3k_pca.py | examples/pbmc68k_pca.py | examples/scale_study.py`.

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
