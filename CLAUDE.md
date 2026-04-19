# Building tinybio — skill guide for Claude Code

**Mission.** Build a minimal, publishable, GPU-accelerated single-cell RNA-seq
preprocessing library on top of [tinygrad](https://github.com/tinygrad/tinygrad),
targeting **AMD eGPU on macOS** — the one niche where every other GPU
bioinformatics tool fails (everything else assumes CUDA or Linux ROCm).

**Publication target.** [Journal of Open Source Software (JOSS)](https://joss.theoj.org).
Short paper (~1000 words), peer-reviewed, citable DOI, 4–8 week turnaround.

**Session author.** A clinician-resident who's already built `tinygrad-ft`
(LoRA fine-tuning for tinygrad) and is using this eGPU + tinygrad stack
for real clinical NLP work. Assume they know some ML, some Python, not
deep on bioinformatics tooling — treat the user as a competent researcher
who asks for what they need but doesn't want hand-holding on basics.

This file is loaded automatically by Claude Code when a session starts in
this directory. Read it once and keep its gotchas in mind for the whole
session.

---

## TL;DR — what to do first

Before anything else, build the **initial prototype**: can we even make
tinygrad's GPU SVD beat scanpy's CPU PCA on a real single-cell dataset?
If yes, we've got a paper. If no, we pivot or stop. Everything else
(packaging, tests, docs, the JOSS paper) is downstream of this working.

The prototype is intentionally small — **~80 lines of code**:

1. Download PBMC3k (2700 cells × 32k genes, the canonical tutorial dataset)
2. Load into AnnData
3. Normalize (CPM → log1p)
4. Select top N highly-variable genes (~2000)
5. Scale (mean-center each gene)
6. Compute top-50 principal components via **tinygrad SVD** on `DEV=AMD`
7. Numerically verify against scanpy's PCA
8. Time both

If (tinygrad PCA) < (scanpy CPU PCA) in wall time, we're golden.
If the numerical verification disagrees by more than a sign flip (PCA is
sign-ambiguous per component), something's wrong with the SVD call.

**Milestone 0: prove the technical thesis.** Once that's done, consult
the ROADMAP and expand.

---

## Sibling project: `tinygrad-ft`

This repo's big sibling is at `~/claude_projects/tinygrad-ft/` and lives
publicly at `https://github.com/dre1667/tinygrad-ft`. It's an LoRA
fine-tuning library on tinygrad, built in a single intensive session.
`tinybio` should feel like its younger sibling — same architectural
taste, same `tinygrad_ft`-style module layout, same discipline about
tests and docs.

### What to reuse from `tinygrad-ft`

- **Patterns, not code:** The `hf_load / build / forward` module split
  works. `tinybio` wants `io / normalize / pca / hvg / benchmark`.
- **`CLAUDE.md` + `BUGS.md` + `ROADMAP.md` style** — mirror the docs setup.
- **Test discipline:** fast unit tests that run in 1s + slow integration
  tests marked `@pytest.mark.slow` that download real data and take
  ~minutes. Both pass, both separately runnable.
- **Venv pattern:** `python3.13 -m venv .venv && source .venv/bin/activate`.
  Use Python 3.13 (NOT 3.14 — the ABI weirdness from the earlier session
  is too close to debug).
- **Installing tinygrad from master:**
  `pip install git+https://github.com/tinygrad/tinygrad.git`
  (PyPI's 0.12.0 doesn't ship the `llm/` subpackage, and master has
  bugfixes we want).

### What NOT to reuse

- `tinygrad_ft.forward.get_logits_train` etc. — that's for transformers.
  We're doing matrix decomposition, not autoregressive models.
- `LoRALinear` — no fine-tuning here. `tinybio` is numerical linear
  algebra on expression matrices.
- The `prepare_for_training` monkey-patch — we're never calling any
  `TransformerBlock`.

---

## Tinygrad gotchas you must know (condensed from tinygrad-ft)

The full debug log is at `~/claude_projects/tinygrad-ft/BUGS.md`. For
`tinybio`, the relevant gotchas are:

### 1. Always set `DEV=AMD` at process start

Never rely on `Device.DEFAULT`, never use deprecated `AMD=1`, never call
`.to('AMD')` on a tensor created under a different default device.
Cross-device transfers on macOS 26 crash with SIGBUS.

```bash
# ALWAYS:
DEV=AMD JITBEAM=2 python3 examples/pbmc3k_pca.py

# NEVER:
python3 examples/pbmc3k_pca.py   # defaults to METAL; then .to('AMD') crashes
```

### 2. Tinygrad is lazy — realize to force compute

`tensor.sum()` doesn't compute anything. To materialize values:

```python
result = (X @ Y).sum()
value = result.item()       # forces realization; returns Python float
# or
result.realize()            # explicit, returns the tensor
arr = result.numpy()        # also forces realization
```

For benchmarks, **always** `.realize()` before timing — otherwise you're
timing graph construction, not compute.

### 3. Factory-time `requires_grad` only matters if you're training

`tinybio` is inference-only (no gradients, no backward). You don't need
`requires_grad=True` anywhere. If you ever add a training path (e.g.,
variational autoencoder for scVI-like embedding), see tinygrad-ft's
`CLAUDE.md` for the gradient traps. For PCA: not applicable.

### 4. numpy has no bfloat16

If you're working with bf16 tensors (unlikely for scRNA-seq data, which
is typically int32 counts or float32 after normalization), and you need
numpy arrays for plotting or scanpy interop, convert through float32
first:

```python
t.cast('float32').numpy()   # ✓ works
t.numpy()                   # TypeError if t is bf16
```

### 5. `numel()` returns a Python int, not a Tensor

```python
n_elements = int(t.numel())        # ✓
n_elements = t.numel().item()      # ✗ — 'int' object has no attribute 'item'
```

### 6. Performance tuning expectations

- First run with `JITBEAM=2` on a new kernel shape: expect several
  minutes of autotuning before any compute happens. Subsequent runs
  with the same shape are near-instant (cached at `~/Library/Caches/tinygrad/cache.db`).
- Without `JITBEAM`: unoptimized kernels, 5–10× slower than with it.
- `PARALLEL=10` speeds up autotuning on M4 Pro (10 P-cores).

---

## Bio context: what scRNA-seq preprocessing actually is

If you're fresh on this domain, here's the standard single-cell
RNA-sequencing preprocessing pipeline (the "scanpy pipeline"):

```
Raw 10x data (mtx / h5ad files)
      ↓
Filter cells + genes (QC thresholds)
      ↓
Normalize per cell (CPM: counts-per-million)
      ↓
log1p transform (log(1 + x))
      ↓
Select highly variable genes (HVGs, typically top 2000)
      ↓
Scale per gene (mean-center, optionally unit variance)
      ↓
PCA (top 50 components)                   ← MOST EXPENSIVE STEP; our win
      ↓
Neighbor graph (k-NN in PCA space)
      ↓
UMAP / t-SNE embedding
      ↓
Leiden clustering
      ↓
Differential expression per cluster
      ↓
Cell type annotation
```

**PCA is consistently the single most expensive step on large datasets.**
For a 100K-cell dataset with 2000 HVGs, scanpy's truncated-SVD PCA takes
~30s–2min on CPU. If tinygrad on the eGPU hits ~5s, that's a
publishable speedup.

### Key concepts (quick reference)

- **Count matrix**: shape `(n_cells, n_genes)`, integers. Typically
  sparse (>95% zeros in scRNA-seq).
- **AnnData**: Python class from `anndata` library, holds a count matrix
  + per-cell metadata + per-gene metadata. `.h5ad` is the file format.
- **scanpy**: the canonical CPU library for scRNA-seq analysis. We
  benchmark against it.
- **CPM**: counts per million — divide each cell by its total, multiply
  by 1e6. Normalizes for sequencing depth.
- **log1p**: `log(1 + x)`. Reduces skew of CPM-normalized counts.
- **HVG**: highly variable gene. The top ~2000 genes by
  mean-variance-adjusted dispersion. Reduces dimensionality from 30K+
  genes to a meaningful subset.
- **PCA (via truncated SVD)**: compute top-K singular vectors of the
  `(cells × HVGs)` scaled matrix. The resulting `(cells × K)` matrix is
  the PCA embedding — the standard input to downstream UMAP/clustering.

### Canonical datasets (for testing + benchmarks)

All publicly available, no IRB needed:

| Dataset | Size | Source | Use for |
|---|---|---|---|
| **PBMC3k** | 2700 cells × 32738 genes | 10x Genomics | Small / quick verify |
| **PBMC68k** | 68579 cells × 32738 genes | 10x Genomics | Medium / meaningful benchmark |
| **Heart Cell Atlas** | 500K cells × 33K genes | Human Cell Atlas | Large / headline benchmark |
| **Tabula Muris** | ~100K cells × ~23K genes | Mouse atlas | Alternate benchmark |

Scanpy ships loaders: `scanpy.datasets.pbmc3k()`, `pbmc68k_reduced()`.
These download + cache transparently.

---

## Architecture proposed for tinybio

Mirrors tinygrad-ft's module split. All modules are thin — most of the
"smarts" is numerical linear algebra, not framework glue.

```
tinybio/
├── pyproject.toml        # standard: tinygrad, anndata, numpy, scanpy (for tests)
├── CLAUDE.md             # this file
├── ROADMAP.md            # milestones (see sibling file)
├── BUGS.md               # working log of issues hit (start empty)
├── README.md             # will write after M1
├── LICENSE               # MIT
├── .gitignore
├── tinybio/
│   ├── __init__.py
│   ├── io.py             # AnnData load/save, 10x mtx reader
│   ├── normalize.py      # cpm, log1p, scale (z-score)
│   ├── hvg.py            # highly-variable gene selection
│   ├── pca.py            # GPU PCA via tinygrad SVD  ← the headline
│   ├── pl.py             # minimal matplotlib helpers (pca scatter, scree, hvg)
│   └── benchmark.py      # timing helpers, comparison vs scanpy
├── examples/
│   ├── pbmc3k_pca.py     # the initial prototype lives here first
│   ├── pbmc68k_pca.py    # scale-up benchmark
│   └── pipeline.py       # end-to-end normalize → HVG → scale → PCA → pl.pca
└── tests/
    ├── test_normalize.py # unit tests on small synthetic matrices
    ├── test_pca.py       # numerical match vs sklearn SVD, vs scanpy
    ├── test_pl.py        # smoke tests that plotting helpers render
    └── test_io.py
```

---

## Technical approach for each module

### `io.py` — use existing libraries, don't reinvent

Use the `anndata` Python library for `.h5ad` I/O. Use `scanpy.datasets`
for downloading canonical datasets. Do NOT write custom HDF5 parsers —
we'd burn effort with nothing to show.

```python
import anndata as ad
adata = ad.read_h5ad("path/to/data.h5ad")   # (n_cells, n_genes) AnnData
X = adata.X                                  # sparse or dense matrix
```

Sparse matrices matter. scRNA-seq data is >95% zeros. Converting to
dense can OOM on larger datasets. First version can accept sparse input
and densify as needed for tinygrad compute; later versions should handle
sparse operations natively.

### `normalize.py` — straightforward element-wise ops

```python
# CPM: per-cell counts summed, divided, multiplied by 1e6
cell_counts = X.sum(axis=1, keepdim=True)
X_cpm = X / cell_counts * 1e6

# log1p
X_log = (X_cpm + 1).log()

# Scale (per-gene z-score)
mu = X_log.mean(axis=0, keepdim=True)
sigma = X_log.std(axis=0, keepdim=True)
X_scaled = (X_log - mu) / sigma.clip(1e-12, float('inf'))
```

All tinygrad-native, all GPU-resident once loaded.

### `hvg.py` — the tricky bit

Scanpy's `pp.highly_variable_genes` has a few modes. The "seurat_v3"
method is: bin genes by log-mean, within each bin rank by
log-variance, select top-N overall. Pretty tinygrad-friendly once
you've got the per-gene mean and variance.

**Shortcut for v0.1:** Just rank by log(variance) / log(mean) and take
top-N. Documented as "simplified HVG selection" — fine for a tool paper
with scanpy parity as a future improvement.

### `pca.py` — the headline module

```python
from tinygrad import Tensor

def pca(X: Tensor, n_components: int = 50) -> Tensor:
    """X is (n_cells, n_genes), already normalized + scaled.
    Returns (n_cells, n_components) embedding."""
    # Truncated SVD: X = U @ diag(S) @ V.T
    # PCA embedding = U @ diag(S), shape (n_cells, n_components)
    U, S, V = X.svd()    # verify tinygrad has this — may need manual
    return (U[:, :n_components] * S[:n_components])
```

**CRITICAL EARLY CHECK:** Does tinygrad have a working `Tensor.svd()`? If
not, we need to either:
(a) Implement truncated SVD via power iteration (~20 lines of matrix
    multiplies, converges in ~20-50 iterations for top-k components)
(b) Call out to scipy.sparse.linalg.svds for comparison, and implement
    the tinygrad path ourselves
(c) Use the Lanczos method (harder but better for very large matrices)

Check early (before writing any downstream code) by running
`Tensor.rand(100,100).svd()` and seeing what happens. If it works, great. If not, implement truncated SVD by power
iteration — it's short and well-understood.

### `pl.py` — GPU-rasterized plotting with matplotlib display (validated)

Design approach: **compute the bitmap on the GPU via tinygrad, display
via matplotlib.** The expensive part — turning N points into a density
grid — runs as a single `scatter_reduce` kernel on the AMD eGPU. For
scRNA-seq with large datasets (>100K cells), this avoids matplotlib's
overplotting problem (everything collapses to a blob) and is
dramatically faster than datashader's CPU approach.

**Validated on AMD eGPU (7900 XT via TB4), macOS 26.3, 1000×1000 grid
fixed, median of 3 runs each:**

| N points | GPU (tinygrad) | CPU (np.bincount) | GPU speedup |
|---:|---:|---:|---:|
|       10,000 | 1.98 ms |   0.05 ms |  0.03× |
|      100,000 | 1.95 ms |   0.25 ms |  0.13× |
|    1,000,000 | 1.85 ms |   2.01 ms |  1.1×  |
|   10,000,000 | 1.95 ms |  18.78 ms |  9.6×  |
|   50,000,000 | 2.61 ms | 103.37 ms | 39.5×  |

Two important take-aways from the curve:

1. **GPU time is ~flat at 2 ms** across five orders of magnitude of
   input size. That's the Thunderbolt + GPU launch overhead floor —
   you can't go below it regardless of workload. The actual scatter
   work is hidden under that overhead until ~10M points.

2. **The crossover is at ~1M points.** Below that, numpy on CPU is
   faster because it doesn't pay TB overhead. Above 1M, tinygrad
   pulls away fast. At 50M points we're 40× faster than CPU, and
   the curve would keep widening at atlas scale (100M+).

**fp32 precision caveat.** At very large total counts (e.g., 50M
points summed across 1M cells where each cell holds ~50 counts),
fp32 exact-integer representation (2²⁴ ≈ 16.7M) is exceeded in the
grand `.sum()`. Individual cells are always fine — the rasterized
image is correct — only the grand-total verification is imprecise.
For paranoia-level accuracy on very large datasets, use fp64 for
the accumulator (at ~2× memory and ~1.5-2× time cost).

Reference API (~40 LOC for the core rasterizer + ~30 LOC each for the
three plot helpers = ~130 LOC total):

```python
from tinygrad import Tensor
import numpy as np
import matplotlib.pyplot as plt


def _rasterize_scatter(points: Tensor,
                       weights: Tensor | None = None,
                       resolution: int = 500) -> Tensor:
    """GPU-rasterize N points into a (resolution, resolution) density grid.

    points : (N, 2) tensor in any real units; we normalize to [0, resolution)
    weights: optional (N,) tensor of per-point values; if None, each point
             contributes 1 (count density). If provided, the grid contains
             the *sum* of weights per pixel (useful for coloring by gene
             expression etc.).

    Returns: (resolution, resolution) float32 grid.
    """
    p = points - points.min(axis=0, keepdim=True)
    span = (points.max(axis=0, keepdim=True) - points.min(axis=0, keepdim=True))
    p = p / span.clip(1e-12, float('inf')) * (resolution - 1)
    xi = p[:, 0].cast('int32').clip(0, resolution - 1)
    yi = p[:, 1].cast('int32').clip(0, resolution - 1)
    flat_idx = (yi * resolution + xi).cast('int32')
    src = weights if weights is not None else Tensor.ones(points.shape[0])
    # CRITICAL API: use scatter_reduce, NOT scatter(reduce='add').
    # scatter(reduce='add') only works with scalar src; for Tensor src
    # tinygrad requires scatter_reduce with include_self=True.
    return (Tensor.zeros(resolution * resolution)
            .scatter_reduce(0, flat_idx, src, reduce='sum', include_self=True)
            .reshape(resolution, resolution))


# Benchmark-driven heuristic: GPU rasterization is only faster than
# matplotlib.scatter above ~100K points. Below that, fall through to
# a plain ax.scatter(). This is the only place in pl.py where we branch
# on N — keeps the API uniform.
RASTERIZE_THRESHOLD = 100_000

def pca(embedding, color=None, resolution=500, ax=None, cmap="viridis"):
    """PCA embedding plot. Uses ax.scatter() for small N, GPU rasterization
    for large N. Returns the matplotlib Axes."""
    pts_np = embedding.numpy() if isinstance(embedding, Tensor) else np.asarray(embedding)
    n_points = pts_np.shape[0]
    ax = ax or plt.gca()

    if n_points < RASTERIZE_THRESHOLD:
        # Small dataset: plain scatter is faster and looks crisper
        color_np = None if color is None else (
            color.numpy() if isinstance(color, Tensor) else np.asarray(color)
        )
        sc = ax.scatter(pts_np[:, 0], pts_np[:, 1], c=color_np, s=4, alpha=0.7)
        if color_np is not None:
            plt.colorbar(sc, ax=ax)
    else:
        # Large dataset: GPU rasterize to a density grid, imshow it
        pts = embedding if isinstance(embedding, Tensor) else Tensor(pts_np.astype(np.float32))
        w = (color if isinstance(color, Tensor)
             else (None if color is None else Tensor(np.asarray(color).astype(np.float32))))
        grid = _rasterize_scatter(pts[:, :2], w, resolution).realize().numpy()
        display = np.log1p(grid) if color is None else grid
        im = ax.imshow(display, cmap=cmap, origin="lower", aspect="auto")
        plt.colorbar(im, ax=ax)

    ax.set(xlabel="PC1", ylabel="PC2")
    return ax


def variance_ratio(singular_values, n_components=50, ax=None):
    """Scree plot: per-component variance + cumulative."""
    s = singular_values.numpy() if isinstance(singular_values, Tensor) else singular_values
    var_ratio = (s ** 2) / (s ** 2).sum()
    ax = ax or plt.gca()
    ax.bar(range(n_components), var_ratio[:n_components], alpha=0.6)
    ax2 = ax.twinx()
    ax2.plot(range(n_components), var_ratio[:n_components].cumsum(), "r-")
    ax.set(xlabel="PC", ylabel="var ratio")
    ax2.set(ylabel="cumulative")
    return ax


def hvg(adata, ax=None):
    """Mean-vs-log-variance scatter, HVG-selected genes highlighted.
    Small N (number of genes, ~30K); matplotlib.scatter is fine here."""
    ...
```

Design rules:
- Always accept `ax=None` and default to `plt.gca()` — lets users
  compose with their own figures
- Always return the `Axes` — users can add titles, legends, etc.
- No opinionated styling (don't mess with rcParams globally)
- No interactivity — static matplotlib only
- GPU rasterize when N is large (>10K points); fall through to
  `ax.scatter` for small N if it's simpler
- Add `matplotlib` as a runtime dep in `pyproject.toml`; it's small
  and universal, don't gate it behind an extras_require

**Additional reading:** before writing the README figures or the JOSS
paper figures, read [`docs/multipanel_figures_skill.md`](./docs/multipanel_figures_skill.md).
It's a 20-rule checklist distilled from getting the tinybio figures
wrong five different ways in one afternoon (inconsistent backgrounds,
colorbar-induced panel-size mismatch, transpose-flipped imshow,
lazy-scatter-timing, etc.). Saves hours.

**Critical API detail**: use `tensor.scatter_reduce(dim, index, src,
reduce='sum', include_self=True)`, NOT `tensor.scatter_add(...)` (doesn't
exist) and NOT `tensor.scatter(..., reduce='add')` (scalar-src-only).

**Large-N scatter hang caveat (AMD eGPU)**: when running `scatter_reduce`
on very large N (≥ ~5M points) in a tight loop, don't pile up many
iterations without explicit sync. tinygrad's async pipelining queues
work faster than the GPU executes it, and cleanup-time `_free()` calls
then hit a `synchronize` timeout that gets escalated to
`RuntimeError: Device hang detected`. It's not a real hang — just a
queue drain timeout.

**Fix we implemented** (see `figures/gpu_utils.py`):

```python
def drain_gpu_queue():
    """Force GPU to finish queued work. Use between large-N iterations."""
    _ = Tensor([0.0]).realize().numpy()  # small readback forces a sync
```

Usage pattern in any loop over large-N GPU work:

```python
from gpu_utils import drain_gpu_queue

for n in big_sizes:
    for _ in range(n_runs):
        result = Tensor.zeros(...).scatter_reduce(...).realize()
        # ... record timing
    if n >= 5_000_000:
        drain_gpu_queue()   # empty the queue before next size
```

Or as a decorator for functions that do big scatter work:

```python
from gpu_utils import drained

@drained
def rasterize(points, weights, res):
    # ...scatter_reduce...
    return grid
```

**Upstream opportunity**: file an issue asking tinygrad to either scale
the synchronize timeout with queued work, or raise a clearer error
(`SynchronizeTimeout` vs `DeviceHang`) so users don't think their GPU
has died.

**Critical axis convention**: build your density grid as
`grid[yi, xi] = count` (row = y, col = x). Then pass the grid to
`imshow(grid, origin='lower', extent=...)` WITHOUT a transpose. Passing
`grid.T` looks right at first glance but produces a transpose-flipped
image because `imshow` already treats `A[row, col]` as a value at plot
position `(col, row)` — the row/col convention matches y/x out of the
box. This bug caused real clusters to appear mirror-imaged during
development; easy to introduce, hard to spot on symmetric data, trivial
once you know to look.

**Explicitly NOT in scope:** replicating `scanpy.pl` (which has 40+
functions, custom color maps, complex integration with AnnData layers).
If a user needs more than these three, they should use scanpy for
plotting. Ours exists so the 80% case is a one-liner — and it handles
atlas-scale data without overplotting, which scanpy.pl does not.

### `benchmark.py` — timing discipline

Always compare to scanpy on the same (normalized) input matrix, same
target rank. Report:

- Wall time (median of 3+ runs)
- Memory peak (tracemalloc or RSS delta)
- Numerical agreement (cosine similarity of corresponding components,
  accounting for sign ambiguity)

Gotcha: **always `.realize()` tinygrad outputs before stopping the
timer.** Lazy eval will make a "GPU PCA" look like it runs in 1ms.

---

## How to know when v0.1 is ready for JOSS

Checklist:

- [ ] Works end-to-end on PBMC3k (verified numerically vs scanpy)
- [ ] Wins wall-time benchmark on PBMC68k (>2× faster than scanpy CPU
      is convincing; 5×+ is publishable with confidence)
- [ ] Clean README with install + 10-line quickstart
- [ ] pytest suite passes, ≥80% line coverage on `tinybio/` source
- [ ] Correctness tests: unit tests + integration vs scanpy on small
      synthetic matrices
- [ ] `CLAUDE.md` + `BUGS.md` up to date
- [ ] MIT license, proper `pyproject.toml`
- [ ] GitHub repo public, CI badge showing tests pass
- [ ] One paragraph "statement of need" in README

JOSS review criteria: <https://joss.readthedocs.io/en/latest/review_criteria.html>.
Worth reading in full before submission.

---

## Known unknowns — things to verify early

1. **Does `tinygrad.Tensor` have a working SVD?** If not, we implement.
2. **Is tinygrad's SVD numerically stable enough for normalized
   log-CPM expression matrices?** These have dynamic range ~1e-5 to
   ~1e2. Power iteration in fp32 should be fine; bf16 might lose
   signal in low-variance components.
3. **Can tinygrad handle sparse matrices?** Almost certainly not
   natively. We'll densify HVG-selected submatrices (2000 genes ×
   100K cells = 1.6 GB in fp32, fits in 20 GB VRAM easily).
4. **Is Thunderbolt bandwidth the bottleneck for the initial transfer
   of a 2 GB matrix?** At ~3.2 GB/s TB4, 2 GB takes ~0.6s. That's
   significant if the whole PCA runs in 1-2s. Consider whether to
   include or exclude transfer time in headline benchmarks
   (both numbers, clearly labeled).
5. **Do we need `JITBEAM` for the SVD kernel specifically?** First-run
   tuning might add minutes before we see any timing. Document the
   two-regime performance (cold cache vs warm cache) in the paper.

---

## The vibe of this project

- **Minimal.** <1000 lines of tinybio source for v0.1.
- **Correct first, fast second.** Verify numerical agreement with
  scanpy before optimizing anything.
- **Honest benchmarks.** Include disadvantageous cases too — if
  tinygrad loses on small matrices due to fixed GPU-dispatch overhead,
  say so.
- **Tool paper, not research paper.** Claims should be "this works and
  is available" not "this reveals new biology." Leave bio discovery
  for a follow-up paper using the tool.
- **Start narrow.** PCA first, expand later. Do NOT attempt UMAP,
  clustering, or differential expression in v0.1. Each of those is
  its own project.
- **Ship 3 plotting helpers, not 30.** We need PCA scatter + scree +
  HVG dispersion to have a usable README and JOSS figures. We do NOT
  need to compete with `scanpy.pl`. If a user wants more, they reach
  for scanpy.

---

## Specific first-session checklist

If you are the first Claude Code session opening this directory, here's
your ordered to-do list:

1. `git init -b main` and scaffold the directory structure (mirror
   tinygrad-ft's layout)
2. Create `pyproject.toml` with dependencies: `tinygrad`, `anndata`,
   `numpy`, `scipy`, `scanpy` (dev only, for benchmarks), `pytest`
3. `python3.13 -m venv .venv && source .venv/bin/activate`
4. `pip install git+https://github.com/tinygrad/tinygrad.git`
5. `pip install -e ".[dev]"`
6. Check `Tensor.svd()` works: `python3 -c "from tinygrad import Tensor; u,s,v = Tensor.randn(32,32).svd(); print(u.shape, s.shape, v.shape)"`.
   - If works → proceed to PCA prototype
   - If not → build truncated SVD via power iteration first
7. Write `examples/pbmc3k_pca.py` — the 80-line initial prototype
8. Run it with `DEV=AMD JITBEAM=2 python3 examples/pbmc3k_pca.py`
9. If results match scanpy + tinygrad is faster: celebrate, commit,
   push to GitHub, start on v0.1 packaging per ROADMAP
10. If not: diagnose, iterate, don't proceed to packaging until it works

When in doubt about patterns, cross-reference `~/claude_projects/tinygrad-ft/`.

---

## Upstream references

- Tinygrad source: `~/tinygrad/` (installed as editable)
- tinygrad-ft reference: `~/claude_projects/tinygrad-ft/`
- Scanpy docs: <https://scanpy.readthedocs.io>
- AnnData docs: <https://anndata.readthedocs.io>
- JOSS review criteria: <https://joss.readthedocs.io/en/latest/review_criteria.html>
- PBMC3k tutorial (scanpy): <https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html>

Good luck. Ship the prototype first — everything else flows from that.
