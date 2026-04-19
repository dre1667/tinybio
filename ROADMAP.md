# tinybio roadmap

Living document. Check items off as they land. Keep it honest — only
check things that actually work, not things that "should" work.

## Guiding principles

- **Publish v0.1 via JOSS, fast.** 4–8 week review cycle. Don't
  overbuild before submission.
- **PCA is the value proposition.** Everything else (normalize, HVG,
  scale) is supporting infrastructure so the PCA benchmark is
  apples-to-apples with scanpy.
- **Benchmark vs scanpy, honestly.** Include cold-cache + warm-cache
  tinygrad timings separately. Small matrices may not beat scanpy.
  Say so in the paper if true.
- **Mirror tinygrad-ft's discipline:** tests, docs, CLAUDE.md,
  BUGS.md, clean commits.

---

## Milestone 0 — scaffold

- [ ] `git init -b main`
- [ ] Directory structure: `tinybio/`, `tests/`, `examples/`
- [ ] `pyproject.toml` with deps:
      - runtime: `tinygrad`, `anndata`, `numpy`, `scipy`
      - dev: `pytest`, `scanpy` (for comparison benchmarks)
- [ ] `.gitignore`, `LICENSE` (MIT), skeleton `README.md`
- [ ] `CLAUDE.md` (done — you're reading it)
- [ ] `ROADMAP.md` (this file)
- [ ] `BUGS.md` (start empty, fill as issues found)
- [ ] venv: `python3.13 -m venv .venv && source .venv/bin/activate`
- [ ] Install tinygrad from git master (PyPI lacks `llm/`; we may not need
      that here but we want to stay on master for bugfixes)
- [ ] `pip install -e ".[dev]"`
- [ ] Verify `from tinygrad import Tensor` works
- [ ] Run a minimal `DEV=AMD python3 -c "from tinygrad import Tensor; print(Tensor([1.,2.,3.]).sum().item())"` sanity check

## Milestone 1 — prototype (the GATE)

**This is the technical thesis validation. Do not proceed to M2 until
this passes.** Keep scope tight: one dataset, one operation, one
benchmark. If the core PCA comparison doesn't win, everything downstream
is speculation.

- [ ] Check if `Tensor.svd()` exists in tinygrad master. If yes, use it.
      If no, implement truncated SVD via power iteration (`tinybio/pca.py`).
- [ ] `examples/pbmc3k_pca.py` — ~80-line standalone script:
      - [ ] Download PBMC3k via `scanpy.datasets.pbmc3k()`
      - [ ] Normalize: sum per cell → CPM → log1p
      - [ ] Select top 2000 HVGs (simplified: rank by log-variance)
      - [ ] Scale per gene (mean-center + unit variance)
      - [ ] Compute top 50 PCs via tinygrad SVD on AMD eGPU
      - [ ] Cross-check: cosine similarity with scanpy.pp.pca output,
            handling sign flips per component. Expect >0.999 for most
            components.
      - [ ] Wall-time benchmark vs scanpy CPU PCA (median of 3 runs each)
- [ ] Run: `DEV=AMD JITBEAM=2 PARALLEL=10 python3 examples/pbmc3k_pca.py`
- [ ] Decision point:
      - **If tinygrad correct + ≥1.5× faster on PBMC3k:** proceed to M2
      - **If correct but slower (common on small matrices):** check with
        PBMC68k next. Small-matrix GPU overhead often dominates.
      - **If numerically wrong:** debug SVD. Might need power-iteration
        fallback.

## Milestone 2 — package scaffolding

Convert the prototype into a real library with the architecture from
`CLAUDE.md`:

- [ ] `tinybio/io.py` — AnnData wrappers, 10x `.mtx` reader
- [ ] `tinybio/normalize.py` — `cpm`, `log1p`, `scale` as standalone functions
- [ ] `tinybio/hvg.py` — `select_highly_variable(adata, n_top=2000)`
- [ ] `tinybio/pca.py` — `pca(X, n_components=50)` as the canonical API
- [ ] `tinybio/benchmark.py` — `compare_pca(X, n_components, n_runs=3)`
      returns `{tinybio_time, scanpy_time, cosine_sim}`
- [ ] `tests/test_normalize.py` — synthetic 10×5 matrix checks
- [ ] `tests/test_pca.py` — vs `sklearn.utils.extmath.randomized_svd` on
      synthetic data, plus vs scanpy on PBMC3k marked `@pytest.mark.slow`
- [ ] `tests/test_hvg.py` — verify HVG selection matches scanpy on known
      dataset within some tolerance
- [ ] `pytest -m "not slow"` runs in < 10s
- [ ] `pytest` full suite passes

### M2 sub-milestone — GPU-rasterized plotting

Design: compute the bitmap on the GPU via `scatter_reduce`, display
through matplotlib. Avoids overplotting on atlas-scale data.
Benchmark-validated on AMD eGPU (RX 7900 XT via TB4) before roadmap
commitment:

| N points | GPU (tinygrad) | CPU (np.bincount) | speedup |
|---:|---:|---:|---:|
|       10,000 | 1.98 ms |   0.05 ms |  0.03× |
|      100,000 | 1.95 ms |   0.25 ms |  0.13× |
|    1,000,000 | 1.85 ms |   2.01 ms |  1.1×  |
|   10,000,000 | 1.95 ms |  18.78 ms |  9.6×  |
|   50,000,000 | 2.61 ms | 103.37 ms | 39.5×  |

GPU time is ~flat at 2 ms up to 10M points (Thunderbolt + launch
overhead dominates). Crossover is at ~1M points; below that,
ax.scatter is faster. Above 10M points the GPU scales essentially
for free, reaching 40× CPU at 50M.

**Design implication:** fall through to `ax.scatter` for small N
(<100K threshold); use GPU rasterization above that. Single branch,
API stays uniform.

- [ ] `tinybio/pl.py` (module name mirrors scanpy's `sc.pl` convention):
      - [ ] `_rasterize_scatter(points, weights=None, resolution=500)` —
            internal helper using `Tensor.scatter_reduce(0, idx, src,
            reduce='sum', include_self=True)`. Returns a (res, res)
            density grid on GPU.
      - [ ] `pl.pca(embedding, color=None, ax=None, resolution=500)` —
            below `RASTERIZE_THRESHOLD=100_000` points: plain
            `ax.scatter`. Above: rasterize PC1/PC2 to a density grid
            via `_rasterize_scatter`, `imshow` with log1p scaling.
            Optional `color` weights (per-cell gene expression etc.)
            change the accumulation mode in the rasterization path.
      - [ ] `pl.variance_ratio(singular_values, n_components=50, ax=None)` —
            scree plot. Small N, matplotlib direct.
      - [ ] `pl.hvg(adata, ax=None)` — mean vs log-variance scatter with
            HVG-selected genes highlighted. Small N (~30K genes),
            matplotlib direct.
- [ ] Each helper accepts `ax=None`, returns the `Axes` object
- [ ] `tests/test_pl.py` — smoke tests (renders without error, returned
      `Axes` has the right number of artists). No image-comparison tests.
- [ ] `matplotlib` as a runtime dependency (small, universal)
- [ ] `examples/pipeline.py` demos `pl.pca` at the end
- [ ] README embeds a generated figure from `pl.pca` on PBMC3k or larger

## Milestone 3 — scale-up benchmark

- [ ] `examples/pbmc68k_pca.py` — same flow on PBMC68k (68K cells × 32K genes)
- [ ] Memory + wall-time profile vs scanpy
- [ ] Document cold-cache vs warm-cache timing separately
- [ ] `examples/pipeline.py` — end-to-end demonstrating all four steps
      (normalize → HVG → scale → PCA) with sensible defaults
- [ ] Benchmark on one larger dataset (Tabula Muris or Heart Cell Atlas
      subset, ~100K–500K cells) to have a headline number

## Milestone 4 — documentation + paper prep

- [ ] `README.md`:
      - Badges (License, Python, tests passing)
      - 1-paragraph statement of need ("first GPU scRNA-seq on AMD macOS")
      - Install section (with the tinygrad-from-git note)
      - Quickstart code block
      - Benchmark table (PBMC3k, PBMC68k, one larger)
      - Comparison with scanpy / rapids-singlecell / scVI
      - Supported hardware table
- [ ] `CLAUDE.md` updated with any new gotchas discovered
- [ ] `BUGS.md` with any issues we hit during build
- [ ] GitHub repo public, README renders properly
- [ ] Pin a release tag (v0.1.0)

## Milestone 5 — JOSS submission

- [ ] Write `paper.md` (JOSS format, ~1000 words) with these sections:
      - Summary
      - Statement of need (the AMD-macOS gap)
      - Functionality (what the API does)
      - Performance (benchmark table)
      - Acknowledgements
- [ ] Write `paper.bib` with citations (scanpy, RAPIDS, tinygrad, PCA lit,
      seminal 10x papers)
- [ ] Submit to JOSS: <https://joss.theoj.org/papers/new>
- [ ] Respond to reviewer comments as they come in (expect 2–4 reviewers)

## Milestone 6 — v0.2 and beyond (post-publication)

Don't try to do these before the JOSS submission. Keep v0.1 focused.

- [ ] `tinybio/neighbors.py` — k-NN graph via brute-force on GPU
      (exact, simple, fast enough for ~1M cells)
- [ ] `tinybio/umap.py` — UMAP-like force-directed embedding using
      gradient descent on tinygrad tensors (harder; consider using
      `umap-learn` as a fallback that takes tinygrad PCA output)
- [ ] `tinybio/cluster.py` — Leiden via external library (`leidenalg`),
      or simple k-means in tinygrad
- [ ] `tinybio/de.py` — differential expression via Wilcoxon (requires
      ranking; tricky in tinygrad) or Welch's t-test (straightforward)
- [ ] NVIDIA GPU testing — verify it also works on Linux CUDA hardware
      (expands the paper's reach)
- [ ] Apple Silicon (M-series) Metal testing — does it work on internal
      GPU too? If yes, bigger potential user base than AMD eGPU only.

## Non-goals (be disciplined)

Things we will NOT build, at least not in v0.1:

- Deep learning models (scVI, scGPT, etc.) — separate paper
- Batch correction (Harmony, scVI, etc.) — complex, separate paper
- **Interactive / dashboard plotting** (plotly, bokeh, shiny) — use `scanpy.pl` or write your own downstream
- **Trying to re-implement scanpy.pl** — its surface area is huge and unnecessary for us
- Spatial transcriptomics — separate domain
- Full AnnData pipeline in a single function — keep modules composable

What we WILL do (added per feedback): **a small set of static matplotlib
helpers** for the results we compute. Enough that the README and JOSS
paper can show figures out of the box, and users can `tinybio.pl.pca(...)`
as a one-liner without reaching for scanpy. See the plotting sub-milestone
in M2.

## Success criteria (will it be a real paper?)

For JOSS acceptance, we need demonstrable:

1. **Functionality** — it computes correct PCA matching scanpy within
   numerical tolerance
2. **Performance** — ≥1.5× faster than scanpy on at least one realistic
   dataset size (more impressive if 5×+)
3. **Uniqueness** — no existing tool covers AMD eGPU on macOS;
   "statement of need" is defensible
4. **Engineering** — tests, docs, reproducible install, CI

For Bioinformatics Applications Note (if we upgrade the target later):

- Everything above, plus
- Application to a real biological question (reanalysis of a published
  dataset with tinybio, confirming known findings)
- More thorough benchmarks (multiple dataset sizes, multiple hardware
  configurations)

Start with JOSS. Upgrade later if momentum supports it.
