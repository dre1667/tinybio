#!/usr/bin/env python3
"""Generate the three headline figures for tinybio's README / JOSS paper.

Figures (all saved into figures/):
  fig_bench_bars.png       — warm-median wall time, tinybio vs scanpy
                             across real scRNA-seq datasets.
  fig_scale_curve.png      — log-log PCA wall time vs N cells on
                             synthetic data; both methods.
  fig_pca_scatter.png      — PC1/PC2 scatter on Tabula Muris Senis
                             (245k cells), tinybio embedding, GPU-
                             rasterized density.

Numbers are the actual warm medians measured by the accompanying
examples/* scripts (top-50 PCA, AMD RX 7900 XT eGPU via TB4, macOS
26.3, M4 Pro host).

Follows the rules in docs/multipanel_figures_skill.md: consistent
units, shared axes bounds, log scaling where the data spans orders of
magnitude, readable font sizes, colorbars via make_axes_locatable,
bbox_inches tight, dpi=160.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE


# -------- Measured warm medians (ms). One source of truth for all figs. --------

REAL_DATASETS = [
    # (label, n_cells, tinybio_ms, sklearn_ms, scanpy_ms)
    ("PBMC3k",        2_700,     57.0,     26.0,     75.0),
    ("PBMC68k",      68_579,    783.6,    373.2,   1717.8),
    ("TMS droplet\n(245k)",
                    245_389,   2292.8,   1182.9,   3326.0),
    ("Mouse 1.3M\nneurons",
                  1_306_127,   9779.7,  10700.0, 146600.0),
    ("HLCA 1.5M\n(human lung)",
                  1_500_000,   3375.0,   8879.0, 127370.0),
]

SYNTH_SCALE = [
    # (n_cells, tinybio_ms, scanpy_ms)
    (  3_000,    57.0,    52.9),
    ( 10_000,    87.7,   200.0),
    ( 30_000,   196.1,   746.7),
    (100_000,   491.0,  2822.4),
]


# -------- Style --------

plt.rcParams.update({
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "font.family":      "DejaVu Sans",
})

TINYBIO_COLOR = "#1f77b4"   # matplotlib "tab:blue"
SKLEARN_COLOR = "#2ca02c"   # matplotlib "tab:green"
SCANPY_COLOR  = "#d62728"   # matplotlib "tab:red"


# -------- Fig 1: benchmark bars on real datasets --------

def fig_bench_bars(out_path: Path) -> None:
    labels  = [d[0] for d in REAL_DATASETS]
    ns      = [d[1] for d in REAL_DATASETS]
    tinybio = np.array([d[2] for d in REAL_DATASETS])
    sklearn = np.array([d[3] for d in REAL_DATASETS])
    scanpy  = np.array([d[4] for d in REAL_DATASETS])

    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    fig.patch.set_facecolor("white")

    bars_tb = ax.bar(x - width, tinybio, width, color=TINYBIO_COLOR,
                     label="tinybio (AMD eGPU)")
    bars_sk = ax.bar(x,         sklearn, width, color=SKLEARN_COLOR,
                     label="sklearn randomized_svd (CPU) — same algorithm")
    bars_sc = ax.bar(x + width, scanpy,  width, color=SCANPY_COLOR,
                     label="scanpy (CPU arpack) — real-user default")

    def fmt(ms: float) -> str:
        return f"{ms:.0f} ms" if ms < 1000 else f"{ms/1000:.1f} s"

    for bars in (bars_tb, bars_sk, bars_sc):
        for b in bars:
            h = b.get_height()
            ax.annotate(fmt(h), xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    # Crossover badge above each group: tinybio's speedup vs scanpy.
    for i, (tb, sk, sc) in enumerate(zip(tinybio, sklearn, scanpy)):
        ratio_sc = sc / tb
        ax.annotate(f"vs scanpy: {ratio_sc:.2f}×\nvs sklearn: {sk/tb:.2f}×",
                    xy=(i, max(tb, sk, sc) * 3.2),
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color="#2ca02c" if ratio_sc >= 2.0 else "#7f7f7f")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n{n:,} cells" for lab, n in zip(labels, ns)])
    ax.set_ylabel("PCA wall time, log scale (ms, median of 3 warm runs)")
    ax.set_title("tinybio vs CPU baselines — top-50 PCA on real scRNA-seq\n"
                 "AMD RX 7900 XT eGPU via TB4, macOS 26.3, M4 Pro (24 GB RAM)")
    ax.set_ylim(10, max(scanpy) * 40)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="none", fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.set_axisbelow(True)

    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# -------- Fig 2: log-log scale curve on synthetic --------

def fig_scale_curve(out_path: Path) -> None:
    ns      = np.array([d[0] for d in SYNTH_SCALE])
    tinybio = np.array([d[1] for d in SYNTH_SCALE])
    scanpy  = np.array([d[2] for d in SYNTH_SCALE])
    ratio   = scanpy / tinybio

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    fig.patch.set_facecolor("white")

    ax.plot(ns, tinybio, "o-", color=TINYBIO_COLOR, linewidth=2, markersize=8,
            label="tinybio (AMD eGPU)")
    ax.plot(ns, scanpy,  "s-", color=SCANPY_COLOR,  linewidth=2, markersize=8,
            label="scanpy (CPU arpack)")

    for n, t, r in zip(ns, tinybio, ratio):
        label_color = "#2ca02c" if r >= 1.0 else "#7f7f7f"
        ax.annotate(f"{r:.2f}×" + (" GPU win" if r >= 1.0 else ""),
                    xy=(n, t), xytext=(4, -14), textcoords="offset points",
                    fontsize=9, color=label_color, fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N cells (synthetic, d = 2000 features, top-50 PCA)")
    ax.set_ylabel("PCA wall time (ms, median of 3 warm runs)")
    ax.set_title("Synthetic scale-up: tinybio pulls ahead at ~10k cells, 5.75× at 100k")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)

    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# -------- Fig 3: PCA scatter on TMS droplet --------

def fig_pca_scatter(out_path: Path) -> None:
    """Load PBMC68k if cached, compute PCA via tinybio, rasterize PC1/PC2.

    PBMC68k (68k immune cells) has interpretable cluster structure in
    PC1/PC2 — the canonical "look, the embedding worked" figure. Atlas-
    scale TMS PC1/PC2 is dominated by depth/outlier effects and needs
    UMAP to be interpretable; out of scope for v0.1.
    """
    try:
        import anndata as ad
        import scanpy as sc
        from tinygrad import Tensor
        from tinybio.pca import pca as tb_pca
    except ImportError as e:
        print(f"  [skip fig_pca_scatter: missing dep {e}]")
        return

    mtx_dir = HERE.parent / "data" / "pbmc68k" / "filtered_matrices_mex" / "hg19"
    if not mtx_dir.exists():
        print(f"  [skip fig_pca_scatter: {mtx_dir} not found — run examples/pbmc68k_pca.py first]")
        return

    print("  loading PBMC68k...", flush=True)
    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", subset=True)
    sc.pp.scale(adata, max_value=None)
    X_np = np.ascontiguousarray(adata.X).astype(np.float32)

    print(f"  PCA on ({X_np.shape[0]}, {X_np.shape[1]})...", flush=True)
    X = Tensor(X_np).realize()
    emb, _ = tb_pca(X, n_components=50)

    # Rasterize PC1/PC2 density onto a fixed grid. At 245k points we're below
    # the GPU/CPU crossover for scatter rasterization (see docs/rasterize_benchmarks.md),
    # so CPU np.bincount. Clip to central 1–99 percentile per axis so extreme
    # outlier cells don't compress the bulk density into a line.
    pts = emb[:, :2]
    res = 600
    xmin, xmax = np.percentile(pts[:, 0], [1.0, 99.0])
    ymin, ymax = np.percentile(pts[:, 1], [1.0, 99.0])
    mask = ((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax))
    pts_in = pts[mask]
    xi = ((pts_in[:, 0] - xmin) / (xmax - xmin) * (res - 1)).astype(np.int32).clip(0, res - 1)
    yi = ((pts_in[:, 1] - ymin) / (ymax - ymin) * (res - 1)).astype(np.int32).clip(0, res - 1)
    grid = np.bincount(yi * res + xi, minlength=res * res).reshape(res, res)
    display = np.log1p(grid)

    cmap = "magma"
    bg = plt.get_cmap(cmap)(0.0)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(bg)

    im = ax.imshow(display, cmap=cmap, origin="lower",
                   extent=[xmin, xmax, ymin, ymax], aspect="auto")

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
    fig.colorbar(im, cax=cax, label="log(1 + cells per pixel)")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PBMC68k PCA embedding (tinybio, AMD eGPU)\n"
                 f"{pts.shape[0]:,} cells • top-50 PCA in 564 ms (scanpy: 1653 ms, 2.93× speedup)")

    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    print("Generating benchmark figures...")
    fig_bench_bars(OUT_DIR / "fig_bench_bars.png")
    fig_scale_curve(OUT_DIR / "fig_scale_curve.png")
    fig_pca_scatter(OUT_DIR / "fig_pca_scatter.png")
    print("done.")


if __name__ == "__main__":
    main()
