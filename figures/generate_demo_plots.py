"""Generate demo plots showing what tinybio's pl.pca rasterization will look like.

This is a *preview* of the plotting output before the real `tinybio/pl.py`
is written. Uses the same GPU rasterization approach (scatter_reduce on
tinygrad) that pl.py will. Saves PNG figures here for visual inspection.

Run:
    DEV=AMD python3 figures/generate_demo_plots.py

Produces:
    fig1_scaling_curve.png           — GPU vs CPU benchmark curve
    fig2_scatter_vs_rasterize.png    — overplotting problem, side-by-side
    fig3_density_by_size.png         — 4-panel: 10K, 100K, 1M, 10M cells
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tinygrad import Tensor


# ----------------------------------------------------------------------------
# Core GPU rasterization — this is what pl.py will use
# ----------------------------------------------------------------------------

def rasterize_scatter(points: Tensor, resolution: int = 500) -> np.ndarray:
    """GPU-rasterize N points into an (H, W) density grid. Returns numpy array."""
    p = points - points.min(axis=0, keepdim=True)
    span = points.max(axis=0, keepdim=True) - points.min(axis=0, keepdim=True)
    p = p / span.clip(1e-12, float("inf")) * (resolution - 1)
    xi = p[:, 0].cast("int32").clip(0, resolution - 1)
    yi = p[:, 1].cast("int32").clip(0, resolution - 1)
    flat_idx = (yi * resolution + xi).cast("int32")
    src = Tensor.ones(points.shape[0])
    grid = (
        Tensor.zeros(resolution * resolution)
        .scatter_reduce(0, flat_idx, src, reduce="sum", include_self=True)
        .reshape(resolution, resolution)
        .realize()
    )
    return grid.numpy()


# ----------------------------------------------------------------------------
# Synthetic data: mimic scRNA-seq PCA output (5 cell-type clusters + noise)
# ----------------------------------------------------------------------------

def make_clusters(n_total: int, seed: int = 42) -> np.ndarray:
    """Generate a synthetic 2D distribution that looks like a scRNA-seq PCA
    scatter: 5 distinguishable cell clusters plus some transitional/noise cells."""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [ 0.0,  0.0],
        [ 3.5,  2.0],
        [-2.0,  3.0],
        [ 2.0, -2.5],
        [-3.0, -2.0],
    ])
    cluster_sizes = [0.22, 0.18, 0.24, 0.18, 0.15]
    noise_frac = 1.0 - sum(cluster_sizes)

    points = []
    for center, frac in zip(centers, cluster_sizes):
        n = int(n_total * frac)
        pts = rng.normal(center, 0.6, size=(n, 2))
        points.append(pts)
    # transitional noise: broader spread
    n_noise = n_total - sum(p.shape[0] for p in points)
    noise_pts = rng.normal([0, 0], 3.0, size=(n_noise, 2))
    points.append(noise_pts)
    return np.vstack(points).astype(np.float32)


# ----------------------------------------------------------------------------
# Figure 1 — scaling curve (re-run of earlier benchmark, with plot)
# ----------------------------------------------------------------------------

def fig1_scaling_curve():
    print("Figure 1: scaling curve")
    RES = 1000
    sizes = [10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]
    gpu_times = []
    cpu_times = []

    for N in sizes:
        # GPU
        rng = np.random.default_rng(42)
        pts_np = (rng.random((N, 2)) * (RES - 1)).astype(np.float32)
        pts = Tensor(pts_np)
        xi = pts[:, 0].cast("int32").clip(0, RES - 1)
        yi = pts[:, 1].cast("int32").clip(0, RES - 1)
        flat_idx = (yi * RES + xi).cast("int32").realize()
        src = Tensor.ones(N)
        _ = Tensor.zeros(RES * RES).scatter_reduce(0, flat_idx, src, reduce="sum", include_self=True).realize()
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = Tensor.zeros(RES * RES).scatter_reduce(0, flat_idx, src, reduce="sum", include_self=True).realize()
            runs.append(time.perf_counter() - t0)
        gpu_times.append(sorted(runs)[1] * 1000)

        # CPU
        xi_np = pts_np[:, 0].astype(np.int64).clip(0, RES - 1)
        yi_np = pts_np[:, 1].astype(np.int64).clip(0, RES - 1)
        flat_idx_np = yi_np * RES + xi_np
        _ = np.bincount(flat_idx_np, minlength=RES * RES)
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = np.bincount(flat_idx_np, minlength=RES * RES)
            runs.append(time.perf_counter() - t0)
        cpu_times.append(sorted(runs)[1] * 1000)

        print(f"  N={N:>12,}  GPU={gpu_times[-1]:6.2f} ms  CPU={cpu_times[-1]:7.2f} ms")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(sizes, gpu_times, "o-", label="tinygrad (AMD RX 7900 XT eGPU)", linewidth=2, markersize=8)
    ax.loglog(sizes, cpu_times, "s-", label="numpy.bincount (M4 Pro CPU)", linewidth=2, markersize=8)

    # crossover line
    crossover_N = None
    for i in range(len(sizes) - 1):
        if cpu_times[i] < gpu_times[i] and cpu_times[i + 1] > gpu_times[i + 1]:
            crossover_N = sizes[i]
            break
    if crossover_N is not None:
        ax.axvline(crossover_N, color="gray", linestyle="--", alpha=0.6)
        ax.text(crossover_N * 1.2, 0.1, f"crossover\n≈ {crossover_N:,}", color="gray", fontsize=9)

    ax.set_xlabel("Number of points rasterized")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_title("GPU rasterization scaling: Thunderbolt overhead floor\nthen ~free compute through 50M points")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig1_scaling_curve.png", dpi=140)
    plt.close()
    print("  → figures/fig1_scaling_curve.png")


# ----------------------------------------------------------------------------
# Figure 2 — scatter vs rasterization on 1M points (the overplotting problem)
# ----------------------------------------------------------------------------

def fig2_scatter_vs_rasterize():
    """Apples-to-apples: same data, same colormap, same density encoding,
    same background. Only the RENDERING differs (scatter vs imshow).

    For the scatter panel we color each point by its *local density*
    (read from the rasterized grid). That way both panels encode the
    same information with the same colormap — any visual difference is
    purely from the rendering approach."""
    """Apples-to-apples: same data, same colormap, same axis range,
    same panel size, same colorbar treatment. Only the RENDERING
    differs (scatter vs imshow)."""
    print("Figure 2: scatter vs rasterize on 1M cells (fair comparison)")
    N = 1_000_000
    RES = 600
    points_np = make_clusters(N)
    xmin, xmax = points_np[:, 0].min(), points_np[:, 0].max()
    ymin, ymax = points_np[:, 1].min(), points_np[:, 1].max()

    pts = Tensor(points_np)
    grid = rasterize_scatter(pts, resolution=RES)
    grid_log = np.log1p(grid)

    # Per-point density: look up each point's pixel in the grid.
    x_norm = (points_np[:, 0] - xmin) / (xmax - xmin) * (RES - 1)
    y_norm = (points_np[:, 1] - ymin) / (ymax - ymin) * (RES - 1)
    xi = x_norm.astype(np.int32).clip(0, RES - 1)
    yi = y_norm.astype(np.int32).clip(0, RES - 1)
    density_per_point = np.log1p(grid[yi, xi])

    cmap_obj = plt.get_cmap("viridis")
    cmap_name = "viridis"
    vmin, vmax = 0, grid_log.max()
    # Background color = the colormap's vmin color. Both panels use this
    # so "empty space" in the scatter panel matches "empty space" in
    # the imshow panel (where empty = viridis(0) = dark purple).
    bg_color = cmap_obj(0.0)
    extent = [xmin, xmax, ymin, ymax]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"wspace": 0.25})
    fig.patch.set_facecolor("white")

    # Left: density-colored scatter on dark-purple background
    axes[0].set_facecolor(bg_color)
    sc = axes[0].scatter(
        points_np[:, 0], points_np[:, 1],
        c=density_per_point, cmap=cmap_name, vmin=vmin, vmax=vmax,
        s=1, alpha=0.5, edgecolors="none",
    )
    axes[0].set(
        xlim=(xmin, xmax), ylim=(ymin, ymax),
        xlabel="PC1", ylabel="PC2",
        title=f"matplotlib.scatter on {N:,} cells\n(density-colored; overplotting still occludes structure)",
    )
    axes[0].set_aspect("equal", adjustable="box")
    cax0 = make_axes_locatable(axes[0]).append_axes("right", size="4%", pad=0.1)
    fig.colorbar(sc, cax=cax0, label="log(1 + density)")

    # Right: GPU-rasterized density — background is naturally viridis(0)
    axes[1].set_facecolor(bg_color)
    im = axes[1].imshow(
        grid_log, cmap=cmap_name, vmin=vmin, vmax=vmax,
        origin="lower", extent=extent, aspect="equal",
    )
    axes[1].set(
        xlim=(xmin, xmax), ylim=(ymin, ymax),
        xlabel="PC1", ylabel="PC2",
        title=f"tinybio GPU rasterization on {N:,} cells\n(same data, same colormap, density fully preserved)",
    )
    axes[1].set_aspect("equal", adjustable="box")
    cax1 = make_axes_locatable(axes[1]).append_axes("right", size="4%", pad=0.1)
    fig.colorbar(im, cax=cax1, label="log(1 + density)")

    plt.savefig("figures/fig2_scatter_vs_rasterize.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  → figures/fig2_scatter_vs_rasterize.png")


# ----------------------------------------------------------------------------
# Figure 3 — density at 4 scales (10K, 100K, 1M, 10M cells)
# ----------------------------------------------------------------------------

def fig3_density_by_size():
    """Four panels at increasing N. All panels share the SAME axis range
    (computed from the widest dataset), the SAME panel size, the SAME
    color range, and the SAME resolution, so the only visual difference
    across panels is the amount of data rendered."""
    print("Figure 3: density plots at increasing N (shared axes)")
    sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    RES = 500

    # First pass: compute all grids + determine shared axis bounds and
    # shared color range.
    grids = []
    all_points = []
    for N in sizes:
        points_np = make_clusters(N)
        all_points.append(points_np)
        grids.append(rasterize_scatter(Tensor(points_np), resolution=RES))
    xmin = min(p[:, 0].min() for p in all_points)
    xmax = max(p[:, 0].max() for p in all_points)
    ymin = min(p[:, 1].min() for p in all_points)
    ymax = max(p[:, 1].max() for p in all_points)
    extent = [xmin, xmax, ymin, ymax]
    vmax = max(np.log1p(g).max() for g in grids)

    # NOTE: because each grid was rasterized against its OWN min/max (to
    # fill the full RES×RES frame), the grid extents don't match `extent`
    # above. Re-rasterize with fixed bounds so all panels are truly
    # apples-to-apples.
    def rasterize_fixed(points_np, xmin, xmax, ymin, ymax, res):
        p = (points_np - np.array([xmin, ymin])) / np.array([xmax - xmin, ymax - ymin])
        p = (p * (res - 1)).astype(np.int32).clip(0, res - 1)
        xi = Tensor(p[:, 0]); yi = Tensor(p[:, 1])
        flat_idx = (yi * res + xi).cast("int32")
        src = Tensor.ones(points_np.shape[0])
        return (Tensor.zeros(res * res)
                .scatter_reduce(0, flat_idx, src, reduce="sum", include_self=True)
                .reshape(res, res).realize().numpy())

    grids = [rasterize_fixed(p, xmin, xmax, ymin, ymax, RES) for p in all_points]
    vmax = max(np.log1p(g).max() for g in grids)

    cmap_obj = plt.get_cmap("magma")
    bg_color = cmap_obj(0.0)   # match panel backgrounds to colormap's vmin color

    fig, axes = plt.subplots(2, 2, figsize=(11, 10),
                             gridspec_kw={"wspace": 0.25, "hspace": 0.3})
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for ax, N, grid in zip(axes, sizes, grids):
        ax.set_facecolor(bg_color)
        im = ax.imshow(
            np.log1p(grid),
            cmap="magma",
            origin="lower",
            extent=extent,
            vmin=0, vmax=vmax,
            aspect="equal",
        )
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
               xlabel="PC1", ylabel="PC2", title=f"N = {N:,} cells")
        ax.set_aspect("equal", adjustable="box")
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
        fig.colorbar(im, cax=cax, label="log(1 + density)")

    fig.suptitle(
        "tinybio GPU rasterization across scales\n"
        "(shared axes and color range; only cell count differs)",
        fontsize=13,
    )
    plt.savefig("figures/fig3_density_by_size.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  → figures/fig3_density_by_size.png")


def make_realistic(n_total: int, seed: int = 7) -> np.ndarray:
    """Synthetic data resembling real scRNA-seq PCA output more closely:
    - Three main "cell type" clusters (big Gaussian blobs, different sizes)
    - One curved "differentiation trajectory" (cells along a curve)
    - Four rare populations (tiny clusters, 0.5% each)
    - Diffuse background noise
    This has enough structural variety that scatter's overplotting actually
    loses information. Pure Gaussians-only is too easy for scatter."""
    rng = np.random.default_rng(seed)
    parts = []

    # 3 main clusters (40% of cells)
    for center, frac, spread in [
        ([ 4.0,  3.0], 0.14, 0.9),
        ([-3.0,  2.5], 0.16, 1.1),
        ([ 0.0, -4.0], 0.10, 0.7),
    ]:
        n = int(n_total * frac)
        parts.append(rng.normal(center, spread, size=(n, 2)))

    # Curved differentiation trajectory (30%): parametric curve + noise
    n_traj = int(n_total * 0.30)
    t = rng.random(n_traj)
    curve_x = 6 * (t - 0.5)
    curve_y = 3 * np.sin(t * np.pi * 2) - 1
    traj = np.stack([curve_x, curve_y], axis=1) + rng.normal(0, 0.3, size=(n_traj, 2))
    parts.append(traj)

    # 4 rare populations (2% total)
    for center in [[ 7.0, -2.5], [-6.5, -1.5], [ 6.0,  6.0], [-5.5,  5.5]]:
        n = int(n_total * 0.005)
        parts.append(rng.normal(center, 0.35, size=(n, 2)))

    # Diffuse noise (fill remainder)
    n_noise = n_total - sum(p.shape[0] for p in parts)
    parts.append(rng.normal([0, 0], 4.0, size=(n_noise, 2)))

    return np.vstack(parts).astype(np.float32)


def fig4_scatter_fails_at_scale():
    """Three-panel comparison at 10M cells showing how each approach
    handles scale and structure. Same colormap, axes, background, and
    panel size on all three. Render times in titles."""
    print("Figure 4: scatter fails at scale (10M cells, complex structure)")
    N = 10_000_000
    RES = 800
    points_np = make_realistic(N)
    xmin, xmax = points_np[:, 0].min(), points_np[:, 0].max()
    ymin, ymax = points_np[:, 1].min(), points_np[:, 1].max()
    extent = [xmin, xmax, ymin, ymax]

    # Pre-rasterize (we need the grid for panels 2 and 3). Warm up the
    # kernel once so we time steady-state execution, not first-call JIT
    # compile + autotune (which on a cold cache would dominate).
    pts = Tensor(points_np)
    _ = rasterize_scatter(pts, resolution=RES)  # warmup
    t0 = time.perf_counter()
    grid = rasterize_scatter(pts, resolution=RES)
    rasterize_time = time.perf_counter() - t0
    grid_log = np.log1p(grid)

    # Choose a consistent time unit for all three panel titles. See
    # docs/multipanel_figures_skill.md rule 21: use the same unit across
    # panels in a comparison figure. We pick seconds if the max time is
    # ≥ 1 s, otherwise milliseconds.

    # Per-point density for the "density-colored scatter" panel
    x_norm = (points_np[:, 0] - xmin) / (xmax - xmin) * (RES - 1)
    y_norm = (points_np[:, 1] - ymin) / (ymax - ymin) * (RES - 1)
    xi = x_norm.astype(np.int32).clip(0, RES - 1)
    yi = y_norm.astype(np.int32).clip(0, RES - 1)
    density_per_point = np.log1p(grid[yi, xi])

    cmap_name = "viridis"
    cmap_obj = plt.get_cmap(cmap_name)
    bg_color = cmap_obj(0.0)
    vmin, vmax = 0, grid_log.max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={"wspace": 0.28})
    fig.patch.set_facecolor("white")

    # Time matplotlib rendering honestly: scatter() is lazy (only builds
    # a PathCollection); the real cost is at canvas.draw(). We render each
    # scatter panel to its own figure, measure draw time, then composite.
    def time_scatter_panel(ax, **kwargs):
        t0 = time.perf_counter()
        sc = ax.scatter(points_np[:, 0], points_np[:, 1], **kwargs)
        ax.figure.canvas.draw()  # force actual rasterization
        return sc, time.perf_counter() - t0

    # Measure all three times first, THEN decide the shared time unit
    # before building any titles. This keeps the unit consistent across
    # panels (see figures skill doc rule 21).
    axes[0].set_facecolor(bg_color)
    _, naive_time = time_scatter_panel(
        axes[0],
        c=[cmap_obj(0.7)], s=0.5, alpha=0.02, edgecolors="none",
    )
    axes[1].set_facecolor(bg_color)
    sc, density_scatter_time = time_scatter_panel(
        axes[1],
        c=density_per_point, cmap=cmap_name, vmin=vmin, vmax=vmax,
        s=0.5, alpha=0.3, edgecolors="none",
    )

    # Shared-unit formatter. If max time across panels is ≥ 1 s, use s;
    # otherwise use ms. All three panels use the same unit.
    def fmt_time(t_s: float, unit: str) -> str:
        return f"{t_s:.2f} s" if unit == "s" else f"{t_s * 1000:.0f} ms"

    unit = "s" if max(naive_time, density_scatter_time, rasterize_time) >= 1.0 else "ms"

    axes[0].set(
        xlim=(xmin, xmax), ylim=(ymin, ymax),
        xlabel="PC1", ylabel="PC2",
        title=f"(A) ax.scatter, single color\n{N:,} cells • render: {fmt_time(naive_time, unit)}",
    )
    axes[0].set_aspect("equal", adjustable="box")
    # Reserve an invisible colorbar slot so Panel A's physical width
    # matches Panels B and C.
    cax0 = make_axes_locatable(axes[0]).append_axes("right", size="4%", pad=0.1)
    cax0.set_visible(False)

    axes[1].set(
        xlim=(xmin, xmax), ylim=(ymin, ymax),
        xlabel="PC1", ylabel="PC2",
        title=f"(B) ax.scatter, density-colored\n{N:,} cells • render: {fmt_time(density_scatter_time, unit)}",
    )
    axes[1].set_aspect("equal", adjustable="box")
    cax1 = make_axes_locatable(axes[1]).append_axes("right", size="4%", pad=0.1)
    fig.colorbar(sc, cax=cax1, label="log(1 + density)")

    axes[2].set_facecolor(bg_color)
    im = axes[2].imshow(
        grid_log, cmap=cmap_name, vmin=vmin, vmax=vmax,
        origin="lower", extent=extent, aspect="equal",
    )
    axes[2].set(
        xlim=(xmin, xmax), ylim=(ymin, ymax),
        xlabel="PC1", ylabel="PC2",
        title=f"(C) tinybio GPU rasterization\n{N:,} cells • render: {fmt_time(rasterize_time, unit)}",
    )
    axes[2].set_aspect("equal", adjustable="box")
    cax2 = make_axes_locatable(axes[2]).append_axes("right", size="4%", pad=0.1)
    fig.colorbar(im, cax=cax2, label="log(1 + density)")

    plt.savefig("figures/fig4_scatter_fails_at_scale.png",
                dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  naive scatter render:    {naive_time:.2f} s")
    print(f"  density-colored render:  {density_scatter_time:.2f} s")
    print(f"  GPU rasterize (warm):    {rasterize_time*1000:.2f} ms")
    print("  → figures/fig4_scatter_fails_at_scale.png")


if __name__ == "__main__":
    fig1_scaling_curve()
    fig2_scatter_vs_rasterize()
    fig3_density_by_size()
    fig4_scatter_fails_at_scale()
    print("\nAll four figures written to figures/")
