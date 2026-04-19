# Multipanel scientific figures: a skill

A concentrated guide for producing publication-quality multipanel figures
with matplotlib. Written after getting it wrong five different ways in
one session while making the `tinybio` benchmark plots. Every rule here
comes from a real thing that looked bad or was misleading.

This is the "skill" — a reusable reference you can copy into any
scientific-plotting context, Python/tinygrad/anywhere. Keep it near the
code that generates your figures.

---

## The cardinal rule

**If two panels compare data, every visual dimension except the one
you're demonstrating must be identical.** Colormap, color range,
background, axis bounds, aspect ratio, panel size, font sizes,
colorbar layout — all of it. A single inconsistency gives the reader
an out to doubt your comparison.

Reviewers are trained to spot uneven panels. They will hold it against
you even if the underlying data is valid.

---

## Checklist for any multipanel figure

### 1. Same colormap across panels that encode the same scalar

```python
cmap_name = "viridis"  # pick one, reuse everywhere
im1 = axes[0].imshow(..., cmap=cmap_name, vmin=vmin, vmax=vmax)
im2 = axes[1].imshow(..., cmap=cmap_name, vmin=vmin, vmax=vmax)
```

Colormap choice matters less than consistency. Default to `viridis` or
`magma` for perceptual uniformity; avoid `jet`/`rainbow` unless a
journal requires them.

### 2. Shared `vmin`/`vmax` across panels

If panel A autoscales to [0, 5] and panel B autoscales to [0, 8], the
reader cannot directly compare pixel colors. Compute the shared range
once, use it everywhere:

```python
all_grids = [rasterize(d) for d in datasets]
vmin = 0
vmax = max(g.max() for g in all_grids)
for ax, g in zip(axes, all_grids):
    ax.imshow(g, vmin=vmin, vmax=vmax, cmap=cmap_name)
```

### 3. Match panel background color to the colormap's `vmin` color

```python
cmap_obj = plt.get_cmap("viridis")
bg_color = cmap_obj(0.0)                  # the color at vmin
axes[0].set_facecolor(bg_color)           # apply to every panel
axes[1].set_facecolor(bg_color)
```

Why: an `imshow` panel has its background "coloured" by the colormap at
vmin (for viridis, dark purple). A `scatter` panel's background is
white by default. Side-by-side that's visually jarring — the viewer
sees the contrast between panels as a difference in your data when it
isn't. Pinning the `scatter` panel's facecolor to `viridis(0)` fixes it.

### 4. Identical axis bounds across panels

```python
xmin = min(d[:, 0].min() for d in datasets)
xmax = max(d[:, 0].max() for d in datasets)
ymin = min(d[:, 1].min() for d in datasets)
ymax = max(d[:, 1].max() for d in datasets)
for ax in axes:
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
```

If you let each panel autoscale to its own data, cluster at coord
`(3, 3)` lands at different physical positions on the page per panel.
Viewer can no longer visually align corresponding features.

### 5. Enforce `aspect="equal"` with `adjustable="box"`

```python
ax.set_aspect("equal", adjustable="box")
```

Without this, matplotlib shrinks axes non-uniformly when subplot space
changes (e.g., when one panel has a colorbar and another doesn't).
`adjustable="box"` keeps the plotting region square (or whatever aspect
your data requires).

### 6. Use `make_axes_locatable` for colorbars — don't use `plt.colorbar(ax=...)`

The default `plt.colorbar(im, ax=ax)` **shrinks the host axes** to make
room. In a multipanel figure this means:
- Panel with colorbar: smaller plotting area
- Panel without colorbar: original size

Result: unequal panels. Fix:

```python
from mpl_toolkits.axes_grid1 import make_axes_locatable

for ax, im in zip(axes, images):
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
    fig.colorbar(im, cax=cax, label="log(1 + density)")
```

`make_axes_locatable` creates a new axes adjacent to the host (not
carved out of it). The host's plotting region is preserved.

### 7. Reserve colorbar slots on panels that don't need one

If one panel has no colorbar (e.g., single-color scatter, no colormap),
add an invisible placeholder so all panels still have the same total
width:

```python
cax = make_axes_locatable(ax_without_colorbar).append_axes("right", size="4%", pad=0.1)
cax.set_visible(False)   # reserves the space, shows nothing
```

### 8. Consistent colorbar label text and formatting

If your panels share a scalar, their colorbars should share the label
text, tick format, and orientation. Don't let one say `"density"` and
another say `"log(1+density)"` — they'd imply different scales.

### 9. `origin='lower'` for `imshow` when axes are Cartesian

Default `imshow` has origin at the top-left (image convention).
Scientific plots usually want origin at bottom-left (Cartesian
convention). Always:

```python
ax.imshow(data, origin="lower", extent=[xmin, xmax, ymin, ymax])
```

`extent=` makes the displayed image coordinates match your data
coordinates, so tick labels are meaningful.

### 10. Don't transpose arrays before `imshow` unless you know why

If you build a density grid with row index = y and column index = x
(the standard convention), pass the array directly — no `.T`. `imshow`
with `origin="lower"` maps `A[row, col]` to plot position `(col, row)`,
which equals `(x, y)` because row=y and col=x. Transposing swaps your
axes and produces mirror-image artifacts. See the bug log in
`tinybio/BUGS.md` (if it exists) or `tinybio/CLAUDE.md`.

### 11. Use `gridspec_kw` to control inter-panel spacing

```python
fig, axes = plt.subplots(
    1, 3, figsize=(18, 6),
    gridspec_kw={"wspace": 0.28, "hspace": 0.3},
)
```

Default spacing often leaves labels crammed into neighboring panels'
axes. `wspace` and `hspace` are fractions of the average panel size.
Tune until titles and colorbars don't collide.

### 12. Save with `bbox_inches="tight"` and adequate `dpi`

```python
plt.savefig("fig.png", dpi=140, bbox_inches="tight")
```

Without `bbox_inches="tight"`, labels can get clipped. For paper
figures, `dpi=150+` is the minimum; `dpi=300` for print.

### 13. Force rendering when timing `matplotlib` operations

Matplotlib's `ax.scatter(...)` is **lazy** — it builds a `PathCollection`
and defers actual rasterization until `savefig` or `canvas.draw()`.
Timing a bare `scatter` call gives you collection construction time,
not render time, which can be orders of magnitude different.

```python
import time

t0 = time.perf_counter()
ax.scatter(x, y, s=0.5, alpha=0.01)
fig.canvas.draw()              # force actual rasterization
scatter_time = time.perf_counter() - t0
```

### 14. Warm up GPU kernels before timing

Any GPU-backed call (tinygrad, cupy, JAX) JIT-compiles on first use.
The first call timing includes autotune + compile, which can be
seconds. Steady-state timing is what you want in a benchmark.

```python
_ = gpu_rasterize(points, res)     # warmup, discard result
t0 = time.perf_counter()
grid = gpu_rasterize(points, res)  # timed
elapsed = time.perf_counter() - t0
```

### 15. Font sizes that survive downsizing

Journal figures get printed small. Set font sizes large enough to be
readable at half-size:

```python
plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
```

Or pass `fontsize=` per-call. Avoid defaults that assume large screen
viewing.

### 16. Use log scales when the data is log-distributed

Don't force linear axes on data that spans orders of magnitude. A
benchmark that runs from 0.05 ms to 103 ms compresses everything below
20 ms into invisibility on a linear axis.

```python
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)   # minor gridlines too
```

### 17. Color should encode exactly one thing at a time

If you're encoding density via colormap, don't also vary marker size
by the same quantity. Pick one channel per variable. Reviewers
actively penalize "double-encoded" dimensions because it implies the
author doesn't know which channel to trust.

### 18. Colorbar tick density should be similar across panels

If one colorbar has 3 ticks and another has 7, readers suspect a
different range. Matplotlib auto-ticks based on figure size; if panels
have different dimensions the tick counts diverge. Fix by either:
- Matching panel sizes exactly (see checklist items above)
- Setting explicit ticks: `cbar.set_ticks([0, 1, 2, 3, 4, 5])`

### 19. Titles annotated with important metadata

Especially for benchmarks: put the key number *in the title*, not
only in figure text. If panels A and B show 20 s vs 3 s, the reader
should see that at a glance.

```python
axes[0].set_title(f"(A) ax.scatter\n{N:,} cells • render: {time_a:.1f} s")
axes[1].set_title(f"(B) rasterize\n{N:,} cells • render: {time_b*1000:.0f} ms")
```

### 21. Consistent units in titles, captions, and annotations across panels

If panel A's title says `render: 21 s` and panel C's title says
`render: 3075 ms`, readers have to do mental arithmetic to compare.
Always use the same unit in every panel of a comparison figure.

```python
# Pick one unit based on the largest time, use for all panels.
max_time = max(t_a, t_b, t_c)
unit = "s" if max_time >= 1.0 else "ms"

def fmt(t):
    return f"{t:.2f} s" if unit == "s" else f"{t * 1000:.0f} ms"

axes[0].set_title(f"(A) {fmt(t_a)}")
axes[1].set_title(f"(B) {fmt(t_b)}")
axes[2].set_title(f"(C) {fmt(t_c)}")   # all three in seconds OR all in ms
```

Same principle for other units — memory in GB vs MB, dataset sizes
in millions vs thousands, etc. Reviewers penalize unit inconsistency
because it implies you weren't careful with the comparison.

### 22. Before saving: sanity-check the output

- Do the panels have identical axis ranges? (read them off the tick labels)
- Do corresponding features appear at identical positions? (pick a
  cluster, verify it's at the same `(x, y)` on every panel)
- Are the colorbar ranges identical where they should be?
- Does any panel look noticeably larger/smaller than the others?
- Is the background color the same across all panels?

If anything fails these checks, fix before you call the figure done.

---

## A reference template

Minimum-viable multipanel figure scaffold that respects all the rules:

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def good_multipanel_figure(datasets, out_path):
    # 1. Compute shared ranges
    xmin = min(d[:, 0].min() for d in datasets)
    xmax = max(d[:, 0].max() for d in datasets)
    ymin = min(d[:, 1].min() for d in datasets)
    ymax = max(d[:, 1].max() for d in datasets)
    grids = [compute_density_grid(d) for d in datasets]
    vmax  = max(g.max() for g in grids)
    vmin  = 0

    # 2. Colormap + matched background
    cmap  = "viridis"
    bg    = plt.get_cmap(cmap)(0.0)

    # 3. Figure + axes
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(6 * len(datasets), 6),
        gridspec_kw={"wspace": 0.25},
    )
    fig.patch.set_facecolor("white")
    if len(datasets) == 1:
        axes = [axes]

    # 4. Per-panel plot + colorbar
    for ax, g, label in zip(axes, grids, LABELS):
        ax.set_facecolor(bg)
        im = ax.imshow(
            g, cmap=cmap, vmin=vmin, vmax=vmax,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            aspect="equal",
        )
        ax.set(
            xlim=(xmin, xmax), ylim=(ymin, ymax),
            xlabel="PC1", ylabel="PC2", title=label,
        )
        ax.set_aspect("equal", adjustable="box")
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
        fig.colorbar(im, cax=cax, label="log(1 + density)")

    # 5. Save
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
```

---

## Summary

| # | Rule |
|---|---|
| 1 | Same colormap across panels encoding the same scalar |
| 2 | Shared `vmin`/`vmax` across panels |
| 3 | Background color = colormap's `vmin` color |
| 4 | Identical axis bounds on every panel |
| 5 | `aspect="equal", adjustable="box"` |
| 6 | `make_axes_locatable` for colorbars, not `plt.colorbar(ax=...)` |
| 7 | Reserve (invisible) colorbar slots where needed for width parity |
| 8 | Consistent colorbar label text and tick format |
| 9 | `origin="lower"` for Cartesian imshow |
| 10 | Don't `.T` a row=y / col=x density grid |
| 11 | Tune `gridspec_kw["wspace"]`/`hspace` to avoid label collisions |
| 12 | Save with `bbox_inches="tight"`, dpi ≥ 140 |
| 13 | `fig.canvas.draw()` to force matplotlib rendering for honest timing |
| 14 | Warm up GPU kernels before timing |
| 15 | Font sizes readable at half-scale |
| 16 | Log axes when data spans orders of magnitude |
| 17 | One visual channel per variable |
| 18 | Similar colorbar tick density across panels |
| 19 | Put key metadata in titles, not just figure caption |
| 20 | Eyeball-check before shipping |
| 21 | Consistent units across panel titles/annotations (all s, or all ms — not mixed) |
