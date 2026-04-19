# eGPU bottleneck hunting: a skill for tinygrad on AMD eGPU + macOS

A reusable method for finding why a workload that *should* win on the
eGPU is slow, and fixing it. Written after a full cycle on `tinybio`
where the GPU PCA went from **2.56× slower** to **1.35–5.75× faster**
than scanpy CPU across four real datasets by finding and killing three
separate surprise bottlenecks.

This is the "skill" — a portable checklist you can copy into any
future tinygrad-eGPU-on-macOS project. Every rule below comes from a
real thing we tripped on.

---

## The mental model

On an AMD eGPU over Thunderbolt 4 on macOS, performance is almost
never about GPU FLOPs. It is about:

1. **Kernel-launch latency.** Each GPU operation dispatched from Python
   costs roughly 0.5–5 ms of TB4 round-trip + Python + tinygrad graph
   overhead. If you launch 20 kernels in a workload, you've already
   paid 10–100 ms regardless of how fast the compute is.
2. **Cross-device transfers.** Moving an ``(m, l)`` tensor to CPU for a
   "small helper" call (e.g. `np.linalg.qr`) costs
   ``m*l*4 / 3.2 GB/s`` each direction. For ``(245000, 80)`` that's
   ~25 ms one way, 50 ms round-trip — dwarfing the ~20 ms the helper
   itself takes.
3. **Platform quirks.** Apple Accelerate's LAPACK has surprisingly
   slow paths for shapes that NumPy/SciPy on Linux breeze through.
   Assume nothing about CPU-side libraries' performance without
   measuring.
4. **GPU FLOPs are almost never the limit** at this scale. The
   7900 XT has ~50 fp32 TFLOPs; a top-50 PCA on 245k cells needs
   under 2 TFLOPs of real compute. Compute is ~always <5% of wall
   time on these problems.

So the question "why is this slow?" reduces to *"which of launch,
transfer, or platform-quirk is eating the budget?"*

---

## The ordered playbook

### Step 1 — Measure the baseline end-to-end

Get one number that characterizes "tinygrad end-to-end wall time" vs
"scanpy / numpy / sklearn CPU wall time" on a representative input.
Report both warm-median and minimum. Three warm runs is enough for a
ratio; five is better for a paper plot.

```python
t0 = time.perf_counter()
result = gpu_fn(X_np)
wall = time.perf_counter() - t0
```

**Gotcha.** The first call pays JIT compile + autotune + graph
capture. Use a separate "cold" timing and never include it in the
median. tinygrad's `@TinyJit` takes 2 calls before it dispatches the
captured graph; treat the first 2 calls of any JIT-wrapped function
as warm-up, not benchmark.

### Step 2 — Break the workload into phases

Don't speculate about where time goes — isolate each phase. For the
tinybio PCA the phases were:

- data transfer numpy → GPU (`Tensor(X_np).realize()`)
- per-iteration GPU matmul chunk (`_apply_A`)
- per-iteration CPU helper (`_thin_qr_cpu`)
- final SVD + emit

Time each independently with a fixed input. The cheapest way: a small
`examples/profile_<workload>.py` that warms each sub-function, then
runs 10 tight loops over each.

**Gotcha.** `Tensor.realize()` alone does NOT sync the GPU. If you
want an honest timing of a GPU step, end it with
`.realize().numpy().shape` (or `.item()`) to force a CPU readback,
which blocks until the GPU is idle. Otherwise you're timing graph
construction, not compute.

### Step 3 — Count kernel launches

Grep the hot function and count every Python-level tensor operation
that produces a new tensor. Each line ≈ one dispatch. If you see more
than ~5–10 per call, you're almost certainly launch-bound on TB4.

### Step 4 — If launch-bound: `@TinyJit` the chunks that can be fused

The single most effective optimization we found. Wrap any contiguous
sequence of pure-Tensor ops in a `TinyJit`-decorated function so
tinygrad captures them as one graph:

```python
from tinygrad import TinyJit

@TinyJit
def inner_step(X: Tensor, Q: Tensor) -> Tensor:
    return (X @ (X.T @ Q)).realize()
```

Measured impact on PBMC3k: two matmuls went from 160 ms combined to
90 ms — 1.8× faster with no algorithmic change.

**Gotcha (shape mismatch).** A module-level `TinyJit` captures one
shape and will raise `JitError: args mismatch` when called with a
different shape. For anything that sweeps shapes, use a per-shape
cache:

```python
_JIT_CACHE: dict[tuple, "TinyJit"] = {}
def inner_step(X, Q):
    key = (tuple(X.shape), tuple(Q.shape))
    fn = _JIT_CACHE.get(key)
    if fn is None:
        fn = TinyJit(lambda X_, Q_: (X_ @ (X_.T @ Q_)).realize())
        _JIT_CACHE[key] = fn
    return fn(X, Q)
```

### Step 5 — If transfer-bound: cut round-trips ruthlessly

Every CPU↔GPU hop is expensive. The two patterns we hit:

- **"Small helper on CPU" pattern.** Algorithm naturally wants a
  `numpy.linalg.qr` or similar on an intermediate tensor. Cost:
  ``(m, l)`` transfer each direction = tens to hundreds of ms.
- **"Multiple PCA calls with fresh input" pattern.** Every call pays
  the full ``m*d`` transfer. For a 550 MB matrix that's 170 ms per
  call.

Fixes:

- Replace CPU helpers with GPU-resident equivalents whose round-trip
  is a small fixed-size block, not the full tensor. For
  orthonormalization: compute the `(l, l)` Gram matrix on the GPU,
  factor on CPU (~25 kB transfer, sub-ms), apply back on GPU.
- Expose a "resident" API variant (see [`examples/tabula_muris_senis_pca.py`](../examples/tabula_muris_senis_pca.py))
  where the caller pre-transfers and amortizes. Always report both
  end-to-end and resident timings honestly.

**Gotcha (transfer isn't always the bottleneck).** Before optimizing
for transfer, actually measure the resident-mode timing. In our TMS
run the resident number was 1.97 s vs 2.66 s end-to-end — the 0.7 s
of transfer was real but **smaller than the compute cost**. We
optimized compute first, transfer second.

### Step 6 — Suspect CPU-library shapes you haven't profiled

Apple Accelerate's `numpy.linalg.qr` at `(245389, 80)` takes **680 ms
per call**. Theoretical LAPACK on that shape is ~20 ms — a 30× slowdown.
The issue is a platform-specific tall-skinny-QR dispatch that
effectively runs single-threaded.

If you're on macOS and a "small" CPU call shows up in your profile
eating hundreds of ms, time it in isolation against a tiny synthetic
matrix, then scale up and watch for non-linearity. `scipy.linalg.qr`
uses a different code path and is usually closer to theoretical.

More broadly: when a CPU helper in a GPU pipeline shows up in the
profile, **always verify it's actually fast on your platform for your
shape** before building the rest of the optimization around it.

### Step 7 — Trust tinygrad's built-ins, but verify

Tinygrad has a rich Tensor API, but some methods are placeholders that
don't scale:

- ``Tensor.svd()`` at commit `f28ea84` runs ``num*log2(num)*2 + 2``
  Jacobi sweeps per call — ~44k sweeps on a 2000-wide matrix. Each
  sweep dispatches 15–20 kernels. **~10⁶ kernel launches per call.**
  Unusable over TB4. Implement randomized SVD yourself (~100 LOC).
- ``Tensor.qr()`` creates a full ``(m, m)`` Q matrix even in economy
  mode — OOM on tall-skinny inputs at atlas scale. Use
  Cholesky-QR / eigh-QR of the Gram matrix instead.

Always check by running the method at your target shape for 1 second
before wiring it into a hot loop. If the method takes >10 s to return
on a representative input, find or implement an alternative.

### Step 8 — `JITBEAM=2` is a trap for launch-bound workloads

`JITBEAM=2` autotunes each kernel's inner-loop shape. It helps when
*compute* is the bottleneck; it does **nothing** for dispatch
latency. On our PCA workload, `DEV=AMD JITBEAM=2` produced identical
timing to `DEV=AMD` (157 vs 160 ms on PBMC3k). Don't bother paying
the 5–15 min first-run autotune cost unless you've measured that the
kernel compute is the hot path.

### Step 9 — Watch for AMD eGPU device-lock collisions

Only one process can hold the eGPU lock at a time. If you background
two AMD benchmarks concurrently, the second will die with
`RuntimeError: Failed to acquire lock file am_usb4.lock`. Run GPU
benchmarks sequentially; save parallelism for CPU-side work.

### Step 10 — Watch for the `scatter_reduce` synchronize-timeout hang

If a loop over large-N `scatter_reduce` calls raises
`RuntimeError: Device hang detected` during cleanup, it's actually a
synchronize timeout from queued work not a real hang. Drain the queue
between iterations:

```python
def drain_gpu_queue():
    _ = Tensor([0.0]).realize().numpy()   # forces a pipeline flush
```

See [`figures/gpu_utils.py`](../figures/gpu_utils.py) and the original
rasterization bug note in [`BUGS.md`](../BUGS.md).

---

## Checklist: "is my eGPU code going to be fast?"

Before shipping, answer these:

1. ☐ Can I count the kernel launches per call on a hand? (If > ~20,
   can I fuse via `@TinyJit`?)
2. ☐ Is every CPU helper in the hot path timed in isolation on the
   real shape? (Surprise Accelerate quirks.)
3. ☐ Does any helper transfer more than a few MB per call? (Replace
   with GPU-resident equivalent if ``rows >> l``.)
4. ☐ Have I measured compute-only vs end-to-end? (To know whether to
   fix transfer or compute next.)
5. ☐ Does my warm median exclude the first 2 calls? (JIT warm-up.)
6. ☐ Do I sync before stopping the timer? (`.numpy()` or `.item()`.)
7. ☐ Does my benchmark cover multiple scales? (A bottleneck invisible
   at 10k rows can dominate at 250k.)
8. ☐ If using `@TinyJit`, is there a shape-keyed cache for
   multi-shape workloads?

If any answer is "no / I didn't check," that's where your next 2×
speedup probably hides.

---

## Concrete example: tinybio PCA optimization log

The tinybio PCA went through five regimes in one session:

| step | change | PBMC3k warm | TMS (245k) warm | vs scanpy at TMS |
|---|---|---:|---:|---:|
| 1 | `Tensor.svd()` directly            |  ∞ (killed at 18 min)  | — | N/A |
| 2 | Randomized SVD, numpy QR            | 160 ms | — | — |
| 3 | + `@TinyJit` on power step          |  90 ms | 1.70 s | 0.89× slower |
| 4 | (add real benchmarks at TMS scale)  | — | 7.58 s | **2.56× slower** |
| 5 | + profile → `np.linalg.qr` at (245k, 80) = 680 ms/call | — | — | — |
| 6 | swap numpy QR → on-GPU eigh-Gram    |  56 ms | 1.97 s | **1.81× faster** |

Total improvement on TMS: **3.8×** from two changes, no hardware
upgrade. Both changes found via profile, not speculation.
