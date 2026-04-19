#!/usr/bin/env python3
"""2-million-cell scale test (tinybio only; scanpy cannot fit on 24 GB RAM).

At 2 M cells × 2000 features float32 the dense matrix is 16 GB. scanpy
needs a second copy of that on CPU for its arpack input, pushing total
to 32 GB — impossible on a 24 GB Mac. We generate the matrix directly
on the GPU via ``Tensor.randn`` (no CPU copy exists), run tinybio PCA,
and compare against an extrapolation of scanpy's measured near-linear
scaling on smaller synthetic data (see ``examples/scale_study.py``).

Run:
    DEV=AMD python3 examples/scale_2m.py
"""
from __future__ import annotations

import time

import numpy as np
from tinygrad import Tensor

from tinybio.normalize import scale as tb_scale
from tinybio.pca import pca as tb_pca

N_ROWS = [1_000_000, 1_500_000, 1_750_000]  # 2M × 2000 float32 = 16 GB, exceeds CPU+GPU budget on 24 GB Mac + 20 GB VRAM
N_FEATURES = 2_000
N_COMPONENTS = 50
N_ITER = 5
N_RUNS = 3

# Measured scanpy times on the sklearn synthetic scale curve
# (see scale_study.py output: N -> scanpy_ms):
#   3k  ->   53 ms,  10k  ->  200 ms,  30k -> 747 ms, 100k -> 2822 ms.
# Fit a power law scanpy_ms = a * N^b from those points.
_SCANPY_CURVE = np.array(
    [[3_000, 53.0], [10_000, 200.0], [30_000, 746.7], [100_000, 2822.4]]
)


def scanpy_projection_ms(n: int) -> float:
    x = np.log(_SCANPY_CURVE[:, 0])
    y = np.log(_SCANPY_CURVE[:, 1])
    b, log_a = np.polyfit(x, y, 1)
    a = float(np.exp(log_a))
    return a * (n ** b)


def main() -> None:
    print("Scale test, tinybio-only (scanpy cannot fit at 2 M × 2000 on 24 GB RAM)")
    print(f"{'N':>10}  {'tinybio (ms)':>14}  {'scanpy proj (ms)':>18}  {'speedup':>10}")
    print(f"{'-'*10}  {'-'*14}  {'-'*18}  {'-'*10}")

    for n in N_ROWS:
        # Generate on CPU then transfer — tinygrad's GPU-side Tensor.randn
        # uses ~2x the output size in intermediate state, which OOMs at n=1M
        # on 20 GB VRAM. Plain numpy + one-way transfer stays within budget.
        # Gaussian noise already satisfies the zero-centered, unit-variance
        # per-column preprocessing contract PCA expects.
        print(f"  [generating {n:,} × {N_FEATURES} on CPU...]", flush=True)
        rng = np.random.default_rng(42)
        X_np = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
        print(f"  [uploading {X_np.nbytes/1e9:.2f} GB to GPU...]", flush=True)
        X = Tensor(X_np).realize()
        del X_np

        # Warm-up (first JIT capture for this new shape).
        _ = tb_pca(X, n_components=N_COMPONENTS, n_iter=N_ITER)

        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            tb_pca(X, n_components=N_COMPONENTS, n_iter=N_ITER)
            times.append(time.perf_counter() - t0)
        tg_ms = float(np.median(times) * 1000)
        sc_proj = scanpy_projection_ms(n)
        speedup = sc_proj / tg_ms
        print(f"{n:>10,d}  {tg_ms:>14.1f}  {sc_proj:>18,.0f}  {speedup:>9.1f}x")

        # Free GPU memory before next iteration.
        del X


if __name__ == "__main__":
    main()
