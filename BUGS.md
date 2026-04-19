# BUGS.md

Working log of issues hit while building tinybio. Append new entries at the
top. Format: symptom / cause / fix.

---

## tinygrad `Tensor.svd()` is unusably slow at realistic scRNA-seq scales

**Symptom.** `Tensor(X).svd()` where `X` is a (2700, 2000) z-scored scaled
matrix did not return after >18 minutes of wall time on AMD eGPU
(DEV=AMD, JITBEAM=2, PARALLEL=10) — single core at ~98% CPU, ~3.2 GB RSS,
no output, no error. Process had to be killed.

**Cause.** tinygrad/tensor.py at commit f28ea84 implements SVD as QR
preprocessing followed by one-sided Jacobi sweeps:

```python
max_iterations, iterations_per_round = 1, int(num * math.log2(num) * 2 + 2)
for _ in range(max_iterations * iterations_per_round):
    U, V, permute, inverse_permute = one_round_jacobi(U, V, permute, inverse_permute)
```

For `num = min(m, n) = 2000` that's ~44k Jacobi rounds. Each round does a
`split`, two `cat`s, a `gather`, a `scatter`, and several element-wise ops
over a `(num, num)` tensor — on the order of 20+ GPU kernel launches per
round. Roughly 10^6 kernel launches per SVD call. Each launch pays TB4
dispatch overhead (~0.1–0.5 ms), so the total is tens of minutes of pure
overhead before any useful compute. Unusable for scRNA-seq.

**Fix.** Added `tinybio/pca.py` implementing randomized truncated SVD
(Halko, Martinsson & Tropp 2011) with subspace iteration. Keeps the large
matmuls on the GPU and uses numpy for thin QRs and the final small SVD.
For top-50 on (2700, 2000): constant ~14 large GPU matmuls regardless of
iteration count. 5 orders of magnitude fewer kernel launches than
`Tensor.svd()`.

**Upstream opportunity.** Worth filing an issue on tinygrad to (a)
provide a `truncated_svd` primitive, or (b) fuse the Jacobi round into a
single captured graph via JIT so kernel launch overhead amortizes.
Probably not a blocker for us — the randomized SVD path is the standard
production choice for truncated SVD anyway, and it's what scanpy and
scikit-learn use under the hood.

---

## `JITBEAM=2` does not accelerate `tinybio.pca.pca` on PBMC3k

**Symptom.** Running `examples/pbmc3k_pca.py` with `DEV=AMD JITBEAM=2
PARALLEL=10` vs plain `DEV=AMD` yields nearly identical timings (warm
median: 157 vs 160 ms; cold: 1400 vs 1460 ms).

**Cause.** The randomized SVD issues ~14 separate GPU matmul launches
per call across Thunderbolt 4. At this scale the wall time is dominated
by per-kernel *dispatch latency* (TB4 round-trip ~0.1–0.5 ms × 14 +
Python overhead), not kernel *compute time*. `JITBEAM` autotunes each
individual kernel's inner loop shape, which helps when compute dominates
but does nothing for launch overhead.

**Fix (landed).** Two changes stacked in M2 took PBMC68k from 1.7 s
(tied with scanpy) to 564 ms (2.93× faster):

1. `@TinyJit` on the one-sided power step `X @ (X.T @ Q)` with a
   shape-keyed cache. Fuses the two matmuls into one captured graph
   per `(X.shape, Q.shape)` pair.
2. GPU-resident orthonormalization: compute `G = Y.T @ Y` on the GPU,
   eigendecompose the tiny `(l, l)` block on CPU, apply `V diag(w^-1/2)
   V.T` back on the GPU — transferring ~25 kB per step instead of the
   full `(rows, l)` probe. At 245 k cells this is the difference
   between 0.5 ms and 680 ms per orthonormalization; see the profile
   in `examples/profile_tms.py`.

---

## `numpy.linalg.qr` on tall-thin matrices is mystery-slow on Apple Accelerate

**Symptom.** At `(245389, 80)` shape, `numpy.linalg.qr(Y, mode="reduced")`
takes ~680 ms per call on an M4 Pro. Eleven calls per PCA invocation =
7.5 s wall time, dominating everything else. The theoretical work
(Householder reduction plus Q accumulation, ~6.3 GFLOPs) should take
~20 ms on threaded LAPACK.

**Cause.** Not conclusively identified — Apple Accelerate's tall-skinny
QR path appears to run effectively single-threaded for this shape class
in the numpy binding. Same code on scipy/OpenBLAS is much closer to the
theoretical throughput.

**Fix.** Skip numpy QR entirely: use Cholesky-QR / eigendecomposition
of the `(l, l)` Gram matrix instead, which keeps everything but the
tiny `l × l` block on the GPU. See the `_chol_qr` note above.
