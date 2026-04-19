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

**Fix (deferred, M2).** Wrap the whole randomized-SVD call in
`@TinyJit` so tinygrad captures the entire matmul sequence as a single
graph and amortizes dispatch. That's the path that usually flips
launch-bound small-workload GPU code from loss to win. Not done this
session — in scope for M2 package work.
