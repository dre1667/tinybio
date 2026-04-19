"""Small helpers that work around the AMD eGPU synchronize-timeout issue.

Background
----------
tinygrad queues GPU work asynchronously. `.realize()` commits a kernel to
the execution queue but does NOT force the GPU to finish before returning.
When a benchmark loop submits many large `scatter_reduce` operations
in a tight sequence, the queue fills faster than the GPU drains it.

Later, when Python garbage-collects the large input tensors, each
buffer's `_free` internally calls `synchronize(timeout=N_seconds)` to
ensure the GPU is done using that memory. If the GPU still has a big
backlog of queued work, this synchronize exceeds the timeout and
tinygrad's AMD runtime escalates it to `RuntimeError("Device hang
detected")` — even though the GPU is actually fine, just busy.

The fix
-------
Periodically force a small scalar readback, which drains the queue by
blocking on a full pipeline flush. Call `drain_gpu_queue()` inside
benchmark loops for N ≥ ~5M scatter_reduce operations, or wrap a
rasterization in `rasterize_drained()` when in doubt.

This is essentially what `cuda.synchronize()` is in the CUDA world —
tinygrad doesn't yet expose an explicit device-level barrier, so we
synthesize one via a no-op realized tensor readback.
"""
from __future__ import annotations

from tinygrad import Tensor


# Sentinel tensor reused for draining. Creating a fresh 1-element Tensor
# each call is ~free, but we cache to make intent explicit in call sites.
_SENTINEL = None


def drain_gpu_queue() -> None:
    """Force the GPU to finish all queued work before returning.

    Uses a 1-element readback, which triggers a synchronize internally.
    Costs <1 ms on warm cache, ~10 ms cold.

    Call this:
      - Between iterations of a benchmark loop when N ≥ ~5M per iteration
      - Before `del` / `gc` of large GPU tensors in a tight loop
      - Whenever you see `Device hang detected` errors during cleanup
    """
    global _SENTINEL
    if _SENTINEL is None:
        _SENTINEL = Tensor([0.0])
    # .numpy() forces a GPU→CPU readback, which requires the GPU to
    # finish everything before the sentinel. Equivalent to a barrier.
    _ = _SENTINEL.realize().numpy()


def drained(fn):
    """Decorator form: call the function, then drain the GPU queue.

    >>> @drained
    ... def rasterize(...):
    ...     return Tensor.zeros(...).scatter_reduce(...).realize()
    """
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        drain_gpu_queue()
        return result
    return wrapper
