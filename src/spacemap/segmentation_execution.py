"""Bounded GPU batch tuning; never changes image scale or segmentation settings."""

import math
import statistics
import time


def batch_candidates(free_bytes, block_size, limit=256):
    """Conservative starting budget; actual allocations still have an OOM fallback."""
    budget = max(0, free_bytes) * 0.5
    per_block = 0.12 * 1024**3 * (block_size / 224) ** 2
    ceiling = max(1, min(limit, int(budget / per_block)))
    return [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256) if n <= ceiling]


def fastest_batch(candidates, measure, recover, *, seconds=20, clock=time.perf_counter):
    """Warm each size, then compare median seconds/block. Prefer smaller ties.

    ``measure`` includes synchronisation and transfers, like real inference.
    The time budget is checked between probes, never by cancelling CUDA work.
    Only allocation failures are recoverable; other errors reach the caller.
    """
    started = clock()
    best, cost, stalls = candidates[0], math.inf, 0
    for batch in candidates:
        if cost < math.inf and clock() - started >= seconds:
            break
        try:
            measure(batch)  # warmup includes convolution algorithm selection
            elapsed = statistics.median(measure(batch) for _ in range(3))
        except RuntimeError as error:
            if not recover(error):
                raise
            break
        per_block = elapsed / batch
        if per_block < cost * 0.97:
            best, cost, stalls = batch, per_block, 0
        else:
            stalls += 1
            if stalls >= 2:
                break
    return best
