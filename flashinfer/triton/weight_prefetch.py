# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read contiguous weight storage into L2 with a bounded number of resident blocks."""

import triton
import triton.language as tl


@triton.jit
def _prefetch_weight_l2(
    words, word_count, sums, BLOCK: tl.constexpr, UNROLL: tl.constexpr
):
    block = tl.program_id(0)
    grid = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)
    partial = tl.full((BLOCK,), 0, tl.int64)
    for base in range(block * BLOCK * UNROLL, word_count, grid * BLOCK * UNROLL):
        for step in tl.static_range(UNROLL):
            indices = base + step * BLOCK + lanes
            partial += tl.load(words + indices, mask=indices < word_count, other=0)
    # Storing one sum per block keeps every weight read live.
    tl.store(sums + block, tl.sum(partial, axis=0))


def prefetch_weight_l2(words, sums, programs):
    """Read a contiguous int64 view. The caller supplies one int64 result per block."""
    _prefetch_weight_l2[(programs,)](
        words, words.numel(), sums, BLOCK=2048, UNROLL=8, num_warps=16
    )
