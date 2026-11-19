# SPDX-License-Identifier: Apache-2.0
"""Expose the interleaved KV slot selector through its established PCIe import.

The implementation in :mod:`b12x.comm.interleaved_topk` builds its optional
CUDA C++ kernel when a selector plan is materialized.
"""

from b12x.comm.interleaved_topk import InterleavedTopKQuery, jit, plan, supports_channel

__all__ = ["InterleavedTopKQuery", "jit", "plan", "supports_channel"]
