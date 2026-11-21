# SPDX-License-Identifier: Apache-2.0
"""Expose the interleaved KV slot selector through its established PCIe import.

The implementation in :mod:`b12x.comm.interleaved_topk` compiles its pure
CuTeDSL kernel through ``b12x._lib.compiler``.
"""

from b12x.comm.interleaved_topk import InterleavedTopKQuery, plan, supports_channel

__all__ = ["InterleavedTopKQuery", "plan", "supports_channel"]
