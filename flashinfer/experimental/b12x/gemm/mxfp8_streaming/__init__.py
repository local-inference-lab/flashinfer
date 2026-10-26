"""Streaming small-row MXFP8 candidate for the prepared dense GEMM contract.

Operands are already encoded E4M3 values with per-32 UE8M0 scales in
F8_128x4 storage. Activation encoding belongs to the producer; this candidate
performs FP32 accumulation and writes BF16 using one K partition. Its producer
contract permits scale bytes 64–190, whose pairwise FP32 products are
normal and finite, and byte 0 for encoded blocks containing only zeros.
Every other scale byte is excluded. The full UE8M0 input
domain retains the CuTe candidate.
"""
