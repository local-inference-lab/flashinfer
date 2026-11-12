"""SM120 FP8 TMA prefill with a relative-distance bias and a per-row sink.

Shape: 12 query heads over 2 K/V heads, head dim 128, NHD pages of 128 tokens,
one request whose query rows end its K/V sequence.  The key ``d`` positions
before a row scores ``softmax_scale * q.k + bias[row, head, d]`` for ``0 <= d <
extent``; ``sink[row, head]`` is one more logit with a zero value.

The kernel multiplies FP8 Q by FP8 K into FP32, keeps the softmax statistics
in FP32, rounds each probability, scaled by 2**8 relative to the row maximum,
to E4M3 for the P.V product, and rounds the output to BF16.  Against an FP64
evaluation on the same FP8 Q:

* E4M3 keeps 4 significant bits, so a normal probability moves by at most
  ``2**-4`` of itself: ``2**-4 * mass`` with ``mass = sum_j p_j |v_j|``.
* Scaled probabilities below ``2**-6`` are subnormal with spacing ``2**-9``,
  so each key adds at most ``2**-10 / 2**8`` relative to the row maximum:
  ``keys * 2**-18 * max|v| * p_max``.
* The BF16 output rounding adds ``2**-8 * |out|`` and FP32 arithmetic is
  covered by ``2**-11 * mass``.
"""

from __future__ import annotations

import math

import pytest
import torch

HQ, HKV, DIM, PAGE, EXTENT = 12, 2, 128, 128, 1024
GROUP = HQ // HKV


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("SM120 or SM121 required")
    pytest.importorskip("cutlass")


def _kernel(*, bias, sink, gate=False):
    from flashinfer.cute_dsl.attention.fmha.sm120 import (
        compile_sm120_fmha_fp8_paged_kernel,
    )

    return compile_sm120_fmha_fp8_paged_kernel(
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=HQ,
        num_kv_heads=HKV,
        head_dim=DIM,
        is_causal=True,
        kv_tile=128,
        q_tile=128,
        num_tokens_per_page=PAGE,
        device=torch.device("cuda"),
        balanced_scheduler=True,
        kv_layout="NHD",
        with_output_gate=gate,
        with_relative_bias=bias,
        with_row_sink=sink,
    )


class _Case:
    def __init__(self, *, rows, capacity, seed):
        self.rows, self.capacity = rows, capacity
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.gen = gen
        pages = (capacity + PAGE - 1) // PAGE

        def randn(shape, std):
            return torch.randn(shape, generator=gen, device="cuda") * std

        self.randn = randn
        self.key = randn((pages + 2, PAGE, HKV, DIM), 0.5).to(torch.float8_e4m3fn)
        self.value = randn((pages + 2, PAGE, HKV, DIM), 0.5).to(torch.float8_e4m3fn)
        self.table = torch.randperm(pages + 2, generator=gen, device="cuda")[:pages]
        self.table = self.table.to(torch.int32).view(1, pages).contiguous()
        self.q = randn((rows, HQ, DIM), 0.5).to(torch.bfloat16)
        self.q8 = self.q.to(torch.float8_e4m3fn)
        self.bias = randn((rows, HQ, EXTENT), 2.0).to(torch.bfloat16)
        self.sink = randn((rows, HQ), 3.0)
        self.lengths = torch.tensor([capacity], dtype=torch.int32, device="cuda")
        self.cu = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
        self.out = torch.empty_like(self.q)

    def run(self, kernel, *, bias, sink, gate=None, extended=True):
        from cutlass import Float32, Int32

        self.out.fill_(float("nan"))
        kernel(
            self.q8,
            self.key.permute(0, 2, 1, 3),
            self.value.permute(0, 2, 1, 3),
            self.out,
            None,
            Float32(math.log2(math.e) / math.sqrt(DIM)),
            Float32(1),
            None,
            self.lengths,
            self.cu,
            self.table,
            None,
            Int32(self.rows),
            False,
            # The plain program takes only the K/V tile ranges after use_pdl.
            *(
                (gate, None, self.bias if bias else None, self.sink if sink else None)
                if extended
                else (None,)
            ),
        )

    def reference(self, *, bias, sink, query=None):
        """FP64 output, error bound, and probability mass inside the extent."""
        length = int(self.lengths[0])
        pages = self.table[0, : (length + PAGE - 1) // PAGE].long()
        key = self.key[pages].reshape(-1, HKV, DIM)[:length].double()
        value = self.value[pages].reshape(-1, HKV, DIM)[:length].double()
        key, value = (t.repeat_interleave(GROUP, 1) for t in (key, value))
        query = (self.q8 if query is None else query).double()
        position = length - self.rows + torch.arange(self.rows, device="cuda")
        distance = position[:, None] - torch.arange(length, device="cuda")[None]
        scores = torch.einsum("thd,jhd->thj", query, key) * DIM**-0.5
        in_extent = (distance >= 0) & (distance < EXTENT)
        if bias:
            index = distance.clamp(0, EXTENT - 1)[:, None].expand(-1, HQ, -1)
            scores = scores + self.bias.double().gather(2, index) * in_extent[:, None]
        scores = scores.masked_fill(~(distance >= 0)[:, None], float("-inf"))
        if sink:
            scores = torch.cat((scores, self.sink.double()[:, :, None]), -1)
        probability = scores.softmax(-1)[..., :length]
        out = torch.einsum("thj,jhd->thd", probability, value)
        mass = torch.einsum("thj,jhd->thd", probability, value.abs())
        # The kernel normalizes probabilities to the largest key score.
        p_max = probability.amax(-1, keepdim=True)
        bound = (
            (2.0**-4 + 2.0**-11) * mass
            + (position + 1)[:, None, None] * 2.0**-18 * value.abs().max() * p_max
            + 2.0**-8 * out.abs()
        )
        return out, bound, (probability * in_extent[:, None]).sum(-1)


@pytest.mark.parametrize("bias, sink", [(True, False), (False, True), (True, True)])
def test_graph_replay_matches_reference(bias, sink):
    """One captured graph serves contexts on both sides of the bias extent."""
    _require_sm120()
    kernel = _kernel(bias=bias, sink=sink)
    case = _Case(rows=320, capacity=4096, seed=3)
    case.run(kernel, bias=bias, sink=sink)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.run(kernel, bias=bias, sink=sink)
    for length in (4096, 320, EXTENT, EXTENT + 1, 320 + EXTENT - 1, 2500):
        case.lengths.fill_(length)
        case.q8.copy_(case.randn(case.q.shape, 0.5).to(torch.float8_e4m3fn))
        case.bias.copy_(case.randn(case.bias.shape, 2.0).to(torch.bfloat16))
        case.sink.copy_(case.randn(case.sink.shape, 3.0))
        graph.replay()
        torch.cuda.synchronize()
        ref, bound, _ = case.reference(bias=bias, sink=sink)
        assert torch.isfinite(case.out).all()
        excess = (case.out.double() - ref).abs() - bound
        assert excess.max().item() <= 0.0, (length, excess.max().item())
        plain, _, _ = case.reference(bias=False, sink=False)
        assert ((case.out.double() - plain).abs() > bound).any()


@pytest.mark.parametrize("target", [0, 1, EXTENT - 1])
def test_bias_peak_and_row_sink_select_exact_values(target):
    """A dominant bias returns the value at its distance bit-exactly; a
    dominant sink zeroes exactly its own rows and heads."""
    _require_sm120()
    kernel = _kernel(bias=True, sink=True)
    case = _Case(rows=320, capacity=EXTENT - 1 + 320, seed=target + 5)
    case.bias.zero_()
    case.bias[:, :, target] = 96.0
    case.sink.fill_(-200.0)
    dominant = torch.zeros_like(case.sink, dtype=torch.bool)
    dominant[1::3, ::5] = True
    dominant[2::7, 3] = True
    case.sink[dominant] = 400.0
    case.run(kernel, bias=True, sink=True)
    torch.cuda.synchronize()
    position = case.capacity - case.rows + torch.arange(case.rows, device="cuda")
    slot = position - target
    pages = case.table[0, slot // PAGE].long()
    value = case.value[pages, slot % PAGE].float().repeat_interleave(GROUP, 1)
    assert torch.equal(case.out.float(), value.masked_fill(dominant[:, :, None], 0))


def test_keys_at_and_beyond_the_extent_get_no_bias():
    _require_sm120()
    kernel = _kernel(bias=True, sink=False)
    case = _Case(rows=320, capacity=EXTENT + 320, seed=9)
    case.bias.fill_(-96.0)
    case.run(kernel, bias=True, sink=False)
    torch.cuda.synchronize()
    ref, bound, biased_mass = case.reference(bias=True, sink=False)
    assert biased_mass.max().item() < 1e-30
    assert ((case.out.double() - ref).abs() <= bound).all()
    # The first row sees exactly one unbiased key, at distance 1024.
    value = case.value[int(case.table[0, 0]), 0].float().repeat_interleave(GROUP, 0)
    assert torch.equal(case.out[0].float(), value)


def test_bias_and_sink_free_kernel_is_unchanged():
    """Zero bias and a negligible sink reproduce the plain kernel bit-exactly."""
    _require_sm120()
    case = _Case(rows=320, capacity=3000, seed=13)
    case.run(_kernel(bias=False, sink=False), bias=False, sink=False, extended=False)
    torch.cuda.synchronize()
    plain = case.out.clone()
    case.bias.zero_()
    case.sink.fill_(-float("inf"))
    case.run(_kernel(bias=True, sink=True), bias=True, sink=True)
    torch.cuda.synchronize()
    assert torch.equal(case.out, plain)
