"""paged_decode relative-distance bias and per-row sink against an FP64 reference.

The layer has 12 query heads over 2 KV heads (GQA group 6), 128-wide heads and
FP8-E4M3 K/V pages.  Query row ``t`` at position ``p`` scores the key at
position ``j`` as ``softmax_scale * q.k + bias[t, h, p - j]`` when
``0 <= p - j < extent`` and as ``softmax_scale * q.k`` otherwise.  A per-row
sink adds the logit ``sink[t, h]`` with a zero value to the row's softmax, so
the output is ``o * sigmoid(lse - sink[t, h])``.

Error bound.  The kernel accumulates q.k, the bias and the softmax statistics
in FP32, rounds each probability to BF16 for the P.V product and rounds the
output to BF16 once.  BF16 keeps 8 significant bits, so round-to-nearest
moves a value by at most ``2**-8`` of its magnitude.  With ``mass = sum_j p_j
|v_j|``, an output element differs from the exact one by at most ``2**-8 *
mass`` (probabilities) plus ``2**-8 * |out|`` (output), plus the FP32 terms,
which ``2**-11 * mass`` covers.  FP32 flushes a probability below ``2**-126``
to zero, which adds at most ``2**-126 * max|v|`` per key.
"""

from __future__ import annotations

import pytest
import torch
from b12x.preparation import PreparationSession, PreparedCall

from b12x.attention import paged_decode

from ..conftest import require_b12x

HQ, HKV, DIM, PAGE, EXTENT = 12, 2, 128, 128, 1024
GROUP = HQ // HKV

# name: (window_left, max_batch, max_q_per_req, max_seq, k_descale, v_descale,
#        per-row sink)
CASES = {
    # Full context, split-KV; unit dequant scales.
    "global_verify": (-1, 3, 8, 6000, 1.0, 1.0, False),
    "global_decode": (-1, 8, 1, 3000, 1.0, 1.0, False),
    # Non-unit dequant scales: the bias must not be scaled with K.
    "global_verify_descaled": (-1, 3, 8, 6000, 0.25, 3.0, False),
    # 1023 previous keys plus the row itself: every visible key has a bias.
    "window_verify": (1023, 4, 8, 4000, 1.0, 1.0, False),
    "window_decode": (1023, 8, 1, 3000, 1.0, 1.0, False),
    # Per-row sinks: the split merge adds them for full-context layers, the
    # forward kernel for windowed ones.
    "global_verify_sink": (-1, 3, 8, 6000, 1.0, 1.0, True),
    "global_decode_sink": (-1, 8, 1, 3000, 0.25, 3.0, True),
    "window_verify_sink": (1023, 4, 8, 4000, 1.0, 1.0, True),
    "window_decode_sink": (1023, 8, 1, 3000, 1.0, 1.0, True),
}
# Context lengths around the bias extent: a row whose farthest key sits at
# distance 1022, 1023 (last biased), 1024 (first unbiased) and far beyond.
BOUNDARY_LENGTHS = (EXTENT - 1, EXTENT, EXTENT + 1, 5, 1)


class _Layer:
    def __init__(self, name: str, seed: int):
        (
            self.window,
            self.max_batch,
            self.max_q,
            self.max_seq,
            k_descale,
            v_descale,
            self.row_sink,
        ) = CASES[name]
        self.name = name
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.width = (self.max_seq + PAGE - 1) // PAGE
        self.pool = self.max_batch * self.width + 3
        shape = (self.pool, PAGE, HKV, DIM)
        self.k_cache = self._randn(shape, 0.5).to(torch.float8_e4m3fn)
        self.v_cache = self._randn(shape, 0.5).to(torch.float8_e4m3fn)
        self.k_descale = torch.tensor([k_descale], dtype=torch.float32, device="cuda")
        self.v_descale = torch.tensor([v_descale], dtype=torch.float32, device="cuda")
        self.rows = self.max_batch * self.max_q

    def _randn(self, shape, std):
        return torch.randn(shape, generator=self.gen, device="cuda") * std

    def caps(self, *, has_relative_bias=True):
        return paged_decode.Caps(
            device="cuda",
            num_q_heads=HQ,
            num_kv_heads=HKV,
            head_dim_qk=DIM,
            head_dim_vo=DIM,
            page_size=PAGE,
            max_batch=self.max_batch,
            max_q_per_req=self.max_q,
            kv_dtype=torch.float8_e4m3fn,
            window_left=self.window,
            has_relative_bias=has_relative_bias,
            has_sinks=self.row_sink,
            sink_layout="row" if self.row_sink else "head",
        )

    def draw(self, *, lengths=None, q_lens=None, bias_std=2.0):
        """A batch at full row capacity so one captured graph serves every draw."""
        batch = self.max_batch
        if lengths is None:
            lengths = torch.randint(
                self.max_q,
                self.max_seq + 1,
                (batch,),
                generator=self.gen,
                device="cuda",
            ).tolist()
            lengths[0] = self.max_seq
        if q_lens is None:
            q_lens = [self.max_q] * batch
        q_lens = [min(q, s) for q, s in zip(q_lens, lengths, strict=True)]
        cu = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
        cu[1:] = torch.tensor(q_lens, device="cuda").cumsum(0)
        perm = torch.randperm(self.pool, generator=self.gen, device="cuda")
        q = torch.zeros((self.rows, HQ, DIM), dtype=torch.bfloat16, device="cuda")
        bias = torch.zeros((self.rows, HQ, EXTENT), dtype=torch.bfloat16, device="cuda")
        total = int(cu[-1])
        q[:total] = self._randn((total, HQ, DIM), 0.5).to(torch.bfloat16)
        bias[:total] = self._randn((total, HQ, EXTENT), bias_std).to(torch.bfloat16)
        sink = {}
        if self.row_sink:
            sink["attention_sink_bias"] = self._randn((self.rows, HQ), 3.0)
        return dict(
            **sink,
            q=q,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            page_table=perm[: batch * self.width]
            .to(torch.int32)
            .view(batch, self.width)
            .contiguous(),
            cache_seqlens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
            cu_seqlens_q=cu,
            k_descale=self.k_descale,
            v_descale=self.v_descale,
            relative_attention_bias=bias,
        )

    def reference(self, inputs, *, use_bias=True):
        """FP64 output, its error bound and each row's biased probability mass."""
        q, bias = inputs["q"].double(), inputs["relative_attention_bias"].double()
        out = torch.zeros_like(q)
        bound = torch.zeros_like(q)
        biased_mass = torch.zeros(q.shape[:2], dtype=torch.float64, device="cuda")
        cu = inputs["cu_seqlens_q"].tolist()
        for request, length in enumerate(inputs["cache_seqlens"].tolist()):
            start, end = cu[request], cu[request + 1]
            if end == start or length == 0:
                continue
            pages = inputs["page_table"][request, : (length + PAGE - 1) // PAGE].long()
            key = self.k_cache[pages].reshape(-1, HKV, DIM)[:length].double()
            value = self.v_cache[pages].reshape(-1, HKV, DIM)[:length].double()
            key = (key * float(self.k_descale)).repeat_interleave(GROUP, 1)
            value = (value * float(self.v_descale)).repeat_interleave(GROUP, 1)
            position = length - (end - start) + torch.arange(end - start, device="cuda")
            distance = position[:, None] - torch.arange(length, device="cuda")[None]
            scores = torch.einsum("thd,jhd->thj", q[start:end], key) * DIM**-0.5
            in_extent = (distance >= 0) & (distance < EXTENT)
            if use_bias:
                index = distance.clamp(0, EXTENT - 1)[:, None].expand(-1, HQ, -1)
                scores = scores + bias[start:end].gather(2, index) * in_extent[:, None]
            visible = distance >= 0
            if self.window >= 0:
                visible = visible & (distance <= self.window)
            scores = scores.masked_fill(~visible[:, None], float("-inf"))
            if self.row_sink:
                sink = inputs["attention_sink_bias"][start:end].double()
                scores = torch.cat((scores, sink[:, :, None]), -1)
            probability = scores.softmax(-1)[..., :length]
            out[start:end] = torch.einsum("thj,jhd->thd", probability, value)
            mass = torch.einsum("thj,jhd->thd", probability, value.abs())
            bound[start:end] = (
                2.0**-8 * mass
                + 2.0**-8 * out[start:end].abs()
                + 2.0**-11 * mass
                + 2.0**-126 * length * value.abs().max()
            )
            biased_mass[start:end] = (probability * in_extent[:, None]).sum(-1)
        return out, bound, biased_mass


def _scratch(plan):
    return [
        torch.empty(s.shape, dtype=s.dtype, device="cuda") for s in plan.scratch_specs()
    ]


def _prepare(layer: _Layer, session: PreparationSession, inputs):
    declaration = paged_decode.plan(layer.caps())

    def prepare_call(state):
        out = torch.empty_like(inputs["q"])
        binding = state.bind(
            plan=declaration, scratch=_scratch(declaration), output=out, **inputs
        )
        return PreparedCall(run=lambda: state.run(binding), output=out)

    result = session.prepare(
        (declaration.request(name=layer.name, prepare_call=prepare_call),)
    )
    return result, result.plans[layer.name]


def _check(layer, out, inputs):
    total = int(inputs["cu_seqlens_q"][-1])
    ref, bound, _ = layer.reference(inputs)
    excess = (out[:total].double() - ref[:total]).abs() - bound[:total]
    assert torch.isfinite(out[:total]).all()
    assert excess.max().item() <= 0.0, excess.max().item()
    # The bias must matter: the unbiased result violates the same bound.
    plain, _, _ = layer.reference(inputs, use_bias=False)
    assert ((out[:total].double() - plain[:total]).abs() > bound[:total]).any()


@pytest.mark.parametrize("name", sorted(CASES))
def test_relative_bias_graph_replay_matches_reference(name):
    """One captured graph serves new lengths, pages, queries and bias values."""
    require_b12x()
    layer = _Layer(name, seed=11)
    boundary = [
        min(max(length, 1), layer.max_seq)
        for length in (BOUNDARY_LENGTHS * layer.max_batch)[: layer.max_batch]
    ]
    draws = [
        layer.draw(),
        layer.draw(lengths=boundary),
        layer.draw(q_lens=[max(1, layer.max_q - i) for i in range(layer.max_batch)]),
    ]
    live = {
        key: value.clone() if key in _LIVE else value for key, value in draws[0].items()
    }
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, plan = _prepare(layer, session, live)
        out = torch.empty_like(live["q"])
        binding = paged_decode.bind(plan, scratch=_scratch(plan), output=out, **live)
        paged_decode.run(binding)
        torch.cuda.synchronize()
        _check(layer, out, live)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged_decode.run(binding)
        for draw in draws:
            for key in _LIVE:
                if key in live:
                    live[key].copy_(draw[key])
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            _check(layer, out, live)
        result.close()


_LIVE = (
    "q",
    "page_table",
    "cache_seqlens",
    "cu_seqlens_q",
    "relative_attention_bias",
    "attention_sink_bias",
)


@pytest.mark.parametrize("name", ["global_verify", "window_verify"])
@pytest.mark.parametrize("target", [0, 1, EXTENT - 1])
def test_relative_bias_peak_selects_the_key_at_its_distance(name, target):
    """A dominant bias at one distance returns the value stored at that distance."""
    require_b12x()
    layer = _Layer(name, seed=target + 3)
    # Request 1's first row sits at position 1023 and request 2's at 1024, so
    # the farthest biased key is the first key of the sequence or of the window.
    lengths = [layer.max_seq, EXTENT + 7, EXTENT + 8, 3000][: layer.max_batch]
    inputs = layer.draw(lengths=lengths, bias_std=0.0)
    inputs["relative_attention_bias"][:, :, target] = 96.0
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, plan = _prepare(layer, session, inputs)
        out = torch.empty_like(inputs["q"])
        paged_decode.run(
            paged_decode.bind(plan, scratch=_scratch(plan), output=out, **inputs)
        )
        torch.cuda.synchronize()
        result.close()
    cu = inputs["cu_seqlens_q"].tolist()
    for request, length in enumerate(inputs["cache_seqlens"].tolist()):
        for row in range(cu[request], cu[request + 1]):
            position = length - (cu[request + 1] - cu[request]) + row - cu[request]
            assert position >= target
            slot = position - target
            page = int(inputs["page_table"][request, slot // PAGE])
            value = layer.v_cache[page, slot % PAGE].float().repeat_interleave(GROUP, 0)
            # The peak outweighs every other key by e**(96 - |q.k| scale) and
            # FP8 values are exact in BF16, so only the output rounding remains.
            assert torch.equal(out[row].float(), value), (request, row)
    _check(layer, out, inputs)


def test_keys_at_and_beyond_the_extent_get_no_bias():
    """A bias that suppresses every distance below the extent leaves the result
    of plain attention over the keys at distance ``extent`` and beyond."""
    require_b12x()
    layer = _Layer("global_verify", seed=5)
    inputs = layer.draw(lengths=[6000, EXTENT + 8, 2 * EXTENT], bias_std=0.0)
    inputs["relative_attention_bias"].fill_(-96.0)
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, plan = _prepare(layer, session, inputs)
        out = torch.empty_like(inputs["q"])
        paged_decode.run(
            paged_decode.bind(plan, scratch=_scratch(plan), output=out, **inputs)
        )
        torch.cuda.synchronize()
        result.close()
    ref, bound, biased_mass = layer.reference(inputs)
    # Every row has at least one key at distance >= extent, which holds all
    # the probability; a kernel that biased distance 1024 would lose it.
    assert biased_mass.max().item() < 1e-30
    assert ((out.double() - ref).abs() <= bound).all()
    # Request 1's first row sees exactly one unbiased key, at distance 1024.
    row = int(inputs["cu_seqlens_q"][1])
    page = int(inputs["page_table"][1, 0])
    value = layer.v_cache[page, 0].float().repeat_interleave(GROUP, 0)
    assert torch.equal(out[row].float(), value)


@pytest.mark.parametrize("name", ["global_verify_sink", "window_verify_sink"])
def test_row_sink_is_indexed_by_row_and_head(name):
    """A dominant sink zeroes exactly its own rows and heads; a negligible one
    leaves the sink-free output."""
    require_b12x()
    layer = _Layer(name, seed=9)
    inputs = layer.draw()
    sink = inputs["attention_sink_bias"]
    sink.fill_(-200.0)
    dominant = torch.zeros_like(sink, dtype=torch.bool)
    dominant[1::3, ::5] = True
    dominant[2::7, 3] = True
    sink[dominant] = 200.0
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, plan = _prepare(layer, session, inputs)
        out = torch.empty_like(inputs["q"])
        paged_decode.run(
            paged_decode.bind(plan, scratch=_scratch(plan), output=out, **inputs)
        )
        torch.cuda.synchronize()
        result.close()
    total = int(inputs["cu_seqlens_q"][-1])
    assert (out[:total][dominant[:total]] == 0).all()
    layer.row_sink = False
    free, bound, _ = layer.reference(inputs)
    kept = ~dominant[:total]
    assert ((out[:total].double() - free[:total]).abs() <= bound[:total])[kept].all()
    assert out[:total][kept].abs().max() > 0.1


# Zero queries and a zero bias give every visible key the same probability,
# which BF16 holds exactly, and sums of at most a few thousand FP8 values are
# exact in FP32.  The output is then the mean of the visible values with one
# BF16 rounding (2**-8) and an approximate reciprocal (covered by 2**-16), so
# one key more or fewer in the window is far outside the tolerance.
UNIFORM_TOLERANCE = 2.0**-8 + 2.0**-16


def uniform_reference(layer, inputs, window):
    """FP64 mean of the values each row sees through ``window`` (-1: all)."""
    out = torch.zeros_like(inputs["q"], dtype=torch.float64)
    cu = inputs["cu_seqlens_q"].tolist()
    for request, length in enumerate(inputs["cache_seqlens"].tolist()):
        start, end = cu[request], cu[request + 1]
        pages = inputs["page_table"][request, : (length + PAGE - 1) // PAGE].long()
        value = layer.v_cache[pages].reshape(-1, HKV, DIM)[:length].double()
        value = (value * float(layer.v_descale)).repeat_interleave(GROUP, 1)
        for row in range(start, end):
            position = length - (end - start) + row - start
            first = 0 if window < 0 else max(0, position - window)
            out[row] = value[first : position + 1].mean(0)
    return out


def check_uniform_window(layer, out, inputs):
    """The output matches the layer's window and neither neighbor of it."""
    total = int(inputs["cu_seqlens_q"][-1])
    got = out[:total].double()

    def within(window):
        ref = uniform_reference(layer, inputs, window)[:total]
        return (got - ref).abs() <= UNIFORM_TOLERANCE * ref.abs() + 1e-12

    assert within(layer.window).all()
    # A neighboring window differs only for a row with that many earlier keys.
    last_position = int(inputs["cache_seqlens"].max()) - 1
    if last_position >= layer.window:
        assert not within(layer.window - 1).all()
    if last_position > layer.window:
        assert not within(layer.window + 1).all()


@pytest.mark.parametrize(
    "name, lengths",
    [
        # First rows at positions 1024, 1039, 1151 and 3992: window starts at
        # keys 1, 16 (a sub-tile edge), 128 (a page edge) and mid-sequence.
        ("window_verify", [4000, 1032, 1047, 1159]),
        ("window_verify_sink", [4000, 1032, 1047, 1159]),
        ("window_decode", [3000, 1024, 1025, 1040, 1151, 1152, 2047, 2048]),
    ],
)
def test_window_edge_with_uniform_attention(name, lengths):
    require_b12x()
    layer = _Layer(name, seed=41)
    inputs = layer.draw(lengths=lengths, bias_std=0.0)
    inputs["q"].zero_()
    if layer.row_sink:
        inputs["attention_sink_bias"].fill_(-float("inf"))
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, plan = _prepare(layer, session, inputs)
        out = torch.empty_like(inputs["q"])
        paged_decode.run(
            paged_decode.bind(plan, scratch=_scratch(plan), output=out, **inputs)
        )
        torch.cuda.synchronize()
        result.close()
    check_uniform_window(layer, out, inputs)


def nonfinite_sinks(layer, inputs):
    """Set +inf and -inf sinks on disjoint rows and heads; return their masks
    and the inputs of an equivalent finite reference."""
    sink = inputs["attention_sink_bias"]
    positive = torch.zeros_like(sink, dtype=torch.bool)
    negative = torch.zeros_like(sink, dtype=torch.bool)
    positive[::3, 1::4] = True
    negative[1::3, ::4] = True
    sink[positive] = float("inf")
    sink[negative] = -float("inf")
    finite = dict(inputs, attention_sink_bias=sink.masked_fill(positive, 1.0e4))
    return positive, finite


def check_nonfinite_sinks(layer, out, inputs, positive, finite):
    total = int(inputs["cu_seqlens_q"][-1])
    ref, bound, _ = layer.reference(finite)
    assert torch.isfinite(out[:total]).all()
    assert (out[:total][positive[:total]] == 0).all()
    assert ((out[:total].double() - ref[:total]).abs() <= bound[:total]).all()


@pytest.mark.parametrize("name", ["global_verify_sink", "window_verify_sink"])
def test_infinite_sinks_agree_between_split_and_unsplit_kernels(name):
    """A +inf sink zeroes its row and a -inf sink is no sink, in the split
    merge kernel (full context) and in the unsplit forward kernel (window)."""
    require_b12x()
    layer = _Layer(name, seed=43)
    inputs = layer.draw()
    positive, finite = nonfinite_sinks(layer, inputs)
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, plan = _prepare(layer, session, inputs)
        out = torch.empty_like(inputs["q"])
        paged_decode.run(
            paged_decode.bind(plan, scratch=_scratch(plan), output=out, **inputs)
        )
        torch.cuda.synchronize()
        result.close()
    check_nonfinite_sinks(layer, out, inputs, positive, finite)


def _bias_state(has_relative_bias):
    from b12x.attention.paged_decode._preparation import PreparedPagedDecode
    from b12x.attention.paged_decode._tuning import PagedDecodeConfig, PagedDecodeQuery

    query = PagedDecodeQuery(
        kv_dtype="float8_e4m3fn",
        num_q_heads=HQ,
        num_kv_heads=HKV,
        head_dim_qk=DIM,
        head_dim_vo=DIM,
        page_size=PAGE,
        max_batch=1,
        max_q_per_req=8,
        window_left=1023,
        causal=True,
        has_sinks=False,
        descale_layout="tensor",
        sm_count=20,
        has_relative_bias=has_relative_bias,
    )
    config = PagedDecodeConfig(
        split_kv=False,
        max_splits=1,
        head_splits=2,
        stage_rows=16,
        num_stages=9,
        min_tiles_per_split=4,
    )
    state = PreparedPagedDecode(
        query=query,
        config=config,
        device=torch.device("cpu"),
        programs={"forward": object()},
        dummy_part=(torch.empty((1, HQ, 1, DIM)), torch.empty((1, HQ, 1))),
        dummy_sinks=torch.zeros(HQ),
        dummy_descale=torch.ones(1),
    )
    inputs = dict(
        q=torch.empty((8, HQ, DIM), dtype=torch.bfloat16),
        output=torch.empty((8, HQ, DIM), dtype=torch.bfloat16),
        k_cache=torch.empty((9, PAGE, HKV, DIM), dtype=torch.float8_e4m3fn),
        v_cache=torch.empty((9, PAGE, HKV, DIM), dtype=torch.float8_e4m3fn),
        page_table=torch.zeros((1, 9), dtype=torch.int32),
        cache_seqlens=torch.tensor([1031], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
        k_descale=torch.ones(1),
        v_descale=torch.ones(1),
    )
    return state, inputs


def test_bind_takes_the_bias_only_when_the_plan_declares_it():
    state, inputs = _bias_state(True)
    bias = torch.empty((8, HQ, EXTENT), dtype=torch.bfloat16)
    binding = state.bind(plan=None, scratch=(), relative_attention_bias=bias, **inputs)
    assert binding.relative_bias is bias
    with pytest.raises(ValueError, match="relative_attention_bias must be BF16"):
        state.bind(plan=None, scratch=(), **inputs)
    plain, inputs = _bias_state(False)
    with pytest.raises(ValueError, match="declared without a relative bias"):
        plain.bind(plan=None, scratch=(), relative_attention_bias=bias, **inputs)
    assert plain.bind(plan=None, scratch=(), **inputs).relative_bias is inputs["q"]


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((7, HQ, EXTENT), torch.bfloat16),  # fewer rows than q
        ((8, HQ - 1, EXTENT), torch.bfloat16),
        ((8, HQ, EXTENT), torch.float32),
        ((8, HQ * EXTENT), torch.bfloat16),
    ],
)
def test_bind_rejects_a_bias_outside_the_contract(shape, dtype):
    state, inputs = _bias_state(True)
    with pytest.raises(ValueError, match="relative_attention_bias"):
        state.bind(
            plan=None,
            scratch=(),
            relative_attention_bias=torch.empty(shape, dtype=dtype),
            **inputs,
        )


def test_bind_takes_a_row_sink_only_in_the_row_layout():
    from dataclasses import replace

    state, inputs = _bias_state(False)
    per_head = replace(state, query=replace(state.query, has_sinks=True))
    per_row = replace(
        state, query=replace(state.query, has_sinks=True, sink_layout="row")
    )
    rows, heads = torch.zeros((8, HQ)), torch.zeros(HQ)
    assert (
        per_row.bind(plan=None, scratch=(), attention_sink_bias=rows, **inputs).sinks
        is rows
    )
    for bound, sink in ((per_row, heads), (per_row, rows[:7]), (per_head, rows)):
        with pytest.raises(ValueError, match="attention_sink_bias must be float32"):
            bound.bind(plan=None, scratch=(), attention_sink_bias=sink, **inputs)
    assert not paged_decode.supports(
        num_q_heads=HQ,
        num_kv_heads=HKV,
        head_dim_qk=DIM,
        head_dim_vo=DIM,
        page_size=PAGE,
        sink_layout="row",
    )
