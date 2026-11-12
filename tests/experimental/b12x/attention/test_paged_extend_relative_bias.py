"""Paged extend (prefill) with a relative-distance bias and a per-row sink.

One request extends its paged FP8 K/V history by a chunk of query rows at 12
query heads over 2 KV heads.  The reference and its error bound are those of
``test_paged_decode_relative_bias``: FP32 scores and softmax statistics, one
BF16 rounding of each probability and one of the output.
"""

from __future__ import annotations

import pytest
import torch
from b12x.preparation import PreparationSession, PreparedCall

from b12x.attention import paged

from ..conftest import require_b12x
from . import test_paged_decode_relative_bias as shared
from .test_paged_decode_relative_bias import DIM, EXTENT, HKV, HQ, PAGE

ROWS = 320
# name: (window_left, longest context, per-row sink)
LAYERS = {
    "global": (-1, 4096, False),
    "global_sink": (-1, 4096, True),
    # 1023 previous keys plus the chunk, as a linearized sliding window.
    "window": (1023, 1023 + ROWS, False),
    "window_sink": (1023, 1023 + ROWS, True),
}
_LIVE = (
    "q",
    "cache_seqlens",
    "page_table",
    "relative_attention_bias",
    "attention_sink_bias",
)


def _layer(name: str, seed: int) -> shared._Layer:
    window, max_seq, row_sink = LAYERS[name]
    shared.CASES["extend:" + name] = (window, 1, ROWS, max_seq, 1.0, 1.0, row_sink)
    try:
        return shared._Layer("extend:" + name, seed)
    finally:
        del shared.CASES["extend:" + name]


def _prepare(layer, session, inputs):
    arguments = dict(inputs, output=torch.empty_like(inputs["q"]))
    geometry = dict(
        num_q_heads=HQ,
        num_kv_heads=HKV,
        head_dim_qk=DIM,
        head_dim_vo=DIM,
        page_size=PAGE,
    )
    capacity = paged.extend_graph_capacity(
        device="cuda",
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        batch=1,
        max_cache_page_count=layer.width,
        window_left=layer.window,
        total_q_capacity=ROWS,
        **geometry,
    )
    caps = paged.Caps(
        device=torch.device("cuda", torch.cuda.current_device()),
        mode="extend",
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        max_total_q=ROWS,
        max_batch=1,
        max_page_table_width=layer.width,
        num_cache_pages=layer.pool,
        max_work_items=capacity.max_work_items,
        max_partial_rows=0,
        use_cuda_graph=True,
        copy_runtime_metadata=False,
        **geometry,
    )
    invocation = dict(
        paged.invocation_from_tensors(caps, **arguments), window_left=layer.window
    )
    plan = paged.plan(caps, invocation=invocation)

    def prepare_call(state):
        scratch = [
            torch.empty(s.shape, dtype=s.dtype, device="cuda")
            for s in state.scratch_plan.scratch_specs()
        ]
        binding = state.bind(scratch=scratch, **arguments, window_left=layer.window)
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))

    result = session.prepare(
        (plan.request(name=layer.name, prepare_call=prepare_call),)
    )
    scratch = [
        torch.empty(s.shape, dtype=s.dtype, device="cuda") for s in plan.scratch_specs()
    ]

    def run():
        paged.run(
            binding=paged.bind(
                plan, scratch=scratch, **arguments, window_left=layer.window
            ),
            plan=plan,
        )

    return result, run, arguments["output"]


@pytest.mark.parametrize("name", sorted(LAYERS))
def test_extend_graph_replay_matches_reference(name):
    """One captured graph serves new context lengths, pages, queries, bias and
    sink values, including contexts that end at the bias extent."""
    require_b12x()
    layer = _layer(name, seed=21)
    lengths = [layer.max_seq, ROWS, EXTENT, EXTENT + 1, ROWS + EXTENT - 1]
    lengths = [min(max(length, ROWS), layer.max_seq) for length in lengths]
    draws = [layer.draw(lengths=[length]) for length in lengths]
    live = {
        key: value.clone() if key in _LIVE else value for key, value in draws[0].items()
    }
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, run, out = _prepare(layer, session, live)
        run()
        torch.cuda.synchronize()
        shared._check(layer, out, live)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for draw in draws:
            for key in _LIVE:
                if key in live:
                    live[key].copy_(draw[key])
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            shared._check(layer, out, live)
        result.close()


@pytest.mark.parametrize("name", ["global_sink", "window_sink"])
@pytest.mark.parametrize("target", [0, EXTENT - 1])
def test_extend_bias_peak_and_row_sink_select_exact_values(name, target):
    """A dominant bias returns the value at its distance bit-exactly; a
    dominant sink zeroes exactly its own rows and heads."""
    require_b12x()
    layer = _layer(name, seed=target + 31)
    inputs = layer.draw(lengths=[layer.max_seq], bias_std=0.0)
    inputs["relative_attention_bias"][:, :, target] = 96.0
    sink = inputs["attention_sink_bias"]
    sink.fill_(-200.0)
    dominant = torch.zeros_like(sink, dtype=torch.bool)
    dominant[1::3, ::5] = True
    dominant[2::7, 3] = True
    sink[dominant] = 400.0
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, run, out = _prepare(layer, session, inputs)
        run()
        torch.cuda.synchronize()
        result.close()
    position = layer.max_seq - ROWS + torch.arange(ROWS, device="cuda") - target
    pages = inputs["page_table"][0, position // PAGE].long()
    value = layer.v_cache[pages, position % PAGE].float()
    expected = value.repeat_interleave(HQ // HKV, 1).masked_fill(
        dominant[:, :, None], 0
    )
    assert torch.equal(out.float(), expected)
