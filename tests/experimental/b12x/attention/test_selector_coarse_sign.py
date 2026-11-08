"""Coarse radix selection preserves FP32 sign ordering before exact refinement."""

import pytest
import torch

from ..conftest import require_b12x


_TOPK = 2048
_SPECIAL = (
    0xFFFFFFFF,
    0xFFC12345,
    0xFF800001,
    0xFF800000,
    0xBF800000,
    0x80000000,
    0x00000000,
    0x3F800000,
    0x7F800000,
    0x7F800001,
    0x7FC12345,
    0x7FFFFFFF,
)


def _key(word):
    word &= 0xFFFFFFFF
    return (~word & 0xFFFFFFFF) if word & 0x80000000 else word | 0x80000000


def _rows(width, *, underfull):
    words, lengths, labels = [], [], []

    def add(label, row, length=width):
        words.append(row)
        lengths.append(length)
        labels.append(label)

    if width == 4096:
        # These threshold buckets contain fewer than 8192 candidates, so they
        # exercise buffered refinement rather than the full-key overflow scan.
        for name, word, position in (
            ("negative_quiet_nan", 0xFFC12345, 0),
            ("negative_signaling_nan", 0xFF812345, width - 1),
        ):
            row = [0] * width
            row[position] = word
            add(name, row)
        patterns = (
            (
                "negative_nan_payloads",
                (0xFF800001, 0xFF812345, 0xFFC12345, 0xFFFFFFFF, 0xFF800000),
            ),
            (
                "positive_nan_payloads",
                (0x7F800001, 0x7F812345, 0x7FC12345, 0x7FFFFFFF, 0x7F800000),
            ),
            ("mixed_signs_and_payloads", _SPECIAL),
            ("signed_zeros", (0x80000000, 0)),
            ("infinities", (0xFF800000, 0xBF800000, 0x3F800000, 0x7F800000)),
        )
        for label, pattern in patterns:
            add(label, [pattern[i % len(pattern)] for i in range(width)])
    else:
        assert width == 32768
        # Both equal scores and nearby finite FP32 values overflow the coarse
        # bucket's 8192-position buffer, covering the existing exact rescan.
        add("finite_equal_overflow", [0x3F000000] * width)
        add("finite_distinct_overflow", [0x3F000000 + i for i in range(width)])
        add(
            "finite_mixed_signs",
            [(0xBF000000 if i % 2 else 0x3F000000) + i for i in range(width)],
        )
    if underfull:
        for length in (0, 17, _TOPK):
            add(
                f"copied_prefix_{length}",
                [_SPECIAL[i % len(_SPECIAL)] for i in range(width)],
                length,
            )
    bits = torch.tensor(words, dtype=torch.int64).to(torch.int32)
    return bits.view(torch.float32), torch.tensor(lengths, dtype=torch.int32), labels


def _check_bits(scores, lengths, values, indices, labels):
    source = scores.view(torch.int32).cpu()
    observed = values.view(torch.int32).cpu()
    selected = indices.cpu()
    for row, length in enumerate(lengths.tolist()):
        count = min(length, _TOPK)
        positions = selected[row, :count].tolist()
        assert len(set(positions)) == count, labels[row]
        assert all(0 <= position < length for position in positions), labels[row]
        assert observed[row, :count].tolist() == source[row, positions].tolist(), (
            labels[row]
        )
        expected_keys = sorted(
            (_key(v) for v in source[row, :length].tolist()), reverse=True
        )[:count]
        actual_keys = sorted(
            (_key(v) for v in observed[row, :count].tolist()), reverse=True
        )
        assert actual_keys == expected_keys, labels[row]
        if length <= _TOPK:
            assert positions == list(range(length)), labels[row]
        assert (selected[row, count:] == -1).all(), labels[row]
        assert (observed[row, count:] == -8388608).all(), labels[row]  # FP32 -inf bits.


def test_integer_oracle_orders_nan_sign_payload_zero_and_infinity():
    assert sorted(_SPECIAL, key=_key) == list(_SPECIAL)
    assert _key(0x80000000) < _key(0) < _key(0x7F800000) < _key(0x7F800001)


def test_negative_nan_regression_does_not_require_overflow():
    scores, lengths, labels = _rows(4096, underfull=False)
    assert 4096 < 8192
    for label in ("negative_quiet_nan", "negative_signaling_nan"):
        row = labels.index(label)
        keys = [_key(v) for v in scores[row].view(torch.int32).tolist()]
        assert lengths[row] == 4096
        assert sum(key == _key(0) for key in keys) == 4095
        assert sorted(keys, reverse=True)[_TOPK - 1] == _key(0)


def _fused_launcher(scores, ids, values, indices):
    """Call the production fused selector without its preceding dot-product scorer."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import from_dlpack

    from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
    from b12x._lib.intrinsics.memory import shared_ptr_to_u32
    from b12x._lib.utils.device import current_cuda_stream
    from b12x.attention.dsa_indexer.fused_indexer import _fused_radix_select

    rows, width = scores.shape
    candidates = 8192

    @cute.struct
    class Storage:
        hist0: cute.struct.MemRange[Int32, 257]
        hist1: cute.struct.MemRange[Int32, 256]
        output: cute.struct.MemRange[Int32, _TOPK]
        candidate0: cute.struct.MemRange[Int32, candidates]
        candidate1: cute.struct.MemRange[Int32, candidates]
        counters: cute.struct.MemRange[Int32, 5]

    class FusedSelection:
        @cute.jit
        def __call__(
            self,
            scores: cute.Tensor,
            ids: cute.Tensor,
            values: cute.Tensor,
            indices: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.kernel(scores, ids, values, indices).launch(
                grid=(rows, 1, 1), block=(1024, 1, 1), stream=stream
            )

        @cute.kernel
        def kernel(
            self,
            scores: cute.Tensor,
            ids: cute.Tensor,
            values: cute.Tensor,
            indices: cute.Tensor,
        ):
            row, _, _ = cute.arch.block_idx()
            thread, _, _ = cute.arch.thread_idx()
            storage = cutlass.utils.SmemAllocator().allocate(Storage)
            h0 = storage.hist0.get_tensor(cute.make_layout((257,), stride=(1,)))
            h1 = storage.hist1.get_tensor(cute.make_layout((256,), stride=(1,)))
            output = storage.output.get_tensor(cute.make_layout((_TOPK,), stride=(1,)))
            c0 = storage.candidate0.get_tensor(
                cute.make_layout((candidates,), stride=(1,))
            )
            c1 = storage.candidate1.get_tensor(
                cute.make_layout((candidates,), stride=(1,))
            )
            counter = shared_ptr_to_u32(storage.counters.data_ptr())
            source_layout = cute.make_layout((width,), stride=(1,))
            output_layout = cute.make_layout((_TOPK,), stride=(1,))
            _fused_radix_select(
                cute.make_tensor(scores.iterator + Int64(row) * width, source_layout),
                cute.make_tensor(ids.iterator + Int64(row) * width, source_layout),
                cute.make_tensor(values.iterator + Int64(row) * _TOPK, output_layout),
                cute.make_tensor(indices.iterator + Int64(row) * _TOPK, output_layout),
                Int32(width),
                Int32(_TOPK),
                candidates,
                Int32(thread),
                h0,
                h1,
                output,
                c0,
                c1,
                shared_ptr_to_u32(storage.hist0.data_ptr()),
                counter,
                counter + 4,
                counter + 8,
                counter + 12,
                counter + 16,
            )

    args = tuple(
        from_dlpack(value, assumed_align=4) for value in (scores, ids, values, indices)
    )
    compiled = b12x_compile(
        FusedSelection(),
        *args,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "test.dsa.coarse_sign.fused_selection", 1, (rows, width, _TOPK, candidates)
        ),
    )
    return lambda: compiled(*args, current_cuda_stream())


@pytest.mark.parametrize("width", (4096, 32768))
@pytest.mark.parametrize("selector", ("row", "fused"))
def test_selection_key_order_and_score_bits_under_graph_replay(width, selector):
    device = require_b12x()
    device = torch.device(device.type, torch.cuda.current_device())
    from b12x.attention import row_topk
    from b12x.preparation import PreparationSession, PreparedCall

    source, host_lengths, labels = _rows(width, underfull=selector == "row")
    scores = source.to(device)
    lengths = host_lengths.to(device)
    ids = (
        torch.arange(width, dtype=torch.int32, device=device)
        .expand(scores.shape)
        .clone()
    )
    values = torch.empty((scores.shape[0], _TOPK), dtype=torch.float32, device=device)
    indices = torch.empty_like(values, dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        if selector == "row":
            plan = row_topk.plan(
                row_topk.RowTopKQuery(rows=scores.shape[0], width=width, topk=_TOPK),
                device=device,
            )
            session.prepare(
                (
                    plan.request(
                        name="coarse_sign",
                        prepare_call=lambda state: PreparedCall(
                            run=lambda: state.run(
                                scores, lengths, ids, values, indices
                            ),
                            owners=(scores, lengths, ids, values, indices),
                        ),
                    ),
                )
            )
            run = lambda: row_topk.topk(plan, scores, lengths, ids, values, indices)
        else:
            run = _fused_launcher(scores, ids, values, indices)
        run()
        torch.cuda.synchronize(device)
        _check_bits(source, host_lengths, values, indices, labels)
        with session.capture(), torch.cuda.graph(graph):
            run()
        session.freeze()
        try:
            for shift in (1, 513, 0):
                changed = (
                    source.view(torch.int32).roll(shift, dims=1).view(torch.float32)
                )
                scores.view(torch.int32).copy_(changed.view(torch.int32))
                values.view(torch.int32).fill_(0x7FC12345)
                indices.fill_(-91)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize(device)
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == before
                )
                _check_bits(changed, host_lengths, values, indices, labels)
                assert torch.equal(
                    scores.view(torch.int32).cpu(), changed.view(torch.int32)
                )
        finally:
            graph.reset()
