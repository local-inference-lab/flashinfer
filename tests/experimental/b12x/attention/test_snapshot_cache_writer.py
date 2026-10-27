"""Snapshot-and-append records preserve history, padding, and slot masking."""

import pytest
import torch

from b12x.attention import compressed_sparse_mla as mla
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


@pytest.mark.parametrize("requests", [1, 3])
@pytest.mark.parametrize("width", [7, 8])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_snapshot_writer_matches_separate_byte_copy_and_prepared_encoder(
    requests, width, slot_dtype
):
    require_b12x()
    device = torch.device("cuda:0")
    page = 128 + width
    prefix_bytes, payload_bytes = 128 * 528, page * 528
    backing = torch.randint(
        0, 256, (7, prefix_bytes + 512), dtype=torch.uint8, device=device
    )
    history = backing[:, 256 : 256 + prefix_bytes]
    history_rows = torch.arange(requests, dtype=torch.int64, device=device)
    storage = torch.full(
        (requests, payload_bytes + 32), 173, dtype=torch.uint8, device=device
    )
    cache = storage[:, :payload_bytes]
    reference = torch.empty_like(storage)
    reference_cache = reference[:, :payload_bytes]
    source = (
        torch.randn((requests * width, 512), dtype=torch.bfloat16, device=device) * 0.25
    )
    slots = torch.tensor(
        [r * page + 128 + j for r in range(requests) for j in range(width)],
        dtype=slot_dtype,
        device=device,
    )
    plans = [
        mla.plan_cache_writer(
            mla.CacheWriterQuery(
                max_rows=source.shape[0],
                page_size=page,
                cache_kind="swa",
                slot_dtype=str(slot_dtype).removeprefix("torch."),
                snapshot_tokens=128 if snapshot else 0,
                max_requests=requests if snapshot else 0,
            ),
            device=device,
        )
        for snapshot in (False, True)
    ]
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare(
            tuple(
                plan.request(
                    name=f"writer.{index}",
                    prepare_call=lambda state, index=index: PreparedCall(
                        run=lambda: state.run(
                            source,
                            (reference_cache, cache)[index],
                            slots,
                            **(
                                dict(history=history, history_rows=history_rows)
                                if index
                                else {}
                            ),
                        )
                    ),
                )
                for index, plan in enumerate(plans)
            )
        )

        def forward():
            mla.write_cache(
                source,
                cache,
                slots,
                plan=plans[1],
                history=history,
                history_rows=history_rows,
            )

        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        try:
            for step in range(8):
                backing.random_(0, 256)
                source.normal_(std=0.25)
                source[0].zero_()
                source[-1].fill_(2**-20)
                selected = [(step + r) % 7 for r in range(requests)]
                if step == 7:
                    selected[0] = 2**32 + 1
                history_rows.copy_(torch.tensor(selected, device=device))
                mapping = [
                    r * page + 128 + j for r in range(requests) for j in range(width)
                ]
                if step % 2:
                    mapping[:5] = [
                        0,
                        127,
                        -1,
                        requests * page,
                        2**40 + 128 if slot_dtype == torch.int64 else 2**30 + 128,
                    ]
                slots.copy_(torch.tensor(mapping, dtype=slot_dtype, device=device))
                storage.fill_(173)
                reference.fill_(173)
                original = backing.clone()
                graph.replay()
                torch.cuda.synchronize()
                for request, selected_row in enumerate(selected):
                    if 0 <= selected_row < 7:
                        reference_cache[request, :prefix_bytes].copy_(
                            history[selected_row]
                        )
                filtered = torch.where((slots >= 0) & (slots % page >= 128), slots, -1)
                mla.write_cache(source, reference_cache, filtered, plan=plans[0])
                torch.cuda.synchronize()
                assert torch.equal(storage, reference)
                assert torch.equal(backing, original)
        finally:
            graph.reset()


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_prepared_glm_next_writer_retains_record_and_padding_arithmetic(
    monkeypatch, slot_dtype
):
    from b12x.attention import sparse_mla
    from . import test_glm_next_mla as checks

    write = sparse_mla.concat_and_cache_glm_next_mla

    def prepared(source, cache, slots):
        plan = sparse_mla.plan_cache_writer(source, cache, slots)
        with PreparationSession(
            device=source.device, autotune=False, compile_workers=2
        ) as session:
            session.prepare(
                (
                    plan.request(
                        name="glm-writer",
                        prepare_call=lambda state: PreparedCall(
                            run=lambda: state.run(source, cache, slots)
                        ),
                    ),
                )
            )
            write(source, cache, slots, plan=plan)

    monkeypatch.setattr(sparse_mla, "concat_and_cache_glm_next_mla", prepared)
    checks.test_glm_next_cache_writer_preserves_padded_tail_and_record_abi(slot_dtype)
