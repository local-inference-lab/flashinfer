"""GPU parity tests for request-grouped multi-query MXFP4 and Layer-20 indexers."""

from __future__ import annotations

import pytest
import torch

from b12x.attention import dsa_indexer as api
from b12x.preparation import PreparationSession, PreparedCall

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)),
    reason="multi-query MXFP4 indexer requires SM120 or SM121",
)

_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def _decode_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode packed E2M1 nibbles + UE8M0 scales to exact float64 values."""
    packed_cpu = packed.cpu()
    scales_cpu = scales.cpu()
    codes = torch.stack((packed_cpu & 15, packed_cpu >> 4), dim=-1).flatten(-2)
    mag = _E2M1_MAG[(codes & 7).long()]
    sign = torch.where((codes & 8) == 0, 1.0, -1.0)
    exp = (scales_cpu.to(torch.int32) - 127).repeat_interleave(32, dim=-1)
    return torch.ldexp(mag * sign, exp)


def _reference_scores(
    q_mxfp4: torch.Tensor,
    q_scales: torch.Tensor,
    query_weights: torch.Tensor,
    pool: torch.Tensor,
    page_table: torch.Tensor,
    cache_lengths: torch.Tensor,
    active_width: int,
    page_size: int,
    *,
    width: int,
    row_to_req: torch.Tensor | None = None,
) -> torch.Tensor:
    """Independent CPU reference matching MXFP4 m16n8k64 + BF16 rounding boundaries."""
    rows, heads, _ = q_mxfp4.shape
    pool_cpu = pool.cpu()
    pool_pages = pool_cpu.shape[0]
    k_packed = pool_cpu[:, : page_size * 64].view(pool_pages, page_size, 64)
    k_scales = pool_cpu[:, page_size * 64 : page_size * 68].view(
        pool_pages, page_size, 4
    )
    q_f = _decode_mxfp4(q_mxfp4, q_scales)  # [rows, heads, 128] float64
    k_f = _decode_mxfp4(k_packed, k_scales)  # [pool_pages, page_size, 128] float64
    w_bf = query_weights.cpu().to(torch.bfloat16)
    pt = page_table.cpu()
    lens = cache_lengths.cpu()
    req_map = (
        torch.arange(rows, dtype=torch.int64)
        if row_to_req is None
        else row_to_req.cpu().long()
    )
    if pt.shape[0] == 1:
        req_map = torch.zeros(rows, dtype=torch.int64)

    out = torch.full((rows, width), float("-inf"), dtype=torch.float32)
    for r in range(rows):
        vis = min(max(int(lens[r].item()), 0), max(int(active_width), 0))
        ext = min(vis, width)
        if ext <= 0:
            continue
        prow = int(req_map[r].item())
        pos = torch.arange(ext, dtype=torch.int64)
        pcol = pos // page_size
        tok = pos % page_size
        in_pt = pcol < pt.shape[1]
        pid = torch.full((ext,), -1, dtype=torch.int64)
        pid[in_pt] = pt[prow, pcol[in_pt]].long()
        valid = in_pt & (pid >= 0) & (pid < pool_pages)
        if not valid.any():
            continue
        vpos = pos[valid]
        kv = k_f[pid[valid], tok[valid]]  # [nv, 128] float64
        # Exact dot per head in float64 (E2M1 * E2M1 sum over 128 is exact in f32/f64).
        dots = (kv @ q_f[r].T).to(torch.float32)  # [nv, heads]
        dots_bf = torch.clamp_min(dots.to(torch.bfloat16), 0.0)
        prods_bf = (dots_bf * w_bf[r].unsqueeze(0)).to(torch.float32)
        # Preserve exact left-to-right FP32 head accumulation order.
        total = torch.zeros(vpos.numel(), dtype=torch.float32)
        for h in range(heads):
            total = total + prods_bf[:, h]
        out[r, vpos] = total.to(torch.bfloat16).to(torch.float32)
    return out


def _assert_topk512_exact(
    got_indices: torch.Tensor,
    got_scores: torch.Tensor,
    ref_scores: torch.Tensor,
) -> None:
    """Assert 0-ULP bit-exact top-512 indices and scores (handling cutoff ties)."""
    gi = got_indices.cpu()
    gv = got_scores.cpu()
    rows, width = ref_scores.shape
    for r in range(rows):
        row_ref = ref_scores[r]
        valid_mask = torch.isfinite(row_ref) | (row_ref > 0)
        valid_pos = torch.nonzero(valid_mask, as_tuple=False).flatten()
        expected_n = min(512, int(valid_pos.numel()))

        sel_i = gi[r]
        sel_v = gv[r]
        got_valid = sel_i >= 0
        assert int(got_valid.sum().item()) == expected_n, (
            f"row {r}: valid count {int(got_valid.sum())} != {expected_n}"
        )
        # Valid entries must be contiguous prefix sorted strictly ascending.
        assert torch.all(got_valid[:expected_n]), f"row {r}: non-contiguous prefix"
        assert torch.all(sel_i[expected_n:] == -1), f"row {r}: tail indices != -1"
        assert torch.all(
            sel_v[expected_n:].view(torch.int32)
            == torch.tensor(float("-inf"), dtype=torch.float32).view(torch.int32)
        ), f"row {r}: tail scores != -inf"
        if expected_n == 0:
            continue
        pos = sel_i[:expected_n]
        vals = sel_v[:expected_n]
        if expected_n > 1:
            assert torch.all(pos[1:] > pos[:-1]), (
                f"row {r}: indices not strictly sorted"
            )
        assert torch.all((pos >= 0) & (pos < width))
        ref_at_pos = row_ref[pos.long()]
        # 0-ULP exact IEEE-754 bit match for every selected score.
        assert torch.equal(vals.view(torch.int32), ref_at_pos.view(torch.int32)), (
            f"row {r}: score bit mismatch"
        )
        # Verify exact top-512 optimality with respect to the 512th score cutoff.
        valid_scores = row_ref[valid_pos]
        sorted_desc = torch.sort(valid_scores, descending=True).values
        cutoff = sorted_desc[expected_n - 1]
        assert torch.all(vals >= cutoff), f"row {r}: selected score below cutoff"
        strictly_above = set(valid_pos[valid_scores > cutoff].tolist())
        chosen = set(pos.tolist())
        assert strictly_above.issubset(chosen), (
            f"row {r}: missing strictly-above-cutoff indices {strictly_above - chosen}"
        )


def _assert_hierarchical_exact(
    got_cand: torch.Tensor,
    got_cand_len: torch.Tensor,
    ref_scores: torch.Tensor,
    cache_lengths: torch.Tensor,
    active_width: int,
) -> None:
    """Assert exact top-2048 8-token block selection, newest priority, and 8x expansion."""
    gc = got_cand.cpu()
    gl = got_cand_len.cpu()
    lens = cache_lengths.cpu()
    rows, width = ref_scores.shape
    for r in range(rows):
        vis = min(max(int(lens[r].item()), 0), max(int(active_width), 0))
        ext = min(vis, width)
        num_blocks = (ext + 7) // 8
        newest = (vis - 1) // 8 if vis > 0 else -1
        row_c = gc[r]
        got_len = int(gl[r].item())
        if num_blocks == 0:
            assert got_len == 0
            assert torch.all(row_c == -1)
            continue

        row_ref = ref_scores[r, :ext]
        pad = num_blocks * 8 - ext
        if pad:
            row_ref = torch.cat(
                [row_ref, torch.full((pad,), float("-inf"), dtype=torch.float32)]
            )
        blk_max = row_ref.view(num_blocks, 8).max(dim=1).values
        if 0 <= newest < num_blocks:
            blk_max[newest] = float("inf")
        valid_blks = torch.nonzero(
            torch.isfinite(blk_max) | (blk_max > 0), as_tuple=False
        ).flatten()
        expected_nb = min(2048, int(valid_blks.numel()))

        # Reconstruct selected blocks from 8-token expanded output.
        sel_blocks = []
        expected_tokens = 0
        for slot in range(2048):
            chunk = row_c[slot * 8 : (slot + 1) * 8]
            if int(chunk[0].item()) == -1:
                assert torch.all(row_c[slot * 8 :] == -1), (
                    f"row {r}: non-contiguous candidate tail at slot {slot}"
                )
                break
            b = int(chunk[0].item()) // 8
            sel_blocks.append(b)
            n_tok = min(8, vis - b * 8)
            assert n_tok > 0
            expected_chunk = torch.full((8,), -1, dtype=torch.int32)
            expected_chunk[:n_tok] = torch.arange(
                b * 8, b * 8 + n_tok, dtype=torch.int32
            )
            assert torch.equal(chunk, expected_chunk), (
                f"row {r} block {b}: expansion mismatch"
            )
            expected_tokens += n_tok
        assert len(sel_blocks) == expected_nb, (
            f"row {r}: selected blocks {len(sel_blocks)} != {expected_nb}"
        )
        assert got_len == expected_tokens, (
            f"row {r}: candidate_output_lengths {got_len} != {expected_tokens}"
        )
        if expected_nb > 1:
            assert all(
                sel_blocks[i + 1] > sel_blocks[i] for i in range(expected_nb - 1)
            ), f"row {r}: blocks not strictly ascending"
        if 0 <= newest < num_blocks:
            assert newest in sel_blocks, (
                f"row {r}: mandatory newest block {newest} missing"
            )
        if expected_nb > 0:
            blk_scores = blk_max[valid_blks]
            cutoff = torch.sort(blk_scores, descending=True).values[expected_nb - 1]
            sel_t = torch.tensor(sel_blocks, dtype=torch.long)
            assert torch.all(blk_max[sel_t] >= cutoff), (
                f"row {r}: selected block below cutoff"
            )
            strictly_above = set(valid_blks[blk_scores > cutoff].tolist())
            assert strictly_above.issubset(set(sel_blocks)), (
                f"row {r}: missing strictly-above-cutoff blocks"
            )


def _make_mxfp4_inputs(
    *,
    rows: int,
    heads: int,
    width: int,
    page_size: int,
    pool_pages: int,
    seed: int,
    device: torch.device,
):
    gen = torch.Generator(device=device).manual_seed(seed)
    q_mxfp4 = torch.randint(
        0, 256, (rows, heads, 64), dtype=torch.uint8, device=device, generator=gen
    )
    q_scales = torch.randint(
        122, 132, (rows, heads, 4), dtype=torch.uint8, device=device, generator=gen
    )
    query_weights = (
        torch.randn((rows, heads), dtype=torch.float32, device=device, generator=gen)
        / max(1, heads // 4)
    ).bfloat16()
    page_bytes = api.index_mxfp4_page_bytes(page_size)
    pool = torch.randint(
        0,
        256,
        (pool_pages, page_bytes),
        dtype=torch.uint8,
        device=device,
        generator=gen,
    )
    pool[:, page_size * 64 : page_size * 68] = torch.randint(
        122,
        132,
        (pool_pages, page_size * 4),
        dtype=torch.uint8,
        device=device,
        generator=gen,
    )
    return q_mxfp4, q_scales, query_weights, pool


def _prepare_and_run(
    caps: api.Caps,
    arguments: dict,
    *,
    request_ids: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    poison_scratch: bool = True,
    override=None,
):
    device = caps.device
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        plan = api.plan(
            caps,
            invocation=api.invocation_from_tensors(caps, **arguments),
            override=override,
        )
        retained = []

        def prepare(state):
            (spec,) = state.layout.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            if poison_scratch:
                scratch.fill_(0xA5)
            binding = state.bind(
                scratch=scratch,
                request_ids=request_ids,
                query_start_loc=query_start_loc,
                **arguments,
            )
            retained.append((state, binding, scratch))
            return PreparedCall(
                run=lambda: state.run(binding), owners=(binding, scratch)
            )

        session.prepare((plan.request(name="mq_test", prepare_call=prepare),))
        state, binding, scratch = retained[0]
        session.freeze()
        state.run(binding)
        torch.cuda.synchronize()
        return state, binding, scratch


@pytest.mark.parametrize("page_size", [8, 24, 64, 128])
@pytest.mark.parametrize("heads", [1, 4, 32, 64])
def test_multiquery_dense_bit_exact_page_sizes_and_heads(page_size: int, heads: int):
    device = torch.device("cuda:0")
    rows = 16
    page_width = max(12, (1536 + page_size - 1) // page_size)
    width = page_width * page_size
    pool_pages = page_width * 4
    q_mxfp4, q_scales, query_weights, pool = _make_mxfp4_inputs(
        rows=rows,
        heads=heads,
        width=width,
        page_size=page_size,
        pool_pages=pool_pages,
        seed=100 + page_size * 7 + heads,
        device=device,
    )
    # Mix grouped, straddling, and independent request page tables + invalid pages.
    gen = torch.Generator(device=device).manual_seed(2026)
    req_pages = torch.randint(
        0, pool_pages, (4, page_width), dtype=torch.int32, device=device, generator=gen
    )
    req_pages[1, 0] = -1
    req_pages[2, 2] = pool_pages
    req_pages[3, 1] = 2**31 - 1
    # Straddle request boundaries inside q_tile=4 groups: [3, 5, 2, 6].
    row_to_req = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3],
        dtype=torch.int32,
        device=device,
    )
    page_table = req_pages[row_to_req.long()].contiguous()
    cache_lengths = torch.tensor(
        [
            0,
            1,
            7,
            8,
            9,
            15,
            16,
            17,
            511,
            512,
            513,
            600,
            777,
            width - 3,
            width,
            width + 50,
        ],
        dtype=torch.int32,
        device=device,
    )
    active_width = torch.tensor([width - 5], dtype=torch.int32, device=device)
    out_i = torch.full((rows, 512), -7, dtype=torch.int32, device=device)
    out_v = torch.full((rows, 512), 123.0, dtype=torch.float32, device=device)

    caps = api.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=page_width,
        topk=512,
        page_size=page_size,
        cache_format="mxfp4",
        mode="decode",
    )
    args = dict(
        q_mxfp4=q_mxfp4,
        q_scales=q_scales,
        query_weights=query_weights,
        index_k_cache=pool,
        page_table=page_table,
        cache_lengths=cache_lengths,
        active_width=active_width,
        output_indices=out_i,
        output_scores=out_v,
    )
    state, binding, _ = _prepare_and_run(caps, args)
    assert state.multiquery_dense is not None
    assert int(binding.runtime.scratch["merge_state"].count_nonzero().item()) == 0
    ref = _reference_scores(
        q_mxfp4,
        q_scales,
        query_weights,
        pool,
        page_table,
        cache_lengths,
        int(active_width.item()),
        page_size,
        width=width,
    )
    _assert_topk512_exact(out_i, out_v, ref)


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("mode", ["request_ids", "query_start_loc"])
def test_multiquery_explicit_request_mapping(hierarchical: bool, mode: str):
    """Explicit [B, W] page_table with request_ids or query_start_loc (variable q_len)."""
    device = torch.device("cuda:0")
    # Variable q_len per request including 1, 2, 0 (empty req), 4, 7, 8 -> total 22 rows.
    q_lens = [1, 2, 0, 4, 7, 8]
    rows = sum(q_lens)
    batch = len(q_lens)
    heads = 32
    page_size = 64
    page_width = 40  # width = 2560
    width = page_width * page_size
    pool_pages = batch * page_width
    q_mxfp4, q_scales, query_weights, pool = _make_mxfp4_inputs(
        rows=rows,
        heads=heads,
        width=width,
        page_size=page_size,
        pool_pages=pool_pages,
        seed=501 + int(hierarchical) * 13,
        device=device,
    )
    req_pages = (
        torch.arange(pool_pages, dtype=torch.int32, device=device)
        .view(batch, page_width)
        .contiguous()
    )
    if mode == "request_ids":
        # Non-monotonic request_ids order to ensure general indirection works.
        row_to_req_list = []
        for req_idx, ql in zip([5, 1, 2, 0, 4, 3], q_lens, strict=True):
            row_to_req_list.extend([req_idx] * ql)
        row_to_req = torch.tensor(row_to_req_list, dtype=torch.int32, device=device)
        request_ids = row_to_req
        query_start_loc = None
    else:
        starts = [0]
        row_to_req_list = []
        for req_idx, ql in enumerate(q_lens):
            row_to_req_list.extend([req_idx] * ql)
            starts.append(starts[-1] + ql)
        row_to_req = torch.tensor(row_to_req_list, dtype=torch.int32, device=device)
        request_ids = None
        query_start_loc = torch.tensor(starts, dtype=torch.int32, device=device)

    cache_lengths = torch.linspace(
        0, width, rows, dtype=torch.int32, device=device
    ).contiguous()
    cache_lengths[1] = 1
    cache_lengths[3] = 513
    active_width = torch.tensor([width], dtype=torch.int32, device=device)
    out_i = torch.empty((rows, 512), dtype=torch.int32, device=device)
    out_v = torch.empty((rows, 512), dtype=torch.float32, device=device)
    extra = {}
    if hierarchical:
        extra["candidate_output"] = torch.empty(
            (rows, 16384), dtype=torch.int32, device=device
        )
        extra["candidate_output_lengths"] = torch.empty(
            (rows,), dtype=torch.int32, device=device
        )

    caps = api.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=page_width,
        topk=512,
        page_size=page_size,
        cache_format="mxfp4",
        mode="decode",
        candidate_topk_blocks=2048 if hierarchical else 0,
    )
    args = dict(
        q_mxfp4=q_mxfp4,
        q_scales=q_scales,
        query_weights=query_weights,
        index_k_cache=pool,
        page_table=req_pages,
        cache_lengths=cache_lengths,
        active_width=active_width,
        output_indices=out_i,
        output_scores=out_v,
        **extra,
    )
    state, _, _ = _prepare_and_run(
        caps, args, request_ids=request_ids, query_start_loc=query_start_loc
    )
    if hierarchical:
        assert state.multiquery_hierarchical is not None
    else:
        assert state.multiquery_dense is not None

    ref = _reference_scores(
        q_mxfp4,
        q_scales,
        query_weights,
        pool,
        req_pages,
        cache_lengths,
        int(active_width.item()),
        page_size,
        width=width,
        row_to_req=row_to_req,
    )
    _assert_topk512_exact(out_i, out_v, ref)
    if hierarchical:
        _assert_hierarchical_exact(
            extra["candidate_output"],
            extra["candidate_output_lengths"],
            ref,
            cache_lengths,
            int(active_width.item()),
        )


@pytest.mark.parametrize(
    ("rows", "width", "page_size"),
    [
        (8, 8192, 64),  # num_blocks=1024 <= 2048 fast compaction path
        (16, 20480, 64),  # num_blocks=2560 in (2048, 16384] SMEM radix path
        (8, 2400, 24),  # page_size % 16 != 0 path
        (8, 1280, 8),  # page_size == 8 path
        (2, 131584, 128),  # num_blocks=16448 > 16384 GMEM radix path
    ],
)
def test_multiquery_hierarchical_all_regimes_and_newest_priority(
    rows: int, width: int, page_size: int
):
    device = torch.device("cuda:0")
    heads = 32
    page_width = width // page_size
    # Reuse a compact physical page pool so even 132K tests stay tiny in VRAM.
    pool_pages = min(page_width, 64)
    q_mxfp4, q_scales, query_weights, pool = _make_mxfp4_inputs(
        rows=rows,
        heads=heads,
        width=width,
        page_size=page_size,
        pool_pages=pool_pages,
        seed=700 + width % 97 + page_size,
        device=device,
    )
    gen = torch.Generator(device=device).manual_seed(909)
    base_pt = torch.randint(
        0, pool_pages, (1, page_width), dtype=torch.int32, device=device, generator=gen
    )
    page_table = base_pt.repeat(rows, 1).contiguous()
    # Force row 0's newest block to live on an invalid page (-inf score) so
    # mandatory newest-block priority is strictly exercised even on -inf!
    cache_lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    cache_lengths[0] = width - 3
    if rows >= 2:
        cache_lengths[1] = 0
    if rows >= 4:
        cache_lengths[2] = 1
        cache_lengths[3] = 9
    if rows >= 8:
        cache_lengths[4] = 513
        cache_lengths[5] = 16385 if width > 16385 else width - 1
    newest_page_r0 = (int(cache_lengths[0].item()) - 1) // page_size
    page_table[0, newest_page_r0] = -1

    active_width = torch.tensor([width], dtype=torch.int32, device=device)
    out_i = torch.empty((rows, 512), dtype=torch.int32, device=device)
    out_v = torch.empty((rows, 512), dtype=torch.float32, device=device)
    cand_i = torch.empty((rows, 16384), dtype=torch.int32, device=device)
    cand_n = torch.empty((rows,), dtype=torch.int32, device=device)

    caps = api.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=page_width,
        topk=512,
        page_size=page_size,
        cache_format="mxfp4",
        mode="decode",
        candidate_topk_blocks=2048,
    )
    args = dict(
        q_mxfp4=q_mxfp4,
        q_scales=q_scales,
        query_weights=query_weights,
        index_k_cache=pool,
        page_table=page_table,
        cache_lengths=cache_lengths,
        active_width=active_width,
        output_indices=out_i,
        output_scores=out_v,
        candidate_output=cand_i,
        candidate_output_lengths=cand_n,
    )
    state, binding, _ = _prepare_and_run(caps, args)
    assert state.multiquery_hierarchical is not None
    assert int(binding.runtime.scratch["state"].count_nonzero().item()) == 0
    ref = _reference_scores(
        q_mxfp4,
        q_scales,
        query_weights,
        pool,
        page_table,
        cache_lengths,
        int(active_width.item()),
        page_size,
        width=width,
    )
    _assert_topk512_exact(out_i, out_v, ref)
    _assert_hierarchical_exact(
        cand_i, cand_n, ref, cache_lengths, int(active_width.item())
    )


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("pattern", ["all_equal", "all_zero", "all_negative_relu"])
def test_multiquery_duplicate_ties_and_extreme_scores(hierarchical: bool, pattern: str):
    """Exact cutoff tie handling when >2048 blocks and >512 tokens share identical scores."""
    device = torch.device("cuda:0")
    rows = 16
    heads = 32
    page_size = 64
    page_width = 288  # 18432 tokens -> 2304 blocks > 2048
    width = page_width * page_size
    pool_pages = 4
    q_mxfp4 = torch.full((rows, heads, 64), 0x22, dtype=torch.uint8, device=device)
    q_scales = torch.full((rows, heads, 4), 127, dtype=torch.uint8, device=device)
    if pattern == "all_zero":
        query_weights = torch.zeros((rows, heads), dtype=torch.bfloat16, device=device)
    elif pattern == "all_negative_relu":
        # Negative E2M1 nibbles (0xA = -1.0) against positive Q (0x2 = +1.0) -> ReLU=0.
        query_weights = torch.ones((rows, heads), dtype=torch.bfloat16, device=device)
    else:
        query_weights = torch.full(
            (rows, heads), 0.25, dtype=torch.bfloat16, device=device
        )
    page_bytes = api.index_mxfp4_page_bytes(page_size)
    pool = torch.full(
        (pool_pages, page_bytes),
        0xAA if pattern == "all_negative_relu" else 0x22,
        dtype=torch.uint8,
        device=device,
    )
    pool[:, page_size * 64 :] = 127
    # Give page 1 slightly higher values so some blocks are strictly above the tie!
    if pattern == "all_equal":
        pool[1, : page_size * 64] = 0x33
    page_table = torch.zeros((rows, page_width), dtype=torch.int32, device=device)
    page_table[:, ::17] = 1
    cache_lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    active_width = torch.tensor([width], dtype=torch.int32, device=device)
    out_i = torch.empty((rows, 512), dtype=torch.int32, device=device)
    out_v = torch.empty((rows, 512), dtype=torch.float32, device=device)
    extra = {}
    if hierarchical:
        extra["candidate_output"] = torch.empty(
            (rows, 16384), dtype=torch.int32, device=device
        )
        extra["candidate_output_lengths"] = torch.empty(
            (rows,), dtype=torch.int32, device=device
        )
    caps = api.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=page_width,
        topk=512,
        page_size=page_size,
        cache_format="mxfp4",
        mode="decode",
        candidate_topk_blocks=2048 if hierarchical else 0,
    )
    args = dict(
        q_mxfp4=q_mxfp4,
        q_scales=q_scales,
        query_weights=query_weights,
        index_k_cache=pool,
        page_table=page_table,
        cache_lengths=cache_lengths,
        active_width=active_width,
        output_indices=out_i,
        output_scores=out_v,
        **extra,
    )
    _prepare_and_run(caps, args)
    ref = _reference_scores(
        q_mxfp4,
        q_scales,
        query_weights,
        pool,
        page_table,
        cache_lengths,
        int(active_width.item()),
        page_size,
        width=width,
    )
    _assert_topk512_exact(out_i, out_v, ref)
    if hierarchical:
        _assert_hierarchical_exact(
            extra["candidate_output"],
            extra["candidate_output_lengths"],
            ref,
            cache_lengths,
            int(active_width.item()),
        )


@pytest.mark.parametrize("rows", [1, 8, 32, 128])
@pytest.mark.parametrize("hierarchical", [False, True])
def test_multiquery_scale_shapes_P1_8_32_128(rows: int, hierarchical: bool):
    """Verify P = 1, 8, 32, 128 across both dense and hierarchical prepared plans."""
    device = torch.device("cuda:0")
    heads = 32
    page_size = 64
    page_width = 32  # width = 2048
    width = page_width * page_size
    batch = max(1, (rows + 7) // 8)
    pool_pages = min(batch * page_width, 128)
    q_mxfp4, q_scales, query_weights, pool = _make_mxfp4_inputs(
        rows=rows,
        heads=heads,
        width=width,
        page_size=page_size,
        pool_pages=pool_pages,
        seed=1200 + rows + int(hierarchical),
        device=device,
    )
    gen = torch.Generator(device=device).manual_seed(42 + rows)
    req_pages = torch.randint(
        0,
        pool_pages,
        (batch, page_width),
        dtype=torch.int32,
        device=device,
        generator=gen,
    )
    if batch == 1:
        page_table = req_pages.expand(rows, -1)
    else:
        page_table = req_pages.repeat_interleave(8, dim=0)[:rows].contiguous()
    cache_lengths = torch.randint(
        0, width + 1, (rows,), dtype=torch.int32, device=device, generator=gen
    )
    active_width = torch.tensor([width], dtype=torch.int32, device=device)
    out_i = torch.empty((rows, 512), dtype=torch.int32, device=device)
    out_v = torch.empty((rows, 512), dtype=torch.float32, device=device)
    extra = {}
    if hierarchical:
        extra["candidate_output"] = torch.empty(
            (rows, 16384), dtype=torch.int32, device=device
        )
        extra["candidate_output_lengths"] = torch.empty(
            (rows,), dtype=torch.int32, device=device
        )
    caps = api.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=page_width,
        topk=512,
        page_size=page_size,
        cache_format="mxfp4",
        mode="decode",
        candidate_topk_blocks=2048 if hierarchical else 0,
    )
    args = dict(
        q_mxfp4=q_mxfp4,
        q_scales=q_scales,
        query_weights=query_weights,
        index_k_cache=pool,
        page_table=page_table,
        cache_lengths=cache_lengths,
        active_width=active_width,
        output_indices=out_i,
        output_scores=out_v,
        **extra,
    )
    _prepare_and_run(caps, args)
    ref = _reference_scores(
        q_mxfp4,
        q_scales,
        query_weights,
        pool,
        page_table,
        cache_lengths,
        int(active_width.item()),
        page_size,
        width=width,
    )
    _assert_topk512_exact(out_i, out_v, ref)
    if hierarchical:
        _assert_hierarchical_exact(
            extra["candidate_output"],
            extra["candidate_output_lengths"],
            ref,
            cache_lengths,
            int(active_width.item()),
        )


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("use_request_ids", [False, True])
def test_multiquery_cuda_graph_capture_and_live_mutation(
    hierarchical: bool, use_request_ids: bool
):
    """Capture under frozen resolution, mutate all live inputs in-place, replay bit-exact."""
    device = torch.device("cuda:0")
    rows = 16
    batch = 4
    heads = 32
    page_size = 64
    page_width = 48  # width = 3072
    width = page_width * page_size
    pool_pages = 64
    q_mxfp4, q_scales, query_weights, pool = _make_mxfp4_inputs(
        rows=rows,
        heads=heads,
        width=width,
        page_size=page_size,
        pool_pages=pool_pages,
        seed=3001 + int(hierarchical) * 5 + int(use_request_ids),
        device=device,
    )
    gen = torch.Generator(device=device).manual_seed(777)
    req_pages = torch.randint(
        0,
        pool_pages,
        (batch, page_width),
        dtype=torch.int32,
        device=device,
        generator=gen,
    )
    if use_request_ids:
        page_table = req_pages.clone()
        request_ids = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            dtype=torch.int32,
            device=device,
        )
    else:
        page_table = req_pages.repeat_interleave(4, dim=0).contiguous()
        request_ids = None
    cache_lengths = torch.randint(
        10, width + 1, (rows,), dtype=torch.int32, device=device, generator=gen
    )
    active_width = torch.tensor([width], dtype=torch.int32, device=device)
    out_i = torch.empty((rows, 512), dtype=torch.int32, device=device)
    out_v = torch.empty((rows, 512), dtype=torch.float32, device=device)
    extra = {}
    if hierarchical:
        extra["candidate_output"] = torch.empty(
            (rows, 16384), dtype=torch.int32, device=device
        )
        extra["candidate_output_lengths"] = torch.empty(
            (rows,), dtype=torch.int32, device=device
        )

    caps = api.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=page_width,
        topk=512,
        page_size=page_size,
        cache_format="mxfp4",
        mode="decode",
        candidate_topk_blocks=2048 if hierarchical else 0,
    )
    args = dict(
        q_mxfp4=q_mxfp4,
        q_scales=q_scales,
        query_weights=query_weights,
        index_k_cache=pool,
        page_table=page_table,
        cache_lengths=cache_lengths,
        active_width=active_width,
        output_indices=out_i,
        output_scores=out_v,
        **extra,
    )
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        plan = api.plan(caps, invocation=api.invocation_from_tensors(caps, **args))
        retained = []

        def prepare(state):
            (spec,) = state.layout.scratch_specs()
            scratch = torch.full(spec.shape, 0xA5, dtype=spec.dtype, device=device)
            binding = state.bind(scratch=scratch, request_ids=request_ids, **args)
            retained.append((state, binding, scratch))
            return PreparedCall(
                run=lambda: state.run(binding), owners=(binding, scratch)
            )

        session.prepare((plan.request(name="cg_mq", prepare_call=prepare),))
        state, binding, _ = retained[0]
        session.freeze()

        # Warmup and capture CUDA graph.
        state.run(binding)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state.run(binding)
        graph.replay()
        torch.cuda.synchronize()

        ref1 = _reference_scores(
            q_mxfp4,
            q_scales,
            query_weights,
            pool,
            page_table,
            cache_lengths,
            int(active_width.item()),
            page_size,
            width=width,
            row_to_req=request_ids,
        )
        _assert_topk512_exact(out_i, out_v, ref1)
        if hierarchical:
            _assert_hierarchical_exact(
                extra["candidate_output"],
                extra["candidate_output_lengths"],
                ref1,
                cache_lengths,
                int(active_width.item()),
            )

        # Mutate EVERY live input in-place and replay the captured CUDA graph.
        q2, qs2, w2, _ = _make_mxfp4_inputs(
            rows=rows,
            heads=heads,
            width=width,
            page_size=page_size,
            pool_pages=pool_pages,
            seed=9999,
            device=device,
        )
        q_mxfp4.copy_(q2)
        q_scales.copy_(qs2)
        query_weights.copy_(w2)
        page_table.copy_(
            torch.randint(
                0,
                pool_pages,
                page_table.shape,
                dtype=torch.int32,
                device=device,
                generator=gen,
            )
        )
        page_table[0, 0] = -1
        if request_ids is not None:
            request_ids.copy_(
                torch.tensor(
                    [3, 2, 1, 0, 3, 3, 2, 2, 1, 1, 0, 0, 2, 1, 3, 0],
                    dtype=torch.int32,
                    device=device,
                )
            )
        cache_lengths.copy_(
            torch.tensor(
                [
                    0,
                    1,
                    8,
                    17,
                    511,
                    512,
                    513,
                    1000,
                    1200,
                    1500,
                    1800,
                    2000,
                    2048,
                    2500,
                    2900,
                    3072,
                ],
                dtype=torch.int32,
                device=device,
            )
        )
        active_width.fill_(2400)
        graph.replay()
        torch.cuda.synchronize()

        ref2 = _reference_scores(
            q_mxfp4,
            q_scales,
            query_weights,
            pool,
            page_table,
            cache_lengths,
            int(active_width.item()),
            page_size,
            width=width,
            row_to_req=request_ids,
        )
        _assert_topk512_exact(out_i, out_v, ref2)
        if hierarchical:
            _assert_hierarchical_exact(
                extra["candidate_output"],
                extra["candidate_output_lengths"],
                ref2,
                cache_lengths,
                int(active_width.item()),
            )


def test_multiquery_validation_errors():
    device = torch.device("cuda:0")
    rows, heads, page_size, page_width = 16, 32, 64, 16
    width = page_size * page_width
    q_mxfp4, q_scales, query_weights, pool = _make_mxfp4_inputs(
        rows=rows,
        heads=heads,
        width=width,
        page_size=page_size,
        pool_pages=16,
        seed=1,
        device=device,
    )
    page_table = torch.zeros((4, page_width), dtype=torch.int32, device=device)
    cache_lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    active_width = torch.tensor([width], dtype=torch.int32, device=device)
    out_i = torch.empty((rows, 512), dtype=torch.int32, device=device)
    out_v = torch.empty((rows, 512), dtype=torch.float32, device=device)
    caps = api.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=page_width,
        topk=512,
        page_size=page_size,
        cache_format="mxfp4",
        mode="decode",
    )
    args = dict(
        q_mxfp4=q_mxfp4,
        q_scales=q_scales,
        query_weights=query_weights,
        index_k_cache=pool,
        page_table=page_table,
        cache_lengths=cache_lengths,
        active_width=active_width,
        output_indices=out_i,
        output_scores=out_v,
    )
    req_ids = torch.zeros((rows,), dtype=torch.int32, device=device)
    qsl = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="at most one"):
        _prepare_and_run(caps, args, request_ids=req_ids, query_start_loc=qsl)
    with pytest.raises(ValueError, match="query_start_loc"):
        _prepare_and_run(
            caps,
            args,
            query_start_loc=torch.tensor(
                [0, 3, 6, 9, 12, 16], dtype=torch.int32, device=device
            ),
        )
    with pytest.raises(ValueError, match="query_start_loc"):
        _prepare_and_run(
            caps,
            args,
            query_start_loc=torch.tensor([0], dtype=torch.int32, device=device),
        )
    with pytest.raises(ValueError, match="one shared row or one row per query"):
        _prepare_and_run(caps, args)


@pytest.mark.parametrize("hierarchical", [False, True])
def test_multiquery_q256_ctas32_repeated_merge_no_war_race(hierarchical: bool):
    """Repeated Q=256, ctas=32 multi-partition merge with carry trims (WAR barrier regression)."""
    from b12x.attention.dsa_indexer._mxfp4_preparation import (
        _multiquery_dense_spec,
        _multiquery_hierarchical_spec,
    )
    from b12x.attention.dsa_indexer._mxfp4_tuning import default_fused_config
    from b12x.attention.dsa_indexer._preparation import _query
    from b12x.attention.dsa_indexer._tuning import DsaIndexerConfig
    from b12x.preparation.device import detect_device

    device = torch.device("cuda:0")
    rows = 256
    batch = 4
    heads = 32
    page_size = 64
    page_width = 256  # width = 16384 (32 x 512-token partitions)
    width = page_width * page_size
    pool_pages = 64
    q_mxfp4, q_scales, query_weights, pool = _make_mxfp4_inputs(
        rows=rows,
        heads=heads,
        width=width,
        page_size=page_size,
        pool_pages=pool_pages,
        seed=4096 + int(hierarchical),
        device=device,
    )
    gen = torch.Generator(device=device).manual_seed(4099 + int(hierarchical))
    page_table = torch.randint(
        0,
        pool_pages,
        (batch, page_width),
        dtype=torch.int32,
        device=device,
        generator=gen,
    )
    request_ids = (
        torch.arange(batch, dtype=torch.int32, device=device)
        .repeat_interleave(rows // batch)
        .contiguous()
    )
    cache_lengths = torch.randint(
        12288, width + 1, (rows,), dtype=torch.int32, device=device, generator=gen
    )
    active_width = torch.tensor([width], dtype=torch.int32, device=device)
    out_i = torch.empty((rows, 512), dtype=torch.int32, device=device)
    out_v = torch.empty((rows, 512), dtype=torch.float32, device=device)
    extra = {}
    if hierarchical:
        extra["candidate_output"] = torch.empty(
            (rows, 16384), dtype=torch.int32, device=device
        )
        extra["candidate_output_lengths"] = torch.empty(
            (rows,), dtype=torch.int32, device=device
        )
    caps = api.Caps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_page_table_width=page_width,
        topk=512,
        page_size=page_size,
        cache_format="mxfp4",
        mode="decode",
        candidate_topk_blocks=2048 if hierarchical else 0,
    )
    args = dict(
        q_mxfp4=q_mxfp4,
        q_scales=q_scales,
        query_weights=query_weights,
        index_k_cache=pool,
        page_table=page_table,
        cache_lengths=cache_lengths,
        active_width=active_width,
        output_indices=out_i,
        output_scores=out_v,
        **extra,
    )
    detected = detect_device(device)
    invocation = api.invocation_from_tensors(caps, **args)
    query = _query(caps, invocation)
    override = DsaIndexerConfig(
        backend="native",
        mxfp4=default_fused_config(query, detected.identity, ctas_per_row=32),
    )
    spec = (
        _multiquery_hierarchical_spec(query, override, detected)
        if hierarchical
        else _multiquery_dense_spec(query, override, detected)
    )
    assert spec is not None and spec[1] == 32

    state, binding, scratch = _prepare_and_run(
        caps, args, request_ids=request_ids, override=override
    )
    ref = _reference_scores(
        q_mxfp4,
        q_scales,
        query_weights,
        pool,
        page_table,
        cache_lengths,
        width,
        page_size,
        width=width,
        row_to_req=request_ids,
    )
    expected_scores = ref.topk(512, dim=1).values
    lengths = cache_lengths.cpu()
    if hierarchical:
        # Width fits in 2048 eight-token blocks and every visible token is valid.
        positions = torch.arange(width, dtype=torch.int32)
        expected_candidates = torch.where(positions < lengths[:, None], positions, -1)

    for _ in range(10):
        scratch.fill_(0xA5)
        out_i.fill_(-17)
        out_v.fill_(float("nan"))
        if hierarchical:
            extra["candidate_output"].fill_(-17)
            extra["candidate_output_lengths"].fill_(-17)
        state.run(binding)
        torch.cuda.synchronize()
        indices = out_i.cpu().long()
        scores = out_v.cpu()
        assert torch.all((indices >= 0) & (indices < lengths[:, None]))
        assert torch.all(indices[:, 1:] > indices[:, :-1])
        assert torch.equal(
            scores.view(torch.int32), ref.gather(1, indices).view(torch.int32)
        )
        # Warps reserve equal-score slots with atomics before sorting by position.
        # Require the exact score multiset and unique matching positions, so only
        # the choice among equal scores at the cutoff can vary between launches.
        assert torch.equal(
            scores.sort(dim=1, descending=True).values.view(torch.int32),
            expected_scores.view(torch.int32),
        )
        if hierarchical:
            assert torch.equal(extra["candidate_output"].cpu(), expected_candidates)
            assert torch.equal(extra["candidate_output_lengths"].cpu(), lengths)
