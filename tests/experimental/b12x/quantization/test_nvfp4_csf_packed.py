"""The stage-readable packed CSF index rebuilds exactly what the W4A16 expansion pass writes."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import numpy as np
import pytest
import torch
from cutlass.cutlass_dsl import Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import ld_shared_u32, shared_ptr_to_u32
from b12x._lib.quant.nvfp4_csf import (
    Nvfp4CsfDecoder,
    make_nvfp4_csf_batch,
    repack_nvfp4_csf_batch,
)
from b12x._lib.quant.nvfp4_csf_packed import (
    HEADER_BYTES,
    RECORD_HEADER_BYTES,
    TAIL_BYTES,
    PackedCsfPlane,
    build_packed_csf_scales,
    configured_inline_words,
    expand_packed_csf_scales,
    inline_words,
    packed_slab_position,
    record_bytes,
)
from b12x.moe._shared.kernels.w4a16.prepare import _process_nvfp4_packed_scales
from b12x.moe._shared.kernels.w4a16.prefill_a4 import A4PackedPrefillGemm
from b12x._lib.utils import current_cuda_stream, make_ptr
from ..conftest import require_b12x


def _logical_scales(experts, rows, columns, seed, outliers=0.03):
    """E4M3 scale bytes with a narrow per-row range and some out-of-window values.

    Two rows hold zero and subnormal scales, which W4A16's value table does not
    re-bias like normal ones: one as out-of-window values, one within its window.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0x28, 0x48, size=(experts, rows, 1))
    scales = base + rng.integers(0, 14, size=(experts, rows, columns))
    hot = rng.random((experts, rows, columns)) < outliers
    scales[hot] = rng.integers(0x40, 0x7E, size=int(hot.sum()))
    scales[:, 5, ::7] = rng.integers(0, 8, size=scales[:, 5, ::7].shape)
    scales[:, 9] = 4 + np.arange(columns) % 13
    return scales.astype(np.uint8)


def _value_table(kind, device):
    """W4A16's E4M3 re-bias table, or a permutation no constant folds."""
    if kind == "permutation":
        table = np.random.default_rng(7).permutation(256).astype(np.uint8)
        return torch.from_numpy(table).to(device)
    alphabet = torch.arange(256, dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    alphabet = alphabet[:, None].expand(256, 4).contiguous().to(torch.bfloat16)
    packed = _process_nvfp4_packed_scales(alphabet, scale_factor=2.0)
    return packed.view(torch.uint8)[:, 0].contiguous()


def _native_batch(source, device):
    """Byte-window planes (row base, 4-bit offsets, exceptions) of logical scales."""
    experts, rows, columns = source.shape
    fixed, exceptions = [], []
    for plane in source:
        base = np.minimum(plane.min(axis=1), 240).astype(np.uint8)
        offsets = plane.astype(np.int16) - base[:, None]
        outside = offsets > 15
        offsets[outside] = 0
        offsets = offsets.astype(np.uint8)
        packed = offsets[:, ::2] | (offsets[:, 1::2] << 4)
        fixed.append(
            np.concatenate((base.reshape(-1, 16), packed.reshape(rows // 16, -1)), 1)
        )
        position = np.flatnonzero(outside).astype(np.uint32)
        exceptions.append(position | (plane.ravel()[position].astype(np.uint32) << 24))
    return make_nvfp4_csf_batch(
        fixed, exceptions, rows=rows, columns=columns, device=device
    )


def test_packed_slab_positions_invert_the_plane_permutation():
    perm = (
        np.arange(64)
        .reshape(8, 8)
        .T.reshape(-1)
        .reshape(-1, 4)[:, [0, 2, 1, 3]]
        .reshape(-1)
    )
    perm = np.concatenate((perm, perm + 64))
    positions = packed_slab_position(torch.arange(128)).numpy()
    assert np.array_equal(perm[positions], np.arange(128))


@pytest.mark.parametrize("inline", [0, 32])
@pytest.mark.parametrize("table", ["w4a16", "permutation"])
@pytest.mark.parametrize(
    "rows,columns,rotation", [(256, 64, 128), (512, 16, 0), (128, 256, 0)]
)
def test_index_matches_the_expansion_pass(rows, columns, rotation, table, inline):
    device = require_b12x()
    experts = 5
    lut = _value_table(table, device)
    first = repack_nvfp4_csf_batch(
        _native_batch(_logical_scales(experts, rows, columns, 1), device),
        row_rotation=rotation,
        value_lut=lut,
    )
    second = repack_nvfp4_csf_batch(
        _native_batch(_logical_scales(experts, 256, 32, 2), device),
        row_rotation=0,
        value_lut=lut,
    )
    out13 = torch.empty(experts, columns, rows, dtype=torch.uint8, device=device)
    out2 = torch.empty(experts, 32, 256, dtype=torch.uint8, device=device)
    outputs = (out13.view(torch.float8_e4m3fn), out2.view(torch.float8_e4m3fn))
    decoder = Nvfp4CsfDecoder.prepare(first, second, *outputs)
    decoder.decode(torch.arange(experts, dtype=torch.int32, device=device), *outputs)
    torch.cuda.synchronize()
    stored = []
    for batch, expected in ((first, out13), (second, out2)):
        scales = build_packed_csf_scales(batch, inline=inline)
        stored.append(scales)
        assert torch.equal(expand_packed_csf_scales(scales), expected)
        assert scales.max_atom_words <= 128
        if table == "w4a16":
            # The table folds into the bases: most words carry no exception.
            exception_bytes = scales.storage.numel() - scales.words_offset - 512
            assert exception_bytes < 0.25 * experts * batch.rows * batch.columns
    # Calls too large for stage-wise reads expand routed experts from the storage.
    again = tuple(torch.full_like(out, 0xFF) for out in (out13, out2))
    views = tuple(out.view(torch.float8_e4m3fn) for out in again)
    expander = Nvfp4CsfDecoder.prepare(*(PackedCsfPlane.of(s) for s in stored), *views)
    expander.decode(torch.tensor([[3, 1]], dtype=torch.int32, device=device), *views)
    torch.cuda.synchronize()
    for out, expected in zip(again, (out13, out2), strict=True):
        assert torch.equal(out[[1, 3]], expected[[1, 3]])


@pytest.mark.parametrize("packed_index", [0, 1])
def test_pair_decodes_each_projection_storage_layout(packed_index):
    device = require_b12x()
    lut = _value_table("w4a16", device)
    planes = [
        repack_nvfp4_csf_batch(
            _native_batch(_logical_scales(3, 256, 16, seed), device),
            row_rotation=0,
            value_lut=lut,
        )
        for seed in (9, 13)
    ]
    expected = [
        torch.empty(3, 16, 256, dtype=torch.float8_e4m3fn, device=device)
        for _ in planes
    ]
    ids = torch.arange(3, device=device, dtype=torch.int32)
    Nvfp4CsfDecoder.prepare(*planes, *expected).decode(ids, *expected)
    planes[packed_index] = PackedCsfPlane.of(
        build_packed_csf_scales(planes[packed_index])
    )
    actual = [torch.empty_like(t) for t in expected]
    Nvfp4CsfDecoder.prepare(*planes, *actual).decode(ids, *actual)
    torch.cuda.synchronize()
    for output, reference in zip(actual, expected):
        assert torch.equal(output.view(torch.uint8), reference.view(torch.uint8))


def _w4a16_scale_bytes(logical, lut, rotation):
    """W4A16 packed scale bytes ``[E, C, R]`` of logical ``[E, R, C]`` scales."""
    experts, rows, columns = logical.shape
    rotated = np.roll(logical, -rotation, axis=1)
    row = np.arange(rows)
    position = (row // 128) * 128 + packed_slab_position(
        torch.from_numpy(row % 128)
    ).numpy()
    out = np.empty((experts, columns, rows), dtype=np.uint8)
    out[:, :, position] = lut.cpu().numpy()[rotated].transpose(0, 2, 1)
    return torch.from_numpy(out)


@pytest.mark.parametrize("inline", [0, 32])
@pytest.mark.parametrize(
    "rows,columns,rotation", [(192, 22, 96), (704, 6, 352), (128, 10, 0)]
)
def test_index_of_planes_off_the_native_grid(rows, columns, rotation, inline):
    """2048/TP6 = 352 channels: 704 gate/up rows (64-row slabs), 22 down k-groups."""
    device = require_b12x()
    experts = 3
    lut = _value_table("w4a16", device)
    logical = _logical_scales(experts, rows, columns, 5)
    native = _native_batch(logical, device)
    assert (native.rows, native.columns) == (
        -(-rows // 128) * 128,
        -(-columns // 4) * 4,
    )
    assert (
        native.logical_rows or native.rows,
        native.logical_columns or native.columns,
    ) == (
        rows,
        columns,
    )
    batch = repack_nvfp4_csf_batch(native, row_rotation=rotation, value_lut=lut)
    scales = build_packed_csf_scales(batch, inline=inline)
    assert (scales.rows, scales.columns) == (rows, columns)
    assert scales.slab_rows == (128 if rows % 128 == 0 else 64)
    expected = _w4a16_scale_bytes(logical, lut, rotation).to(device)
    assert torch.equal(expand_packed_csf_scales(scales), expected)
    out = torch.full((experts, columns, rows), 0xFF, dtype=torch.uint8, device=device)
    other = torch.empty(experts, 32, 256, dtype=torch.uint8, device=device)
    second = build_packed_csf_scales(
        repack_nvfp4_csf_batch(
            _native_batch(_logical_scales(experts, 256, 32, 2), device),
            row_rotation=0,
            value_lut=lut,
        ),
        inline=inline,
    )
    views = (out.view(torch.float8_e4m3fn), other.view(torch.float8_e4m3fn))
    expander = Nvfp4CsfDecoder.prepare(
        PackedCsfPlane.of(scales), PackedCsfPlane.of(second), *views
    )
    expander.decode(torch.tensor([2, 0], dtype=torch.int32, device=device), *views)
    torch.cuda.synchronize()
    assert torch.equal(out[[0, 2]], expected[[0, 2]])
    assert torch.equal(other[[0, 2]], expand_packed_csf_scales(second)[[0, 2]])


@pytest.mark.parametrize("inline", [4, 32])
def test_records_spill_past_their_inline_words(inline):
    """Atoms with more replacement words than a record holds read the rest from storage."""
    device = require_b12x()
    experts, rows, columns = 2, 128, 8
    lut = _value_table("w4a16", device)
    logical = _logical_scales(experts, rows, columns, 9, outliers=0.5)
    batch = repack_nvfp4_csf_batch(
        _native_batch(logical, device), row_rotation=0, value_lut=lut
    )
    scales = build_packed_csf_scales(batch, inline=inline)
    assert scales.max_atom_words > inline_words(scales.slab_rows, inline)
    assert torch.equal(
        expand_packed_csf_scales(scales), _w4a16_scale_bytes(logical, lut, 0).to(device)
    )


def test_inline_words_parameter(monkeypatch):
    """B12X_NVFP4_CSF_INLINE_WORDS: 0 (default) keeps 32-byte records, header only."""
    monkeypatch.delenv("B12X_NVFP4_CSF_INLINE_WORDS", raising=False)
    assert configured_inline_words() == 0
    for value, words in (("0", 0), ("32", 32), ("8", 8), ("64", 64)):
        monkeypatch.setenv("B12X_NVFP4_CSF_INLINE_WORDS", value)
        assert configured_inline_words() == words
    for value in ("3", "-4", "68", "many"):
        monkeypatch.setenv("B12X_NVFP4_CSF_INLINE_WORDS", value)
        with pytest.raises(ValueError, match="multiple of 4"):
            configured_inline_words()
    # 64-row records hold five eighths of the 128-row count, in whole 16 bytes.
    assert [record_bytes(128, w) for w in (0, 4, 8, 32)] == [32, 48, 64, 160]
    assert [record_bytes(64, w) for w in (0, 4, 8, 32)] == [32, 48, 64, 112]


@pytest.mark.parametrize("rows,columns", [(256, 64), (704, 22)])
def test_storage_without_inline_words_matches_uninlined_storage_size(
    rows, columns, monkeypatch
):
    """The default keeps the memory of 32-byte records plus every replacement word once."""
    device = require_b12x()
    experts = 3
    lut = _value_table("w4a16", device)
    batch = repack_nvfp4_csf_batch(
        _native_batch(_logical_scales(experts, rows, columns, 11), device),
        row_rotation=0,
        value_lut=lut,
    )
    monkeypatch.delenv("B12X_NVFP4_CSF_INLINE_WORDS", raising=False)
    plain = build_packed_csf_scales(batch)
    assert plain.inline_words == 0 and plain.record_bytes == RECORD_HEADER_BYTES
    slabs, atoms = plain.slabs, plain.atoms
    records = plain.storage[HEADER_BYTES : plain.words_offset].view(experts, -1)
    records = records[:, slabs * plain.slab_bytes :].contiguous().view(torch.int32)
    words = int(records.reshape(-1, RECORD_HEADER_BYTES // 4)[:, 2].sum())
    assert words > 0
    assert plain.storage.numel() == (
        HEADER_BYTES
        + experts * slabs * (plain.slab_bytes + atoms * RECORD_HEADER_BYTES)
        + 4 * words
        + TAIL_BYTES
    )
    inlined = build_packed_csf_scales(batch, inline=32)
    assert torch.equal(
        expand_packed_csf_scales(inlined), expand_packed_csf_scales(plain)
    )
    assert inlined.storage.numel() > plain.storage.numel()


class _A4ScaleStages:
    def __init__(self, phase, inline, intermediate_size):
        self.gemm = A4PackedPrefillGemm(
            phase=phase,
            hidden_size=512,
            intermediate_size=intermediate_size,
            top_k=2,
            csf_inline_words=inline,
        )

    @cute.jit
    def __call__(self, source, output, byte_count: Int64, experts: Int32, stream):
        source = cute.make_tensor(source, cute.make_layout((byte_count,)))
        output = cute.make_tensor(
            output,
            cute.make_layout(
                (Int64(experts) * Int64(self.gemm.n_tiles * self.gemm.k_tiles * 256),)
            ),
        )
        self.kernel(source, output).launch(
            grid=(self.gemm.n_tiles, self.gemm.k_tiles, experts),
            block=(self.gemm.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source, output):
        tid, _, _ = cute.arch.thread_idx()
        n_tile, k_tile, expert = cute.arch.block_idx()
        shared = cutlass_utils.SmemAllocator().allocate_tensor(
            cutlass.Uint8,
            cute.make_layout(self.gemm.shared_bytes),
            byte_alignment=16,
        )
        base = shared_ptr_to_u32(shared.iterator)
        stage = base + (Int32(k_tile) % Int32(self.gemm.stages)) * Int32(
            self.gemm.stage_bytes
        )
        self.gemm._stage_csf_tile(source, base, tid, expert, n_tile)
        self.gemm._issue_csf(source, stage, k_tile, tid, expert, n_tile)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        self.gemm._expand_csf_stage(source, base, stage, tid)
        cute.arch.sync_threads()
        if tid < 256:
            tile = (Int64(expert) * Int64(self.gemm.n_tiles) + Int64(n_tile)) * Int64(
                self.gemm.k_tiles
            ) + Int64(k_tile)
            output[tile * Int64(256) + Int64(tid)] = ld_shared_u32(
                stage + Int32(self.gemm.s_off) + Int32(tid) * Int32(4)
            )


@pytest.mark.parametrize("phase", ["fc1", "fc2"])
@pytest.mark.parametrize("inline", [0, 4])
@pytest.mark.parametrize("intermediate_size", [64, 256, 320])
def test_a4_stage_words_match_dense_scale_tiles(phase, inline, intermediate_size):
    """A4 reads separated gate/up slabs and global replacement-word spills exactly."""
    device = require_b12x()
    probe = _A4ScaleStages(phase, inline, intermediate_size)
    gemm = probe.gemm
    experts = 3
    logical = _logical_scales(
        experts, gemm.size_n, gemm.size_k // 16, 31, outliers=0.5
    ).clip(max=0x6F)
    lut = _value_table("w4a16", device)
    rotation = gemm.intermediate_size if gemm.fc1 else 0
    scales = build_packed_csf_scales(
        repack_nvfp4_csf_batch(
            _native_batch(logical, device), row_rotation=rotation, value_lut=lut
        ),
        inline=inline,
    )
    assert scales.max_atom_words > max(32, inline)
    dense = _w4a16_scale_bytes(logical, lut, rotation)
    expected = []
    for tile in range(gemm.n_tiles):
        if gemm.fc1:
            first = tile * 128
            live = min(128, gemm.intermediate_size - first)
            halves = []
            for offset in (0, gemm.intermediate_size):
                half = torch.zeros(experts, gemm.size_k // 16, 128, dtype=torch.uint8)
                half[:, :, :live] = dense[:, :, offset + first : offset + first + live]
                halves.append(half)
            expected.append(torch.cat(halves, dim=-1))
        else:
            expected.append(dense[:, :, tile * 256 : (tile + 1) * 256])
    expected = torch.stack(expected, dim=1).reshape(
        experts, gemm.n_tiles, gemm.k_tiles, 4, 256
    )
    output = torch.full(expected.shape, 0xD6, dtype=torch.uint8, device=device)
    program = b12x_compile(
        probe,
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int64(1),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "quant.w4a16_a4_csf_stage_oracle", 2, (phase, inline, intermediate_size)
        ),
    )
    program(
        make_ptr(
            cutlass.Uint8,
            scales.storage.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint32, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        Int64(scales.storage.numel()),
        Int32(experts),
        current_cuda_stream(),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
