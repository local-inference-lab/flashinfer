"""WO projection declarations and ready-plan lowering."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x.preparation import FrozenMapping, MemoryRequirements, PersistentMemory, Plan

from .._shared.wo_mxfp8 import (
    WOProjectionScratchCaps,
    _materialize_wo_projection_scratch,
)
from ._tuning import TUNING, WoProjectionQuery


def _fused_tile_override():
    value = os.environ.get("B12X_WO_B_FUSED_TILE", "").strip().lower()
    if value and value not in ("16x64", "16x128"):
        raise ValueError("B12X_WO_B_FUSED_TILE must be 16x64 or 16x128")
    return value


def _query(
    caps: WOProjectionScratchCaps, invocation: FrozenMapping
) -> WoProjectionQuery:
    operation = invocation.get("operation", "plain")
    if operation not in ("plain", "inv_rope"):
        raise ValueError("WO invocation operation must be 'plain' or 'inv_rope'")
    inv_rope = operation == "inv_rope"
    fields = ("heads_per_group", "nope_dim", "rope_dim") if inv_rope else ()
    if any(name not in invocation for name in fields):
        raise ValueError(
            "inverse-RoPE WO declaration requires heads_per_group, nope_dim, and rope_dim"
        )
    allowed = {
        "operation",
        "heads_per_group",
        "nope_dim",
        "rope_dim",
        "return_3d",
        "positions_dtype",
        "cos_sin_dtype",
        "dynamic_tokens",
        "sfb_k_replicated",
        "wo_a_tiled",
        "wo_b_tiled",
    }
    if set(invocation) - allowed:
        raise ValueError("unknown WO invocation field")
    dtype_names = {"bfloat16", "float16"}
    if str(caps.dtype).removeprefix("torch.") not in dtype_names:
        raise ValueError("WO projection supports bfloat16 or float16 inputs")
    if inv_rope and (
        str(invocation.get("positions_dtype", "int64")).removeprefix("torch.")
        not in {"int32", "int64"}
        or str(invocation.get("cos_sin_dtype", "bfloat16")).removeprefix("torch.")
        not in dtype_names | {"float32"}
    ):
        raise ValueError(
            "inverse-RoPE WO requires int32/int64 positions and BF16/FP16/FP32 cos/sin"
        )
    from .._shared.wo_mxfp8 import _WO_QUANT_CHUNKS_PER_PROGRAM

    return WoProjectionQuery(
        dtype=str(caps.dtype).removeprefix("torch."),
        max_tokens=caps.max_tokens,
        dynamic_tokens=invocation.get("dynamic_tokens", False),
        groups=caps.groups,
        group_width=caps.group_width,
        rank=caps.rank,
        hidden=caps.hidden,
        operation=operation,
        heads_per_group=int(invocation["heads_per_group"]) if inv_rope else None,
        nope_dim=int(invocation["nope_dim"]) if inv_rope else None,
        rope_dim=int(invocation["rope_dim"]) if inv_rope else None,
        return_3d=bool(invocation.get("return_3d", False)),
        positions_dtype=str(invocation.get("positions_dtype", "int64")),
        cos_sin_dtype=str(invocation.get("cos_sin_dtype", "bfloat16")),
        sfb_k_replicated=invocation.get("sfb_k_replicated", False),
        wo_a_tiled=invocation.get("wo_a_tiled", False),
        wo_b_tiled=invocation.get("wo_b_tiled", False),
        codegen=FrozenMapping(
            {
                "quant_chunks_per_program": _WO_QUANT_CHUNKS_PER_PROGRAM,
                "wo_b_fused_tile": _fused_tile_override(),
            }
        ),
    )


def _fused_b_tile(query: WoProjectionQuery, config, device):
    override = query.codegen["wo_b_fused_tile"]
    if override:
        return tuple(map(int, override.split("x")))
    if config.decode_tile_n:
        return (16, config.decode_tile_n)
    if query.wo_b_tiled:
        from b12x._lib.dense_gemm.policy import _select_default_dense_gemm_plan

        selected = _select_default_dense_gemm_plan(
            query.max_tokens,
            query.hidden,
            query.rank * query.groups,
            device.identity.sm_count,
            is_mxfp8=True,
            expected_m=query.max_tokens,
        )
        tile = selected.mma_tiler_mn
        return tile if tile in ((16, 64), (16, 128)) else (16, 128)
    return None


def _compile_query_payload(query: WoProjectionQuery):
    return FrozenMapping(
        {
            name: value
            for name, value in TUNING.encode_query(query).items()
            if name != "autotune_effort"
        }
    )


def plan(caps: WOProjectionScratchCaps, *, invocation=FrozenMapping(), override=None):
    """Declare WO projection at a planned token count.

    ``invocation["dynamic_tokens"]=True`` allows any positive live count up to
    ``caps.max_tokens`` using the capacity's launch configuration. The default
    requires the exact planned count and preserves decode specialization.
    Packed loaders may declare ``sfb_k_replicated``, ``wo_a_tiled`` and ``wo_b_tiled`` in
    ``invocation``; binding verifies those weight-layout contracts.
    """
    if not isinstance(caps, WOProjectionScratchCaps):
        raise TypeError("caps must be WOProjectionScratchCaps")
    invocation = FrozenMapping(invocation)
    query = _query(caps, invocation)
    use_fused_b = caps.max_tokens <= 8 and not query.dynamic_tokens
    fused = {}

    def fused_b(config, device):
        if not use_fused_b:
            return None
        key = (device.identity, config)
        if key not in fused:
            from b12x._lib.dense_gemm.preparation import _lower_dense_gemm_fused_quant_a
            from b12x.gemm._preparation import _metadata_operands
            from b12x.gemm._tuning import DenseGemmQuery

            width = caps.rank * caps.groups
            dense_query = DenseGemmQuery(
                recipe="mxfp8",
                entry_point="gemm.mm",
                weight_storage="native",
                output_dtype="bfloat16",
                batch=1,
                max_rows=caps.max_tokens,
                in_features=width,
                out_features=caps.hidden,
                output_mode="provided",
                alpha_mode="unit",
                expected_m=caps.max_tokens,
            )
            _, rhs, _, _, _ = _metadata_operands(dense_query)
            output = torch.empty_strided(
                (caps.max_tokens, caps.hidden, 1),
                (caps.hidden, 1, caps.max_tokens * caps.hidden),
                dtype=torch.bfloat16,
                device="meta",
            )
            if caps.groups == 1:
                source = torch.empty(
                    (caps.max_tokens, caps.rank), dtype=torch.bfloat16, device="meta"
                )
                span = 0
            else:
                source = torch.empty_strided(
                    (caps.max_tokens, caps.rank, caps.groups),
                    (caps.rank, 1, caps.max_tokens * caps.rank),
                    dtype=torch.bfloat16,
                    device="meta",
                )
                span = caps.rank
            fused[key] = _lower_dense_gemm_fused_quant_a(
                source,
                rhs[0],
                rhs[1],
                sm_count=device.identity.sm_count,
                out=output,
                expected_m=caps.max_tokens,
                sfb_k_replicated=query.sfb_k_replicated,
                rhs_values_tiled=(
                    torch.empty(
                        (1, 32, 32, 128, 128), dtype=torch.float8_e4m3fn, device="meta"
                    )
                    if query.wo_b_tiled
                    else None
                ),
                a_inner_span=span,
                mma_tiler_mn=_fused_b_tile(query, config, device),
            )
        return fused[key]

    ordinary = {}

    def ordinary_states(config, device):
        key = device.identity
        if key not in ordinary:
            from b12x.gemm._preparation import _default_lowering, _lower_query
            from b12x.gemm._tuning import DenseGemmQuery
            from .._shared.wo_mxfp8 import _wo_a_mma_tiler

            a_tile = (
                None
                if query.dynamic_tokens
                else _wo_a_mma_tiler(
                    caps.max_tokens,
                    rank=caps.rank,
                    group_width=caps.group_width,
                    groups=caps.groups,
                )
            )
            a_query = DenseGemmQuery(
                recipe="mxfp8",
                entry_point="gemm.mm",
                weight_storage="native",
                output_dtype="bfloat16",
                batch=caps.groups,
                max_rows=caps.max_tokens,
                in_features=caps.group_width,
                out_features=caps.rank,
                output_mode="provided",
                alpha_mode="unit",
                expected_m=caps.max_tokens,
                sfb_k_replicated=query.sfb_k_replicated,
            )
            b_query = DenseGemmQuery(
                recipe="mxfp8",
                entry_point="gemm.mm",
                weight_storage="native",
                output_dtype="bfloat16",
                batch=1,
                max_rows=caps.max_tokens,
                in_features=caps.rank * caps.groups,
                out_features=caps.hidden,
                output_mode="provided",
                alpha_mode="unit",
                expected_m=caps.max_tokens,
                sfb_k_replicated=query.sfb_k_replicated,
            )
            a_options = {"mma_tiler_mn": a_tile}
            if query.wo_a_tiled and a_tile in ((16, 64), (32, 64), (64, 64)):
                a_options["rhs_values_tiled"] = torch.empty(
                    (4, 16, 32, 64, 128),
                    dtype=torch.float8_e4m3fn,
                    device="meta",
                )
            ordinary[key] = (
                _lower_query(a_query, device.identity, a_options),
                _default_lowering(b_query, device.identity),
            )
        return ordinary[key]

    def memory(config, device):
        if config.backend == "a16_tc_gemv":
            from b12x._lib.scratch import scratch_buffer_spec
            from ._tc_gemv import (
                _WOA_FLAGS_CACHE,
                wo_a16_tc_gemv_flags_bytes,
                wo_a16_tc_gemv_scratch_bytes,
            )

            nbytes = wo_a16_tc_gemv_scratch_bytes(
                caps.max_tokens, caps.groups, caps.rank, config.wo_a_split_k
            )
            scratch = (
                scratch_buffer_spec(
                    "wo_projection.a16_scratch",
                    nbytes=nbytes,
                    device=torch.device("cuda", device.ordinal),
                ),
            )
            persistent = ()
            if config.wo_a_split_k > 1:
                fbytes = wo_a16_tc_gemv_flags_bytes(
                    caps.groups, caps.rank, config.wo_a_split_k
                )
                fkey = (
                    device.ordinal,
                    caps.groups,
                    caps.rank,
                    caps.group_width,
                    config.wo_a_split_k,
                )
                persistent = (
                    PersistentMemory(
                        ("wo_projection.a16_woa_flags", *fkey),
                        fbytes,
                        fbytes if fkey in _WOA_FLAGS_CACHE else 0,
                    ),
                )
            return MemoryRequirements(scratch=scratch, persistent=persistent)
        scratch = _materialize_wo_projection_scratch(
            caps, config=config
        ).scratch_specs()
        # Fused dense retains alpha-one; only its state may allocate that scalar.
        from b12x._lib.dense_gemm.buffers import _ALPHA_ONE_CACHE

        alpha = _ALPHA_ONE_CACHE.get(("cuda", device.ordinal))
        resident = 0 if alpha is None else alpha.numel() * alpha.element_size()
        persistent = (
            PersistentMemory(("dense.alpha_one", device.ordinal), 4, resident),
        )
        return MemoryRequirements(scratch=scratch, persistent=persistent)

    def _a16_woa_args(config):
        inv_rope = query.operation == "inv_rope"
        head_dim = (query.nope_dim + query.rope_dim) if inv_rope else 512
        return (
            caps.groups,
            caps.rank,
            caps.group_width,
            caps.max_tokens,
            config.wo_a_split_k,
            config.wo_a_warps,
            inv_rope,
            head_dim,
            query.nope_dim if inv_rope else 448,
            query.rope_dim if inv_rope else 64,
            query.positions_dtype,
        )

    def _a16_wob_args(config):
        k_b = caps.groups * caps.rank
        return (
            caps.hidden,
            k_b,
            k_b,
            caps.max_tokens,
            1,
            config.wo_b_warps,
            config.decode_tile_n,
        )

    def compile_jobs(config, device):
        if config.backend == "a16_tc_gemv":
            return (
                CompileJob.create(
                    "b12x.gemm.wo_projection._tc_gemv:compile_grouped_woa_tc_gemv",
                    CompileJob.DEVICE_ORDINAL,
                    *_a16_woa_args(config),
                ),
                CompileJob.create(
                    "b12x.gemm.blockscaled._mxfp8_tc_gemv:compile_mxfp8_tc_gemv",
                    CompileJob.DEVICE_ORDINAL,
                    *_a16_wob_args(config),
                ),
            )
        return (
            CompileJob.create(
                "b12x._lib.dense_gemm.preparation:_compile_dense_lowering",
                ordinary_states(config, device)[0].to_dict(),
                CompileJob.DEVICE_ORDINAL,
            ),
            CompileJob.create(
                "b12x._lib.dense_gemm.preparation:_compile_dense_lowering",
                ordinary_states(config, device)[1].to_dict(),
                CompileJob.DEVICE_ORDINAL,
            ),
            *(
                (
                    CompileJob.create(
                        "b12x._lib.dense_gemm.preparation:_compile_dense_fused_quant_lowering",
                        fused_b(config, device).to_dict(),
                        CompileJob.DEVICE_ORDINAL,
                    ),
                )
                if use_fused_b
                else ()
            ),
            CompileJob.create(
                "b12x.gemm._shared.wo_mxfp8:compile_wo_quantizers",
                _compile_query_payload(query),
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def materialize(selection, device):
        config = selection.config
        if config.backend == "a16_tc_gemv":
            from b12x.gemm.blockscaled._mxfp8_tc_gemv import compile_mxfp8_tc_gemv
            from ._tc_gemv import (
                cached_wo_a16_tc_gemv_flags,
                compile_grouped_woa_tc_gemv,
            )

            resolved = torch.device("cuda", device.ordinal)
            a_fn = compile_grouped_woa_tc_gemv(device.ordinal, *_a16_woa_args(config))
            b_fn = compile_mxfp8_tc_gemv(device.ordinal, *_a16_wob_args(config))
            woa_flags = (
                cached_wo_a16_tc_gemv_flags(
                    resolved,
                    caps.groups,
                    caps.rank,
                    caps.group_width,
                    config.wo_a_split_k,
                )
                if config.wo_a_split_k > 1
                else None
            )
            return _PreparedWO(
                _A16WOScratchState(caps, config),
                query,
                None,
                None,
                None,
                None,
                a16_wo_a=a_fn,
                a16_wo_b=b_fn,
                a16_woa_flags=woa_flags,
                a16_split_k=config.wo_a_split_k,
            )
        from b12x._lib.dense_gemm.preparation import (
            _materialize_dense,
            _materialize_dense_fused_quant,
        )
        from .._shared.wo_mxfp8 import compile_wo_quantizers

        a_lowering, b_lowering = ordinary_states(config, device)
        return _PreparedWO(
            _materialize_wo_projection_scratch(caps, config=config),
            query,
            _materialize_dense(a_lowering, torch.device("cuda", device.ordinal)),
            _materialize_dense(b_lowering, torch.device("cuda", device.ordinal)),
            (
                _materialize_dense_fused_quant(
                    fused_b(config, device),
                    torch.device("cuda", device.ordinal),
                )
                if use_fused_b
                else None
            ),
            compile_wo_quantizers(_compile_query_payload(query), device.ordinal),
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=caps.device,
        shared=True,
    )


class _A16WOScratchState:
    """Minimal scratch state for the A16 TC-GEMV WO backend."""

    def __init__(self, caps: WOProjectionScratchCaps, config):
        from b12x._lib.scratch import scratch_buffer_spec
        from ._tc_gemv import wo_a16_tc_gemv_scratch_bytes

        self.caps = caps
        self.config = config
        nbytes = wo_a16_tc_gemv_scratch_bytes(
            caps.max_tokens, caps.groups, caps.rank, config.wo_a_split_k
        )
        self._specs = (
            scratch_buffer_spec(
                "wo_projection.a16_scratch", nbytes=nbytes, device=caps.device
            ),
        )

    def scratch_specs(self):
        return self._specs

    def _slice_scratch(self, scratch, tokens, out):
        from b12x._lib.scratch import scratch_tensor
        from ._tc_gemv import wo_a16_tc_gemv_tmp_bytes

        storage = scratch_tensor(scratch, self._specs, owner="WO A16 projection")
        tmp_bytes = wo_a16_tc_gemv_tmp_bytes(
            self.caps.max_tokens, self.caps.groups, self.caps.rank
        )
        tmp = (
            storage[: tokens * self.caps.groups * self.caps.rank * 2]
            .view(torch.bfloat16)
            .view(tokens, self.caps.groups * self.caps.rank)
        )
        splitk_scratch = (
            storage[tmp_bytes:].view(torch.float32)
            if self.config.wo_a_split_k > 1
            else None
        )
        if out is None:
            out = torch.empty(
                (tokens, self.caps.hidden),
                dtype=torch.bfloat16,
                device=storage.device,
            )
        if out.device != storage.device or not out.is_contiguous():
            raise ValueError("WO output must be contiguous on the scratch device")
        if out.shape not in ((tokens, self.caps.hidden), (tokens, self.caps.hidden, 1)):
            raise ValueError(
                "WO output must have shape [tokens, hidden] or [tokens, hidden, 1]"
            )
        output = out.as_strided(
            (tokens, self.caps.hidden, 1),
            (self.caps.hidden, 1, tokens * self.caps.hidden),
        )
        return tmp, splitk_scratch, output

    def bind(
        self,
        *,
        scratch,
        source_tgd,
        weights,
        out=None,
        return_3d=False,
        expected_m=None,
    ):
        from .._shared.wo_mxfp8 import (
            WOProjectionBinding,
            _validate_wo_projection_inputs,
        )

        tokens = _validate_wo_projection_inputs(source_tgd, weights)
        tmp, splitk_scratch, output = self._slice_scratch(scratch, tokens, out)
        return WOProjectionBinding(
            source_tgd=source_tgd,
            weights=weights,
            x_q=splitk_scratch,
            tmp=tmp,
            tmp_q=None,
            output=output,
            return_3d=return_3d,
            expected_m=expected_m,
        )

    def bind_inv_rope(
        self,
        *,
        scratch,
        o,
        positions,
        cos_sin_cache,
        weights,
        heads_per_group,
        out=None,
        nope_dim=448,
        rope_dim=64,
        return_3d=False,
        expected_m=None,
    ):
        from .._shared.wo_mxfp8 import (
            WOProjectionInvRopeBinding,
            _validate_wo_projection_inv_rope_inputs,
        )

        tokens = _validate_wo_projection_inv_rope_inputs(
            o=o,
            weights=weights,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )
        tmp, splitk_scratch, output = self._slice_scratch(scratch, tokens, out)
        return WOProjectionInvRopeBinding(
            o=o,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            weights=weights,
            x_q=splitk_scratch,
            tmp=tmp,
            tmp_q=None,
            output=output,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            return_3d=return_3d,
            expected_m=expected_m,
        )


@dataclass(frozen=True)
class _PreparedWO:
    """Private ready WO state; public callers receive a prepared Plan only."""

    _scratch_state: object
    query: WoProjectionQuery
    ordinary_a: object
    ordinary_b: object
    fused_b: object
    quantizers: object
    a16_wo_a: object = None
    a16_wo_b: object = None
    a16_woa_flags: object = None
    a16_split_k: int = 1

    def bind(self, **kwargs):
        if self.query.operation != "plain":
            raise ValueError("inverse-RoPE declaration requires bind_inv_rope")
        if kwargs.get("return_3d", False) != self.query.return_3d:
            raise ValueError("WO output form differs from declaration")
        source = kwargs["source_tgd"]
        if str(source.dtype).removeprefix("torch.") != self.query.dtype:
            raise ValueError("WO source dtype differs from declaration")
        self._check_tokens(source.shape[0])
        self._check_weights(kwargs["weights"])
        return self._scratch_state.bind(**kwargs)

    def bind_inv_rope(self, **kwargs):
        if self.query.operation != "inv_rope":
            raise ValueError("plain WO declaration requires bind")
        actual = (
            kwargs.get("heads_per_group"),
            kwargs.get("nope_dim", 448),
            kwargs.get("rope_dim", 64),
        )
        expected = (
            self.query.heads_per_group,
            self.query.nope_dim,
            self.query.rope_dim,
        )
        if actual != expected or kwargs.get("return_3d", False) != self.query.return_3d:
            raise ValueError("inverse-RoPE WO invocation differs from declaration")
        o, positions, cos_sin_cache = (
            kwargs[name] for name in ("o", "positions", "cos_sin_cache")
        )
        if (
            str(o.dtype).removeprefix("torch.") != self.query.dtype
            or str(positions.dtype).removeprefix("torch.") != self.query.positions_dtype
            or str(cos_sin_cache.dtype).removeprefix("torch.")
            != self.query.cos_sin_dtype
        ):
            raise ValueError("inverse-RoPE WO tensor dtypes differ from declaration")
        self._check_tokens(o.shape[0])
        self._check_weights(kwargs["weights"])
        return self._scratch_state.bind_inv_rope(**kwargs)

    def _check_weights(self, weights):
        if self.query.sfb_k_replicated and not weights.sfb_k_replicated:
            raise ValueError("WO weights do not guarantee replicated block scales")
        if self.query.wo_a_tiled and weights.wo_a.values_tiled is None:
            raise ValueError("WO weights do not provide the declared tiled WO-A layout")
        if self.query.wo_b_tiled and weights.wo_b.values_tiled is None:
            raise ValueError("WO weights do not provide the declared tiled WO-B layout")

    def _check_tokens(self, tokens):
        if not 1 <= int(tokens) <= self.query.max_tokens:
            raise ValueError("WO execution exceeds its planned token capacity")
        if not self.query.dynamic_tokens and int(tokens) != self.query.max_tokens:
            raise ValueError("WO plan requires its exact prepared token count")

    def _get_quantizers(self, device):
        if self.quantizers is not None:
            return self.quantizers
        from .._shared.wo_mxfp8 import compile_wo_quantizers

        return compile_wo_quantizers(_compile_query_payload(self.query), device.index)

    def quantize_a(self, source_tgd, *, out=None):
        from .._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm

        self._check_tokens(source_tgd.shape[0])
        if out is None:
            out = empty_mxfp8_rows_for_dense_gemm(
                source_tgd.shape[0],
                self.query.group_width,
                num_groups=self.query.groups,
                device=source_tgd.device,
            )
        self._get_quantizers(source_tgd.device).quantize_a(source_tgd, out)
        return out

    def quantize_a_inv_rope(
        self,
        o,
        positions,
        cos_sin_cache,
        *,
        groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        out=None,
    ):
        from .._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm

        self._check_tokens(o.shape[0])
        if (groups, heads_per_group, nope_dim, rope_dim) != (
            self.query.groups,
            self.query.heads_per_group,
            self.query.nope_dim,
            self.query.rope_dim,
        ):
            raise ValueError("inverse-RoPE WO quantization differs from declaration")
        if out is None:
            out = empty_mxfp8_rows_for_dense_gemm(
                o.shape[0],
                self.query.group_width,
                num_groups=self.query.groups,
                device=o.device,
            )
        self._get_quantizers(o.device).quantize_a_inv_rope(
            o,
            positions,
            cos_sin_cache,
            out,
            groups=groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )
        return out

    def quantize_b(self, tmp_trg, *, out=None):
        from .._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm

        self._check_tokens(tmp_trg.shape[0])
        if out is None:
            out = empty_mxfp8_rows_for_dense_gemm(
                tmp_trg.shape[0],
                self.query.rank * self.query.groups,
                num_groups=1,
                device=tmp_trg.device,
            )
        self._get_quantizers(tmp_trg.device).quantize_b(tmp_trg, out)
        return out

    def run(self, binding, *, stream=None):
        if self.a16_wo_a is not None:
            from b12x._lib.utils.device import cuda_stream_to_int

            stream_int = cuda_stream_to_int(stream)
            tokens = binding.source_tgd.shape[0]
            self.a16_wo_a(
                binding.source_tgd,
                None,
                None,
                binding.weights.wo_a.values,
                binding.weights.wo_a.scale_mma,
                binding.tmp,
                binding.x_q if self.a16_split_k > 1 else None,
                self.a16_woa_flags,
                stream_int,
            )
            self.a16_wo_b(
                binding.tmp,
                binding.weights.wo_b.values,
                binding.weights.wo_b.scale_mma,
                binding.output[:, :, 0],
                None,
                None,
                stream_int,
            )
            return binding.output if binding.return_3d else binding.output[:, :, 0]
        tokens = binding.source_tgd.shape[0]
        self.quantizers.quantize_a(binding.source_tgd, binding.x_q)
        a_values = binding.weights.wo_a.values
        if a_values.ndim == 2:
            a_values = a_values.unsqueeze(-1)
        self.ordinary_a.run(
            (
                binding.x_q.values.view(
                    tokens, self.query.group_width, self.query.groups
                ),
                binding.x_q.scale_mma,
            ),
            (a_values, binding.weights.wo_a.scale_mma),
            out=binding.tmp,
            alpha=None,
            stream=stream,
            rhs_values_tiled=(
                binding.weights.wo_a.values_tiled
                if self.ordinary_a.lowering.b_tile_major
                else None
            ),
        )
        if self.fused_b is not None:
            source = binding.tmp
            if binding.weights.groups == 1:
                source = source.as_strided(
                    (tokens, binding.weights.rank),
                    (binding.weights.rank, 1),
                )
            self.fused_b.run(
                source,
                binding.weights.wo_b.values.reshape(binding.weights.hidden, -1, 1),
                binding.weights.wo_b.scale_mma,
                out=binding.output,
                stream=stream,
                rhs_values_tiled=(
                    binding.weights.wo_b.values_tiled if self.query.wo_b_tiled else None
                ),
            )
        else:
            self.quantizers.quantize_b(binding.tmp, binding.tmp_q)
            self.ordinary_b.run(
                (binding.tmp_q.values.unsqueeze(-1), binding.tmp_q.scale_mma),
                (
                    binding.weights.wo_b.values.unsqueeze(-1),
                    binding.weights.wo_b.scale_mma,
                ),
                out=binding.output,
                alpha=None,
                stream=stream,
            )
        return binding.output if binding.return_3d else binding.output[:, :, 0]

    def run_inv_rope(self, binding, *, stream=None):
        if self.a16_wo_a is not None:
            from b12x._lib.utils.device import cuda_stream_to_int

            stream_int = cuda_stream_to_int(stream)
            self.a16_wo_a(
                binding.o,
                binding.positions,
                binding.cos_sin_cache,
                binding.weights.wo_a.values,
                binding.weights.wo_a.scale_mma,
                binding.tmp,
                binding.x_q if self.a16_split_k > 1 else None,
                self.a16_woa_flags,
                stream_int,
            )
            self.a16_wo_b(
                binding.tmp,
                binding.weights.wo_b.values,
                binding.weights.wo_b.scale_mma,
                binding.output[:, :, 0],
                None,
                None,
                stream_int,
            )
            return binding.output if binding.return_3d else binding.output[:, :, 0]
        tokens = binding.o.shape[0]
        self.quantizers.quantize_a_inv_rope(
            binding.o,
            binding.positions,
            binding.cos_sin_cache,
            binding.x_q,
            groups=binding.weights.groups,
            heads_per_group=binding.heads_per_group,
            nope_dim=binding.nope_dim,
            rope_dim=binding.rope_dim,
        )
        a_values = binding.weights.wo_a.values
        if a_values.ndim == 2:
            a_values = a_values.unsqueeze(-1)
        self.ordinary_a.run(
            (
                binding.x_q.values.view(
                    tokens, self.query.group_width, self.query.groups
                ),
                binding.x_q.scale_mma,
            ),
            (a_values, binding.weights.wo_a.scale_mma),
            out=binding.tmp,
            alpha=None,
            stream=stream,
            rhs_values_tiled=(
                binding.weights.wo_a.values_tiled
                if self.ordinary_a.lowering.b_tile_major
                else None
            ),
        )
        if self.fused_b is not None:
            source = (
                binding.tmp
                if binding.weights.groups != 1
                else binding.tmp.as_strided(
                    (tokens, binding.weights.rank), (binding.weights.rank, 1)
                )
            )
            self.fused_b.run(
                source,
                binding.weights.wo_b.values.reshape(binding.weights.hidden, -1, 1),
                binding.weights.wo_b.scale_mma,
                out=binding.output,
                stream=stream,
                rhs_values_tiled=(
                    binding.weights.wo_b.values_tiled if self.query.wo_b_tiled else None
                ),
            )
        else:
            self.quantizers.quantize_b(binding.tmp, binding.tmp_q)
            self.ordinary_b.run(
                (binding.tmp_q.values.unsqueeze(-1), binding.tmp_q.scale_mma),
                (
                    binding.weights.wo_b.values.unsqueeze(-1),
                    binding.weights.wo_b.scale_mma,
                ),
                out=binding.output,
                alpha=None,
                stream=stream,
            )
        return binding.output if binding.return_3d else binding.output[:, :, 0]


__all__ = ["plan"]
