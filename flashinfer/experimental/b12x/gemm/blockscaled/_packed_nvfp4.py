"""Prepared BF16 dense projections over the compact W4A16 weight layout."""

from dataclasses import dataclass, replace
import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, PersistentMemory, Plan
from b12x.preparation.tuning import Knob, ParameterSpace
from ._tuning import BlockscaledConfig, BlockscaledQuery, TUNING


@dataclass(frozen=True)
class PackedNVFP4LinearWeight:
    values: torch.Tensor
    scale_mma: torch.Tensor
    global_scale: torch.Tensor
    in_features: int
    out_features: int
    global_scale_kind: str = "multiplier"

    @property
    def padded_in_features(self):
        return self.in_features


def pack_nvfp4_a16_linear_weight(weight, weight_scale, global_scale):
    from b12x.moe._shared.kernels.w4a16.prepare import (
        _repack_weight,
        _permute_nvfp4_scales,
        _nvfp4_compute_scale_factor,
        _scale_perms,
    )

    if weight.ndim != 2 or weight.dtype != torch.uint8 or not weight.is_cuda:
        raise ValueError("packed NVFP4 requires CUDA uint8 [N,K/2] weights")
    n, half_k = weight.shape
    k = half_k * 2
    if n % 64 or k % 32:
        raise ValueError("packed NVFP4 requires N divisible by 64 and K by 32")
    if (
        weight_scale.shape != (n, k // 16)
        or weight_scale.dtype != torch.float8_e4m3fn
        or weight_scale.device != weight.device
    ):
        raise ValueError("NVFP4 scales must be colocated E4M3 [N,K/16]")
    if (
        global_scale.numel() != 1
        or global_scale.dtype != torch.float32
        or global_scale.device != weight.device
        or not bool(torch.isfinite(global_scale).all())
        or not bool((global_scale > 0).all())
    ):
        raise ValueError("NVFP4 global scale must be a finite positive FP32 multiplier")
    factor = _nvfp4_compute_scale_factor(weight_scale[None], torch.bfloat16)
    sf = weight_scale.float()
    if (
        not bool(torch.isfinite(sf).all())
        or bool((sf < 0).any())
        or bool(((sf * factor * 128 < 2) & (sf != 0)).any())
    ):
        raise ValueError(
            "NVFP4 scales are not losslessly representable in packed W4A16"
        )
    values = _repack_weight(weight[None], size_k=k, size_n=n, row_rotation=None)
    scales, gain = _permute_nvfp4_scales(
        weight_scale[None],
        global_scale,
        size_k=k,
        size_n=n,
        a_dtype=torch.bfloat16,
        row_rotation=None,
    )
    lifted = (scales.view(torch.uint8).to(torch.int16) << 7).view(torch.float16).float()
    recovered = (lifted / (factor * 128)).reshape(-1, 4)[:, [0, 2, 1, 3]]
    permutation, _ = _scale_perms()
    inverse = torch.argsort(torch.tensor(permutation, device=weight.device))
    recovered = recovered.reshape(-1, 64)[:, inverse].reshape(k // 16, n).T
    if not torch.equal(recovered, sf) or not bool(torch.isfinite(gain).all()):
        raise ValueError("packed W4A16 conversion changes NVFP4 scale values")
    return PackedNVFP4LinearWeight(values.reshape(n, k // 8), scales, gain, k, n)


@dataclass(frozen=True, kw_only=True)
class PackedNVFP4Config(BlockscaledConfig):
    pipeline_stages: int = 4
    grid_blocks: int | None = None

    @classmethod
    def from_config(cls, payload):
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("packed NVFP4 configuration fields differ from its schema")
        return cls(**dict(payload))


def _validate_query(q, device):
    if device is not None and device.compute_capability not in ((12, 0), (12, 1)):
        raise ValueError("packed NVFP4 dense projection requires SM120/SM121")
    if (
        q.recipe != "nvfp4_packed"
        or q.activation_mode != "a16"
        or q.source_activation != "identity"
        or q.activation_scale_available
        or q.global_scale_kind != "multiplier"
        or not q.source_contiguous
        or not q.source_aligned
        or q.in_features != q.padded_in_features
        or q.in_features % 32
        or q.out_features % 64
        or q.num_tokens < 1
        or q.workspace_form not in ("owned", "provided")
    ):
        raise ValueError(
            "packed NVFP4 requires aligned contiguous BF16 rows without activation scaling"
        )


def _validate_config(q, c, device):
    from b12x.moe._shared.kernels.w4a16.kernel.config import (
        _candidate_tile_fits,
        _w4a16_num_regs,
    )

    _validate_query(q, device)
    if (
        c.mode != "a16"
        or c.tile_m not in (8, 16, 32, 64)
        or c.tile_n not in (64, 128, 256, 512)
        or c.tile_k not in (32, 64, 128)
        or c.split_k not in (None, 1)
        or c.mma_warps is not None
        or c.dense_config is not None
        or c.fuse_swiglu
        or c.pipeline_stages not in (4, 5)
        or (c.grid_blocks is not None and c.grid_blocks <= 0)
    ):
        raise ValueError("unsupported packed NVFP4 launch configuration")
    _w4a16_num_regs(
        cta_threads=c.tile_n * c.tile_k // 64,
        cta_m_blocks=(c.tile_m + 15) // 16,
        cta_n_blocks=c.tile_n // 16,
        cta_k_blocks=c.tile_k // 16,
        uses_m_block_8=c.tile_m == 8,
        weight_layout="packed",
    )
    if not _candidate_tile_fits(
        problem_n=q.out_features,
        problem_k=q.in_features,
        cta_m_blocks=(c.tile_m + 15) // 16,
        tile_n=c.tile_n,
        tile_k=c.tile_k,
        cta_threads=c.tile_n * c.tile_k // 64,
        max_shared_mem=101376 - 512,
        scale_format="e4m3_k16",
        weight_layout="packed",
        weight_bits=4,
        allow_qualified_fc2_tile=True,
        uses_m_block_8=c.tile_m == 8,
        pipeline_stages=c.pipeline_stages,
    ):
        raise ValueError("packed NVFP4 tile exceeds device resources")


def _default(q, device):
    from b12x.moe._shared.kernels.w4a16.kernel.config import _select_tile_config

    m = 8 if q.num_tokens <= 8 else 16 if q.num_tokens <= 16 else 32
    k, n, _, _ = _select_tile_config(
        problem_m=q.num_tokens,
        problem_n=q.out_features,
        problem_k=q.in_features,
        top_k=1,
        moe_block_size=m,
        sms=device.sm_count,
        max_shared_mem=101376,
        scale_format="e4m3_k16",
        weight_layout="packed",
        weight_bits=4,
    )
    return PackedNVFP4Config(mode="a16", tile_m=m, tile_n=n, tile_k=k)


def _parameters(q, device):
    rows = (
        (8,)
        if q.num_tokens <= 8
        else (16,)
        if q.num_tokens <= 16
        else (16, 32)
        if q.num_tokens <= 32
        else (32, 64)
    )
    return ParameterSpace(
        knobs=(
            Knob(name="mode", values=("a16",)),
            Knob(name="tile_m", values=rows),
            Knob(name="tile_n", values=(64, 128, 256, 512)),
            Knob(name="tile_k", values=(32, 64, 128)),
            Knob(name="split_k", values=(None,)),
            Knob(name="mma_warps", values=(None,)),
            Knob(name="dense_config", values=(None,)),
            Knob(name="a16_backend", values=("auto",)),
            Knob(name="fuse_swiglu", values=(False,)),
            Knob(name="pipeline_stages", values=(4, 5)),
            Knob(
                name="grid_blocks",
                values=(None, max(1, (q.out_features // 512 + 1) // 2)),
            ),
        ),
        predicates=(
            lambda x: (x["tile_n"], x["tile_k"])
            in ((128, 64), (64, 128), (128, 128), (256, 64), (512, 32)),
        ),
        autotune_effort=q.autotune_effort,
    )


CONTRACT = replace(
    TUNING,
    query_schema_version=25,
    config_schema_version=27,
    config_fields=frozenset(PackedNVFP4Config.__dataclass_fields__),
    decode_config=PackedNVFP4Config.from_config,
    candidate_contract_version=26,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default,
    parameters=_parameters,
    knobs=(Knob(name="mode", values=("a16",)),),
    equivalence_key=lambda q, d, c: c.to_dict(),
)


@program_cache(scope="preparation")
def compile_projection(query_payload, config_payload, ordinal):
    from b12x.moe._shared.kernels.w4a16.kernel.gemm_launch import compile_w4a16_gemm

    q = BlockscaledQuery(**dict(query_payload))
    c = PackedNVFP4Config.from_config(FrozenMapping(config_payload))
    with torch.cuda.device(ordinal):
        return compile_w4a16_gemm(
            size_m=q.num_tokens,
            size_n=q.out_features,
            size_k=q.in_features,
            num_experts=1,
            top_k=1,
            mul_topk_weights=False,
            tile_n=c.tile_n,
            tile_k=c.tile_k,
            moe_block_size=c.tile_m,
            max_m_blocks=(q.num_tokens + c.tile_m - 1) // c.tile_m,
            element_dtype="bf16",
            weight_layout="packed",
            scale_format="e4m3_k16",
            w13_layout="packed",
            dense_route_fast_path=True,
            pipeline_stages=c.pipeline_stages,
        )


@dataclass
class _State:
    query: BlockscaledQuery
    config: BlockscaledConfig
    device: torch.device
    kernel: object
    workspace: torch.Tensor | None
    locks: torch.Tensor
    dummy: torch.Tensor
    sms: int
    required_workspace: int = 0

    @property
    def owned_nbytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.workspace, self.locks, self.dummy)
            if t is not None
        )

    def run(
        self,
        source,
        values,
        scales,
        global_scale,
        *,
        activation_scale=None,
        out=None,
        workspace=None,
        stream=None,
    ):
        import cutlass
        import cutlass.cute as cute
        from b12x._lib.utils.pointers import make_ptr
        from b12x._lib.utils.device import current_cuda_stream
        from ._a16 import _stream_context, _overlap

        q = self.query
        m = source.numel() // q.in_features
        if (
            source.dtype != torch.bfloat16
            or source.device != self.device
            or not source.is_contiguous()
            or source.shape[-1] != q.in_features
            or m > q.num_tokens
            or activation_scale is not None
            or source.data_ptr() % 16
        ):
            raise ValueError(
                "packed NVFP4 invocation differs from its prepared contract"
            )
        if q.workspace_form == "owned":
            if workspace is not None:
                raise ValueError(
                    "owned packed NVFP4 scratch does not accept caller storage"
                )
            temporary = self.workspace
        else:
            if (
                workspace is None
                or workspace.dtype != torch.uint8
                or workspace.device != self.device
                or not workspace.is_contiguous()
                or workspace.numel() < self.required_workspace
                or workspace.data_ptr() % 16
            ):
                raise ValueError(
                    "packed NVFP4 scratch does not satisfy the prepared size/alignment"
                )
            temporary = workspace[: self.required_workspace].view(torch.float32)
        if (
            values.dtype != torch.int32
            or values.shape != (q.out_features, q.in_features // 8)
            or not values.is_contiguous()
            or values.device != self.device
            or scales.numel() != q.out_features * q.in_features // 16
            or scales.element_size() != 1
            or scales.device != self.device
            or not scales.is_contiguous()
            or global_scale.dtype != torch.float32
            or global_scale.numel() != 1
            or global_scale.device != self.device
        ):
            raise ValueError("packed NVFP4 weight geometry differs from preparation")
        shape = (*source.shape[:-1], q.out_features)
        if (out is None) != (q.output_mode == "functional"):
            raise ValueError("packed NVFP4 output form differs from preparation")
        if out is None:
            out = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        else:
            if (
                out.shape != shape
                or out.dtype != torch.bfloat16
                or not out.is_contiguous()
            ):
                raise ValueError("packed NVFP4 output layout differs from preparation")
            if any(_overlap(out, t) for t in (source, values, scales, global_scale)):
                raise ValueError("packed NVFP4 output aliases an input")
        if out.device != self.device or out.data_ptr() % 16:
            raise ValueError(
                "packed NVFP4 output device/alignment differs from preparation"
            )
        if any(
            _overlap(temporary, t) for t in (source, values, scales, global_scale, out)
        ):
            raise ValueError("packed NVFP4 scratch aliases an operand")
        if m == 0:
            return out
        grid = min(
            ((q.out_features + self.kernel.tile_n - 1) // self.kernel.tile_n)
            * ((m + self.config.tile_m - 1) // self.config.tile_m),
            self.config.grid_blocks or self.sms * self.kernel.blocks_per_sm,
        )
        with torch.cuda.device(self.device), _stream_context(stream, self.device):
            a = make_ptr(
                cutlass.BFloat16,
                source.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            c = make_ptr(
                cutlass.BFloat16,
                out.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            self.kernel.compiled(
                a,
                a,
                values.flatten(),
                c,
                scales.view(torch.uint8).view(torch.int32).flatten(),
                global_scale,
                self.dummy,
                self.dummy,
                self.dummy,
                global_scale,
                temporary,
                self.locks,
                self.dummy,
                m,
                grid,
                current_cuda_stream(),
            )
        return out


def plan(q, *, invocation=FrozenMapping(), override=None):
    from b12x.moe._shared.kernels.w4a16.host import packed_gemm_scratch_elements

    def count(c, device):
        return packed_gemm_scratch_elements(
            size_n=q.out_features,
            route_slots=q.num_tokens,
            moe_block_size=c.tile_m,
            sms=device.identity.sm_count,
            weight_layout="packed",
        )

    def jobs(c, device):
        return (
            CompileJob.create(
                "b12x.gemm.blockscaled._packed_nvfp4:compile_projection",
                CONTRACT.encode_query(q),
                c.to_dict(),
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def memory(c, device):
        from b12x.preparation import current_plan
        from b12x._lib.scratch import scratch_buffer_spec

        needed = count(c, device) * 4
        scratch = (
            (
                scratch_buffer_spec(
                    "blockscaled.scratch",
                    nbytes=needed,
                    device=torch.device("cuda", device.ordinal),
                ),
            )
            if q.workspace_form == "provided"
            else ()
        )
        size = (device.identity.sm_count * 4 + 6) * 4
        if q.workspace_form == "owned":
            size += needed
        return MemoryRequirements(
            scratch, (PersistentMemory(("packed_nvfp4", current_plan()), size, 0),)
        )

    def materialize(selection, device):
        c = selection.config
        d = torch.device("cuda", device.ordinal)
        kernel = compile_projection(
            CONTRACT.encode_query(q), c.to_dict(), device.ordinal
        )
        return _State(
            q,
            c,
            d,
            kernel,
            torch.empty(count(c, device), dtype=torch.float32, device=d)
            if q.workspace_form == "owned"
            else None,
            torch.zeros(device.identity.sm_count * 4 + 2, dtype=torch.int32, device=d),
            torch.zeros(4, dtype=torch.int32, device=d),
            device.identity.sm_count,
            count(c, device) * 4,
        )

    return Plan(
        contract=CONTRACT,
        query=q,
        invocation=invocation,
        override=override,
        shared=True,
        _compile_jobs=jobs,
        _memory_requirements=memory,
        _materialize=materialize,
    )
