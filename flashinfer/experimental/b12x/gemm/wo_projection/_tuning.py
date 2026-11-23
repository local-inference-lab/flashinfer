"""Startup selection for W_o projection launch geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from b12x.preparation import (
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)
from b12x.preparation._efficiency import (
    AutotuneEffort,
    capture_autotune_effort,
)


@dataclass(frozen=True, kw_only=True)
class WoProjectionQuery:
    dtype: str
    max_tokens: int
    dynamic_tokens: bool = False
    groups: int
    group_width: int
    rank: int
    hidden: int
    codegen: FrozenMapping = FrozenMapping()
    operation: str = "plain"
    heads_per_group: int | None = None
    nope_dim: int | None = None
    rope_dim: int | None = None
    return_3d: bool = False
    positions_dtype: str = "int64"
    cos_sin_dtype: str = "bfloat16"
    sfb_k_replicated: bool = False
    wo_a_tiled: bool = False
    wo_b_tiled: bool = False
    autotune_effort: str = field(default_factory=capture_autotune_effort)

    def __post_init__(self) -> None:
        object.__setattr__(self, "codegen", FrozenMapping(self.codegen))
        object.__setattr__(
            self, "autotune_effort", AutotuneEffort(self.autotune_effort).value
        )


@dataclass(frozen=True, kw_only=True)
class WoProjectionConfig:
    backend: str = "mxfp8"
    decode_tile_n: int = 0
    wo_a_split_k: int = 1
    wo_a_warps: int = 4
    wo_b_warps: int = 4


def _a16_tc_gemv_eligible(query, device):
    return (
        device is not None
        and device.compute_capability in ((12, 0), (12, 1))
        and query.dtype == "bfloat16"
        and not query.wo_a_tiled
        and not query.wo_b_tiled
        and not query.sfb_k_replicated
        and 1 <= query.max_tokens <= 32
        and query.group_width % 64 == 0
        and (query.groups * query.rank) % 64 == 0
        and query.rank % 16 == 0
        and query.hidden % 8 == 0
        and (
            query.operation == "plain"
            or (
                query.nope_dim % 16 == 0
                and query.rope_dim % 16 == 0
                and query.cos_sin_dtype == "float32"
            )
        )
    )


def _decode_domain(query):
    return (
        not query.dynamic_tokens
        and query.dtype == "bfloat16"
        and 1 <= query.max_tokens <= 8
        and query.groups * query.rank % 128 == 0
        and (query.groups == 1 or query.rank % 32 == 0)
    )


def _decode_tuning_geometry(query):
    return _decode_domain(query) and (
        query.groups,
        query.group_width,
        query.rank,
        query.hidden,
    ) == (2, 4096, 1024, 5120)


def _validate_query(query, device):
    if not isinstance(query, WoProjectionQuery):
        raise TypeError("WO query must be WoProjectionQuery")
    if query.dtype not in ("bfloat16", "float16"):
        raise TypeError("WO input must be BF16 or FP16")
    if any(
        type(value) is not int or value <= 0
        for value in (
            query.max_tokens,
            query.groups,
            query.group_width,
            query.rank,
            query.hidden,
        )
    ):
        raise ValueError("WO geometry must contain positive integer dimensions")
    if type(query.dynamic_tokens) is not bool:
        raise TypeError("WO dynamic_tokens must be a boolean")
    if any(
        type(value) is not bool
        for value in (query.sfb_k_replicated, query.wo_a_tiled, query.wo_b_tiled)
    ):
        raise TypeError("WO packed-weight contracts must be boolean")
    if query.wo_a_tiled and (query.groups, query.rank, query.group_width) != (
        4,
        1024,
        4096,
    ):
        raise ValueError("tiled WO-A requires four packed 1024-by-4096 weight groups")
    if query.wo_b_tiled and (query.hidden, query.groups * query.rank) != (4096, 4096):
        raise ValueError("tiled WO-B requires the packed 4096-by-4096 weight layout")
    if query.operation not in ("plain", "inv_rope"):
        raise ValueError("WO operation must be plain or inv_rope")
    if query.operation == "inv_rope" and (
        any(
            type(value) is not int or value <= 0
            for value in (query.heads_per_group, query.nope_dim, query.rope_dim)
        )
        or query.heads_per_group * (query.nope_dim + query.rope_dim)
        != query.group_width
        or query.rope_dim % 2
        or query.positions_dtype not in ("int32", "int64")
        or query.cos_sin_dtype not in ("bfloat16", "float16", "float32")
    ):
        raise ValueError("inverse-RoPE WO geometry or pointer dtypes are invalid")


def _validate(query, config, device):
    if not isinstance(config, WoProjectionConfig) or config.backend not in (
        "mxfp8",
        "a16_tc_gemv",
    ):
        raise ValueError("WO projection requires an MXFP8 or A16 TC-GEMV configuration")
    if config.backend == "a16_tc_gemv":
        if not _a16_tc_gemv_eligible(query, device):
            raise ValueError("WO A16 TC-GEMV backend is ineligible for this query")
        if (
            type(config.decode_tile_n) is not int
            or config.decode_tile_n not in (16, 32)
            or type(config.wo_a_split_k) is not int
            or config.wo_a_split_k not in (1, 2)
            or (query.group_width // 64) % config.wo_a_split_k != 0
            or type(config.wo_a_warps) is not int
            or config.wo_a_warps not in (4, 8)
            or type(config.wo_b_warps) is not int
            or config.wo_b_warps not in (4, 8)
        ):
            raise ValueError("invalid WO A16 TC-GEMV configuration")
        return
    if config.wo_a_split_k != 1 or config.wo_a_warps != 4 or config.wo_b_warps != 4:
        raise ValueError("WO A16 TC-GEMV knobs require backend='a16_tc_gemv'")
    if type(config.decode_tile_n) is not int or config.decode_tile_n not in (
        0,
        64,
        128,
    ):
        raise ValueError("WO decode_tile_n must be 0, 64 or 128")
    if config.decode_tile_n and not _decode_domain(query):
        raise ValueError("WO decode tile requires static BF16 M1..8 and aligned K/span")
    override = query.codegen.get("wo_b_fused_tile", "")
    if override and config.decode_tile_n not in (0, int(override.split("x")[1])):
        raise ValueError("WO decode tile conflicts with the captured tile control")


def _default(query, device):
    if (
        _a16_tc_gemv_eligible(query, device)
        and not query.dynamic_tokens
        and query.operation == "inv_rope"
        and query.cos_sin_dtype == "float32"
    ):
        wo_b_tn = 32 if query.max_tokens >= 25 else 16
        warps = 8 if query.max_tokens <= 8 else 4
        return WoProjectionConfig(
            backend="a16_tc_gemv",
            decode_tile_n=wo_b_tn,
            wo_a_split_k=1,
            wo_a_warps=warps,
            wo_b_warps=warps,
        )
    if not _decode_tuning_geometry(query):
        return WoProjectionConfig()
    override = query.codegen.get("wo_b_fused_tile", "")
    tile_n = (
        int(override.split("x")[1])
        if override
        else (64 if query.max_tokens == 1 else 128)
    )
    return WoProjectionConfig(decode_tile_n=tile_n)


def _parameters(query, device):
    override = query.codegen.get("wo_b_fused_tile", "")
    mxfp8_tiles = (0, int(override.split("x")[1])) if override else (0, 64, 128)
    mxfp8_tiles = mxfp8_tiles if _decode_domain(query) else (0,)
    a16_ok = _a16_tc_gemv_eligible(query, device)
    a16_tiles = (16, 32) if query.max_tokens >= 17 else (16,)
    a16_splits = (
        (1, 2)
        if (
            a16_ok
            and (query.group_width // 64) % 2 == 0
            and query.groups * ((query.rank + 15) // 16) <= 192
        )
        else (1,)
    )
    backends = ("mxfp8", "a16_tc_gemv") if a16_ok else ("mxfp8",)
    all_tiles = tuple(dict.fromkeys((*mxfp8_tiles, *(a16_tiles if a16_ok else ()))))

    def eligible(choice):
        if choice["backend"] == "mxfp8":
            return (
                choice["decode_tile_n"] in mxfp8_tiles
                and choice["wo_a_split_k"] == 1
                and choice["wo_a_warps"] == 4
                and choice["wo_b_warps"] == 4
            )
        return (
            a16_ok
            and choice["decode_tile_n"] in a16_tiles
            and choice["wo_a_split_k"] in a16_splits
            and choice["wo_a_warps"] in (4, 8)
            and choice["wo_b_warps"] in (4, 8)
        )

    def prefer_a16(choice):
        # Single-launch grouped A16 TC-GEMV streams WO-A and WO-B once from DRAM
        # without activation quantization, consistently beating MXFP8 DenseGemm
        # in cold DRAM for M <= 32 even when hot-L2 single-layer tuning prefers MXFP8.
        return not a16_ok or query.dynamic_tokens or choice["backend"] == "a16_tc_gemv"

    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "backend": backends,
            "decode_tile_n": all_tiles,
            "wo_a_split_k": a16_splits if a16_ok else (1,),
            "wo_a_warps": (4, 8) if a16_ok else (4,),
            "wo_b_warps": (4, 8) if a16_ok else (4,),
        },
        predicates=(eligible,),
        efficiency_predicates=(prefer_a16,),
        autotune_effort=query.autotune_effort,
    )


TUNING = TuningContract(
    component_id="gemm.wo_projection",
    query_schema_version=23,
    config_schema_version=23,
    query_fields=frozenset(WoProjectionQuery.__dataclass_fields__),
    config_fields=frozenset(WoProjectionConfig.__dataclass_fields__),
    encode_query=lambda query: {
        name: getattr(query, name) for name in query.__dataclass_fields__
    },
    encode_config=asdict,
    decode_config=lambda payload: WoProjectionConfig(**dict(payload)),
    validate_query=_validate_query,
    validate_config=_validate,
    default_config=_default,
    knobs=(
        Knob(
            name="backend",
            values=("mxfp8", "a16_tc_gemv"),
            binding=ParameterBinding.COMPILE,
        ),
        Knob(name="decode_tile_n", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="wo_a_split_k", values=(1, 2), binding=ParameterBinding.COMPILE),
        Knob(name="wo_a_warps", values=(4, 8), binding=ParameterBinding.COMPILE),
        Knob(name="wo_b_warps", values=(4, 8), binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=7,
    parameters=_parameters,
    materialize=lambda query, device, choice: WoProjectionConfig(**dict(choice)),
)


__all__ = ["TUNING", "WoProjectionConfig", "WoProjectionQuery"]
