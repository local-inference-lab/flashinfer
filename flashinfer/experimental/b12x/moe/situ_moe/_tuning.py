# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tuning contract and configuration space for fused route-weighted SiTU MoE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from b12x.preparation import (
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)
from b12x.preparation._efficiency import capture_autotune_effort

FORMATS = ("nvfp4", "mxfp8")
DECODE_MAX_TOKENS = 16
DEFAULT_DECODE_MAX_TOKENS = 4


@dataclass(frozen=True, kw_only=True)
class SituMoeQuery:
    """Geometry and quantization contract for one fused SiTU MoE family."""

    weight_format: str  # "nvfp4" or "mxfp8"
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    max_tokens: int
    has_csf: bool = False
    deterministic_output: bool = False
    sm_count: int = 1
    autotune_effort: str = field(default_factory=capture_autotune_effort)

    @property
    def is_decode(self) -> bool:
        return self.max_tokens <= DECODE_MAX_TOKENS

    @property
    def max_routes(self) -> int:
        return self.max_tokens * self.top_k


@dataclass(frozen=True, kw_only=True)
class SituMoeConfig:
    """Kernel launch knobs for SiTU MoE."""

    use_decode_kernel: bool
    prefill_tile_m: int
    fused_route_sum: bool = False

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "SituMoeConfig":
        expected = {f.name for f in fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                f"situ_moe config fields must be {sorted(expected)}, got {sorted(payload)}"
            )
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_query(query: SituMoeQuery) -> None:
    if not isinstance(query, SituMoeQuery):
        raise TypeError("query must be SituMoeQuery")
    if query.weight_format not in FORMATS:
        raise ValueError(
            f"weight_format must be one of {FORMATS}, got {query.weight_format!r}"
        )
    if query.has_csf and query.weight_format != "nvfp4":
        raise ValueError("inline CSF scale decoding is only supported for nvfp4")
    if query.num_experts <= 0 or query.top_k <= 0 or query.max_tokens <= 0:
        raise ValueError("num_experts, top_k, and max_tokens must be positive")
    if query.top_k > query.num_experts:
        raise ValueError("top_k cannot exceed num_experts")
    if query.sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if query.weight_format == "nvfp4":
        if query.hidden % 256 != 0:
            raise ValueError(
                f"nvfp4 situ_moe requires hidden % 256 == 0, got {query.hidden}"
            )
        if query.intermediate % 128 != 0:
            raise ValueError(
                f"nvfp4 situ_moe requires intermediate % 128 == 0, got {query.intermediate}"
            )
    else:
        if query.hidden % 128 != 0:
            raise ValueError(
                f"mxfp8 situ_moe requires hidden % 128 == 0, got {query.hidden}"
            )
        if query.intermediate % 64 != 0:
            raise ValueError(
                f"mxfp8 situ_moe requires intermediate % 64 == 0, got {query.intermediate}"
            )


def default_config(query: SituMoeQuery, device=None) -> SituMoeConfig:
    del device
    # The warp-specialized SIMT decode kernel wins at M <= 4 (1-4 tokens/expert).
    # At M >= 8, skewed/concentrated routing puts 4-16 tokens on hot experts where
    # the Tensor-Core BM=16 kernel processes all routed tokens in one weight pass.
    # At M >= 2048, BM=128 halves HBM weight streaming across active experts.
    use_decode = query.max_tokens <= DEFAULT_DECODE_MAX_TOKENS
    tile_m = (
        16
        if query.max_tokens <= 256
        else (128 if query.max_tokens >= 2048 else 64)
    )
    fused_sum = not use_decode and not query.deterministic_output
    return SituMoeConfig(
        use_decode_kernel=use_decode,
        prefill_tile_m=tile_m,
        fused_route_sum=fused_sum,
    )


def validate_config(query: SituMoeQuery, config: SituMoeConfig, device=None) -> None:
    del device
    if not isinstance(config, SituMoeConfig):
        raise TypeError("config must be SituMoeConfig")
    if config.use_decode_kernel and query.max_tokens > DECODE_MAX_TOKENS:
        raise ValueError(
            f"use_decode_kernel requires max_tokens <= {DECODE_MAX_TOKENS}, got {query.max_tokens}"
        )
    if config.use_decode_kernel and config.fused_route_sum:
        raise ValueError("fused_route_sum is only supported on prefill MMA kernels")
    if query.deterministic_output and config.fused_route_sum:
        raise ValueError("fused_route_sum cannot be enabled when deterministic_output=True")
    if config.prefill_tile_m not in (16, 32, 64, 128):
        raise ValueError(
            f"prefill_tile_m must be 16, 32, 64, or 128, got {config.prefill_tile_m}"
        )
    if not config.use_decode_kernel:
        if (
            (2 * query.intermediate) % 128 != 0
            or query.hidden % 128 != 0
            or query.intermediate % 64 != 0
        ):
            raise ValueError(
                "prefill MMA kernels require hidden % 128 == 0 and intermediate % 64 == 0"
            )


def _validate_query(query, device):
    del device
    validate_query(query)


def _parameters(query: SituMoeQuery, device):
    base = default_config(query, device)
    decode_choices = (
        (True, False) if query.max_tokens <= DECODE_MAX_TOKENS else (False,)
    )
    tile_m_choices = (16, 32, 64, 128)
    fused_choices = (False,) if query.deterministic_output else (False, True)
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "use_decode_kernel": decode_choices,
            "prefill_tile_m": tile_m_choices,
            "fused_route_sum": fused_choices,
        },
        efficiency_predicates=(
            lambda p: (
                p["use_decode_kernel"] == base.use_decode_kernel
                and p["prefill_tile_m"] == base.prefill_tile_m
                and p["fused_route_sum"] == base.fused_route_sum
            ),
        ),
        autotune_effort=query.autotune_effort,
    )


def _materialize(query: SituMoeQuery, device, choice) -> SituMoeConfig:
    base = default_config(query, device)
    return SituMoeConfig(**{**base.to_dict(), **dict(choice)})


TUNING = TuningContract(
    component_id="moe.situ_moe",
    query_schema_version=2,
    config_schema_version=2,
    query_fields=frozenset(f.name for f in fields(SituMoeQuery)),
    config_fields=frozenset(f.name for f in fields(SituMoeConfig)),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda payload: SituMoeConfig(**dict(payload)),
    validate_query=_validate_query,
    validate_config=validate_config,
    default_config=default_config,
    knobs=(
        Knob(name="use_decode_kernel", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="prefill_tile_m", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="fused_route_sum", values=None, binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=2,
    parameters=_parameters,
    materialize=_materialize,
)
