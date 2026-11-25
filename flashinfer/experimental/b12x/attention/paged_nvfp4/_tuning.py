# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tuning contract and configuration space for paged NVFP4 (4-or-6 MAE K) attention."""

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

from ._decode_kernel import SMEM_LIMIT, default_max_splits
from ._math import kv_stage_sizes

HEAD_DIMS = ((128, 128),)
KV_FORMATS = ("nvfp4_4o6k",)
MODES = ("decode", "verify", "prefill")
PAGE_SIZES = (128,)
MAX_DECODE_Q_PER_REQ = 16
MAX_SPLITS = 128
SINK_LAYOUTS = ("head", "row")
SMEM_HEADROOM_BYTES = 2048


def _consumer_warps(query: "PagedNVFP4Query") -> int:
    if query.mode == "prefill":
        return 4
    rows = query.group_size * query.max_q_per_req
    return 4 if (rows + 15) // 16 <= 4 else 8


@dataclass(frozen=True, kw_only=True)
class PagedNVFP4Query:
    """Geometry and feature contract for one paged NVFP4 attention family."""

    mode: str
    kv_format: str
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    page_size: int
    max_batch: int
    max_q_per_req: int
    window_left: int
    causal: bool
    has_sinks: bool
    sink_layout: str = "head"
    has_relative_bias: bool = False
    has_factored_relative_bias: bool = False
    sm_count: int = 1
    caller_owned_placeholders: bool = False
    autotune_effort: str = field(default_factory=capture_autotune_effort)

    @property
    def group_size(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def max_total_q(self) -> int:
        return self.max_batch * self.max_q_per_req


@dataclass(frozen=True, kw_only=True)
class PagedNVFP4Config:
    """Kernel launch knobs."""

    split_kv: bool
    max_splits: int
    head_splits: int
    stage_rows: int
    num_stages: int
    min_tiles_per_split: int

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "PagedNVFP4Config":
        expected = {f.name for f in fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                f"paged_nvfp4 config fields must be {sorted(expected)}, got {sorted(payload)}"
            )
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _head_split_choices(query: PagedNVFP4Query) -> tuple[int, ...]:
    if query.mode == "prefill":
        return (1,)
    return tuple(hs for hs in (1, 2, 4, 8, 16) if query.group_size % hs == 0)


def _rows_fit(query: PagedNVFP4Query, head_splits: int) -> bool:
    if query.mode == "prefill":
        return head_splits == 1
    rows = (query.group_size // head_splits) * query.max_q_per_req
    return (rows + 15) // 16 <= _consumer_warps(query)


def _stage_bytes(query: PagedNVFP4Query, stage_rows: int, num_stages: int) -> int:
    *_, per_stage = kv_stage_sizes(stage_rows, query.head_dim_qk, query.head_dim_vo)
    ring = num_stages * per_stage
    if query.mode == "prefill":
        return ring
    ncw = _consumer_warps(query)
    combine = (ncw * (query.head_dim_vo // 16) * 8 * 32 + 2 * ncw * 2 * 32) * 4
    return max(ring, combine)


def validate_query(query: PagedNVFP4Query) -> None:
    if not isinstance(query, PagedNVFP4Query):
        raise TypeError("query must be PagedNVFP4Query")
    if query.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {query.mode!r}")
    if query.kv_format not in KV_FORMATS:
        raise ValueError(
            f"kv_format must be one of {KV_FORMATS}, got {query.kv_format!r}"
        )
    if (query.head_dim_qk, query.head_dim_vo) not in HEAD_DIMS:
        raise ValueError(f"head dims must be one of {HEAD_DIMS}")
    if query.page_size not in PAGE_SIZES:
        raise ValueError(f"page_size must be one of {PAGE_SIZES}")
    if not query.causal:
        raise ValueError("paged_nvfp4 currently supports causal=True only")
    if (
        query.num_kv_heads <= 0
        or query.num_q_heads <= 0
        or query.num_q_heads % query.num_kv_heads
    ):
        raise ValueError("num_q_heads must be a positive multiple of num_kv_heads")
    if query.max_batch <= 0 or query.sm_count <= 0 or query.max_q_per_req <= 0:
        raise ValueError("max_batch, max_q_per_req, and sm_count must be positive")
    if (
        query.mode in ("decode", "verify")
        and query.max_q_per_req > MAX_DECODE_Q_PER_REQ
    ):
        raise ValueError(
            f"decode/verify max_q_per_req must be <= {MAX_DECODE_Q_PER_REQ}"
        )
    if query.window_left < -1:
        raise ValueError("window_left must be -1 or nonnegative")
    if query.sink_layout not in SINK_LAYOUTS:
        raise ValueError(f"sink_layout must be one of {SINK_LAYOUTS}")
    if query.sink_layout == "row" and not query.has_sinks:
        raise ValueError("sink_layout='row' requires has_sinks=True")
    if query.has_relative_bias and query.has_factored_relative_bias:
        raise ValueError("dense and factored relative bias cannot be combined")
    if query.has_relative_bias and query.mode == "prefill":
        raise ValueError(
            "prefill mode uses factored relative bias, not dense relative bias"
        )
    if query.has_factored_relative_bias and query.mode != "prefill":
        raise ValueError(
            "decode/verify mode uses dense relative bias, not factored relative bias"
        )
    if not any(_rows_fit(query, hs) for hs in _head_split_choices(query)):
        raise ValueError("no head split fits query rows into one CTA")


def default_config(query: PagedNVFP4Query, device=None) -> PagedNVFP4Config:
    del device
    if query.mode == "prefill":
        stages = 6
        while (
            stages > 2
            and _stage_bytes(query, 32, stages) + SMEM_HEADROOM_BYTES > SMEM_LIMIT
        ):
            stages -= 1
        return PagedNVFP4Config(
            split_kv=False,
            max_splits=1,
            head_splits=1,
            stage_rows=32,
            num_stages=stages,
            min_tiles_per_split=4,
        )
    hs_choices = [hs for hs in _head_split_choices(query) if _rows_fit(query, hs)]
    split = query.window_left < 0 or (
        query.window_left >= 1024 and query.max_batch * query.num_kv_heads <= 8
    )
    hs = 1 if split else min(hs_choices, key=lambda v: abs(v - 4))
    if hs not in hs_choices:
        hs = hs_choices[0]
    stages = 12 if _consumer_warps(query) == 4 else 16
    while (
        stages > 2
        and _stage_bytes(query, 16, stages) + SMEM_HEADROOM_BYTES > SMEM_LIMIT
    ):
        stages -= 1
    target_ctas = query.sm_count * (2 if _consumer_warps(query) == 4 else 1)
    max_splits = (
        default_max_splits(
            query.max_batch,
            query.num_kv_heads,
            target_ctas,
            MAX_SPLITS,
            head_splits=hs,
        )
        if split
        else 1
    )
    return PagedNVFP4Config(
        split_kv=split,
        max_splits=max_splits,
        head_splits=hs,
        stage_rows=16,
        num_stages=stages,
        min_tiles_per_split=4,
    )


def validate_config(
    query: PagedNVFP4Query, config: PagedNVFP4Config, device=None
) -> None:
    del device
    if not isinstance(config, PagedNVFP4Config):
        raise TypeError("config must be PagedNVFP4Config")
    if config.head_splits not in _head_split_choices(query) or not _rows_fit(
        query, config.head_splits
    ):
        raise ValueError(f"head_splits={config.head_splits} invalid for query")
    if query.mode == "prefill":
        if config.stage_rows != 32:
            raise ValueError("prefill mode requires stage_rows == 32")
    elif config.stage_rows not in (16, 32) or query.page_size % config.stage_rows:
        raise ValueError("decode/verify stage_rows must be 16 or 32 dividing page_size")
    if (
        config.num_stages < 2
        or _stage_bytes(query, config.stage_rows, config.num_stages)
        + SMEM_HEADROOM_BYTES
        > SMEM_LIMIT
    ):
        raise ValueError("stage ring exceeds shared memory limit")
    if config.min_tiles_per_split < 1:
        raise ValueError("min_tiles_per_split must be positive")
    if query.mode == "prefill" and config.split_kv:
        raise ValueError("prefill mode does not use split_kv")
    if config.split_kv:
        if not 1 <= config.max_splits <= MAX_SPLITS:
            raise ValueError(f"max_splits must be in [1, {MAX_SPLITS}]")
    elif config.max_splits != 1:
        raise ValueError("unsplit configs require max_splits=1")


def _validate_query(query, device):
    del device
    validate_query(query)


def _parameters(query: PagedNVFP4Query, device):
    hs = default_config(query, device).head_splits
    return ParameterSpace.create(
        TUNING.knobs,
        values={"head_splits": _head_split_choices(query)},
        predicates=(lambda p: _rows_fit(query, p["head_splits"]),),
        efficiency_predicates=(lambda p: p["head_splits"] == hs,),
        autotune_effort=query.autotune_effort,
    )


def _materialize(query: PagedNVFP4Query, device, choice):
    base = default_config(query, device)
    return PagedNVFP4Config(**{**base.to_dict(), **dict(choice)})


TUNING = TuningContract(
    component_id="attention.paged_nvfp4",
    query_schema_version=2,
    config_schema_version=1,
    query_fields=frozenset(f.name for f in fields(PagedNVFP4Query)),
    config_fields=frozenset(f.name for f in fields(PagedNVFP4Config)),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda payload: PagedNVFP4Config(**dict(payload)),
    validate_query=_validate_query,
    validate_config=validate_config,
    default_config=default_config,
    knobs=(Knob(name="head_splits", values=None, binding=ParameterBinding.COMPILE),),
    candidate_contract_version=1,
    parameters=_parameters,
    materialize=_materialize,
)
