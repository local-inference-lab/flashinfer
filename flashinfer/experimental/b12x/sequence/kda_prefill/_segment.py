"""Planned distributed segment-parallel KDA prefill (summarize + finalize).

Each rank owns a contiguous segment of tokens (up to ``max_tokens``) for all
heads and executes:

1. ``segment_summarize``: runs prologue, prepare, transfer matrix ``A``
   (``summary_mode=2``, initial slot 1 = I -> final slot 2) and local state ``B``
   + tentative outputs + token-128 convergence probe (``summary_mode=4``,
   initial slot 0 = 0 -> final slot 3, probe slot ``2 + 3 * max_seqs``).
2. Caller exchanges ``(A_r, B_r)`` across ranks and computes each rank's
   incoming state ``S_in`` into slot ``2 + 2 * max_seqs`` (slot 4 for
   ``max_seqs == 1``).
3. ``segment_finalize``: reuses the prepared tiles already in ``scratch`` and
   runs ``summary_mode=5`` (initial slot 4 -> final slot 4), exiting early at
   token 128 when the state matches the tentative probe bit-for-bit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping,
    Knob,
    MemoryRequirements,
    ParameterBinding,
    ParameterSpace,
    Plan,
    TuningContract,
)
from b12x.preparation.types import plan_from_handle, require_prepared
import math
from .._shared.delta_prefill.contract import materialize_layout

_LOG2E = 1.0 / math.log(2.0)
from .._shared.delta_prefill.workspace import (
    BACKEND,
    CHUNK_TOKENS,
    STAGE_CHOICES,
    validate_shared_memory,
)
from . import _impl
from ._tuning import KdaPrefillConfig, KdaPrefillQuery, tiles_capacity

__b12x_source_dependencies__ = (
    "b12x.sequence.kda_prefill._impl",
    "b12x.sequence.kda_prefill._tuning",
    "b12x.sequence._shared.delta_prefill._cute_kernels",
)

_SEGMENT_V_SPLIT_CHOICES = (16, 32, 64)
_PARAMETER_FIELDS = frozenset(("a_log_dtype", "dt_bias_dtype", "state_indices_dtype"))

_SEGMENT_INPUTS = (
    "q",
    "k",
    "v",
    "raw_g",
    "raw_beta",
    "A_log",
    "dt_bias",
    "cu_seqlens",
    "checkpoint_state_indices",
    "checkpoint_offsets",
    "num_seqs",
    "num_tokens",
)


def _segment_default_config(query: KdaPrefillQuery, device) -> KdaPrefillConfig:
    del device
    return KdaPrefillConfig(
        backend=BACKEND,
        v_split=16,
        k_split=1,
        stages=4,
        window_tiles=tiles_capacity(query.max_tokens, query.max_seqs),
    )


def _segment_validate_query(query: KdaPrefillQuery, device) -> None:
    from ._tuning import _validate_query

    _validate_query(query, device)
    if query.null_state_index != 0:
        raise ValueError("KDA segment prefill requires null_state_index == 0")
    min_slots = 2 + 4 * query.max_seqs
    if query.max_state_slots < min_slots:
        raise ValueError(
            f"KDA segment prefill requires at least {min_slots} pool slots, got {query.max_state_slots}"
        )


def _segment_validate_config(
    query: KdaPrefillQuery, config: KdaPrefillConfig, device
) -> None:
    del device
    if config.backend != BACKEND:
        raise ValueError(f"unsupported KDA segment backend {config.backend!r}")
    if config.v_split not in _SEGMENT_V_SPLIT_CHOICES:
        raise ValueError(
            f"unsupported KDA segment v_split {config.v_split!r}; expected {_SEGMENT_V_SPLIT_CHOICES}"
        )
    if config.k_split != 1:
        raise ValueError("KDA segment prefill requires k_split == 1")
    if config.stages not in STAGE_CHOICES:
        raise ValueError(f"unsupported KDA segment stages {config.stages!r}")
    expected_window = tiles_capacity(query.max_tokens, query.max_seqs)
    if config.window_tiles != expected_window:
        raise ValueError(
            f"KDA segment prefill requires single-window window_tiles == {expected_window}, got {config.window_tiles}"
        )
    max_seq_tiles = -(-query.max_tokens // CHUNK_TOKENS)
    validate_shared_memory(
        config.v_split,
        config.k_split,
        config.stages,
        config.window_tiles,
        max_sequence_tiles=max_seq_tiles,
        summary_mode=5,
        reuse_value_buffer=False,
    )


def _segment_tuning_parameters(query: KdaPrefillQuery, device):
    del device
    window = tiles_capacity(query.max_tokens, query.max_seqs)
    return ParameterSpace.create(
        SEGMENT_TUNING.knobs,
        values={"window_tiles": (window,)},
        autotune_effort=query.autotune_effort,
    )


_KEY_FIELDS = frozenset(KdaPrefillQuery.__dataclass_fields__) - {"max_state_slots"}


def _encode_query(query: KdaPrefillQuery) -> dict[str, object]:
    return {name: value for name, value in query.to_dict().items() if name in _KEY_FIELDS}


SEGMENT_TUNING = TuningContract(
    component_id="sequence.kda_prefill.segment",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=_KEY_FIELDS,
    config_fields=frozenset({"backend", "v_split", "k_split", "stages", "window_tiles"}),
    encode_query=_encode_query,
    encode_config=KdaPrefillConfig.to_dict,
    decode_config=KdaPrefillConfig.from_config,
    validate_query=_segment_validate_query,
    validate_config=_segment_validate_config,
    default_config=_segment_default_config,
    knobs=(
        Knob(name="backend", values=(BACKEND,), binding=ParameterBinding.COMPILE),
        Knob(name="v_split", values=_SEGMENT_V_SPLIT_CHOICES, binding=ParameterBinding.COMPILE),
        Knob(name="k_split", values=(1,), binding=ParameterBinding.COMPILE),
        Knob(name="stages", values=STAGE_CHOICES, binding=ParameterBinding.COMPILE),
        Knob(name="window_tiles", values=None, binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=1,
    parameters=_segment_tuning_parameters,
)


def _materialize_segment_layout(caps: _impl.Caps, config: KdaPrefillConfig) -> _impl._Layout:
    window = tiles_capacity(caps.max_tokens, caps.max_seqs)
    max_seq_tiles = -(-caps.max_tokens // CHUNK_TOKENS)
    return materialize_layout(
        caps,
        layout_type=_impl._Layout,
        v_split=config.v_split,
        k_split=1,
        stages=config.stages,
        window_tiles=window,
        workspace_windows=1,
        max_sequence_tiles=max_seq_tiles,
    )


@program_cache(scope="preparation")
def compile_segment(
    query_dict: Mapping[str, object],
    config_dict: Mapping[str, object],
    device_ordinal: int,
) -> tuple:
    from .._shared.delta_prefill._cute_kernels import (
        _compile_prepare,
        _compile_prologue,
        _compile_recurrence,
    )
    from .._shared.delta_prefill.preparation import _metadata_binding

    query = KdaPrefillQuery(**dict(query_dict))
    config = KdaPrefillConfig.from_config(FrozenMapping(config_dict))
    caps = _impl.Caps(
        device=torch.device("cuda", device_ordinal),
        max_tokens=query.max_tokens,
        max_seqs=query.max_seqs,
        max_state_slots=max(query.max_state_slots, 2 + 4 * query.max_seqs),
        heads=query.heads,
        head_dim=query.head_dim,
        model_dtype=getattr(torch, query.model_dtype),
        state_dtype=getattr(torch, query.state_dtype),
        qk_l2norm=query.qk_l2norm,
        checkpoint_export=query.checkpoint_export,
        null_state_index=query.null_state_index,
        chunk_tokens=CHUNK_TOKENS,
    )
    layout = _materialize_segment_layout(caps, config)
    meta = _metadata_binding(layout, query)
    return (
        _compile_prologue(meta)[1],
        _compile_prepare(meta)[1],
        _compile_recurrence(meta, 2)[1],
        _compile_recurrence(meta, 4)[1],
        _compile_recurrence(meta, 5)[1],
    )


@dataclass(frozen=True)
class _SegmentState:
    query: KdaPrefillQuery
    layout: _impl._Layout
    programs: tuple
    parameter_dtypes: tuple
    zero_indices: torch.Tensor
    identity_indices: torch.Tensor
    transfer_indices: torch.Tensor
    local_indices: torch.Tensor
    output_indices: torch.Tensor
    no_checkpoint_indices: torch.Tensor
    no_checkpoint_offsets: torch.Tensor

    def _make_binding(
        self,
        *,
        plan,
        scratch,
        pool,
        output,
        initial_state_indices,
        final_state_indices,
        checkpoint_state_indices,
        checkpoint_offsets,
        **inputs,
    ):
        binding = _impl._bind(
            self.layout,
            _plan=plan,
            scratch=scratch,
            recurrent_state=pool,
            output=output,
            initial_state_indices=initial_state_indices,
            final_state_indices=final_state_indices,
            checkpoint_state_indices=checkpoint_state_indices,
            checkpoint_offsets=checkpoint_offsets,
            **inputs,
        )
        actual = (
            binding.A_log.dtype,
            binding.dt_bias.dtype,
            binding.initial_state_indices.dtype,
        )
        if actual != self.parameter_dtypes:
            raise ValueError("KDA segment parameter/index dtypes differ from preparation")
        return binding

    def summarize(
        self,
        *,
        plan,
        inputs: dict[str, torch.Tensor],
        scratch: torch.Tensor,
        pool: torch.Tensor,
        output: torch.Tensor,
        lower_bound: float,
        scale: float | None,
        eps: float,
    ) -> None:
        lower_bound_v, scale_v, eps_v = _impl._check_run_scalars(lower_bound, scale, eps)
        prologue, prepare, transfer_prog, local_prog, _ = self.programs
        ckpt_idx = inputs["checkpoint_state_indices"]
        ckpt_off = inputs["checkpoint_offsets"]
        base = {k: v for k, v in inputs.items() if k not in ("checkpoint_state_indices", "checkpoint_offsets")}
        transfer_binding = self._make_binding(
            plan=plan,
            scratch=scratch,
            pool=pool,
            output=output,
            initial_state_indices=self.identity_indices,
            final_state_indices=self.transfer_indices,
            checkpoint_state_indices=self.no_checkpoint_indices,
            checkpoint_offsets=self.no_checkpoint_offsets,
            **base,
        )
        local_binding = self._make_binding(
            plan=plan,
            scratch=scratch,
            pool=pool,
            output=output,
            initial_state_indices=self.zero_indices,
            final_state_indices=self.local_indices,
            checkpoint_state_indices=ckpt_idx,
            checkpoint_offsets=ckpt_off,
            **base,
        )
        prologue(local_binding)
        prepare(local_binding, scale_v, lower_bound_v * _LOG2E, eps_v, 0)
        transfer_prog(transfer_binding, 0)
        local_prog(local_binding, 0)

    def finalize(
        self,
        *,
        plan,
        inputs: dict[str, torch.Tensor],
        scratch: torch.Tensor,
        pool: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        _, _, _, _, correct_prog = self.programs
        ckpt_idx = inputs["checkpoint_state_indices"]
        ckpt_off = inputs["checkpoint_offsets"]
        base = {k: v for k, v in inputs.items() if k not in ("checkpoint_state_indices", "checkpoint_offsets")}
        output_binding = self._make_binding(
            plan=plan,
            scratch=scratch,
            pool=pool,
            output=output,
            initial_state_indices=self.output_indices,
            final_state_indices=self.output_indices,
            checkpoint_state_indices=ckpt_idx,
            checkpoint_offsets=ckpt_off,
            **base,
        )
        correct_prog(output_binding, 0)


def segment_plan(
    caps: _impl.Caps,
    *,
    invocation: Mapping[str, object],
    override: KdaPrefillConfig | None = None,
) -> Plan:
    """Declare a planned distributed segment-parallel KDA recurrence."""
    if not isinstance(caps, _impl.Caps):
        raise TypeError("caps must be kda_prefill.Caps")
    if caps.null_state_index != 0:
        raise ValueError("KDA segment prefill requires null_state_index == 0")
    min_slots = 2 + 4 * caps.max_seqs
    if caps.max_state_slots < min_slots:
        raise ValueError(f"KDA segment prefill requires max_state_slots >= {min_slots}")
    inv = FrozenMapping(invocation)
    if set(inv) - _PARAMETER_FIELDS:
        raise ValueError("unknown delta-prefill invocation metadata")
    query = _impl._query(caps, inv)

    def compile_jobs(config, device):
        del device
        return (
            CompileJob.create(
                "b12x.sequence.kda_prefill._segment:compile_segment",
                FrozenMapping(replace(query, autotune_effort="default").to_dict()),
                FrozenMapping(SEGMENT_TUNING.encode_config(config)),
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def memory(config, device):
        del device
        layout = _materialize_segment_layout(caps, config)
        return MemoryRequirements(scratch=layout.scratch_specs(), persistent=())

    def materialize(selection, device):
        layout = _materialize_segment_layout(caps, selection.config)
        programs = compile_segment(
            FrozenMapping(replace(query, autotune_effort="default").to_dict()),
            FrozenMapping(SEGMENT_TUNING.encode_config(selection.config)),
            device.ordinal,
        )
        idx_dtype = getattr(torch, query.state_indices_dtype)
        dev = caps.device
        m = caps.max_seqs
        arange = torch.arange(m, dtype=idx_dtype, device=dev)
        return _SegmentState(
            query=query,
            layout=layout,
            programs=programs,
            parameter_dtypes=(
                getattr(torch, query.a_log_dtype),
                getattr(torch, query.dt_bias_dtype),
                idx_dtype,
            ),
            zero_indices=torch.zeros((m,), dtype=idx_dtype, device=dev),
            identity_indices=torch.ones((m,), dtype=idx_dtype, device=dev),
            transfer_indices=(2 + arange).to(idx_dtype),
            local_indices=(2 + m + arange).to(idx_dtype),
            output_indices=(2 + 2 * m + arange).to(idx_dtype),
            no_checkpoint_indices=torch.zeros((m,), dtype=idx_dtype, device=dev),
            no_checkpoint_offsets=torch.zeros((m,), dtype=torch.int32, device=dev),
        )

    return Plan(
        contract=SEGMENT_TUNING,
        query=query,
        invocation=inv,
        override=override,
        shared=False,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=caps.device,
    )


@torch.library.custom_op(
    "b12x::kda_segment_summarize",
    mutates_args=("scratch", "pool", "output"),
)
def _segment_summarize_op(
    inputs: list[torch.Tensor],
    scratch: torch.Tensor,
    pool: torch.Tensor,
    output: torch.Tensor,
    plan_handle: int,
    lower_bound: float,
    scale: float | None,
    eps: float,
) -> None:
    plan = plan_from_handle(plan_handle)
    state = require_prepared(plan, "sequence.kda_prefill.segment", output.device)
    state.summarize(
        plan=plan,
        inputs=dict(zip(_SEGMENT_INPUTS, inputs, strict=True)),
        scratch=scratch,
        pool=pool,
        output=output,
        lower_bound=lower_bound,
        scale=scale,
        eps=eps,
    )


@_segment_summarize_op.register_fake
def _fake_segment_summarize(
    inputs, scratch, pool, output, plan_handle, lower_bound, scale, eps
):
    pass


@torch.library.custom_op(
    "b12x::kda_segment_finalize",
    mutates_args=("scratch", "pool", "output"),
)
def _segment_finalize_op(
    inputs: list[torch.Tensor],
    scratch: torch.Tensor,
    pool: torch.Tensor,
    output: torch.Tensor,
    plan_handle: int,
) -> None:
    plan = plan_from_handle(plan_handle)
    state = require_prepared(plan, "sequence.kda_prefill.segment", output.device)
    state.finalize(
        plan=plan,
        inputs=dict(zip(_SEGMENT_INPUTS, inputs, strict=True)),
        scratch=scratch,
        pool=pool,
        output=output,
    )


@_segment_finalize_op.register_fake
def _fake_segment_finalize(inputs, scratch, pool, output, plan_handle):
    pass


def segment_summarize(
    plan: Plan,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    scratch: torch.Tensor,
    pool: torch.Tensor,
    output: torch.Tensor,
    lower_bound: float,
    scale: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    _segment_summarize_op(
        [
            q,
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            cu_seqlens,
            checkpoint_state_indices,
            checkpoint_offsets,
            num_seqs,
            num_tokens,
        ],
        scratch,
        pool,
        output,
        plan.handle,
        lower_bound,
        scale,
        eps,
    )
    return output


def segment_finalize(
    plan: Plan,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    scratch: torch.Tensor,
    pool: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    _segment_finalize_op(
        [
            q,
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            cu_seqlens,
            checkpoint_state_indices,
            checkpoint_offsets,
            num_seqs,
            num_tokens,
        ],
        scratch,
        pool,
        output,
        plan.handle,
    )
    return output


__all__ = [
    "SEGMENT_TUNING",
    "segment_finalize",
    "segment_plan",
    "segment_summarize",
]
