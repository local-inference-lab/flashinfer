"""Prepared BF16 RMSNorm declarations with a fixed FP32 reduction schedule."""

from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
    make_fixed_contract,
)


@dataclass(frozen=True, kw_only=True)
class RMSNormQuery:
    hidden_size: int

    def __post_init__(self):
        if (
            type(self.hidden_size) is not int
            or not 0 < self.hidden_size <= 2**24
            or self.hidden_size % 4
        ):
            raise ValueError(
                "Torch-order RMSNorm width must be a positive multiple of four, at most 2**24"
            )


TUNING = make_fixed_contract(
    component_id="norm.torch_rmsnorm", query_type=RMSNormQuery, backend="cute"
)


@program_cache(scope="preparation")
def compile_program(query_payload, ordinal):
    from .torch_rmsnorm import _compile

    return _compile(RMSNormQuery(**dict(query_payload)).hidden_size, ordinal)


@dataclass(frozen=True)
class RMSNormState:
    query: RMSNormQuery
    compiled: object

    def run(self, source, weight, *, out, eps):
        from .torch_rmsnorm import _run_state

        return _run_state(self, source, weight, out, eps)


def plan(query, *, device, invocation=FrozenMapping(), override=None):
    if not isinstance(query, RMSNormQuery):
        raise TypeError("RMSNorm plan requires RMSNormQuery")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("RMSNorm rows and epsilon are runtime inputs")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("Torch-order RMSNorm requires CUDA")

    def materialize(selection, detected):
        del selection, detected
        from .torch_rmsnorm import _compile

        return RMSNormState(query, _compile(query.hidden_size, target.index))

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=lambda config, detected: (
            CompileJob.create(
                "b12x.norm._torch_rmsnorm_preparation:compile_program",
                TUNING.encode_query(query),
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _memory_requirements=lambda config, detected: MemoryRequirements(),
        _materialize=materialize,
        _device=target,
    )
