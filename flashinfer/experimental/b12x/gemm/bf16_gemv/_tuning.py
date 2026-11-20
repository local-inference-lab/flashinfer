"""Startup selection for unquantized BF16 and FP32 projections."""

from dataclasses import asdict, dataclass, field, fields

from b12x.preparation import Knob, ParameterBinding, ParameterSpace, TuningContract
from b12x.preparation._efficiency import (
    AutotuneEffort,
    capture_autotune_effort,
)


@dataclass(frozen=True, kw_only=True)
class GemvQuery:
    source_dtype: str
    weight_dtype: str
    max_rows: int
    in_features: int
    out_features: int
    source_contiguous: bool
    source_aligned: bool
    weight_contiguous: bool
    weight_aligned: bool
    output_dtype: str = "bfloat16"
    output_contiguous: bool = True
    output_aligned: bool = True
    bias_dtype: str | None = None
    # "fp32" requires the whole sum of products to be carried in FP32 (ordered or
    # compensated), as the FP32-operand product does; "any" also admits backends
    # whose tensor-core or library accumulation is looser.
    accumulation: str = "any"
    autotune_effort: str = field(default_factory=capture_autotune_effort)

    def __post_init__(self):
        object.__setattr__(
            self, "autotune_effort", AutotuneEffort(self.autotune_effort).value
        )


@dataclass(frozen=True, kw_only=True)
class GemvConfig:
    backend: str
    rows_per_tile: int = 8
    block_threads: int = 128


def validate_query(query):
    if not isinstance(query, GemvQuery):
        raise TypeError("projection query must be GemvQuery")
    if any(
        dtype not in ("bfloat16", "float32")
        for dtype in (
            query.source_dtype,
            query.weight_dtype,
            query.output_dtype,
        )
    ) or query.bias_dtype not in (None, "bfloat16", "float32"):
        raise ValueError("unquantized projection requires BF16 or FP32 operands")
    if query.accumulation not in ("any", "fp32"):
        raise ValueError("projection accumulation must be 'any' or 'fp32'")
    if (
        not 0 < query.max_rows < 2**31
        or min(query.in_features, query.out_features) <= 0
    ):
        raise ValueError(
            "projection geometry must be positive and row capacity fit Int32"
        )
    for name in (
        "source_contiguous",
        "source_aligned",
        "weight_contiguous",
        "weight_aligned",
        "output_contiguous",
        "output_aligned",
    ):
        if type(getattr(query, name)) is not bool:
            raise TypeError(f"projection {name} must be a boolean")


# Backends whose complete K reduction is an FP32 sum: the SIMT kernels add FP32
# products in a fixed order and the prefill GEMM adds 16-product tensor-core
# segments with compensated FP32 addition. The warp-MMA, small-row tensor-core
# and library backends accumulate longer runs inside tensor cores.
FP32_ACCUMULATION_BACKENDS = frozenset({"simt", "mixed_vector", "prefill"})


def _accumulation_admits(query, backend):
    return query.accumulation == "any" or backend in FP32_ACCUMULATION_BACKENDS


def _mma_eligible(query):
    return query.source_dtype == query.weight_dtype == "bfloat16"


def _mma_preferred(query):
    return query.out_features >= 256 and query.in_features >= 16


def _tc_eligible(query):
    from ._tensor_core import TC_K_MULTIPLE, TC_MAX_ROWS

    return (
        query.source_dtype == query.weight_dtype == "bfloat16"
        and query.output_dtype in ("bfloat16", "float32")
        and query.bias_dtype is None
        and query.max_rows <= TC_MAX_ROWS
        and query.in_features % TC_K_MULTIPLE == 0
        and query.source_contiguous
        and query.source_aligned
        and query.weight_contiguous
        and query.weight_aligned
        and query.output_contiguous
    )


def _torch_eligible(query):
    if query.source_dtype != "bfloat16" or query.weight_dtype != "bfloat16":
        return False
    if query.output_dtype == "bfloat16":
        return query.bias_dtype in (None, "bfloat16")
    import torch

    return (
        query.output_dtype == "float32"
        and query.bias_dtype is None
        and hasattr(torch.ops.aten.mm, "dtype_out")
    )


def _prefill_eligible(query):
    return (
        query.bias_dtype is None
        and query.source_dtype == query.weight_dtype == "bfloat16"
        and query.in_features % 64 == 0
        and query.source_contiguous
        and query.source_aligned
        and query.weight_contiguous
        and query.weight_aligned
        and query.output_contiguous
        and query.output_aligned
    )


def default_config(query, device):
    if query.accumulation == "fp32":
        if _prefill_eligible(query) and query.max_rows > 8:
            return GemvConfig(backend="prefill")
        return GemvConfig(backend="simt")
    if _torch_eligible(query) and query.max_rows > 8:
        return GemvConfig(backend="torch")
    n_tiles = (query.out_features + 63) // 64
    minimum_mma_rows = 24 if n_tiles >= 64 else ((64 + n_tiles - 1) // n_tiles) * 32
    if (
        _mma_eligible(query)
        and _mma_preferred(query)
        and query.max_rows >= minimum_mma_rows
    ):
        return GemvConfig(backend="mma")
    rows_per_tile = 8
    if (
        query.source_dtype == query.weight_dtype == "bfloat16"
        and query.in_features == 5120
        and query.out_features in (32, 384, 512)
        and query.max_rows <= 8
        and query.source_contiguous
        and query.source_aligned
        and query.weight_contiguous
        and query.weight_aligned
    ):
        rows_per_tile = 2 if query.out_features == 32 else 4
    return GemvConfig(backend="simt", rows_per_tile=rows_per_tile)


def _validate_query(query, device):
    validate_query(query)


def validate_config(query, config, device):
    if not isinstance(config, GemvConfig):
        raise TypeError("projection config must be GemvConfig")
    if not _accumulation_admits(query, config.backend):
        raise ValueError(
            f"projection backend {config.backend!r} does not carry the sum in FP32"
        )
    if config.backend in ("simt", "mixed_vector"):
        if config.backend == "mixed_vector" and not (
            query.source_dtype == "bfloat16" and query.weight_dtype == "float32"
        ):
            raise ValueError(
                "mixed vector projection requires BF16 source and FP32 weights"
            )
        if type(config.rows_per_tile) is not int or config.rows_per_tile not in (
            1,
            2,
            4,
            8,
        ):
            raise ValueError("SIMT projection rows_per_tile must be 1, 2, 4 or 8")
        if type(config.block_threads) is not int or config.block_threads not in (
            128,
            256,
        ):
            raise ValueError("SIMT projection block_threads must be 128 or 256")
        return
    if config.rows_per_tile != 8:
        raise ValueError("rows_per_tile only configures SIMT projection")
    if config.block_threads != 128:
        raise ValueError("block_threads only configures SIMT projection")
    if config.backend == "mma" and _mma_eligible(query):
        return
    if config.backend == "tc" and _tc_eligible(query):
        return
    if config.backend == "torch" and _torch_eligible(query):
        return
    if config.backend == "prefill" and _prefill_eligible(query):
        return
    raise ValueError(
        f"projection backend {config.backend!r} is ineligible for this query"
    )


def _parameters(query, device):
    backends = ["simt"]
    if query.source_dtype == "bfloat16" and query.weight_dtype == "float32":
        backends.append("mixed_vector")
    if _mma_eligible(query):
        backends.append("mma")
    if _tc_eligible(query):
        backends.append("tc")
    if _prefill_eligible(query):
        backends.append("prefill")
    if _torch_eligible(query):
        backends.append("torch")
    backends = [backend for backend in backends if _accumulation_admits(query, backend)]
    return ParameterSpace.create(
        TUNING.knobs,
        values={"backend": tuple(backends)},
        predicates=(_eligible,),
        # Without an accumulation requirement the prefill GEMM is raced only by
        # the audit: it never came within 2% of a winner that may accumulate in
        # tensor cores. Under FP32 accumulation it races the SIMT kernels.
        efficiency_predicates=(
            lambda choice: (
                (choice["backend"] != "mma" or _mma_preferred(query))
                and (choice["backend"] != "prefill" or query.accumulation == "fp32")
            ),
        ),
        autotune_effort=query.autotune_effort,
    )


def _eligible(choice):
    return choice["backend"] in ("simt", "mixed_vector") or (
        choice["rows_per_tile"] == 8 and choice["block_threads"] == 128
    )


TUNING = TuningContract(
    component_id="gemm.bf16_gemv",
    query_schema_version=24,
    config_schema_version=23,
    query_fields=frozenset(field.name for field in fields(GemvQuery)),
    config_fields=frozenset(field.name for field in fields(GemvConfig)),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda payload: GemvConfig(**dict(payload)),
    validate_query=_validate_query,
    validate_config=validate_config,
    default_config=default_config,
    knobs=(
        Knob(name="backend", values=None, binding=ParameterBinding.COMPILE),
        Knob(
            name="rows_per_tile", values=(1, 2, 4, 8), binding=ParameterBinding.COMPILE
        ),
        Knob(name="block_threads", values=(128, 256), binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=13,
    parameters=_parameters,
    materialize=lambda query, device, choice: GemvConfig(**dict(choice)),
)
