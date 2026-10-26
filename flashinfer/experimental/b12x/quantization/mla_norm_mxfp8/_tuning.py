"""Numerical and launch contract for split MLA normalization and MXFP8 encoding."""

from dataclasses import asdict, dataclass, fields
from b12x.preparation import Knob, ParameterBinding, TuningContract


@dataclass(frozen=True, kw_only=True)
class Query:
    rows: int
    norm_weight_dtype: str
    query_features: int = 1280
    kv_features: int = 512
    epsilon: float = 1e-20
    min_amax: float = 1e-4
    affine_prefetch_ready: bool = False


@dataclass(frozen=True)
class Config:
    programmatic_serialization: bool = False


def validate_query(query, device):
    if not isinstance(query, Query):
        raise TypeError("MLA normalization requires Query")
    if type(query.affine_prefetch_ready) is not bool:
        raise TypeError("affine prefetch readiness must be boolean")
    if (
        type(query.rows) is not int
        or not 0 < query.rows <= 2**31 - 1
        or query.norm_weight_dtype not in ("bfloat16", "float32")
        or (query.query_features, query.kv_features) != (1280, 512)
        or query.epsilon != 1e-20
        or query.min_amax != 1e-4
    ):
        raise ValueError(
            "MLA normalization requires BF16 rows, 1280+512 features, epsilon=1e-20 and min_amax=1e-4"
        )


def validate_config(query, config, device):
    if (
        not isinstance(config, Config)
        or type(config.programmatic_serialization) is not bool
    ):
        raise TypeError(
            "MLA normalization launch requires a boolean programmatic-serialization setting"
        )
    if config.programmatic_serialization and not query.affine_prefetch_ready:
        raise ValueError("programmatic affine prefetch requires declared readiness")


TUNING = TuningContract(
    component_id="quantization.mla_norm_mxfp8",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(f.name for f in fields(Query)),
    config_fields=frozenset({"programmatic_serialization"}),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda p: Config(**dict(p)),
    validate_query=validate_query,
    validate_config=validate_config,
    default_config=lambda q, d: Config(),
    candidate_contract_version=1,
    knobs=(
        Knob(
            name="programmatic_serialization",
            values=(False, True),
            binding=ParameterBinding.COMPILE,
        ),
    ),
    parameters=lambda q, d: {
        "programmatic_serialization": (False, True)
        if q.affine_prefetch_ready
        else (False,)
    },
)
