"""Complete per-invocation mHC defaults and legal projection choices."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation._efficiency import AutotuneEffort
from b12x.preparation.tuning import (
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)

_MHC_MULT = 4
_MIXES = 24
# TF32 projection overtakes the native fused path at about fifty tokens on
# SM120; long-M tile shapes start where the first TMA pipeline rules did.
_TF32_MIN_TOKENS = 52
_LONG_M_TOKENS = 384


@dataclass(frozen=True, kw_only=True)
class MhcQuery:
    dtype: str
    max_tokens: int
    hidden_size: int
    split_k: int
    operation: str
    has_norm_weight: bool = False
    norm_weight_dtype: str = "bfloat16"
    has_fn_bf16: bool = False
    collapse_weighted: bool = False
    # V4.1 feeds the coefficients predicted by one sublayer into the next.
    # This changes both the producer and finalizer ABI; it is not a runtime
    # optional operand on an ordinary MHC execution.
    lagged_mix: bool = False
    expanded_residual: bool = False
    bf16x2_eligible: bool = True
    output_mode: str = "provided"
    rms_eps: float = 1.0e-6
    hc_eps: float = 1.0e-6
    sinkhorn_iters: int = 20
    norm_eps: float = 0.0
    block_k: int = 256
    block_h: int = 512
    smem_limit: int = 0
    controls: FrozenMapping = FrozenMapping()
    codegen: FrozenMapping = FrozenMapping()


@dataclass(frozen=True, kw_only=True)
class MhcConfig:
    backend: str
    projection_tile_m: int
    projection_tile_n: int
    projection_tile_k: int
    projection_num_stages: int
    projection_num_m_warps: int
    projection_num_n_warps: int
    projection_k_splits: int
    lagged_prepare: bool = False
    partials_per_cta: int = 4

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "MhcConfig":
        expected = frozenset(cls.__dataclass_fields__)
        if frozenset(payload) != expected:
            raise ValueError(f"mHC config fields must be {sorted(expected)}")
        return cls(
            backend=str(payload["backend"]),
            projection_tile_m=int(payload["projection_tile_m"]),
            projection_tile_n=int(payload["projection_tile_n"]),
            projection_tile_k=int(payload["projection_tile_k"]),
            projection_num_stages=int(payload["projection_num_stages"]),
            projection_num_m_warps=int(payload["projection_num_m_warps"]),
            projection_num_n_warps=int(payload["projection_num_n_warps"]),
            projection_k_splits=int(payload["projection_k_splits"]),
            lagged_prepare=payload["lagged_prepare"],
            partials_per_cta=int(payload["partials_per_cta"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "projection_tile_m": self.projection_tile_m,
            "projection_tile_n": self.projection_tile_n,
            "projection_tile_k": self.projection_tile_k,
            "projection_num_stages": self.projection_num_stages,
            "projection_num_m_warps": self.projection_num_m_warps,
            "projection_num_n_warps": self.projection_num_n_warps,
            "projection_k_splits": self.projection_k_splits,
            "lagged_prepare": self.lagged_prepare,
            "partials_per_cta": self.partials_per_cta,
        }


def _tf32_config(
    *,
    backend: str = "tf32_tma",
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    num_m_warps: int,
    num_n_warps: int,
    k_splits: int,
) -> MhcConfig:
    return MhcConfig(
        backend=backend,
        projection_tile_m=tile_m,
        projection_tile_n=tile_n,
        projection_tile_k=tile_k,
        projection_num_stages=num_stages,
        projection_num_m_warps=num_m_warps,
        projection_num_n_warps=num_n_warps,
        projection_k_splits=k_splits,
    )


def _enough_split_work(query, splits):
    """Long M keeps at least 2048 K columns per split."""
    return query.max_tokens < 2048 or _MHC_MULT * query.hidden_size // splits >= 2048


def _grid_filling_splits(query, device, *, m_warps, tile_n, tile_k):
    """Largest legal K split whose grid fits the resident SMs."""
    if device is None:
        return 1
    total_k = _MHC_MULT * query.hidden_size
    sms = device.sm_count
    m_tiles = (query.max_tokens + 16 * m_warps - 1) // (16 * m_warps)
    n_tiles = (_MIXES + tile_n - 1) // tile_n
    legal = [
        splits
        for splits in range(1, max(2, query.split_k))
        if (total_k // tile_k) % splits == 0 and _enough_split_work(query, splits)
    ]
    fitting = [splits for splits in legal if m_tiles * n_tiles * splits <= sms]
    return max(fitting) if fitting else min(legal)


def _projection_default_config(
    query: MhcQuery,
    device: DeviceIdentity | None,
) -> MhcConfig:
    tokens = int(query.max_tokens)
    eligible = _tf32_eligible(query)
    pin = _tf32_pin(query)
    tf32 = eligible and (pin is True or (pin is None and tokens >= _TF32_MIN_TOKENS))
    if not tf32:
        return _tf32_config(
            backend="native",
            tile_m=16,
            tile_n=8,
            tile_k=256,
            num_stages=1,
            num_m_warps=1,
            num_n_warps=1,
            k_splits=1,
        )
    # Short M splits the N tile across two warps over 16- or 32-row M tiles;
    # long M keeps one N warp over whole-quad M tiles covering every mix. Both
    # keep three TMA stages in flight and fill the SMs through the K split.
    if tokens < _LONG_M_TOKENS:
        m_warps = 1 if tokens <= 64 else 2
        tile_n, n_warps, tile_k = 16, 2, 128
    else:
        m_warps = 4 if tokens < 2048 else 8
        tile_n, n_warps, tile_k = _MIXES, 1, 64
    return _tf32_config(
        tile_m=16 * m_warps,
        tile_n=tile_n,
        tile_k=tile_k,
        num_stages=3,
        num_m_warps=m_warps,
        num_n_warps=n_warps,
        k_splits=_grid_filling_splits(
            query, device, m_warps=m_warps, tile_n=tile_n, tile_k=tile_k
        ),
    )


def _tf32_eligible(query):
    return (
        query.has_norm_weight
        and query.hidden_size in (4096, 5120, 7168)
        and (
            query.operation == "post_pre"
            or (
                query.operation == "pre"
                and query.expanded_residual
                and query.lagged_mix
            )
        )
    )


def _default_config(query, device):
    config = _projection_default_config(query, device)
    threshold = int(query.controls.get("B12X_MHC_PREFILL_MIN_TOKENS", "96"))
    prepared = query.lagged_mix and config.backend == "native"
    if query.operation == "pre":
        prepared = prepared and query.max_tokens < threshold
    else:
        prepared = prepared and (
            not query.has_norm_weight or query.max_tokens < threshold
        )
        splits = query.controls.get("B12X_MHC_DECODE_SPLITS")
        if splits not in (None, ""):
            prepared = prepared and int(splits) == 0
        elif (
            device is not None
            and device.compute_capability == (12, 1)
            and query.hidden_size == 4096
            and query.max_tokens >= 8
        ):
            prepared = False
    partials = 4
    if prepared:
        raw = query.controls.get("B12X_MHC_PARTIALS_PER_CTA")
        if raw not in (None, ""):
            partials = int(raw)
        elif (
            query.operation == "pre"
            and device is not None
            and device.compute_capability == (12, 1)
            and query.hidden_size == 4096
        ):
            partials = (
                25 if query.max_tokens >= 8 else 9 if query.max_tokens >= 4 else 4
            )
    return replace(config, lagged_prepare=prepared, partials_per_cta=partials)


def _encode(query: MhcQuery) -> dict[str, object]:
    return {name: getattr(query, name) for name in query.__dataclass_fields__}


def _validate(
    query: MhcQuery,
    config: MhcConfig,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(config, MhcConfig):
        raise TypeError("config must be MhcConfig")
    if any(
        type(getattr(config, field)) is not int
        for field in config.__dataclass_fields__
        if field not in ("backend", "lagged_prepare")
    ):
        raise TypeError("projection geometry fields must be integers")
    if config.backend not in {"native", "tf32_tma", "tf32_fused"}:
        raise ValueError(f"unsupported mHC backend {config.backend!r}")
    if type(config.lagged_prepare) is not bool:
        raise TypeError("lagged_prepare must be a boolean")
    if config.lagged_prepare and (not query.lagged_mix or config.backend != "native"):
        raise ValueError("lagged preparation requires the native lagged route")
    if not 1 <= config.partials_per_cta <= 25:
        raise ValueError("partials_per_cta must be in [1, 25]")
    if not config.lagged_prepare and config.partials_per_cta != 4:
        raise ValueError("partial grouping requires the fused lagged producer")
    if config.backend == "tf32_fused":
        if config.projection_tile_m not in (16, 32) or config != _fused_config(
            query, config.projection_tile_m
        ):
            raise ValueError(
                "fused TF32 requires its declared row-tile projection geometry"
            )
        if query.hidden_size % 512:
            raise ValueError("fused TF32 requires hidden size divisible by 512")
        if not _fused_eligible(query):
            raise ValueError("fused TF32 requires expanded lagged mHC")
        return
    if config.backend == "native":
        return
    if not _tf32_eligible(query):
        raise ValueError(
            "TF32 requires post-pre with norm or expanded lagged pre with norm"
        )
    if config.projection_num_stages not in range(1, 9):
        raise ValueError("projection_num_stages must be in [1, 8]")
    if config.projection_num_m_warps <= 0 or config.projection_num_n_warps <= 0:
        raise ValueError("projection warp counts must be positive")
    if config.projection_tile_m != 16 * config.projection_num_m_warps:
        raise ValueError("projection_tile_m must equal 16 * projection_num_m_warps")
    if config.projection_tile_n <= 0 or config.projection_tile_n % 8:
        raise ValueError("projection_tile_n must be a positive multiple of 8")
    if config.projection_tile_n == 48:
        raise ValueError("TF32 TMA shared-memory layout cannot slice tile_n=48")
    n_mma_tiles = config.projection_tile_n // 8
    if n_mma_tiles % config.projection_num_n_warps:
        raise ValueError("projection_num_n_warps must divide projection_tile_n / 8")
    total_k = _MHC_MULT * query.hidden_size
    if config.projection_tile_k <= 0 or config.projection_tile_k % 8:
        raise ValueError("projection_tile_k must be a positive multiple of 8")
    if config.projection_tile_k < 32:
        raise ValueError(
            "TF32 TMA projection requires at least 32 FP32 weights per row"
        )
    if total_k % config.projection_tile_k:
        raise ValueError("projection_tile_k must divide the flattened hidden width")
    k_tiles = total_k // config.projection_tile_k
    if config.projection_k_splits <= 0 or k_tiles % config.projection_k_splits:
        raise ValueError("projection_k_splits must divide the projection K tiles")
    if config.projection_k_splits >= query.split_k:
        raise ValueError("projection_k_splits must be smaller than scratch split_k")
    compute_warps = config.projection_num_m_warps * config.projection_num_n_warps
    if (compute_warps + 1) * 32 > 1_024:
        raise ValueError("projection geometry exceeds the CUDA thread-block limit")


def _fused_eligible(query):
    return query.lagged_mix and (
        query.operation == "post_pre"
        or query.operation == "pre"
        and query.expanded_residual
    )


def _fused_config(query, tile_m=16):
    return MhcConfig(
        backend="tf32_fused",
        projection_tile_m=tile_m,
        projection_tile_n=24,
        projection_tile_k=128,
        projection_num_stages=1,
        projection_num_m_warps=tile_m // 16,
        projection_num_n_warps=3,
        projection_k_splits=query.hidden_size // 32,
    )


def _tf32_pin(query):
    value = query.controls.get("B12X_MHC_PREFILL_TF32_MMA")
    if value is None:
        value = query.controls.get("B12X_MHC_PREFILL_BF16_MMA")
    return None if value is None else value != "0"


def _validate_query(query, device):
    if not isinstance(query, MhcQuery):
        raise TypeError("query must be MhcQuery")
    _autotune_effort(query)
    if query.operation not in ("pre", "post", "post_pre", "collapse"):
        raise ValueError("unknown mHC operation")
    if query.dtype != "bfloat16":
        raise ValueError("mHC requires bfloat16 state")
    if any(
        type(value) is not int or value <= 0
        for value in (
            query.max_tokens,
            query.hidden_size,
            query.split_k,
            query.block_k,
            query.block_h,
        )
    ):
        raise ValueError("mHC dimensions must be positive integers")
    hidden_sizes = (4096, 5120, 7168)
    if query.hidden_size not in hidden_sizes:
        raise ValueError(f"{query.operation} supports hidden sizes {hidden_sizes}")
    if query.operation in ("pre", "post_pre"):
        if (
            4 * query.hidden_size % query.block_k
            or query.split_k != 4 * query.hidden_size // query.block_k
            or query.block_k != 256
            or query.block_h != 512
        ):
            raise ValueError("mHC requires the production fused Gram geometry")
        if type(query.sinkhorn_iters) is not int or query.sinkhorn_iters != 20:
            raise ValueError("mHC production kernels require 20 Sinkhorn iterations")
        if query.rms_eps not in (1.0e-20, 1.0e-6, 1.0e-5) or query.hc_eps != 1.0e-6:
            raise ValueError(
                "mHC epsilon values differ from the production numerical contract"
            )
    if query.has_norm_weight and (
        not math.isfinite(query.norm_eps)
        or query.norm_eps < 0
        or query.norm_weight_dtype not in ("bfloat16", "float32")
    ):
        raise ValueError(
            "normalization requires BF16/FP32 weights and nonnegative epsilon"
        )
    if query.output_mode not in ("functional", "provided"):
        raise ValueError("unknown mHC output form")
    if any(
        type(value) is not bool
        for value in (
            query.has_norm_weight,
            query.has_fn_bf16,
            query.collapse_weighted,
            query.bf16x2_eligible,
            query.lagged_mix,
            query.expanded_residual,
        )
    ):
        raise TypeError("operand-presence and layout flags must be boolean")
    if query.expanded_residual and query.operation != "pre":
        raise ValueError("expanded pre layout only applies to the pre operation")
    if query.lagged_mix and query.operation not in ("pre", "post_pre"):
        raise ValueError("lagged MHC supports only pre and post_pre operations")


def _autotune_effort(query: MhcQuery) -> AutotuneEffort:
    value = query.controls.get("B12X_AUTOTUNE_EFFORT")
    return AutotuneEffort.DEFAULT if value is None else AutotuneEffort(value)


def _tuning_parameters(query: MhcQuery, device: DeviceIdentity | None):
    operation = query.operation
    total_k = _MHC_MULT * query.hidden_size
    tf32 = _tf32_eligible(query)
    pin = _tf32_pin(query)
    backends = ("native", "tf32_tma") if tf32 else ("native",)
    if tf32 and pin is not None:
        backends = ("tf32_tma",) if pin else ("native",)
    if _fused_eligible(query) and pin is None:
        backends += ("tf32_fused",)
    partials = query.controls.get("B12X_MHC_PARTIALS_PER_CTA")
    values = {
        "backend": backends,
        "fused_tile_m": (16,) if query.max_tokens < 128 else (16, 32),
        "lagged_prepare": (False, True) if query.lagged_mix else (False,),
        "partials_per_cta": (int(partials),)
        if partials not in (None, "")
        else (4, 9, 13, 25),
        "projection_tile_k": tuple(
            k for k in (8, 16, 32, 64, 128, 256) if total_k % k == 0
        ),
        "projection_k_splits": tuple(
            splits
            for splits in range(1, max(2, query.split_k))
            if total_k % splits == 0
        ),
    }
    if "tf32_tma" not in backends:
        return ParameterSpace.create(
            TUNING.knobs, values=values, autotune_effort=_autotune_effort(query)
        )

    from ._kernels import MHCPrefillTf32ProjectTmaKernel

    smem_limit = query.smem_limit
    if type(smem_limit) is not int or smem_limit <= 0:
        raise ValueError(
            "TF32 preparation requires the declared device's shared-memory limit"
        )
    alignment = MHCPrefillTf32ProjectTmaKernel.buffer_align_bytes
    n_cover = 1 << (_MIXES - 1).bit_length()

    def shared_bytes(p):
        # Both TMA stage buffers and the barrier prefix are independently aligned.
        return sum(
            ((size + alignment - 1) // alignment) * alignment
            for size in (
                16 * p["projection_num_stages"],
                32
                * p["projection_num_m_warps"]
                * p["projection_tile_k"]
                * p["projection_num_stages"],
                4
                * p["projection_tile_n"]
                * p["projection_tile_k"]
                * p["projection_num_stages"],
            )
        )

    m_warp_values = next(
        knob.values for knob in TUNING.knobs if knob.name == "projection_num_m_warps"
    )

    def compact_m_grid(p):
        if p["backend"] != "tf32_tma":
            return True
        warps = p["projection_num_m_warps"]
        grid = (query.max_tokens + 16 * warps - 1) // (16 * warps)
        return not any(
            smaller < warps
            and (query.max_tokens + 16 * smaller - 1) // (16 * smaller) == grid
            for smaller in m_warp_values
        )

    sms = None if device is None else device.sm_count

    both_backends = "native" in backends and "tf32_tma" in backends

    def tf32_regime(p):
        # Native decode beats every TF32 geometry below this token count.
        return not both_backends or p["backend"] != "tf32_tma" or query.max_tokens >= 32

    def native_regime(p):
        return not both_backends or p["backend"] != "native" or query.max_tokens < 128

    def resident_grid(p):
        # Winning grids fill at least 40% of the SMs unless no legal K split
        # for the tile reaches that, and a K split never extends the grid
        # past one wave.
        if p["backend"] != "tf32_tma" or sms is None:
            return True
        tile_m = 16 * p["projection_num_m_warps"]
        grid = (
            ((query.max_tokens + tile_m - 1) // tile_m)
            * ((_MIXES + p["projection_tile_n"] - 1) // p["projection_tile_n"])
            * p["projection_k_splits"]
        )
        splits = p["projection_k_splits"]
        k_tiles = total_k // p["projection_tile_k"]
        largest = max(
            candidate
            for candidate in values["projection_k_splits"]
            if k_tiles % candidate == 0 and _enough_split_work(query, candidate)
        )
        filled = 2 * sms <= 5 * grid or splits == largest
        return filled and (splits == 1 or grid <= sms)

    def warp_layout(p):
        # Short M splits the N tile one MMA tile per warp; long M keeps one N
        # warp and draws its parallelism from M warps.
        if p["backend"] != "tf32_tma":
            return True
        if query.max_tokens >= _LONG_M_TOKENS:
            return p["projection_num_n_warps"] == 1
        return p["projection_num_n_warps"] == p["projection_tile_n"] // 8

    def tile_shape(p):
        # Short M favors small M tiles and N tiles without padding; long M
        # favors M tiles of whole warp quads and one N tile covering all mixes.
        if p["backend"] != "tf32_tma":
            return True
        m_warps = p["projection_num_m_warps"]
        if query.max_tokens >= _LONG_M_TOKENS:
            return m_warps % 4 == 0 and p["projection_tile_n"] >= _MIXES
        return m_warps <= 4 and p["projection_tile_n"] <= _MIXES

    def pipeline(p):
        # Overlap TMA with at least two stages; short M keeps more K columns
        # in flight per CTA because it has fewer M tiles to spread them over.
        if p["backend"] != "tf32_tma":
            return True
        stages = p["projection_num_stages"]
        buffered = stages * p["projection_tile_k"]
        floor = 128 if query.max_tokens >= _LONG_M_TOKENS else 192
        return p["projection_tile_k"] >= 64 and stages >= 2 and floor <= buffered <= 512

    def prefill_split_work(p):
        return p["backend"] != "tf32_tma" or _enough_split_work(
            query, p["projection_k_splits"]
        )

    return ParameterSpace.create(
        TUNING.knobs,
        values=values,
        autotune_effort=_autotune_effort(query),
        predicates=(
            # TMA alignment, complete MMA tiles, CTA limits and scratch bounds.
            lambda p: p["backend"] != "tf32_tma" or p["projection_tile_k"] >= 32,
            # CuTe aborts when slicing the FP32 swizzled layout with 48 rows.
            lambda p: p["backend"] != "tf32_tma" or p["projection_tile_n"] != 48,
            lambda p: (
                p["backend"] != "tf32_tma"
                or (p["projection_num_m_warps"] * p["projection_num_n_warps"] + 1) * 32
                <= 1024
            ),
            lambda p: (
                p["backend"] != "tf32_tma"
                or p["projection_tile_n"] % (8 * p["projection_num_n_warps"]) == 0
            ),
            lambda p: (
                p["backend"] != "tf32_tma"
                or (total_k // p["projection_tile_k"]) % p["projection_k_splits"] == 0
            ),
            lambda p: p["backend"] != "tf32_tma" or shared_bytes(p) <= smem_limit,
            # Larger M tiles with the same grid only add padding and storage.
            compact_m_grid,
            # Keep exact and power-of-two N coverage and avoid wholly masked warps.
            lambda p: p["backend"] != "tf32_tma" or p["projection_tile_n"] <= n_cover,
            lambda p: (
                p["backend"] != "tf32_tma"
                or p["projection_num_n_warps"]
                <= min(p["projection_tile_n"], _MIXES) // 8
            ),
            # More stages than iterations cannot buffer additional work.
            lambda p: (
                p["backend"] != "tf32_tma"
                or p["projection_num_stages"]
                <= total_k // (p["projection_tile_k"] * p["projection_k_splits"])
            ),
        ),
        # Scheduling, warp layout and pipeline guesses; the audit lifts them.
        efficiency_predicates=(
            tf32_regime,
            native_regime,
            resident_grid,
            warp_layout,
            tile_shape,
            pipeline,
            prefill_split_work,
        ),
    )


def _materialize_tuning(query, device, choice):
    if choice["backend"] == "tf32_fused":
        return _fused_config(query, choice["fused_tile_m"])
    values = dict(choice)
    values.pop("fused_tile_m", None)
    # The production M tile is fixed by its MMA warp layout.
    values["projection_tile_m"] = 16 * values["projection_num_m_warps"]
    config = MhcConfig.from_config(FrozenMapping(values))
    return config


TUNING = TuningContract(
    component_id="norm.mhc",
    query_schema_version=22,
    config_schema_version=22,
    query_fields=frozenset(MhcQuery.__dataclass_fields__),
    config_fields=frozenset(MhcConfig.__dataclass_fields__),
    encode_query=_encode,
    encode_config=MhcConfig.to_dict,
    decode_config=MhcConfig.from_config,
    default_config=_default_config,
    validate_query=_validate_query,
    validate_config=_validate,
    candidate_contract_version=30,
    knobs=(
        Knob(
            name="backend",
            values=("native", "tf32_tma", "tf32_fused"),
            binding=ParameterBinding.COMPILE,
        ),
        Knob(
            name="fused_tile_m",
            values=(16, 32),
            binding=ParameterBinding.COMPILE,
            when=FrozenMapping({"backend": "tf32_fused"}),
            otherwise=16,
        ),
        Knob(
            name="lagged_prepare",
            values=None,
            binding=ParameterBinding.COMPILE,
            when=FrozenMapping({"backend": "native"}),
            otherwise=False,
        ),
        Knob(
            name="partials_per_cta",
            values=None,
            binding=ParameterBinding.COMPILE,
            when=FrozenMapping({"backend": "native", "lagged_prepare": True}),
            otherwise=4,
        ),
        Knob(
            name="projection_tile_n",
            values=(8, 16, 24, 32, 48, 64),
            binding=ParameterBinding.COMPILE,
            when=FrozenMapping({"backend": "tf32_tma"}),
            otherwise=8,
        ),
        Knob(
            name="projection_tile_k",
            values=None,
            binding=ParameterBinding.COMPILE,
            when=FrozenMapping({"backend": "tf32_tma"}),
            otherwise=256,
        ),
        Knob(
            name="projection_num_stages",
            values=(1, 2, 3, 4),
            binding=ParameterBinding.COMPILE,
            when=FrozenMapping({"backend": "tf32_tma"}),
            otherwise=1,
        ),
        Knob(
            name="projection_num_m_warps",
            values=(1, 2, 3, 4, 6, 8, 12, 16),
            binding=ParameterBinding.COMPILE,
            when=FrozenMapping({"backend": "tf32_tma"}),
            otherwise=1,
        ),
        Knob(
            name="projection_num_n_warps",
            values=(1, 2, 3, 4),
            binding=ParameterBinding.COMPILE,
            when=FrozenMapping({"backend": "tf32_tma"}),
            otherwise=1,
        ),
        Knob(
            name="projection_k_splits",
            values=None,
            binding=ParameterBinding.COMPILE,
            when=FrozenMapping({"backend": "tf32_tma"}),
            otherwise=1,
        ),
    ),
    parameters=_tuning_parameters,
    materialize=_materialize_tuning,
)


__all__ = ["MhcConfig", "MhcQuery", "TUNING"]
