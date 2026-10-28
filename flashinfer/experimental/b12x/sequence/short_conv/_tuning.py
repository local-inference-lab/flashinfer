"""Packed-request convolution geometry, tile tactics and launch layouts.

Prefill and mixed plans with at least ``TILED_MIN_TOKENS`` rows convolve tiles of
tokens and channels. The tile shape and warp count are raced tactics; every
candidate computes each output element with the same operations in the same
order, so all candidates are bitwise identical. Decode plans and plans below
the tiling threshold keep one fixed geometry.
"""

from dataclasses import dataclass, field

from b12x.preparation import (
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)
from b12x.preparation._efficiency import AutotuneEffort, capture_autotune_effort

TILED_MIN_TOKENS = 256
BACKEND = "triton"
BLOCK_T_CHOICES = (8, 16, 32, 64)
BLOCK_C_CHOICES = (64, 128, 256)
NUM_WARPS_CHOICES = (4, 8)
# Per-thread FP32 values of the convolution tile that the compiler keeps live at
# once: the accumulator and every unrolled tap. The remaining registers of the
# 255-register thread limit hold masks, row addresses and metadata.
_LIVE_REGISTER_BUDGET = 192
_REGISTER_LIMIT = 255
_WARP_SIZE = 32

OPERANDS = (
    "x",
    "conv_weight",
    "query_start_loc",
    "state_slot_ids",
    "state_is_fresh",
    "num_accepted_tokens",
    "request_is_prefill",
    "num_seqs",
    "num_tokens",
    "conv_state",
    "out",
    "gathered_state",
    "request_ids",
    "conv_bias",
)


@dataclass(frozen=True, kw_only=True)
class ShortConvQuery:
    mode: str
    channels: int
    kernel_size: int
    dilation: int
    max_tokens: int
    max_seqs: int
    max_state_slots: int
    max_speculative_tokens: int
    round_before_silu: bool = True
    bias_dtype: str | None = None
    weight_dtype: str = "bfloat16"
    state_strides: tuple[int, int, int] | None = None
    input_row_stride: int | None = None
    input_alignments: tuple[int, ...] = (16,) * len(OPERANDS)
    autotune_effort: str = field(
        default_factory=lambda: capture_autotune_effort().value
    )

    def __post_init__(self):
        object.__setattr__(
            self, "autotune_effort", AutotuneEffort(self.autotune_effort).value
        )
        if (
            self.bias_dtype not in (None, "bfloat16", "float32")
            or type(self.round_before_silu) is not bool
        ):
            raise ValueError(
                "short convolution requires BF16/FP32 bias and a boolean rounding contract"
            )
        if self.weight_dtype not in ("bfloat16", "float32"):
            raise ValueError("short convolution weights require BF16 or FP32")
        if self.mode not in ("decode", "prefill", "mixed"):
            raise ValueError("short convolution mode must be decode, prefill, or mixed")
        if any(
            type(value) is not int or not 0 < value < 2**31
            for value in (
                self.channels,
                self.kernel_size,
                self.dilation,
                self.max_tokens,
                self.max_seqs,
                self.max_state_slots,
            )
        ):
            raise ValueError(
                "short convolution dimensions must be positive Int32 integers"
            )
        if self.kernel_size < 2:
            raise ValueError("short convolution requires at least two taps")
        if (
            type(self.max_speculative_tokens) is not int
            or self.max_speculative_tokens < 0
        ):
            raise ValueError(
                "short convolution speculative capacity must be nonnegative"
            )
        input_stride = (
            self.channels if self.input_row_stride is None else self.input_row_stride
        )
        if type(input_stride) is not int or input_stride < self.channels:
            raise ValueError(
                "short convolution input rows must have a nonoverlapping positive stride"
            )
        if input_stride * self.max_tokens >= 2**63:
            raise ValueError(
                "short convolution input exceeds the Int64 indexing domain"
            )
        object.__setattr__(self, "input_row_stride", input_stride)
        capacity = self.state_capacity
        if capacity >= 2**31:
            raise ValueError(
                "short convolution history exceeds the Int32 indexing domain"
            )
        strides = (
            (self.channels * capacity, capacity, 1)
            if self.state_strides is None
            else tuple(self.state_strides)
        )
        if len(strides) != 3 or any(type(s) is not int or s <= 0 for s in strides):
            raise ValueError(
                "short convolution history requires three positive strides"
            )
        if strides[1:] != (capacity, 1) or strides[0] < self.channels * capacity:
            raise ValueError(
                "short convolution history must be dense within nonoverlapping slots"
            )
        if strides[0] * self.max_state_slots >= 2**63:
            raise ValueError("short convolution pool exceeds the Int64 indexing domain")
        alignments = tuple(self.input_alignments)
        if len(alignments) != len(OPERANDS) or any(
            type(a) is not int or a not in (1, 2, 4, 8, 16) for a in alignments
        ):
            raise ValueError("short convolution requires every operand's alignment")
        object.__setattr__(self, "state_strides", strides)
        object.__setattr__(self, "input_alignments", alignments)

    @property
    def tiled(self):
        """Whether the convolution launches token-and-channel tiles."""
        return self.mode != "decode" and self.max_tokens >= TILED_MIN_TOKENS

    @property
    def int32_addressing(self):
        """Whether every tiled-kernel offset, including partial tiles, fits Int32.

        The bounds extend the live extents by the largest token and channel tile
        so that lanes past the capacities, which are masked, also stay in range.
        The tiled kernel addresses the input rows, the output rows, the gathered
        history scratch and the weights. A masked lane reaches a history position
        up to ``state_length + rows`` and an input row down to ``-state_length``
        (a negative offset of magnitude ``state_length * input_row_stride``).
        """
        rows = self.max_tokens + max(BLOCK_T_CHOICES)
        columns = self.channels + max(BLOCK_C_CHOICES)
        limit = 2**31
        return (
            rows * self.input_row_stride + columns < limit
            and self.state_length * self.input_row_stride < limit
            and rows * self.channels + columns < limit
            and (self.max_seqs * self.channels + columns) * self.state_length
            + self.state_length
            + rows
            < limit
            and columns * self.kernel_size < limit
        )

    @property
    def state_length(self):
        return self.dilation * (self.kernel_size - 1)

    @property
    def state_capacity(self):
        return self.state_length + self.max_speculative_tokens


@dataclass(frozen=True, kw_only=True)
class ShortConvConfig:
    """Token tile, channel tile, warp count and address width of the launch.

    ``block_t == 1`` is the per-token kernel of untiled plans. ``wide_index``
    selects Int64 address arithmetic; Int32 is legal only for queries whose
    offsets all fit below 2**31 (``ShortConvQuery.int32_addressing``).
    """

    backend: str
    block_t: int
    block_c: int
    num_warps: int
    wide_index: bool

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "ShortConvConfig":
        if (
            set(payload)
            != {
                "backend",
                "block_t",
                "block_c",
                "num_warps",
                "wide_index",
            }
            or type(payload["wide_index"]) is not bool
        ):
            raise ValueError(
                "short convolution config fields must be backend, block_t, "
                "block_c, num_warps, wide_index"
            )
        return cls(
            backend=str(payload["backend"]),
            block_t=int(payload["block_t"]),
            block_c=int(payload["block_c"]),
            num_warps=int(payload["num_warps"]),
            wide_index=payload["wide_index"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "block_t": self.block_t,
            "block_c": self.block_c,
            "num_warps": self.num_warps,
            "wide_index": self.wide_index,
        }


FIXED = ShortConvConfig(
    backend=BACKEND, block_t=1, block_c=128, num_warps=4, wide_index=True
)
# The 32 x 128 tile of four warps with Int64 addressing: the tile of every tiled
# plan before tile tactics existed. It is the default of tiled plans and a member
# of every candidate set.
FIXED_TILE = ShortConvConfig(
    backend=BACKEND, block_t=32, block_c=128, num_warps=4, wide_index=True
)


def live_values_per_thread(config: ShortConvConfig, kernel_size: int) -> int:
    """FP32 tile values held per thread: the accumulator plus each unrolled tap."""
    tile = config.block_t * config.block_c
    return -(-tile // (_WARP_SIZE * config.num_warps)) * (kernel_size + 1)


def _encode_query(query: ShortConvQuery) -> dict[str, object]:
    # The state slot count sizes the caller's pool and cannot change which tile
    # is fastest, so it stays out of the selection key.
    return {name: getattr(query, name) for name in _KEY_NAMES}


_KEY_NAMES = tuple(
    name for name in ShortConvQuery.__dataclass_fields__ if name != "max_state_slots"
)
_KEY_FIELDS = frozenset(_KEY_NAMES)


def program_payload(query: ShortConvQuery) -> dict[str, object]:
    """Complete query that rebuilds the compiled programs; effort never changes them."""
    return {
        name: "default" if name == "autotune_effort" else getattr(query, name)
        for name in ShortConvQuery.__dataclass_fields__
    }


def _validate_query(query: ShortConvQuery, device) -> None:
    del device
    if not isinstance(query, ShortConvQuery):
        raise TypeError("query must be ShortConvQuery")


def _validate_config(query: ShortConvQuery, config: ShortConvConfig, device) -> None:
    if not isinstance(config, ShortConvConfig):
        raise TypeError("config must be ShortConvConfig")
    if config.backend != BACKEND:
        raise ValueError(f"unsupported short convolution backend {config.backend!r}")
    if not query.tiled:
        if config != FIXED:
            raise ValueError(
                "decode and untiled short convolution plans use the fixed geometry"
            )
        return
    if config == FIXED_TILE:
        return
    if not config.wide_index and not query.int32_addressing:
        raise ValueError("Int32 addressing requires every offset below 2**31")
    if (
        config.block_t not in BLOCK_T_CHOICES
        or config.block_c not in BLOCK_C_CHOICES
        or config.num_warps not in NUM_WARPS_CHOICES
    ):
        raise ValueError("short convolution tile is outside the supported geometry")
    live = live_values_per_thread(config, query.kernel_size)
    if live > _LIVE_REGISTER_BUDGET:
        raise ValueError(
            f"short convolution tile keeps {live} live values per thread; "
            f"the budget is {_LIVE_REGISTER_BUDGET} of {_REGISTER_LIMIT} registers"
        )


def _default_config(query: ShortConvQuery, device) -> ShortConvConfig:
    return FIXED_TILE if query.tiled else FIXED


def _tuning_parameters(query: ShortConvQuery, device):
    if not query.tiled:
        return {
            "block_t": (FIXED.block_t,),
            "block_c": (FIXED.block_c,),
            "num_warps": (FIXED.num_warps,),
            "wide_index": (FIXED.wide_index,),
        }
    # A channel tile more than twice the channel count is mostly masked lanes.
    widest = max(FIXED_TILE.block_c, 1 << (query.channels - 1).bit_length())
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "block_t": BLOCK_T_CHOICES,
            "block_c": tuple(c for c in BLOCK_C_CHOICES if c <= widest),
            "num_warps": NUM_WARPS_CHOICES,
            "wide_index": (True, False) if query.int32_addressing else (True,),
        },
        autotune_effort=query.autotune_effort,
    )


TUNING = TuningContract(
    component_id="sequence.short_conv",
    query_schema_version=23,
    config_schema_version=23,
    query_fields=_KEY_FIELDS,
    config_fields=frozenset(ShortConvConfig.__dataclass_fields__),
    encode_query=_encode_query,
    encode_config=ShortConvConfig.to_dict,
    decode_config=ShortConvConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    knobs=(
        Knob(name="backend", values=(BACKEND,), binding=ParameterBinding.COMPILE),
        Knob(name="block_t", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="block_c", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="num_warps", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="wide_index", values=None, binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=2,
    parameters=_tuning_parameters,
)
