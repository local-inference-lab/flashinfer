"""Logical PCIe operations with native transport and launch candidates."""

from dataclasses import asdict, dataclass, replace
from functools import lru_cache

from b12x.preparation import (
    AutotuneEffort,
    CollectiveBucket,
    CollectiveGrid,
    FixedMemoryRequirements,
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    Plan,
    TuningContract,
)
from b12x.preparation._efficiency import capture_autotune_effort

from ._tuning import (
    PAIRWISE_THREADS,
    PAIRWISE_BLOCKS,
    pairwise_device,
    PcieQuery,
    TUNING as NATIVE,
)


# Compiled choices for local push, cross-island scatter/gather, cross-island DMA, and cross-island push.
ISLAND_VARIANTS = {
    "island_rs": (False, False, False, False),
    "island_rsag": (False, True, False, False),
    "island_push": (True, False, False, False),
    "island_push_rsag": (True, True, False, False),
    "island_hybrid_rsag": (True, True, True, False),
    "island_allpush_full": (True, False, False, True),
    "island_allpush_rsag": (True, True, False, True),
}


@dataclass(frozen=True, kw_only=True)
class CollectiveConfig:
    implementation: str
    threads: int
    blocks: int
    submode: str = ""
    active_blocks: int = 0


@dataclass(frozen=True, kw_only=True)
class CollectiveQuery:
    operation: str
    world_size: int
    rank: int
    alternatives: FrozenMapping
    default: CollectiveConfig
    inplace: bool = False


def _configs(query, device=None):
    for name, payload in query.alternatives.items():
        call = payload["call"]
        blocks = int(call.get("blocks", call.get("block_limit", 16)))
        threads = int(call["threads"])
        if name in ("dcp_route_separate", "dcp_route_row"):
            if name == "dcp_route_separate" or pairwise_device(device):
                yield CollectiveConfig(
                    implementation=name, threads=threads, blocks=blocks
                )
            continue
        if name in ISLAND_VARIANTS:
            # Physical CTAs keep buffer parity synchronized. Only participating
            # CTAs advance communication epochs, so active counts can vary safely.
            counts = (
                (threads,)
                if payload["setup"].get("threads_override", False)
                else tuple(dict.fromkeys((threads, 128, 256, 512, 768, 1024)))
            )
            fixed = any(
                payload["setup"].get(key, False)
                for key in (
                    "threads_override",
                    "blocks_override",
                    "active_blocks_override",
                )
            )
            active_counts = (
                (int(call["active_blocks"]),) if fixed else range(1, blocks + 1)
            )
            for count in counts:
                for active in active_counts:
                    yield CollectiveConfig(
                        implementation=name,
                        threads=count,
                        blocks=blocks,
                        active_blocks=active,
                    )
            continue
        if name in ("dcp_pairwise", "twoshot_pairwise"):
            if pairwise_device(device):
                yield CollectiveConfig(
                    implementation=name,
                    threads=PAIRWISE_THREADS,
                    blocks=PAIRWISE_BLOCKS,
                )
            continue
        if name in (
            "dcp_phased",
            "dcp_phased_push",
            "dcp_ring",
            "dcp_switch_push",
            "dcp_recursive",
            "dcp_recursive_ready",
            "dcp_recursive_outer_first",
            "dcp_recursive_pix_last",
        ):
            # Selected profiles require SM120 hardware with at least 64 SMs.
            # Phased kernels also have to pass the native cluster occupancy check.
            if not pairwise_device(device):
                continue
            if payload["setup"].get("threads_override", 0):
                counts = (threads,)
            elif name == "dcp_ring":
                counts = (128, 512)
            elif name in (
                "dcp_phased_push",
                "dcp_recursive",
                "dcp_recursive_ready",
                "dcp_recursive_outer_first",
            ):
                counts = (128,)
            else:
                counts = (128, 256, 512)
            is_dcp8_pix = (
                name == "dcp_recursive_pix_last"
                and query.world_size == 8
                and not payload["setup"].get("blocks_override", 0)
            )
            is_tp16_ring16 = (
                name == "dcp_ring"
                and query.world_size == 16
                and int(call.get("block_limit", 8)) == 16
            )
            phys_blocks = 16 if (is_dcp8_pix or is_tp16_ring16) else 8
            active_choices = (
                (8, 16) if is_dcp8_pix else ((16,) if is_tp16_ring16 else (8,))
            )
            for active in active_choices:
                for count in counts:
                    yield CollectiveConfig(
                        implementation=name,
                        threads=count,
                        blocks=phys_blocks,
                        submode="barrier",
                        active_blocks=active,
                    )
            continue
        yield CollectiveConfig(implementation=name, threads=threads, blocks=blocks)
        if (
            query.world_size in (2, 3, 4, 6, 8, 10)
            and query.operation == "all_reduce"
            and call.get("stage_input")
            and call.get("dtype") in ("float16", "bfloat16")
            and name
            in (
                "pull",
                "tp2_remote_push",
                "tp2_remote_push_simple",
                "remote_push_ll",
                "remote_push_simple",
                "remote_push_full",
                "remote_push_rsag",
            )
        ):
            from .pcie_oneshot import _MAX_BLOCKS

            max_blocks = _MAX_BLOCKS
            if name in ("remote_push_full", "remote_push_rsag"):
                max_blocks = max(
                    _MAX_BLOCKS,
                    int(payload["setup"].get("payload_ready_max_blocks", _MAX_BLOCKS)),
                )
            base_grids = (1, 2, 4, 8, 16, 32, _MAX_BLOCKS)
            if max_blocks > _MAX_BLOCKS:
                base_grids = (*base_grids, max_blocks)
            units = int(call["size_packs"]) * (2 if name == "tp2_remote_push" else 1)
            for count_threads in (32, 64, 128, 256, 512):
                grids = (
                    *base_grids,
                    min(max_blocks, max(1, -(-units // count_threads))),
                )
                for count_blocks in dict.fromkeys(grids):
                    yield CollectiveConfig(
                        implementation=name, threads=count_threads, blocks=count_blocks
                    )
        elif name == "dcp_word":
            from math import prod

            element_bytes = 1 if call["dtype"] == "torch.float8_e4m3fn" else 2
            chunks = (prod(call["shape"]) * element_bytes // 4 + 63) // 64
            setup = payload["setup"]
            thread_counts = (
                (threads,) if setup.get("threads_override", 0) else (32, 64, 128, 256)
            )
            for count_threads in thread_counts:
                warps = count_threads // 32
                if warps > 1 << (chunks - 1).bit_length():
                    continue
                grids = (
                    (blocks,)
                    if setup.get("blocks_override", 0)
                    else sorted({1, (chunks + warps - 1) // warps})
                )
                for count_blocks in grids:
                    yield CollectiveConfig(
                        implementation=name, threads=count_threads, blocks=count_blocks
                    )
        elif name == "twoshot":
            from math import prod

            packs = prod(call["shape"]) // 8
            shard = -(-packs // query.world_size)
            for count_threads in (32, 64, 128, 256, 512):
                for count in dict.fromkeys(
                    (
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        min(64, max(1, -(-shard // count_threads))),
                    )
                ):
                    yield CollectiveConfig(
                        implementation=name, threads=count_threads, blocks=count
                    )
        elif name == "dcp":
            setup = payload["setup"]
            thread_values = (
                (threads,)
                if setup["threads_override"]
                else (
                    (128, 256, 512, 1024)
                    if query.operation == "all_gather_heads" and query.world_size == 2
                    else (128, 256, 512)
                )
            )
            block_values = (blocks,) if setup["blocks_override"] else (8, 16, 32, 64)
            for threads in thread_values:
                for count in block_values:
                    yield CollectiveConfig(
                        implementation=name, threads=threads, blocks=count
                    )


def _efficient_plain_grid(query, value):
    name = value["implementation"]
    if query.operation != "all_reduce" or name not in (
        "pull",
        "tp2_remote_push",
        "tp2_remote_push_simple",
        "remote_push_ll",
        "remote_push_simple",
        "remote_push_full",
        "remote_push_rsag",
    ):
        return True
    call = query.alternatives[name]["call"]
    if not call.get("stage_input") or call.get("dtype") not in ("float16", "bfloat16"):
        return True
    from .pcie_oneshot import _MAX_BLOCKS

    max_blocks = _MAX_BLOCKS
    if name in ("remote_push_full", "remote_push_rsag"):
        max_blocks = max(
            _MAX_BLOCKS,
            int(
                query.alternatives[name]["setup"].get(
                    "payload_ready_max_blocks", _MAX_BLOCKS
                )
            ),
        )
    # Retain maximum grids at every thread count even for arbitrarily large payloads.
    if value["blocks"] == max_blocks:
        return True
    units = int(call["size_packs"]) * (2 if name == "tp2_remote_push" else 1)
    # Only exclude extreme serialization; idle blocks can affect synchronization latency.
    return units <= 1024 * value["threads"] * value["blocks"]


def _parameters(query, device):
    configs = tuple(dict.fromkeys(_configs(query, device)))
    effort = capture_autotune_effort()
    if effort is AutotuneEffort.OFF:
        eligible = {config.implementation for config in configs}
        configs = tuple(
            dict.fromkeys(
                CollectiveConfig(
                    implementation=name,
                    threads=int(payload["call"]["threads"]),
                    submode=payload["call"].get("submode", ""),
                    active_blocks=int(payload["call"].get("active_blocks", 0)),
                    blocks=int(
                        payload["call"].get(
                            "blocks", payload["call"].get("block_limit", 16)
                        )
                    ),
                )
                for name, payload in query.alternatives.items()
                if name in eligible
            )
        )
    legal_configs = frozenset(configs)
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "implementation": tuple(dict.fromkeys(c.implementation for c in configs)),
            "threads": tuple(dict.fromkeys(c.threads for c in configs)),
            "blocks": tuple(dict.fromkeys(c.blocks for c in configs)),
            "submode": tuple(dict.fromkeys(c.submode for c in configs)),
            "active_blocks": tuple(dict.fromkeys(c.active_blocks for c in configs)),
        },
        predicates=(lambda value: CollectiveConfig(**value) in legal_configs,),
        # Plain reductions measure pull alongside the payload-ready transports.
        efficiency_predicates=(
            lambda value: (
                value["implementation"] not in ISLAND_VARIANTS
                or value["active_blocks"] == query.default.active_blocks
                or value["active_blocks"] & (value["active_blocks"] - 1) == 0
            ),
            lambda value: (
                value["implementation"] != "pull"
                or query.default.implementation == "pull"
                or query.operation == "all_reduce"
            ),
            lambda value: (
                effort is AutotuneEffort.OFF or _efficient_plain_grid(query, value)
            ),
        ),
        autotune_effort=effort,
    )


def _validate_query(query, device):
    if not isinstance(query, CollectiveQuery) or not query.alternatives:
        raise ValueError("collective query requires native alternatives")
    if (
        query.world_size not in (2, 3, 4, 6, 8, 9, 10, 12, 16)
        or not 0 <= query.rank < query.world_size
    ):
        raise ValueError("collective tuning requires a supported world size and rank")
    for name, payload in query.alternatives.items():
        native = PcieQuery(**payload)
        NATIVE.validate_query(native, device)
        if (native.world_size, native.rank) != (query.world_size, query.rank):
            raise ValueError("native alternatives must use the same rank group")
        if name in ("dcp_route_separate", "dcp_route_row") and (
            query.operation != "all_gather_pair_kimi_topk"
            or query.world_size not in (8, 16)
            or native.call.get("route_transport")
            != {"dcp_route_separate": "separate", "dcp_route_row": "row"}[name]
        ):
            raise ValueError("Kimi row gathering requires a complete four-row route")
        if name == "twoshot_pairwise" and (
            query.operation != "all_reduce"
            or native.surface != "PCIeTwoShotBF16.all_reduce"
            or native.call.get("payload_schedule") != "pairwise"
        ):
            raise ValueError(
                "pairwise two-shot tuning requires its TP4 native schedule"
            )
        if name == "dcp_pairwise" and (
            query.operation != "lse_reduce_scatter"
            or native.call.get("payload_schedule") != "pairwise"
        ):
            raise ValueError("pairwise LSE tuning requires its DCP4 native schedule")
        if name in ISLAND_VARIANTS and (
            query.operation != "all_reduce"
            or query.world_size != 16
            or native.surface != "PCIeIslandRSAllReduce.all_reduce"
        ):
            raise ValueError("island launch tuning requires its TP16 reduction owner")
        if name in ISLAND_VARIANTS:
            from ._island_rs_cute import island_rs_peers

            if (
                bool(native.call.get("local_push", False)),
                bool(native.call.get("cross_island_rsag", False)),
                bool(native.call.get("cross_island_dma", False)),
                bool(native.call.get("cross_island_push", False)),
            ) != ISLAND_VARIANTS[name]:
                raise ValueError(
                    "island implementation differs from its compiled protocol"
                )

            if tuple(native.setup["peer_ranks"]) != island_rs_peers(
                native.rank, native.world_size
            ):
                raise ValueError(
                    "island tuning peers differ from the rank's reduction topology"
                )
        if name in (
            "dcp_phased",
            "dcp_phased_push",
            "dcp_ring",
            "dcp_switch_push",
            "dcp_recursive",
            "dcp_recursive_ready",
            "dcp_recursive_outer_first",
            "dcp_recursive_pix_last",
        ) and (
            query.operation != "all_gather_heads"
            or (
                query.world_size not in (8, 16)
                and not (
                    query.world_size == 4
                    and name
                    in ("dcp_ring", "dcp_switch_push", "dcp_recursive_pix_last")
                )
            )
            or (name == "dcp_recursive_outer_first" and query.world_size != 16)
            or (name == "dcp_switch_push" and query.world_size != 4)
            or not native.surface.endswith(".all_gather_heads")
            or native.call.get("gather_transport")
            != {
                "dcp_ring": "selected_ring",
                "dcp_switch_push": "selected_switch_push",
                "dcp_recursive": "selected_recursive",
                "dcp_recursive_ready": "selected_recursive_ready",
                "dcp_recursive_outer_first": "selected_recursive_outer_first",
                "dcp_recursive_pix_last": "selected_recursive_pix_last",
                "dcp_phased": "selected_phased_barrier",
                "dcp_phased_push": "selected_phased_push",
            }[name]
        ):
            raise ValueError("selected gather tuning requires its DCP barrier profile")


@lru_cache(maxsize=1)
def _legal_configs(query, device):
    return frozenset(_configs(query, device))


def _validate_config(query, config, device):
    if not isinstance(config, CollectiveConfig) or config not in _legal_configs(
        query, device
    ):
        raise ValueError(
            "configuration is outside the native collective parameter space"
        )


def _equivalence_key(query, device, config):
    if config.implementation == "twoshot":
        from math import prod

        packs = prod(query.alternatives["twoshot"]["call"]["shape"]) // 8
        shard = -(-packs // query.world_size)
        return (
            config.implementation,
            config.threads,
            min(config.blocks, max(1, -(-shard // config.threads))),
        )
    if config.implementation != "dcp":
        return asdict(config)
    native = query.alternatives["dcp"]
    heads = native["setup"]["total_heads"]
    if query.operation == "lse_reduce_scatter":
        heads //= query.world_size
    rows = native["call"]["shape"][0] * heads
    warps = config.threads // 32
    return (
        config.implementation,
        config.threads,
        min(config.blocks, (rows + warps - 1) // warps),
    )


def _collective_signature(query):
    alternatives = {}
    for name, payload in query.alternatives.items():
        native = payload.to_dict()
        del native["rank"]
        for section in ("call", "setup"):
            native[section].pop("device_index", None)
        if name in ISLAND_VARIANTS:
            # These peers are determined by the validated rank and TP16 topology.
            native["setup"].pop("peer_ranks")
        alternatives[name] = native
    return dict(
        operation=query.operation,
        world_size=query.world_size,
        alternatives=alternatives,
        inplace=query.inplace,
    )


def _grid_bucket(size, identity):
    step = max(16, (1 << (size.bit_length() - 1)) // 16)
    return CollectiveBucket(
        identity=identity,
        size_bytes=size,
        limit=8,
        grid=CollectiveGrid(
            upper_bytes=-(-size // step) * step,
            band=(size.bit_length() - 1) // 2,
        ),
    )


def _island_bucket(query):
    from math import prod

    signature = _collective_signature(query)
    sizes = set()
    for native in signature["alternatives"].values():
        call, setup = native["call"], native["setup"]
        inp = call["inp"]
        if inp["dtype"] != "torch.bfloat16":
            return None
        size = prod(inp["shape"]) * 2
        sizes.add(size)
        for tensor in (inp, call.get("output")):
            if tensor is None:
                continue
            shape, strides = tensor["shape"], tensor["stride"]
            if len(shape) != len(strides):
                return None
            contiguous_stride = 1
            for extent, stride in zip(reversed(shape), reversed(strides)):
                if extent != 1 and stride != contiguous_stride:
                    return None
                contiguous_stride *= extent
            if prod(shape) * 2 != size:
                return None
            del tensor["shape"], tensor["stride"]
        if not setup.get("active_blocks_override") and not setup.get("blocks_override"):
            del call["active_blocks"]
    if len(sizes) != 1:
        raise ValueError("collective alternatives disagree on payload size")
    return _grid_bucket(sizes.pop(), dict(version=2, signature=signature))


def _bucket(query):
    if query.operation != "all_reduce":
        return None
    if any(name in query.alternatives for name in ISLAND_VARIANTS):
        return _island_bucket(query)
    from math import prod

    alternatives = {}
    sizes = set()
    for name, payload in query.alternatives.items():
        call, setup = payload["call"], payload["setup"]
        if call["dtype"] not in ("float16", "bfloat16"):
            return None
        shape = (
            call["shape"]
            if payload["surface"] == "PCIeTwoShotBF16.all_reduce"
            else setup["shape"]
        )
        sizes.add(prod(shape) * 2)
        alternatives[name] = dict(
            surface=payload["surface"],
            topology=payload["topology"],
            dtype=call["dtype"],
            stage_input=call.get("stage_input"),
            remote_push_region_packs=call.get("remote_push_region_packs"),
            payload_ready_region_packs=call.get("payload_ready_region_packs"),
            eager_buffer_bytes=setup.get("eager_buffer_bytes"),
            eager_storage_shards=setup.get("eager_storage_shards"),
            registered=setup.get("registered"),
            shard_storage={
                field: setup[field]
                for field in ("max_rows", "pack_stride", "slot_bytes", "slots")
                if field in setup
            },
        )
        if call.get("has_addend", False):
            alternatives[name]["has_addend"] = True
        if setup.get("payload_ready_max_blocks", 36) > 36:
            alternatives[name]["payload_ready_max_blocks"] = int(
                setup["payload_ready_max_blocks"]
            )
    if len(sizes) != 1:
        raise ValueError("collective alternatives disagree on payload size")
    size = sizes.pop()
    identity = dict(
        operation=query.operation,
        world_size=query.world_size,
        inplace=query.inplace,
        alternatives=alternatives,
    )
    if "twoshot" in alternatives:
        call = query.alternatives["twoshot"]["call"]
        identity["row_elems"] = call["row_elems"]
        if call["transport"] == "pull":
            identity["twoshot_transport"] = "pull"
    return _grid_bucket(size, dict(version=2, **identity))


TUNING = TuningContract(
    component_id="comm.pcie",
    query_schema_version=8,
    candidate_contract_version=14,
    config_schema_version=2,
    query_fields=frozenset(
        ("operation", "world_size", "rank", "alternatives", "default", "inplace")
    ),
    config_fields=frozenset(
        ("implementation", "threads", "blocks", "submode", "active_blocks")
    ),
    encode_query=lambda q: dict(
        operation=q.operation,
        world_size=q.world_size,
        rank=q.rank,
        alternatives=q.alternatives,
        default=asdict(q.default),
        inplace=q.inplace,
    ),
    encode_config=asdict,
    decode_config=lambda value: CollectiveConfig(**value),
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=lambda q, device: q.default,
    knobs=(
        Knob(name="implementation", values=None),
        Knob(name="threads", values=None),
        Knob(name="blocks", values=None, binding=ParameterBinding.RUNTIME),
        Knob(name="submode", values=None, binding=ParameterBinding.RUNTIME),
        Knob(name="active_blocks", values=None, binding=ParameterBinding.RUNTIME),
    ),
    parameters=_parameters,
    collective_signature=_collective_signature,
    collective_bucket=_bucket,
    equivalence_key=_equivalence_key,
)


def plan(
    alternatives, *, default, inplace=False, invocation=FrozenMapping(), override=None
):
    from ._preparation import _provider
    from ._benchmark import benchmark_call

    first = next(iter(alternatives.values()))[0]
    default_call = alternatives[default][0].call
    query = CollectiveQuery(
        operation=first.surface.split(".", 1)[1],
        world_size=first.world_size,
        rank=first.rank,
        alternatives=FrozenMapping(
            {
                name: NATIVE.encode_query(q)
                for name, (q, runtime) in alternatives.items()
            }
        ),
        default=CollectiveConfig(
            implementation=default,
            submode=default_call.get("submode", ""),
            active_blocks=int(default_call.get("active_blocks", 0)),
            threads=int(default_call["threads"]),
            blocks=int(default_call.get("blocks", default_call.get("block_limit", 16))),
        ),
        inplace=inplace,
    )
    TUNING.validate_query(query, None)

    def native(config):
        q, runtime = alternatives[config.implementation]
        call = dict(q.call)
        call["threads"] = config.threads
        call["blocks" if "blocks" in call else "block_limit"] = config.blocks
        if config.implementation in ISLAND_VARIANTS:
            call["active_blocks"] = config.active_blocks
        if config.implementation in (
            "dcp_phased",
            "dcp_phased_push",
            "dcp_ring",
            "dcp_switch_push",
            "dcp_island",
            "dcp_island_pipe",
            "dcp_recursive",
            "dcp_recursive_ready",
            "dcp_recursive_outer_first",
            "dcp_recursive_pix_last",
        ):
            call.update(submode=config.submode, active_blocks=config.active_blocks)
        q = replace(q, call=FrozenMapping(call))
        return _provider(q.surface).plan(q, runtime=runtime)

    memory_by_implementation = {
        name: _provider(q.surface).plan(q, runtime=runtime)._memory_requirements
        for name, (q, runtime) in alternatives.items()
    }
    first_memory = next(iter(memory_by_implementation.values()))
    if isinstance(first_memory, FixedMemoryRequirements) and all(
        memory == first_memory for memory in memory_by_implementation.values()
    ):
        memory = first_memory
    else:

        def memory(config, device):
            return memory_by_implementation[config.implementation](config, device)

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=lambda config, device: native(config)._compile_jobs(
            config, device
        ),
        _memory_requirements=memory,
        _materialize=lambda selection, device: native(selection.config)._materialize(
            selection, device
        ),
        _benchmark_call=lambda state: benchmark_call(state, inplace=inplace),
        _device=next(iter(alternatives.values()))[1].device,
    )


def oneshot_alternatives(query, runtime):
    alternatives = {}
    field = "transport" if query.surface.endswith(".all_reduce") else "mode"
    selected = query.call[field]
    alternatives[selected] = (query, runtime)
    if not query.call["stage_input"]:
        return alternatives
    modes = ["pull"] if field == "transport" else ["stage_pull"]
    if field == "transport":
        capacity = int(query.setup.get("eager_buffer_bytes") or 0) // 16
        packs = int(query.call["size_packs"])
        graph_slots = all(
            variant in tuple(tuple(v) for v in query.setup["variants"])
            for variant in ((True, 0), (True, 1))
        )
        valid_graph = (
            query.world_size in (2, 3, 4, 6, 8, 10)
            and query.call["dtype"] in ("float16", "bfloat16")
            and graph_slots
            and 0 < packs <= min(capacity, 0x7FFFFFFF)
        )
        if (
            valid_graph
            and query.call["remote_push_region_packs"] > 0
            and query.call["remote_push_region_packs"]
            + 2 * (query.world_size - 1) * capacity
            <= int(query.setup["eager_storage_shards"]) * capacity
        ):
            from .pcie_oneshot import _MAX_BLOCKS

            modes.append("remote_push_ll")
            if query.world_size == 2:
                modes.append("tp2_remote_push")
            if (
                capacity >= _MAX_BLOCKS
                and packs * 16 <= capacity * 32 - _MAX_BLOCKS * 16
            ):
                modes.append("remote_push_simple")
                if query.world_size == 2:
                    modes.append("tp2_remote_push_simple")
        if (
            valid_graph
            and query.call["dtype"] == "bfloat16"
            and query.call.get("payload_ready_region_packs", 0) > 0
            and query.call["payload_ready_region_packs"] + query.world_size * capacity
            <= int(query.setup["eager_storage_shards"]) * capacity
        ):
            modes += ["remote_push_full", "remote_push_rsag"]
    elif query.world_size == 4:
        modes += ["stage_remote_push"]
        # The payload-ready fused specialization uses an isolated incoming region.
        if selected == "stage_bf16_sentinel":
            modes += ["stage_bf16_sentinel"]
    for mode in modes:
        call = {**query.call, field: mode}
        if field == "transport" and mode == "pull":
            call.update(threads=call["eager_threads"], blocks=call["eager_blocks"])
        if field == "transport" and mode in ("tp2_remote_push", "remote_push_ll"):
            call.update(threads=128, blocks=36)
        if field == "transport" and mode in (
            "tp2_remote_push_simple",
            "remote_push_simple",
        ):
            call.update(threads=128, blocks=min(36, -(-call["size_packs"] // 128)))
        if field == "transport" and mode in ("remote_push_full", "remote_push_rsag"):
            call.update(threads=256, blocks=16)
        if mode not in alternatives:
            alternatives[mode] = (replace(query, call=FrozenMapping(call)), runtime)
    return alternatives
