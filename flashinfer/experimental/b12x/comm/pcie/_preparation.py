"""One declaration entry point for established native PCIe resource owners."""

from __future__ import annotations

from importlib import import_module
from dataclasses import replace
from math import prod

from b12x.preparation import FrozenMapping, Plan
from ._tuning import PAIRWISE_THREADS, PAIRWISE_BLOCKS, PcieConfig, PcieQuery, TUNING

from ._collective_tuning import CollectiveConfig


_OWNER_SURFACES = frozenset(
    (
        "DcpTopKOwnerExchange.stage_candidates",
        "VocabParallelArgmax.fused_add_argmax",
        "PCIeHierarchicalAllReduce.all_reduce",
        "PCIeIslandRSAllReduce.all_reduce",
        "PCIeIslandRSAllReduce.reduce_scatter",
        "PCIeIslandRSAllReduce.all_gather",
        "PCIeIslandRSAllReduce.row_to_col_all_to_all",
    )
)


def _provider(surface: str):
    if surface.startswith(("OneshotAllReduce.", "OneshotAllReducePool.")):
        module = "_oneshot_preparation"
    elif surface == "DmaAllReduce.all_reduce":
        module = "_dma_preparation"
    elif surface.startswith(("TwoShotReduceScatter.", "PCIeTwoShotBF16.")):
        module = "_twoshot_preparation"
    elif surface == "kimi_topk16" or surface.startswith(
        ("DcpAllToAll.", "DcpAllToAllPool.")
    ):
        module = "_dcp_preparation"
    elif surface in _OWNER_SURFACES:
        module = "_owner_preparation"
    else:
        raise ValueError(f"unknown native PCIe execution surface {surface!r}")
    return import_module(f"{__package__}.{module}")


def query_from_runtime(runtime, *, surface: str, call) -> PcieQuery:
    """Normalize call metadata for the caller's already-established channel."""
    return _provider(surface).query_from_runtime(runtime, surface=surface, call=call)


def _selected_gather_query(query, transport, thread_counts):
    from ._gather_preparation import validate_query

    call, setup = query.call, query.setup
    if (
        call.get("has_gate", False)
        or (
            query.world_size not in (8, 16)
            and not (
                query.world_size == 4
                and transport
                in (
                    "selected_ring",
                    "selected_switch_push",
                    "selected_recursive_pix_last",
                )
            )
        )
        or not query.surface.endswith(".all_gather_heads")
        or not call.get("disjoint_output", False)
        or not call.get("pack_aligned", False)
        or (setup.get("blocks_override", 0) and call["block_limit"] != 8)
        or (setup.get("threads_override", 0) and call["threads"] not in thread_counts)
    ):
        return None
    element_bytes = 1 if call["dtype"] == "torch.float8_e4m3fn" else 2
    payload_bytes = prod(call["shape"]) * element_bytes
    default_blocks = (
        16
        if (
            not setup.get("blocks_override", 0)
            and (
                (transport == "selected_recursive_pix_last" and query.world_size == 8)
                or (transport == "selected_switch_push" and query.world_size == 4)
                or (
                    transport == "selected_ring"
                    and query.world_size == 16
                    and payload_bytes >= 65536
                )
            )
        )
        else 8
    )
    selected = replace(
        query,
        call=FrozenMapping(
            dict(
                call,
                gather_transport=transport,
                threads=call["threads"]
                if call["threads"] in thread_counts
                else thread_counts[0],
                block_limit=default_blocks,
                submode="barrier",
                active_blocks=default_blocks,
            )
        ),
    )
    try:
        validate_query(selected)
    except ValueError:
        return None
    return selected


def plan(
    query: PcieQuery,
    *,
    runtime=None,
    invocation=FrozenMapping(),
    override: PcieConfig | CollectiveConfig | None = None,
) -> Plan:
    """Declare native work; neither topology nor a communicator is constructed.

    DCP head gather and LSE reduce-scatter expose launch candidates on TP2–8.
    Head gather races ring profiles on TP4/8/16 and phased and recursive
    profiles on TP8/16. TP4 LSE and push two-shot also race pairwise schedules.
    IslandRS reductions tune threads and active CTA counts while retaining a
    fixed physical grid and resource owner. Other low-level declarations retain their explicit configuration.
    ``AllReduce.plan`` composes eligible one-shot and two-shot transports.
    """
    if not isinstance(query, PcieQuery):
        raise TypeError("query must be PcieQuery")
    TUNING.validate_query(query, None)
    if (
        runtime is not None
        and query.surface == "PCIeIslandRSAllReduce.all_reduce"
        and not isinstance(override, PcieConfig)
    ):
        from ._collective_tuning import plan as collective_plan

        from ._collective_tuning import ISLAND_VARIANTS

        flags = (
            query.call.get("local_push", False),
            query.call.get("cross_island_rsag", False),
            query.call.get("cross_island_dma", False),
        )
        default = next(
            name for name, value in ISLAND_VARIANTS.items() if value == flags
        )
        alternatives = {default: (query, runtime)}
        if not query.setup.get("threads_override", False):
            for name, (push, scattered, dma) in ISLAND_VARIANTS.items():
                alternatives[name] = (
                    replace(
                        query,
                        call=FrozenMapping(
                            {
                                **query.call,
                                "local_push": push,
                                "cross_island_rsag": scattered,
                                "cross_island_dma": dma,
                            }
                        ),
                    ),
                    runtime,
                )
        return collective_plan(
            alternatives,
            default=default,
            inplace=bool(query.call.get("inplace", False)),
            invocation=invocation,
            override=override,
        )
    if (
        runtime is not None
        and query.surface.endswith(".all_gather_pair_kimi_topk")
        and query.call.get("rows", 1) > 1
        and not isinstance(override, PcieConfig)
    ):
        from ._collective_tuning import plan as collective_plan

        rows = int(query.call["rows"])
        alternatives = {
            "dcp_route_separate": (
                replace(query, call=FrozenMapping(dict(query.call, block_limit=1))),
                runtime,
            )
        }
        if query.call["threads"] == 512:
            alternatives["dcp_route_row"] = (
                replace(
                    query,
                    call=FrozenMapping(
                        dict(query.call, route_transport="row", block_limit=rows)
                    ),
                ),
                runtime,
            )
        return collective_plan(
            alternatives,
            default="dcp_route_separate",
            invocation=invocation,
            override=override,
        )

    from ._gather_preparation import GATHER_MODES

    if query.surface.endswith(".all_gather_heads") and (
        query.call.get("gather_transport") in GATHER_MODES
        or (query.world_size == 16 and isinstance(override, PcieConfig))
    ):
        return _provider(query.surface).plan(
            query, runtime=runtime, invocation=invocation, override=override
        )
    if (
        runtime is not None
        and (
            query.world_size <= 8
            or (query.world_size == 16 and query.surface.endswith(".all_gather_heads"))
        )
        and query.surface.startswith(("DcpAllToAll.", "DcpAllToAllPool."))
        and query.surface.endswith((".lse_reduce_scatter", ".all_gather_heads"))
    ):
        from ._collective_tuning import plan as collective_plan

        alternatives = {"dcp": (query, runtime)}
        if (
            query.world_size == 4
            and query.surface.endswith(".lse_reduce_scatter")
            and not query.setup.get("threads_override", 0)
            and not query.setup.get("blocks_override", 0)
        ):
            alternatives["dcp_pairwise"] = (
                replace(
                    query,
                    call=FrozenMapping(
                        dict(
                            query.call,
                            threads=PAIRWISE_THREADS,
                            block_limit=PAIRWISE_BLOCKS,
                            payload_schedule="pairwise",
                        )
                    ),
                ),
                runtime,
            )
        specializations = [
            ("dcp_phased", "selected_phased_barrier", (128, 256, 512)),
            ("dcp_ring", "selected_ring", (128, 512)),
            ("dcp_recursive", "selected_recursive", (128,)),
            ("dcp_recursive_ready", "selected_recursive_ready", (128,)),
            ("dcp_recursive_pix_last", "selected_recursive_pix_last", (128, 256, 512)),
        ]
        if query.world_size == 4:
            specializations.append(
                ("dcp_switch_push", "selected_switch_push", (128, 256, 512))
            )
        if query.world_size == 16:
            specializations.append(
                ("dcp_recursive_outer_first", "selected_recursive_outer_first", (128,))
            )
        if query.world_size == 8:
            specializations.append(("dcp_phased_push", "selected_phased_push", (128,)))
        for name, transport, threads in specializations:
            selected = _selected_gather_query(query, transport, threads)
            if selected is not None:
                alternatives[name] = (selected, runtime)
        if (
            query.surface.endswith(".all_gather_heads")
            and query.world_size in (2, 4)
            and query.setup.get("exchange_chunks", 0) > 0
            and query.call.get("word_aligned", False)
            and query.call.get("disjoint_output", False)
            and (
                not query.setup.get("threads_override", 0)
                or query.call["threads"] in (32, 64, 128, 256)
            )
        ):
            element_bytes = 1 if query.call["dtype"] == "torch.float8_e4m3fn" else 2
            nbytes = prod(query.call["shape"]) * element_bytes
            if (
                nbytes % 4 == 0
                and 0 < nbytes // 4 <= query.setup["exchange_chunks"] * 64
            ):
                call = dict(query.call)
                call.update(
                    gather_transport="word_exchange",
                    threads=query.call["threads"]
                    if query.setup.get("threads_override", 0)
                    else 32,
                    block_limit=query.call["block_limit"]
                    if query.setup.get("blocks_override", 0)
                    else 1,
                )
                alternatives["dcp_word"] = (
                    replace(query, call=FrozenMapping(call)),
                    runtime,
                )
        return collective_plan(
            alternatives,
            default="dcp",
            invocation=invocation,
            override=override,
        )
    return _provider(query.surface).plan(
        query,
        runtime=runtime,
        invocation=invocation,
        override=override,
    )


__all__ = ["plan", "query_from_runtime"]
