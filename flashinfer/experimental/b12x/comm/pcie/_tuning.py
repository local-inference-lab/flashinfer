"""Singleton declarations for caller-owned PCIe communication surfaces.

Each surface retains its existing communicator, transport precision, launch
settings and channel ownership. This contract does not select interchangeable
collective backends or qualify a topology from a visible-device count.
"""

from dataclasses import dataclass, replace

from b12x.preparation import BackendConfig, FrozenMapping, make_fixed_contract


PAIRWISE_THREADS = 128
PAIRWISE_BLOCKS = 8


def pairwise_device(device):
    return (
        device is not None
        and device.vendor == "nvidia"
        and device.compute_capability == (12, 0)
        and device.sm_count >= 64
    )


# Native execution surfaces after the all-reduce manager's plan-time routing.
# Values are supported worlds, never a set of transports to race.
SURFACES = {
    "OneshotAllReduce.all_reduce": (2, 3, 4, 6, 8, 10),
    "OneshotAllReduce.all_reduce_fused_add_rms_norm": (2, 3, 4, 6, 8, 10),
    "OneshotAllReduce.all_reduce_fused_add": (2, 3, 4, 6, 8, 10),
    "OneshotAllReducePool.all_reduce": (2, 3, 4, 6, 8, 10),
    "OneshotAllReducePool.all_reduce_fused_add_rms_norm": (2, 3, 4, 6, 8, 10),
    "DmaAllReduce.all_reduce": (2, 3, 4, 6, 8, 10, 16),
    "PCIeTwoShotBF16.all_reduce": (2, 3, 4, 6, 8, 10),
    "PCIeTwoShotBF16.reduce_scatter": (2, 3, 4, 6, 8, 10),
    "PCIeTwoShotBF16.all_gather": (2, 3, 4, 6, 8, 10),
    "TwoShotReduceScatter.reduce_scatter_fp8": (2, 4, 8),
    "TwoShotReduceScatter.all_gather_fp8": (2, 4, 8),
    "DcpAllToAll.lse_reduce_scatter": (2, 4, 8, 9, 10, 12, 16),
    "DcpAllToAll.all_gather_varlen": (2, 4),
    "DcpAllToAll.all_gather_heads": (2, 4, 8, 9, 10, 12, 16),
    "DcpAllToAll.all_gather_pair": (2, 4, 8, 16),
    "DcpAllToAll.all_gather_pair_kimi_topk": (2, 4, 8, 16),
    "DcpAllToAll.kimi_topk16": (2, 4, 8, 16),
    "DcpAllToAllPool.lse_reduce_scatter": (2, 4, 8, 9, 10, 12, 16),
    "DcpAllToAllPool.all_gather_heads": (2, 4, 8, 9, 10, 12, 16),
    "DcpAllToAllPool.all_gather_pair": (2, 4, 8, 16),
    "DcpAllToAllPool.all_gather_pair_kimi_topk": (2, 4, 8, 16),
    "DcpAllToAllPool.kimi_topk16": (2, 4, 8, 16),
    "DcpTopKOwnerExchange.stage_candidates": (2, 3, 4, 6, 8, 16),
    "VocabParallelArgmax.fused_add_argmax": (8, 12, 16),
    "PCIeHierarchicalAllReduce.all_reduce": (9, 10, 12, 16),
    "PCIeIslandRSAllReduce.all_reduce": (16,),
    "PCIeIslandRSAllReduce.reduce_scatter": (16,),
    "PCIeIslandRSAllReduce.all_gather": (16,),
    "PCIeIslandRSAllReduce.row_to_col_all_to_all": (16,),
    "kimi_topk16": (1,),
}


@dataclass(frozen=True, kw_only=True)
class PcieQuery:
    surface: str
    world_size: int
    rank: int
    topology: str
    # Capacity, tensor geometry/dtypes, codec, channel, and existing host launch
    # overrides are concrete caller metadata, never alternative tuning labels.
    call: FrozenMapping
    setup: FrozenMapping


PcieConfig = BackendConfig
TUNING = make_fixed_contract(
    component_id="comm.pcie",
    query_type=PcieQuery,
    backend="native",
)
_validate_fixed_query = TUNING.validate_query


def _validate_query(query: PcieQuery, device) -> None:
    _validate_fixed_query(query, device)
    worlds = SURFACES.get(query.surface)
    if worlds is None:
        raise ValueError(f"unknown PCIe execution surface {query.surface!r}")
    if query.world_size not in worlds:
        raise ValueError(f"{query.surface} supports world sizes {worlds}")
    if not 0 <= query.rank < query.world_size:
        raise ValueError("rank must belong to the caller's process group")
    if query.surface in (
        "PCIeIslandRSAllReduce.all_reduce",
        "PCIeIslandRSAllReduce.reduce_scatter",
        "PCIeIslandRSAllReduce.all_gather",
        "PCIeIslandRSAllReduce.row_to_col_all_to_all",
    ):
        from ._island_rs_cute import MAX_BLOCKS

        if query.surface == "PCIeIslandRSAllReduce.all_reduce":
            push = query.call.get("local_push", False)
            scattered = query.call.get("cross_island_rsag", False)
            dma = query.call.get("cross_island_dma", False)
            if any(type(value) is not bool for value in (push, scattered, dma)):
                raise TypeError("island protocol choices must be booleans")
            if dma and not (push and scattered):
                raise ValueError(
                    "cross-island DMA requires local push and scatter/gather"
                )
        blocks, active = query.call["blocks"], query.call["active_blocks"]
        if (
            type(blocks) is not int
            or type(active) is not int
            or not 1 <= active <= blocks <= MAX_BLOCKS
        ):
            raise ValueError("island participation must fit the fixed physical grid")
    if query.surface.endswith(".all_gather_varlen"):
        capacity, bank = query.call["capacity"], query.call["bank_bytes"]
        limit = (
            query.setup["exchange_offset"]
            if query.setup["exchange_chunks"]
            else query.setup["slot_bytes"]
        )
        if (
            type(capacity) is not int
            or capacity <= 0
            or capacity % 16
            or type(bank) is not int
            or bank <= 0
            or bank % (8 * 128 * 16)
            or bank > limit
        ):
            raise ValueError(
                "variable gather requires aligned rows and a complete CTA bank within staging"
            )
    if (
        query.surface.startswith("PCIeTwoShotBF16.")
        and query.call.get("payload_schedule", "direct") == "pairwise"
    ):
        if (
            query.world_size != 4
            or query.call["threads"] != PAIRWISE_THREADS
            or query.call["block_limit"] != PAIRWISE_BLOCKS
            or query.call["transport"] != "push"
        ):
            raise ValueError(
                "pairwise two-shot requires TP4 push with eight 128-thread CTAs"
            )
    if query.surface.endswith(".lse_reduce_scatter"):
        transport = query.call.get("payload_schedule", "direct")
        if transport not in ("direct", "pairwise") or (
            transport == "pairwise"
            and (
                query.world_size != 4
                or query.call["threads"] != PAIRWISE_THREADS
                or query.call["block_limit"] != PAIRWISE_BLOCKS
            )
        ):
            raise ValueError("pairwise LSE requires DCP4 with eight 128-thread CTAs")
    if query.surface.endswith(".all_gather_heads"):
        transport = query.call.get("gather_transport", "barrier")
        from ._gather_preparation import GATHER_MODES, validate_query as validate_gather

        has_gate = query.call.get("has_gate", False)
        if type(has_gate) is not bool:
            raise ValueError("all_gather_heads has_gate must be a bool")
        if has_gate and transport not in ("barrier", "word_exchange"):
            raise ValueError(
                "gated all_gather_heads requires barrier or word_exchange transport"
            )
        if transport in GATHER_MODES:
            validate_gather(query)
        elif transport not in ("barrier", "word_exchange"):
            raise ValueError("unknown DCP gather transport")
        if transport == "word_exchange":
            from math import prod

            element_bytes = 1 if query.call["dtype"] == "torch.float8_e4m3fn" else 2
            nbytes = prod(query.call["shape"]) * element_bytes
            chunks = query.setup.get("exchange_chunks", 0)
            if (
                query.world_size not in (2, 4)
                or not 1 <= chunks <= 64
                or not 0 < nbytes <= chunks * 64 * 4
                or nbytes % 4
                or not query.call.get("disjoint_output", False)
                or not query.call.get("word_aligned", False)
                or query.call["threads"] not in (32, 64, 128, 256)
                or not 1 <= query.call["block_limit"] <= 64
                or query.setup.get("exchange_offset", -1) < 0
                or query.setup["exchange_offset"] % 256
                or query.setup["exchange_offset"] + query.world_size * chunks * 64 * 8
                > query.setup["slot_bytes"]
                or query.setup.get("exchange_counter_offset", 0)
                != 2 * query.setup["slot_bytes"]
            ):
                raise ValueError(
                    "word exchange requires disjoint aligned data and owned TP2/TP4 state"
                )
    if query.surface.endswith(".all_gather_pair"):
        blocks = query.call.get("row_blocks", 1)
        if type(blocks) is not int or not 1 <= blocks <= 64:
            raise ValueError("paired gather row blocks must fit the signal plane")
        if blocks > 1 and not query.call.get("disjoint_output", False):
            raise ValueError("parallel paired gather requires disjoint outputs")
        load_batch = query.call.get("load_batch", 1)
        if type(load_batch) is not int or load_batch not in (1, 2):
            raise ValueError("paired gather load batch must be one or two")
        if load_batch != 1 and (
            blocks == 1 or not query.call.get("disjoint_output", False)
        ):
            raise ValueError("batched paired loads require disjoint multi-CTA rows")
    if query.surface.endswith(".all_gather_pair_kimi_topk"):
        if "rows" in query.call and (
            type(query.call["rows"]) is not int
            or not 2 <= query.call["rows"] <= 64
            or query.world_size != 16
            or query.setup.get("max_batch_size", 0) < query.call["rows"]
            or query.setup.get("query_head_dim") != 672
            or query.call.get("route_transport") not in ("separate", "row")
        ):
            raise ValueError(
                "multi-row Kimi expert selection requires a sixteen-rank 672-byte channel"
            )
        for field in ("row_blocks", "load_batch"):
            value = query.call.get(field, 1)
            if type(value) is not int or value != 1:
                raise ValueError(
                    "fused Kimi paired gather requires row_blocks=1 and load_batch=1"
                )
    if query.surface == "kimi_topk16":
        if query.topology != "local":
            raise ValueError("stateless kimi_topk16 uses one local device")
    elif query.topology != "pcie_ipc":
        raise ValueError(
            "PCIe collectives require caller-established CUDA IPC topology"
        )


TUNING = replace(TUNING, query_schema_version=43, validate_query=_validate_query)

__all__ = ["PcieQuery", "PcieConfig", "SURFACES", "TUNING"]
