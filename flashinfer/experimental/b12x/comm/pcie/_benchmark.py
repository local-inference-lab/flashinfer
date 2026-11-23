"""Private operands and arithmetic checks for prepared PCIe tuning races."""

from functools import partial
from math import prod
from dataclasses import dataclass
from weakref import WeakValueDictionary

import torch

from b12x.preparation import PreparedCall


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class _Reference:
    seeds: tuple[torch.Tensor, ...]
    expected: tuple[torch.Tensor, ...]
    residual: tuple[torch.Tensor, ...] = ()
    lse: torch.Tensor | None = None


# Live benchmark calls own these immutable fixtures; the cache does not extend their lifetime.
_REFERENCES = WeakValueDictionary()


def _assert_output(actual, expected, *, rtol=0, atol=0):
    if (
        rtol == 0
        and atol == 0
        and actual.dtype == expected.dtype
        and torch.equal(actual, expected)
    ):
        return
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _reduction_reference(query, shape, dtype, device, fused):
    call = query.call
    has_addend = bool(call.get("has_addend", False)) and not fused
    semantics = (
        (
            tuple(query.setup["logical_shape"]),
            *(
                call[name]
                for name in (
                    "group_size",
                    "group_id_limit",
                    "input_addend",
                    "reduction_dtype",
                    "residual_dtype",
                    "normalize",
                    "output_dtype",
                    "zero_centered",
                )
            ),
        )
        if fused
        else (("has_addend", True),)
        if has_addend
        else ()
    )
    key = ("reduction", query.world_size, query.rank, shape, dtype, device, semantics)
    reference = _REFERENCES.get(key)
    if reference is not None:
        return reference
    generator = torch.Generator().manual_seed(81721)
    # Exactly representable partial sums isolate protocol and binding errors.
    values = (
        torch.randint(-16, 17, (query.world_size, *shape), generator=generator).float()
        / 32
    )
    sources = (values, torch.zeros_like(values), -values)
    seeds = tuple(value[query.rank].to(device=device, dtype=dtype) for value in sources)
    expected_outputs, expected_residuals = [], []
    for values in sources:
        if fused:
            if call["group_size"]:
                values = (values * 0.5).sum(dim=2)
                if call["group_id_limit"] == 0:
                    values.zero_()
                values = values.to(dtype).float()
            if call["input_addend"]:
                values = (values + 0.125).to(dtype).float()
        elif has_addend:
            values = (values + 0.125).to(dtype).float()
        expected = values.sum(dim=0)
        if not fused:
            expected_outputs.append(expected.to(device=device, dtype=dtype))
            continue
        if call["reduction_dtype"] is not None:
            expected = expected.to(_DTYPES[call["reduction_dtype"]]).float()
        expected = expected + 0.25
        expected_residuals.append(
            expected.to(device=device, dtype=_DTYPES[call["residual_dtype"]])
        )
        if call["normalize"]:
            expected = expected * torch.rsqrt(
                expected.square().mean(dim=-1, keepdim=True) + 1e-6
            )
            if call["zero_centered"]:
                expected *= 2
            expected_outputs.append(
                expected.to(device=device, dtype=_DTYPES[call["output_dtype"]])
            )
    reference = _Reference(seeds, tuple(expected_outputs), tuple(expected_residuals))
    _REFERENCES[key] = reference
    return reference


def _dcp_reference(query, shape, dtype, device, lse_reduce):
    call = query.call
    out_shape = tuple(call["out_shape"])
    key = (
        "dcp",
        query.world_size,
        query.rank,
        shape,
        out_shape,
        dtype,
        device,
        lse_reduce,
        call.get("natural_log"),
    )
    reference = _REFERENCES.get(key)
    if reference is not None:
        return reference
    generator = torch.Generator().manual_seed(9419)
    if not lse_reduce:
        # Gather transports preserve arbitrary bits, including FP32 views whose
        # BF16 lanes can encode NaNs or signed zero. No arithmetic is permitted.
        row_bytes = prod(shape[1:]) * dtype.itemsize
        stacked = torch.randint(
            0,
            256,
            (query.world_size, shape[0], row_bytes),
            dtype=torch.uint8,
            generator=generator,
        )
        values = (stacked, torch.zeros_like(stacked), torch.bitwise_not(stacked))
        reference = _Reference(
            tuple(
                value[query.rank].to(device).view(dtype).view(shape) for value in values
            ),
            tuple(
                value.permute(1, 0, 2)
                .contiguous()
                .view(dtype)
                .view(out_shape)
                .to(device)
                for value in values
            ),
        )
        _REFERENCES[key] = reference
        return reference
    stacked = torch.randint(0, 16, (query.world_size, *shape), generator=generator).to(
        dtype
    )
    seeds = tuple(
        value[query.rank].to(device) for value in (stacked, torch.zeros_like(stacked))
    )
    from .pcie_dcp_a2a import lse_reduce_scatter_reference

    scores = (
        torch.arange(query.world_size, dtype=torch.float32)[:, None, None]
        .expand(query.world_size, *shape[:2])
        .contiguous()
        / 8
    )
    expected = lse_reduce_scatter_reference(
        stacked, scores, query.rank, is_lse_base_on_e=call["natural_log"]
    )
    lse = scores[query.rank].to(device)
    reference = _Reference(
        seeds, (expected.to(device), torch.zeros_like(expected, device=device)), lse=lse
    )
    _REFERENCES[key] = reference
    return reference


def _reduction_repetitions(shape):
    rows = prod(shape[:-1])
    return max(1, (100 + rows - 1) // rows)


def _empty_island_operand(shape, dtype, device, alignment):
    if alignment == 256:
        return torch.empty(shape, dtype=dtype, device=device)
    # An alignment-sized offset gives the declared sub-256-byte pointer class.
    offset = alignment // dtype.itemsize
    storage = torch.empty(prod(shape) + offset, dtype=dtype, device=device)
    return storage[offset:].view(shape)


class _ReductionInputs:
    def __init__(self, state, *, inplace):
        self.state, self.step = state, 0
        query = state.query
        call = query.call
        self.twoshot = query.surface.startswith("PCIeTwoShotBF16.")
        self.owner = query.surface == "PCIeIslandRSAllReduce.all_reduce"
        self.fused = not query.surface.endswith(".all_reduce")
        shape = tuple(
            call["inp"]["shape"]
            if self.owner
            else call["shape"]
            if self.twoshot
            else query.setup["shape"]
        )
        dtype = _DTYPES[
            (
                call["inp"]["dtype"].removeprefix("torch.")
                if self.owner
                else call["dtype"]
            )
        ]
        device = state.runtime.device
        self.reference = _reduction_reference(query, shape, dtype, device, self.fused)
        if self.owner:
            self.source = _empty_island_operand(
                shape, dtype, device, call["inp"]["alignment"]
            )
            if inplace:
                self.out = self.source
            elif call["output"] is None or call["output"]["alignment"] == 256:
                self.out = torch.empty_like(self.source)
            else:
                self.out = _empty_island_operand(
                    shape, dtype, device, call["output"]["alignment"]
                )
        else:
            self.source = torch.empty(shape, dtype=dtype, device=device)
            self.out = self.source if inplace else torch.empty_like(self.source)
        self.owner_call = None
        self.plain_addend = (
            torch.full(shape, 0.125, dtype=dtype, device=device)
            if call.get("has_addend", False) and not self.owner and not self.fused
            else None
        )
        if self.owner:
            from ._owner_preparation import prepared_call

            self.owner_call = prepared_call(state, inp=self.source, out=self.out)
        self.residual = self.weight = self.addend = self.group_weights = (
            self.group_ids
        ) = None
        self.residual_seed = None
        if self.fused:
            logical = tuple(query.setup["logical_shape"])
            self.residual_seed = torch.full(
                logical, 0.25, dtype=_DTYPES[call["residual_dtype"]], device=device
            )
            self.residual = torch.empty_like(self.residual_seed)
            self.out = (
                torch.empty(logical, dtype=_DTYPES[call["output_dtype"]], device=device)
                if call["normalize"]
                else None
            )
            self.weight = (
                torch.ones(
                    logical[-1], dtype=_DTYPES[call["weight_dtype"]], device=device
                )
                if call["normalize"]
                else None
            )
            if call["input_addend"]:
                self.addend = torch.full(logical, 0.125, dtype=dtype, device=device)
            if call["group_size"]:
                group_shape = shape[:2]
                self.group_weights = torch.full(
                    group_shape, 0.5, dtype=torch.float32, device=device
                )
                self.group_ids = torch.zeros(
                    group_shape,
                    dtype=getattr(torch, call["group_ids_dtype"]),
                    device=device,
                )
        self.repetitions = _reduction_repetitions(
            tuple(query.setup["logical_shape"]) if self.fused else shape
        )
        self.producers = tuple(
            partial(self.produce, index) for index in range(len(self.reference.seeds))
        )

    def produce(self, index):
        self.step = index
        self.source.copy_(self.reference.seeds[index])
        if self.residual is not None:
            self.residual.copy_(self.residual_seed)

    def run(self):
        state, call = self.state, self.state.query.call
        if self.owner:
            return self.owner_call.run()
        if self.twoshot:
            addend_kwargs = (
                {"addend": self.plain_addend} if self.plain_addend is not None else {}
            )
            return state.run(
                self.source,
                None,
                self.out,
                threads=call["threads"],
                block_limit=call["block_limit"],
                **addend_kwargs,
            )
        if not self.fused:
            if self.plain_addend is None:
                state.run_plain(self.source, self.out)
            else:
                state.run_plain(self.source, self.out, self.plain_addend)
            return self.out
        state.run_fused(
            self.source,
            self.residual,
            self.weight,
            self.out,
            self.residual,
            1e-6,
            self.addend,
            self.group_weights,
            self.group_ids,
            call["group_id_limit"],
        )
        return self.residual if self.out is None else self.out

    def validate(self):
        if not self.fused:
            _assert_output(self.out, self.reference.expected[self.step])
            return
        _assert_output(self.residual, self.reference.residual[self.step])
        if self.out is not None:
            tolerance = (
                8e-3
                if self.out.dtype == torch.bfloat16
                else 2e-3
                if self.out.dtype == torch.float16
                else 2e-5
            )
            _assert_output(
                self.out,
                self.reference.expected[self.step],
                rtol=tolerance,
                atol=tolerance,
            )

    def close(self):
        self.producers = ()
        self.reference = None
        self.owner_call = None


def benchmark_call(state, *, inplace=False):
    dcp = state.query.surface.startswith(("DcpAllToAll.", "DcpAllToAllPool."))
    if state.query.surface.endswith(".all_gather_pair_kimi_topk"):
        return _route_benchmark_call(state)
    inputs = _DcpInputs(state) if dcp else _ReductionInputs(state, inplace=inplace)
    if not dcp and not inputs.twoshot and not inputs.owner:
        try:
            state.bind_input(inputs.source)
        except BaseException:
            inputs.close()
            raise
    return PreparedCall(
        run=inputs.run,
        output=inputs.out,
        produce=inputs.producers[0],
        benchmark_producers=inputs.producers,
        validate=inputs.validate,
        close=inputs.close,
        owners=(inputs, state.runtime),
        capture_context=None if dcp else state.runtime.capture,
        benchmark_repetitions=1 if dcp else inputs.repetitions,
    )


class _DcpInputs:
    def __init__(self, state):
        self.state, self.step = state, 0
        query, runtime = state.query, state.runtime
        call = query.call
        self.lse_reduce = query.surface.endswith(".lse_reduce_scatter")
        shape = tuple(call["shape"])
        name = call["dtype"].removeprefix("torch.")
        dtype = _DTYPES.get(name) or getattr(torch, name)
        self.reference = _dcp_reference(
            query, shape, dtype, runtime.device, self.lse_reduce
        )
        self.source = torch.empty_strided(
            shape, tuple(call["stride"]), dtype=dtype, device=runtime.device
        )
        self.out = torch.empty_strided(
            tuple(call["out_shape"]),
            tuple(call["out_stride"]),
            dtype=dtype,
            device=runtime.device,
        )
        self.lse = None
        self.gate = (
            torch.ones((1,), dtype=torch.int32, device=runtime.device)
            if not self.lse_reduce and call.get("has_gate", False)
            else None
        )
        if self.lse_reduce:
            self.lse = torch.empty_strided(
                tuple(call["lse_shape"]),
                tuple(call["lse_stride"]),
                dtype=torch.float32,
                device=runtime.device,
            )
            self.lse.copy_(self.reference.lse)
        self.producers = tuple(
            partial(self.produce, index) for index in range(len(self.reference.seeds))
        )

    def produce(self, index):
        self.step = index
        self.source.copy_(self.reference.seeds[index])
        if not self.lse_reduce:
            self.out.view(torch.uint8).fill_(0xA5)

    def run(self):
        if self.lse_reduce:
            return self.state.run(
                partial_output=self.source,
                partial_lse=self.lse,
                out=self.out,
                is_lse_base_on_e=self.state.query.call["natural_log"],
            )
        if self.gate is not None:
            return self.state.run(local_input=self.source, out=self.out, gate=self.gate)
        return self.state.run(local_input=self.source, out=self.out)

    def validate(self):
        if not self.lse_reduce:
            _assert_output(
                self.out.view(torch.uint8),
                self.reference.expected[self.step].view(torch.uint8),
            )
            _assert_output(
                self.source.view(torch.uint8),
                self.reference.seeds[self.step].view(torch.uint8),
            )
            return
        tolerance = 2e-2 if self.out.dtype == torch.bfloat16 else 2e-3
        _assert_output(
            self.out, self.reference.expected[self.step], rtol=tolerance, atol=tolerance
        )

    def close(self):
        self.producers = ()
        self.reference = None


def _route_benchmark_call(state):
    """Validate whole multi-row rank-major outputs and Kimi's lower-ID tie rule."""
    device, rank = state.runtime.device, state.query.rank
    rows = int(state.query.call.get("rows", 4))
    cases = []
    features = torch.arange(rows * 896, dtype=torch.long).reshape(rows, 896)
    latent_bytes = torch.arange(rows * 3584, dtype=torch.int32).reshape(rows, 3584)
    world = state.query.world_size
    down_cols = 3584 // world
    router_cols = 896 // world
    for seed in (0, 1, 2):
        # These logits give exactly representable sigmoid scores on CPU and GPU.
        # Permute score groups across experts so a wrong rank order remains visible.
        labels = (features * (2 * seed + 3) + seed) % 7
        lookup = torch.tensor(
            [0.0, float("inf"), float("-inf"), -0.0, float("nan"), 0.0, float("inf")]
        )
        logits = lookup[labels]
        raw = torch.where(
            torch.isposinf(logits), 1.0, torch.where(logits == 0, 0.5, 0.0)
        )
        bias = (((torch.arange(896) * (3 * seed + 1)) % 11) - 4).float() * 0.125
        ids = torch.argsort(raw + bias, descending=True, stable=True)[:, :16]
        top = torch.gather(raw, 1, ids)
        denominator = top.sum(1, keepdim=True)
        weights = top * torch.where(denominator > 0, 1.0 / denominator, 1.0)
        latent = (latent_bytes * 97 + seed * 117).to(torch.int16).view(torch.bfloat16)
        cases.append(
            tuple(
                v.to(device)
                for v in (
                    latent[:, rank * down_cols : (rank + 1) * down_cols].contiguous(),
                    logits[:, rank * router_cols : (rank + 1) * router_cols].contiguous(),
                    bias,
                    latent,
                    logits,
                    weights,
                    ids.to(torch.int32),
                )
            )
        )
    down, router, bias, latent, logits, weights, ids = tuple(
        torch.empty_like(v) for v in cases[0]
    )
    active = 0

    def produce(i):
        nonlocal active
        active = i
        for target, source in zip((down, router, bias), cases[i][:3]):
            target.copy_(source)

    def validate():
        values = (down, router, bias, latent, logits, weights, ids)
        for value, expected in zip(values, cases[active]):
            _assert_output(value.view(torch.uint8), expected.view(torch.uint8))

    def run():
        return state.run(
            local_down=down,
            local_router=router,
            correction_bias=bias,
            out_down=latent,
            out_logits=logits,
            topk_weights=weights,
            topk_ids=ids,
        )

    producers = tuple(partial(produce, i) for i in range(len(cases)))
    return PreparedCall(
        run=run,
        output=(latent, logits, weights, ids),
        produce=producers[0],
        benchmark_producers=producers,
        validate=validate,
        owners=(state.runtime,),
    )
