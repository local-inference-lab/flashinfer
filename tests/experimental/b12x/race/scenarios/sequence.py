"""Race scenarios for b12x stateful sequence ops.

Every family here updates recurrent or convolution state in place, so
``reset`` copies the state pool (and output) back from a pristine clone taken
after preparation, and the state pool is compared along with the output.

- ``seq_gdn_prefill`` / ``seq_kda_prefill``: the delta-rule prefill
  (``sequence/_shared/delta_prefill``). The prepare kernel of a window runs on
  a side stream while the recurrence of the previous window runs (CUDA graph
  branches under capture), publishes each record with ``fence.acq_rel`` and a
  release store of a ready flag, and the recurrence spins on acquire loads of
  that flag before bulk-copying the record. Small ``window_tiles`` give many
  windows, so ring slots and their flags are reused within a replay and carry
  stale values from the previous replay. Sequences that span windows carry
  their state through the final slot between recurrence launches.
- ``seq_gdn_decode``: the CuTe grouped-head (Qwen) speculative decode.
- ``seq_short_conv``: the raw-history short convolution pipeline.
"""

from __future__ import annotations

import torch

from ._base import Scenario, _generator


def _frozen_session(device, *requests):
    from b12x.preparation import PreparationSession

    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    session.__enter__()
    try:
        session.prepare(tuple(requests))
        session.freeze()
    except BaseException:
        session.__exit__(None, None, None)
        raise
    return session


def _restorer(pairs):
    """Copy each pristine clone back over its live tensor."""

    def reset():
        for live, pristine in pairs:
            live.copy_(pristine)

    return reset


# --------------------------------------------------------------------------
# Delta-rule prefill (GDN and KDA)


def _prefill(device, case, recipe):
    from b12x.preparation import FrozenMapping, PreparedCall
    from b12x.sequence._shared.delta_prefill.preparation import invocation_from_tensors
    from b12x.sequence._shared.delta_prefill.workspace import BACKEND, tiles_capacity
    from b12x.testing.delta_prefill_cases import PrefillCase, make_inputs, prefill_op

    op = prefill_op(recipe)
    lengths = tuple(case.get("lengths", (256, 150, 129, 64, 33, 200, 17, 100)))
    key_heads = case.get("key_heads", 16 if recipe == "gdn" else 32)
    value_heads = 3 * key_heads if recipe == "gdn" else key_heads
    max_tokens = case.get("max_tokens") or max(16, sum(lengths))
    max_seqs = len(lengths)
    tensors = make_inputs(
        PrefillCase(recipe, key_heads, value_heads, lengths),
        device=device,
        seed=int(case.get("seed", 5)),
        max_tokens=max_tokens,
        max_seqs=max_seqs,
    )
    checkpoint = bool(case.get("checkpoint", True))
    if checkpoint:
        # Snapshot each sequence about halfway, on a chunk boundary.
        offsets = [(length // 2) // 16 * 16 for length in lengths]
        tensors["checkpoint_offsets"].copy_(torch.tensor(offsets, dtype=torch.int32))
    geometry = (
        {"key_heads": key_heads, "value_heads": value_heads}
        if recipe == "gdn"
        else {"heads": value_heads}
    )
    if recipe == "gdn" and case.get("fp32"):
        geometry["intermediate_dtype"] = torch.float32
    caps = op.Caps(
        device=device,
        max_tokens=max_tokens,
        max_seqs=max_seqs,
        max_state_slots=tensors["recurrent_state"].shape[0],
        checkpoint_export=checkpoint,
        **geometry,
    )
    override = None
    if any(
        case.get(name) is not None
        for name in ("v_split", "k_split", "stages", "window_tiles", "algorithm")
    ):
        config = {
            "backend": BACKEND,
            "v_split": case.get("v_split", 64),
            "k_split": case.get("k_split", 1),
            "stages": case.get("stages", 3),
        }
        algorithm = case.get("algorithm", "sequential")
        if algorithm == "chunk_parallel":
            segment = case.get("segment_tokens", 256)
            segments = -(-max_tokens // segment) + max_seqs
            config.update(
                algorithm=algorithm,
                segment_tokens=segment,
                window_tiles=tiles_capacity(max_tokens, segments),
            )
        else:
            from b12x.sequence._shared.delta_prefill.workspace import (
                SM12X_SHARED_MEMORY_LIMIT,
                default_window_tiles,
                recurrence_shared_bytes,
            )

            config["window_tiles"] = case.get("window_tiles") or default_window_tiles(
                value_heads, max_tokens, max_seqs
            )
            # Fuzzed geometry may not fit; shed stages, then value rows.
            while config["stages"] > 2 or config["v_split"] > 16:
                if (
                    recurrence_shared_bytes(
                        config["v_split"],
                        config["k_split"],
                        config["stages"],
                        config["window_tiles"],
                    )
                    <= SM12X_SHARED_MEMORY_LIMIT
                ):
                    break
                if config["stages"] > 2:
                    config["stages"] -= 1
                else:
                    config["v_split"] //= 2
            if recipe == "gdn":
                config["algorithm"] = algorithm
        config_type = op.GdnPrefillConfig if recipe == "gdn" else op.KdaPrefillConfig
        override = config_type.from_config(FrozenMapping(config))
    plan = op.plan(
        caps,
        invocation=FrozenMapping(invocation_from_tensors(**tensors)),
        override=override,
    )
    args = dict(tensors)
    if recipe == "gdn":
        args["a"], args["b"] = args.pop("raw_g"), args.pop("raw_beta")
    run_kwargs = {"lower_bound": -5.0} if recipe == "kda" else {}

    def call(state):
        (spec,) = state.layout.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = state.bind(scratch=scratch, **args)
        return PreparedCall(
            run=lambda: state.run(binding, **run_kwargs), owners=(scratch, binding)
        )

    session = _frozen_session(
        device, plan.request(name=f"race.{recipe}_prefill", prepare_call=call)
    )
    state = plan.prepared.state
    (spec,) = state.layout.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    binding = op.bind(plan, scratch=scratch, **args)
    # make_inputs reads slots [0, S) and writes final slots [S, 2S) and
    # checkpoint slots [2S, 3S): restore and compare only the written ones.
    written = binding.recurrent_state[max_seqs : 3 * max_seqs]
    output = binding.output
    pristine = [(written, written.clone()), (output, output.clone())]

    def forward():
        op.run(binding, **run_kwargs)

    try:
        windows = state.layout.launched_windows(None, None)
    except Exception:  # noqa: BLE001 - description only
        windows = "?"
    return Scenario(
        forward=forward,
        outputs={"output": output, "state": written},
        reset=_restorer(pristine),
        close=lambda: session.__exit__(None, None, None),
        keepalive=[session, plan, binding, scratch, tensors, pristine],
        description=(
            f"{recipe.upper()} prefill heads {key_heads}/{value_heads} lengths {lengths} "
            f"config {override.to_dict() if override is not None else 'default'} windows {windows}"
        ),
    )


def seq_gdn_prefill(device, case) -> Scenario:
    return _prefill(device, case, "gdn")


def seq_kda_prefill(device, case) -> Scenario:
    return _prefill(device, case, "kda")


# --------------------------------------------------------------------------
# GDN speculative decode (CuTe grouped-head kernel + gated RMSNorm)


def seq_gdn_decode(device, case) -> Scenario:
    from b12x.preparation import PreparedCall
    from b12x.sequence import gdn_decode as gdn

    lengths = tuple(
        case.get("lengths", (4, 3, 1, 2, 4, 4, 2, 1, 3, 4, 4, 1, 2, 3, 4, 2))
    )
    key_heads = case.get("key_heads", 8)
    value_heads = 3 * key_heads
    columns = case.get("columns", 4)
    max_seqs = len(lengths)
    max_tokens = sum(lengths)
    slots = max_seqs * columns + 1
    gen = _generator(case)
    accepted = [
        1 + int(torch.randint(0, length, (1,), generator=gen)) for length in lengths
    ]
    caps = gdn.Caps(
        device=device,
        max_tokens=max_tokens,
        max_seqs=max_seqs,
        max_state_slots=slots,
        key_heads=key_heads,
        value_heads=value_heads,
        state_index_columns=columns,
        state_dtype=torch.float32,
        gate_activation=case.get("activation", "sigmoid"),
        qk_l2norm=True,
    )

    def randn(shape, dtype=torch.bfloat16, scale=0.25):
        return (
            (torch.randn(shape, generator=gen) * scale)
            .to(dtype)
            .to(device)
            .contiguous()
        )

    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    tensors = {
        "mixed_qkv": randn((max_tokens, caps.packed_qkv_width)),
        "a": randn((max_tokens, value_heads)),
        "b": randn((max_tokens, value_heads)),
        "z": randn((max_tokens, value_heads, 128)),
        "A_log": randn((value_heads,), torch.float32, 0.1),
        "dt_bias": randn((value_heads,), torch.float32, 0.1),
        "norm_weight": (1.0 + randn((128,), torch.bfloat16, 0.05)).contiguous(),
        "recurrent_state": randn((slots, value_heads, 128, 128), torch.float32, 0.1),
        "query_start_loc": torch.tensor(starts, dtype=torch.int32, device=device),
        "num_accepted_tokens": torch.tensor(accepted, dtype=torch.int32, device=device),
        "state_indices": torch.arange(
            max_seqs * columns, dtype=torch.int32, device=device
        )
        .add_(1)
        .view(max_seqs, columns),
        "num_seqs": torch.tensor([max_seqs], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([max_tokens], dtype=torch.int32, device=device),
        "output": torch.full(
            (max_tokens, value_heads, 128), 7.0, dtype=torch.bfloat16, device=device
        ),
    }
    plan = gdn.plan(caps, invocation=gdn.invocation_from_tensors(caps, **tensors))

    def call(state):
        if "scratch" not in tensors:
            (spec,) = state.layout.scratch_specs()
            tensors["scratch"] = torch.empty(
                spec.shape, dtype=spec.dtype, device=spec.device
            )
        binding = state.bind(**tensors)
        return PreparedCall(run=lambda: state.run(binding))

    session = _frozen_session(
        device, plan.request(name="race.gdn_decode", prepare_call=call)
    )
    binding = gdn.bind(plan, **tensors)
    pool, output = tensors["recurrent_state"], tensors["output"]
    pristine = [(pool, pool.clone()), (output, output.clone())]

    def forward():
        gdn.run(binding)

    return Scenario(
        forward=forward,
        outputs={"output": output, "state": pool},
        reset=_restorer(pristine),
        close=lambda: session.__exit__(None, None, None),
        keepalive=[session, plan, binding, tensors, pristine],
        description=(
            f"GDN decode heads {key_heads}/{value_heads} columns {columns} "
            f"lengths {lengths} accepted {tuple(accepted)}"
        ),
    )


# --------------------------------------------------------------------------
# Short convolution


def seq_short_conv(device, case) -> Scenario:
    from b12x.preparation import PreparedCall
    from b12x.sequence import short_conv

    mode = case.get("mode", "prefill")
    channels, kernel_size = case.get("channels", 10240), case.get("kernel_size", 4)
    dilation, speculative = case.get("dilation", 1), case.get("speculative", 3)
    lengths = list(case.get("lengths", (128, 100, 40, 32)))
    max_seqs = len(lengths)
    max_tokens = case.get("max_tokens") or max(1, sum(lengths))
    slots_total = max_seqs + 2
    gen = _generator(case)
    history = dilation * (kernel_size - 1)
    capacity = history + speculative
    stride = channels * capacity + 32
    storage = (
        (torch.randn((slots_total * stride,), generator=gen) * 0.5)
        .to(torch.bfloat16)
        .to(device)
    )
    state = storage.as_strided((slots_total, channels, capacity), (stride, capacity, 1))
    x = (
        (torch.randn((max_tokens, channels), generator=gen))
        .to(torch.bfloat16)
        .to(device)
    )
    weight = (
        (torch.randn((channels, kernel_size), generator=gen) * 0.2)
        .to(torch.bfloat16)
        .to(device)
    )
    prefill = [
        mode == "prefill" or (mode == "mixed" and i % 2 == 0) for i in range(max_seqs)
    ]
    if mode == "decode":
        lengths = [min(length, 1 + speculative) for length in lengths]
    elif mode == "mixed":
        lengths = [
            length if p else min(length, 1 + speculative)
            for length, p in zip(lengths, prefill)
        ]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    accepted = [
        1 if p else 1 + i % length
        for i, (length, p) in enumerate(zip(lengths, prefill))
    ]
    i32 = dict(dtype=torch.int32, device=device)
    args = dict(
        x=x,
        conv_weight=weight,
        conv_state=state,
        out=torch.full_like(x, float("nan")),
        query_start_loc=torch.tensor(starts, **i32),
        # Slot order differs from request order; slot 0 stays untouched.
        state_slot_ids=torch.tensor(
            list(range(max_seqs, 0, -1)), dtype=torch.int64, device=device
        ),
        state_is_fresh=torch.tensor(
            [i % 3 == 1 for i in range(max_seqs)], dtype=torch.bool, device=device
        ),
        num_accepted_tokens=torch.tensor(accepted, **i32),
        num_seqs=torch.tensor([max_seqs], **i32),
        num_tokens=torch.tensor([sum(lengths)], **i32),
    )
    if mode == "mixed":
        args["request_is_prefill"] = torch.tensor(
            prefill, dtype=torch.bool, device=device
        )
    caps = short_conv.Caps(
        device=device,
        mode=mode,
        channels=channels,
        kernel_size=kernel_size,
        dilation=dilation,
        max_tokens=max_tokens,
        max_seqs=max_seqs,
        max_state_slots=slots_total,
        max_speculative_tokens=speculative,
    )
    plan = short_conv.plan(caps, invocation=short_conv.invocation_from_tensors(**args))
    box = []

    def call(prepared):
        scratch = {
            s.name: torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in plan.scratch_specs()
        }
        binding = prepared.bind(plan=plan, scratch=scratch, **args)
        box.append(binding)
        return PreparedCall(run=lambda: prepared.run(binding), output=binding.out)

    session = _frozen_session(
        device, plan.request(name="race.short_conv", prepare_call=call)
    )
    binding = box[0]
    out = args["out"]
    pristine = [(storage, storage.clone()), (out, torch.full_like(out, float("nan")))]

    def forward():
        short_conv.run(binding)

    return Scenario(
        forward=forward,
        outputs={"out": out, "state": storage},
        reset=_restorer(pristine),
        close=lambda: session.__exit__(None, None, None),
        keepalive=[session, plan, binding, args, storage, pristine],
        description=(
            f"short conv {mode} C={channels} taps={kernel_size} dilation={dilation} "
            f"spec={speculative} lengths {tuple(lengths)}"
        ),
    )


FAMILIES = {
    "seq_gdn_prefill": seq_gdn_prefill,
    "seq_kda_prefill": seq_kda_prefill,
    "seq_gdn_decode": seq_gdn_decode,
    "seq_short_conv": seq_short_conv,
}

CASES = {
    "seq_gdn_prefill": [
        {},
        # Many windows: ring slots and ready flags are reused within each replay.
        {"window_tiles": 8, "v_split": 32},
        {"algorithm": "chunk_parallel", "segment_tokens": 128},
    ],
    "seq_kda_prefill": [
        {},
        {"window_tiles": 6, "k_split": 2, "stages": 2},
    ],
    "seq_gdn_decode": [
        # 16 x 24 heads x 4 value tiles = 1536 work items: one per persistent CTA.
        {},
        # More work items than the 1536 persistent CTAs, so CTAs run a second
        # item through the same shared q/k/decay/beta: single-token requests
        # (ungrouped path) and multi-token MTP requests (grouped-head path).
        # Correct results depend on the CTA barrier that opens each work item
        # (local-inference-lab/flashinfer#26); without it, replays of both cases
        # disagree by 0.9-1.2x the output scale even without contention.
        {"key_heads": 4, "lengths": (1,) * 48, "columns": 1},
        {"lengths": (4, 3, 2, 4, 4, 2, 3, 4) * 4, "columns": 4},
    ],
    "seq_short_conv": [
        {"mode": "prefill"},
        {"mode": "mixed", "lengths": (200, 3, 64, 4, 17, 1), "dilation": 2},
    ],
}


def _fuzz_lengths(rng, longest):
    lengths = [
        rng.choice((0, 1, 15, 16, 17, 64, 100, 129, rng.randint(1, longest)))
        for _ in range(rng.randint(1, 6))
    ]
    if not any(lengths):
        lengths[0] = 16
    return tuple(lengths)


def _fuzz_prefill(rng, recipe):
    case = {
        "lengths": _fuzz_lengths(rng, 600),
        "key_heads": rng.choice((2, 4, 8, 16))
        if recipe == "gdn"
        else rng.choice((4, 8, 16, 32)),
        "checkpoint": rng.random() < 0.7,
        "seed": rng.getrandbits(31),
    }
    roll = rng.random()
    if recipe == "gdn" and roll < 0.2:
        case.update(
            algorithm="chunk_parallel", segment_tokens=rng.choice((128, 256, 512))
        )
    elif recipe == "gdn" and roll < 0.3:
        case["fp32"] = True
    elif roll < 0.8:
        case.update(
            v_split=rng.choice((16, 32, 64, 128)),
            k_split=rng.choice((1, 2, 4)),
            stages=rng.choice((2, 3, 4)),
            window_tiles=rng.randint(1, 16),
        )
        if 2 * case["v_split"] * case["k_split"] + 32 > 1024:
            case["k_split"] = 1
    return case


def _fuzz_gdn_decode(rng):
    columns = rng.randint(1, 8)
    return {
        "lengths": tuple(rng.randint(1, columns) for _ in range(rng.randint(1, 48))),
        "key_heads": rng.choice((1, 2, 4, 8, 16)),
        "columns": columns,
        "activation": rng.choice(("sigmoid", "silu")),
        "seed": rng.getrandbits(31),
    }


def _fuzz_short_conv(rng):
    return {
        "mode": rng.choice(("decode", "prefill", "mixed")),
        "channels": 128 * rng.randint(1, 96),
        "kernel_size": rng.randint(2, 5),
        "dilation": rng.randint(1, 3),
        "speculative": rng.randint(0, 4),
        "lengths": tuple(rng.randint(1, 300) for _ in range(rng.randint(1, 6))),
        "seed": rng.getrandbits(31),
    }


FUZZ = {
    "seq_gdn_prefill": lambda rng: _fuzz_prefill(rng, "gdn"),
    "seq_kda_prefill": lambda rng: _fuzz_prefill(rng, "kda"),
    "seq_gdn_decode": _fuzz_gdn_decode,
    "seq_short_conv": _fuzz_short_conv,
}
