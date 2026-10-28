"""Tile tactics of the prefill short convolution: legal space, selection identity, exactness.

Every tile geometry computes each output element with the same operations in
the same order, so all geometries and both address widths must agree bitwise.
"""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
import triton
from b12x.preparation import (
    DetectedDevice,
    DeviceIdentity,
    FrozenMapping,
    PreparationSession,
    PreparedCall,
)
from b12x.sequence import short_conv
from b12x.sequence.short_conv._tuning import (
    FIXED,
    FIXED_TILE,
    TUNING,
    ShortConvConfig,
    ShortConvQuery,
    live_values_per_thread,
    program_payload,
)

from .test_short_conv import (
    _Case,
    _prepared,
    require_b12x,
    tactic_id,
    tactic_is_legal,
    tactics,
)

IDENTITY = DeviceIdentity("nvidia", (10, 3), 152, "synthetic SM103")
# The default is the fixed 32 x 128 tile with Int64 addressing on every device.
DEFAULT = FIXED_TILE
# GLM-5.3 KDA offsets fit Int32, so the same tile is also offered with narrow
# addressing.
NARROW = replace(FIXED_TILE, wide_index=False)


def _query(**changes):
    values = dict(
        mode="prefill",
        channels=24576,
        kernel_size=4,
        dilation=1,
        max_tokens=4096,
        max_seqs=32,
        max_state_slots=64,
        max_speculative_tokens=0,
        autotune_effort="default",
    )
    return ShortConvQuery(**(values | changes))


def _candidates(query, device=IDENTITY):
    return tuple(c for _, c in TUNING.eligible_plan(query, device).candidates)


def test_prefill_space_contains_the_fixed_geometry_and_only_budgeted_tiles():
    query = _query()
    candidates = tuple(c for c in _candidates(query) if c.backend == "triton")
    assert DEFAULT in candidates
    assert NARROW in candidates
    assert TUNING.default_config(query, IDENTITY) == DEFAULT
    assert len(candidates) == len(set(candidates)) == 40
    for config in candidates:
        assert live_values_per_thread(config, query.kernel_size) <= 192
        assert config.block_t in (8, 16, 32, 64)
        assert config.block_c in (64, 128, 256)
        assert config.num_warps in (4, 8)
    # Tiles of 64 x 256 or 32 x 256 on four warps keep 64 live values per tap.
    assert replace(DEFAULT, block_t=64, block_c=256) not in candidates
    assert replace(DEFAULT, block_t=32, block_c=256) not in candidates


def test_tile_space_follows_channel_count_and_tap_count():
    narrow = _candidates(_query(channels=65, max_tokens=300))
    assert max(c.block_c for c in narrow) == 128
    seven_taps = _query(kernel_size=7)
    wide_taps = _candidates(seven_taps)
    assert all(live_values_per_thread(c, 7) <= 192 for c in wide_taps if c != DEFAULT)
    # The fixed tile stays a candidate and the default with seven taps although
    # its live values exceed the register budget that bounds the other tiles.
    assert live_values_per_thread(DEFAULT, 7) > 192
    assert DEFAULT in wide_taps
    assert TUNING.default_config(seven_taps, IDENTITY) == DEFAULT
    assert sum(live_values_per_thread(c, 7) > 192 for c in wide_taps) == 1


@pytest.mark.parametrize(
    "changes",
    [{"mode": "decode"}, {"max_tokens": 255}, {"mode": "mixed", "max_tokens": 16}],
)
def test_decode_and_untiled_plans_keep_one_fixed_geometry(changes):
    query = _query(**changes)
    assert not query.tiled
    assert _candidates(query) == (FIXED,)
    with pytest.raises(ValueError, match="fixed geometry"):
        TUNING.validate_config(query, DEFAULT, IDENTITY)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0), (12, 1)])
def test_tile_candidates_and_default_do_not_depend_on_the_device(capability):
    device = DeviceIdentity("nvidia", capability, 188, "any")
    query = _query()
    assert _candidates(query, device) == _candidates(query, IDENTITY)
    assert TUNING.default_config(query, device) == DEFAULT
    assert len(_candidates(query, device)) == 40
    for config in (DEFAULT, NARROW, replace(DEFAULT, block_t=16, wide_index=False)):
        TUNING.validate_config(query, config, device)


def test_int32_addressing_needs_a_fitting_query():
    wide = _query(input_row_stride=2**31 // 4096 + 64)
    assert TUNING.default_config(wide, IDENTITY) == DEFAULT
    assert {c.wide_index for c in _candidates(wide)} == {True}
    with pytest.raises(ValueError, match="Int32 addressing"):
        TUNING.validate_config(wide, NARROW, IDENTITY)
    assert {c.wide_index for c in _candidates(_query())} == {True, False}
    TUNING.validate_config(_query(), NARROW, IDENTITY)
    TUNING.validate_config(_query(), DEFAULT, IDENTITY)


def test_mixed_prefill_plans_race_tiles():
    assert _query(mode="mixed").tiled
    assert len(_candidates(_query(mode="mixed"))) == 40


def test_heuristic_effort_prepares_only_the_default_tile():
    query = _query(autotune_effort="off")
    assert _candidates(query) == (DEFAULT,)
    assert _candidates(_query(autotune_effort="audit")).count(DEFAULT) == 1


@pytest.mark.parametrize(
    "config,message",
    [
        (replace(DEFAULT, block_t=128), "outside the supported geometry"),
        (replace(DEFAULT, block_c=512), "outside the supported geometry"),
        (replace(DEFAULT, num_warps=2), "outside the supported geometry"),
        (replace(DEFAULT, block_t=64, block_c=256), "live values per thread"),
        (replace(DEFAULT, backend="cutedsl"), "unsupported short convolution backend"),
    ],
)
def test_illegal_tiles_are_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        TUNING.validate_config(_query(), config, IDENTITY)


def test_int32_addressing_requires_every_offset_to_fit():
    # GLM-5.3 KDA: 4096 rows of 24768 channels, strided input rows.
    assert _query(input_row_stride=24768).int32_addressing
    assert not _query(input_row_stride=2**31 // 4096 + 64).int32_addressing
    # Partial tiles extend the live extents by the largest tile.
    just_fits = (2**31 - 24576 - 256 - 1) // (4096 + 64)
    assert _query(input_row_stride=just_fits).int32_addressing
    assert not _query(input_row_stride=just_fits + 1).int32_addressing
    # The gathered history scratch and the output rows bound the other offsets.
    assert not _query(max_seqs=2**31 // (24576 * 3)).int32_addressing
    assert not _query(max_tokens=2**31 // 24576).int32_addressing


def test_int32_addressing_covers_masked_history_and_negative_input_offsets():
    limit = 2**31
    # A query-sourced masked lane reaches history position
    # state_length + max_tokens + 63, so the history scratch bound includes the
    # token rows. With few channels one more sequence moves the bound across
    # the limit by less than one row count.
    channels, state_length = 64, 3
    columns = channels + 256
    rows = 4096 + 64
    largest_fit = (limit - 1 - state_length - columns * state_length) // (
        channels * state_length
    )
    fits_without_rows = _query(channels=channels, max_seqs=largest_fit)
    assert (
        fits_without_rows.max_seqs * channels + columns
    ) * state_length + state_length < limit
    assert (
        fits_without_rows.max_seqs * channels + columns
    ) * state_length + state_length + rows >= limit
    assert not fits_without_rows.int32_addressing
    assert _query(channels=channels, max_seqs=largest_fit - 64).int32_addressing
    # A lane before the start of its request reads input row -state_length, a
    # negative offset of magnitude state_length * input_row_stride.
    stride = 16_000_000
    short = _query(channels=64, max_tokens=64, max_seqs=1, input_row_stride=stride)
    assert short.int32_addressing
    deep = _query(
        channels=64, max_tokens=64, max_seqs=1, input_row_stride=stride, dilation=64
    )
    assert deep.state_length * stride >= limit
    assert (64 + 64) * stride < limit
    assert not deep.int32_addressing


def test_selection_key_excludes_the_state_pool_size_and_includes_the_effort():
    configuration = lambda query: TUNING.configure(query, device=IDENTITY).encoded_query
    base = _query()
    assert configuration(base) == configuration(replace(base, max_state_slots=26_130))
    assert "max_state_slots" not in configuration(base)
    assert configuration(base) != configuration(replace(base, autotune_effort="audit"))
    assert configuration(base) != configuration(replace(base, input_row_stride=24768))


def test_program_payload_rebuilds_the_query_without_the_effort():
    audit = _query(autotune_effort="audit", input_row_stride=24768)
    payload = program_payload(audit)
    assert payload == program_payload(replace(audit, autotune_effort="off"))
    rebuilt = ShortConvQuery(**dict(FrozenMapping(payload)))
    assert rebuilt == replace(audit, autotune_effort="default")
    assert rebuilt.max_state_slots == audit.max_state_slots


def test_compile_jobs_carry_the_candidate_geometry():
    plan = short_conv.plan(
        short_conv.Caps(
            device="cuda:0",
            mode="prefill",
            channels=24576,
            kernel_size=4,
            max_tokens=4096,
            max_seqs=32,
            max_state_slots=64,
        )
    )
    detected = DetectedDevice(
        ordinal=0,
        identity=IDENTITY,
        uuid="synthetic",
        max_shared_memory_per_block=227 * 1024,
    )
    configs = _candidates(plan.query)
    jobs = {config: plan._compile_jobs(config, detected) for config in configs}
    assert {len(job) for job in jobs.values()} == {1}
    assert len({FrozenMapping(job[0].args[1]) for job in jobs.values()}) == len(configs)
    assert all(
        ShortConvConfig.from_config(FrozenMapping(job[0].args[1])) == config
        for config, job in jobs.items()
    )


def test_config_round_trips_through_its_payload():
    for config in tactics():
        assert ShortConvConfig.from_config(FrozenMapping(config.to_dict())) == config
    with pytest.raises(ValueError, match="config fields"):
        ShortConvConfig.from_config(FrozenMapping({"backend": "triton"}))


# --- GPU checks ----------------------------------------------------------


def _batch(case, lengths, slots, fresh):
    case.state.normal_()
    case.x.normal_()
    case.batch(lengths, slots, fresh, [1] * len(lengths))


_SMALL = [1, 2, 3, 5, 7, 1, 1, 2, 3, 5, 13, 64, 1, 2, 100, 50]


def _scenario(name, tokens):
    """Request lengths, slots, fresh flags and rows cut from num_tokens for ``name``."""
    if name == "one-carried":
        return [tokens], [2], [False], 0
    if name == "four-fresh-and-carried":
        # An inactive second request and rows past num_tokens.
        return (
            [1500, 1, 700, 1796],
            [3, -1, 1, 4],
            [True, False, False, True],
            41,
        )
    if name == "four-equal":
        return [tokens // 4] * 4, [0, 1, 2, 3], [False, True, False, True], 0
    # Seventeen requests: starts fall at many positions within a token tile, the
    # ninth one is inactive, and the last one fills the capacity.
    lengths = _SMALL + [tokens - sum(_SMALL)]
    slots = list(range(len(lengths)))
    slots[8] = -1
    return lengths, slots, [bool(i % 2) for i in range(len(lengths))], 0


@pytest.mark.parametrize(
    "scenario,max_tokens,channels,round_before_silu,bias_dtype",
    [
        ("four-fresh-and-carried", 4096, 24576, True, None),
        ("four-fresh-and-carried", 4093, 24576, True, None),
        ("four-fresh-and-carried", 4093, 24576, False, torch.bfloat16),
        ("four-fresh-and-carried", 4096, 24576, False, torch.float32),
        ("one-carried", 4096, 24576, True, None),
        ("four-equal", 4096, 24576, True, None),
        ("irregular", 4096, 24576, True, None),
        ("irregular", 4093, 24576, False, torch.float32),
        ("four-fresh-and-carried", 4096, 6288, True, None),
        ("irregular", 4096, 6288, False, torch.bfloat16),
        ("four-fresh-and-carried", 4096, 3072, True, None),
    ],
)
def test_every_tactic_is_bitwise_equal_to_the_default_geometry(
    scenario, max_tokens, channels, round_before_silu, bias_dtype
):
    """Prefill with fresh and carried history, strided rows, inactive requests, and
    rows past the live tokens: every candidate equals the device's default tile,
    in the output rows and the updated convolution state."""
    outputs, states = {}, {}
    lengths, slots, fresh, cut = _scenario(scenario, max_tokens)
    assert sum(lengths) <= max_tokens
    for config in (None, *tactics()):
        torch.manual_seed(11)
        case = _Case(
            "prefill",
            channels=channels,
            input_padding=192,
            max_tokens=max_tokens,
            speculative_tokens=0,
            round_before_silu=round_before_silu,
            bias_dtype=bias_dtype,
            max_seqs=len(lengths),
            state_slots=len(lengths) + 2,
        )
        if not tactic_is_legal(case, config):
            continue
        with _prepared(case, config) as binding:
            _batch(case, lengths, slots, fresh)
            case.args["num_tokens"].fill_(sum(lengths) - cut)
            case.args["out"].fill_(float("nan"))
            short_conv.run(binding)
            torch.cuda.synchronize()
            outputs[config] = binding.out.clone()
            states[config] = case.state.clone()
            live = int(case.args["num_tokens"].item())
            assert not outputs[config][live:].any()
            assert torch.isfinite(outputs[config]).all()
    reference = outputs[None], states[None]
    for config in outputs:
        label = tactic_id(config) if config else "default"
        assert torch.equal(outputs[config], reference[0]), label
        assert torch.equal(states[config], reference[1]), label


@pytest.mark.parametrize("dilation", [1, 3])
@pytest.mark.parametrize("tactic", tactics(), ids=tactic_id)
def test_prefill_chunks_carry_history_through_every_tactic(tactic, dilation):
    """Chunked prefill with carried history equals one canonical convolution."""
    case = _Case("prefill", channels=65, dilation=dilation, max_tokens=300)
    sequence = case.x.clone()
    full_input = F.pad(sequence.T.unsqueeze(0), (case.length, 0))
    expected = (
        F.silu(
            F.conv1d(full_input, case.weight.unsqueeze(1), groups=65, dilation=dilation)
        )
        .squeeze(0)
        .T
    )
    outputs = []
    with _prepared(case, tactic) as binding:
        start = 0
        for length in (1, 2, 5, 4, 100, 188):
            case.x[:length].copy_(sequence[start : start + length])
            case.batch([length], [3], [start == 0], [1])
            short_conv.run(binding)
            outputs.append(binding.out[:length].clone())
            start += length
        torch.testing.assert_close(torch.cat(outputs), expected, rtol=0.008, atol=0.002)
        torch.testing.assert_close(
            case.state[3, :, : case.length], sequence[-case.length :].T, rtol=0, atol=0
        )


@pytest.mark.parametrize("tactic", tactics(), ids=tactic_id)
def test_every_tactic_rounds_bf16_before_silu(tactic):
    case = _Case("prefill", channels=64, max_tokens=300, speculative_tokens=0)
    case.x.fill_(1)
    case.weight.zero_()
    case.weight[:, 0] = 1
    case.weight[:, 1] = 1 / 256
    with _prepared(case, tactic) as binding:
        case.state.fill_(1)
        case.batch([200], [0], [False], [1])
        short_conv.run(binding)
        expected = F.silu(torch.ones_like(binding.out[:200]))
        unrounded = F.silu(torch.full((200, 64), 1 + 1 / 256, device=case.x.device)).to(
            torch.bfloat16
        )
        assert not torch.equal(expected, unrounded)
        torch.testing.assert_close(binding.out[:200], expected, rtol=0, atol=0)


def _direct_launch(wide, tile, x, gathered, weight, starts, ids, slots, count):
    from b12x.sequence import _short_conv as kernels

    block_t, block_c, warps = tile
    out = torch.full_like(x, float("nan"))
    channels = x.shape[1]
    kernels._short_conv_tiled_kernel[
        (triton.cdiv(x.shape[0], block_t), triton.cdiv(channels, block_c))
    ](
        x,
        gathered,
        weight,
        starts,
        ids,
        slots,
        count,
        out,
        channels,
        3,
        4,
        1,
        block_c,
        x.stride(0),
        block_t,
        x.shape[0],
        None,
        True,
        wide,
        num_warps=warps,
    )
    return out


@pytest.mark.parametrize("tile", [(32, 128, 4), (8, 256, 8), (64, 64, 4), (16, 64, 8)])
def test_int32_and_int64_addressing_agree_bitwise(tile):
    """Addressing width changes offsets only: rows of strided input, history from the
    gathered scratch, and partial tiles in both dimensions."""
    device = require_b12x()
    tokens, channels, padding = 1000, 1090, 6
    torch.manual_seed(5)
    x = torch.randn((tokens, channels + padding), device=device).to(torch.bfloat16)[
        :, :channels
    ]
    gathered = torch.randn((4, channels, 3), device=device).to(torch.bfloat16)
    weight = (torch.randn((channels, 4), device=device) * 0.2).to(torch.bfloat16)
    starts = torch.tensor([0, 17, 18, 600, 900], dtype=torch.int32, device=device)
    ids = torch.full((tokens,), -1, dtype=torch.int32, device=device)
    for request in range(4):
        ids[int(starts[request]) : int(starts[request + 1])] = request
    slots = torch.tensor([2, 0, -1, 1], dtype=torch.int64, device=device)
    count = torch.tensor([900], dtype=torch.int32, device=device)
    arguments = (tile, x, gathered, weight, starts, ids, slots, count)
    wide = _direct_launch(True, *arguments)
    narrow = _direct_launch(False, *arguments)
    assert torch.equal(wide, narrow)
    assert not wide[900:].any()
    reference = _direct_launch(True, (32, 128, 4), *arguments[1:])
    assert torch.equal(wide, reference)


def test_race_measures_every_candidate_and_replays_the_selected_tile(tmp_path):
    case = _Case("prefill", channels=4096, max_tokens=512, speculative_tokens=0)
    plan = short_conv.plan(
        case.caps, invocation=short_conv.invocation_from_tensors(**case.args)
    )
    lengths, slots, fresh = [200, 1, 311], [3, 1, 4], [True, False, False]
    _batch(case, lengths, slots, fresh)
    expected, state = case.oracle()
    initial = case.state.clone()
    source = case.x.clone()

    def prepare_call(prepared):
        scratch = {
            s.name: torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in plan.scratch_specs()
        }
        binding = prepared.bind(plan=plan, scratch=scratch, **case.args)

        def reset():
            case.state.copy_(initial)

        def run():
            prepared.run(binding)

        reset()
        case.args["out"].fill_(float("nan"))
        run()
        torch.cuda.synchronize()
        torch.testing.assert_close(binding.out, expected, rtol=0.008, atol=0.002)
        reset()
        return PreparedCall(
            run=run,
            output=binding.out,
            reset=reset,
            produce=lambda: case.x.copy_(source),
            restore=reset,
            owners=(binding,),
        )

    with PreparationSession(
        device=case.x.device,
        compile_workers=4,
        rounds=3,
        samples=4,
        cache_dir=tmp_path,
    ) as session:
        result = session.prepare(
            (
                plan.request(
                    name="short_conv",
                    prepare_call=prepare_call,
                    benchmark_call=prepare_call,
                ),
            )
        )
        session.freeze()
        selected = result.selections["short_conv"].config
        legal = _candidates(plan.query, session.device.identity)
        assert selected in legal
        assert result.coverage["short_conv"]["measured_count"] == len(legal)
        case.state.copy_(initial)
        binding = short_conv.bind(
            plan,
            scratch={
                s.name: torch.empty(s.shape, dtype=s.dtype, device=s.device)
                for s in plan.scratch_specs()
            },
            **case.args,
        )
        short_conv.run(binding)
        torch.testing.assert_close(binding.out, expected, rtol=0.008, atol=0.002)
        torch.testing.assert_close(case.state, state, rtol=0, atol=0)
