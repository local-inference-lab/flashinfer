"""Weighted mHC finalization preserves the BF16 collapse and FP32 norm vector."""

import math

import pytest
import torch

from flashinfer.mhc import (
    _mhc_pre_big_fuse_impl,
    _mhc_pre_big_fuse_with_norm_impl,
)
from flashinfer.utils import get_compute_capability


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def _require_bf16():
    if get_compute_capability(torch.device("cuda"))[0] < 8:
        pytest.skip("mHC BF16 requires SM80+")


def _case(rows, hidden, splits, norm_dtype):
    torch.manual_seed(3917 + rows + hidden + splits)
    residual = (torch.randn(rows, 4, hidden, device="cuda") * 0.1).bfloat16()
    flat = residual.flatten(1).float()
    fn = torch.randn(24, 4 * hidden, device="cuda") * 0.02
    span = flat.shape[1] // splits
    dot = torch.stack(
        [
            flat[:, i * span : (i + 1) * span] @ fn[:, i * span : (i + 1) * span].T
            for i in range(splits)
        ]
    )
    squares = torch.stack(
        [flat[:, i * span : (i + 1) * span].square().sum(-1) for i in range(splits)]
    )
    if splits == 1:
        dot, squares = dot[0], squares[0]
    norm = (torch.rand(hidden, device="cuda") + 0.5).to(norm_dtype)
    if norm_dtype == torch.float32:
        assert torch.count_nonzero(norm != norm.bfloat16().float())
    inputs = dict(
        dot_mix=dot,
        sqrsum=squares,
        residual=residual,
        mhc_scale=torch.tensor([0.2, 0.3, 0.4], device="cuda"),
        mhc_base=torch.randn(24, device="cuda") * 0.3,
        norm_weight=norm,
    )
    options = dict(
        rms_eps=1e-5 if hidden == 4096 else 1e-20,
        mhc_pre_eps=1e-6,
        mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        norm_eps=1e-5 if hidden == 4096 else 1e-20,
        num_splits=splits,
        block_size=160,
    )
    return inputs, options


def _guarded(shape, dtype):
    count = math.prod(shape)
    backing = torch.full((count + 16,), 17.0, dtype=dtype, device="cuda")
    return backing[8 : 8 + count].view(shape), backing


def _outputs(inputs):
    rows, _, hidden = inputs["residual"].shape
    buffers = [
        _guarded(shape, dtype)
        for shape, dtype in (
            ((rows, 4), torch.float32),
            ((rows, 4, 4), torch.float32),
            ((rows, hidden), torch.bfloat16),
        )
    ]
    views, guards = zip(*buffers, strict=True)
    return dict(zip(("post_mix", "comb_mix", "normalized"), views, strict=True)), guards


def _reference(inputs, options):
    dot, squares = inputs["dot_mix"], inputs["sqrsum"]
    if options["num_splits"] > 1:
        dot, squares = dot.sum(0), squares.sum(0)
    residual = inputs["residual"].float()
    logits = dot * torch.rsqrt(
        squares[:, None] / residual.shape[1:].numel() + options["rms_eps"]
    )
    scale, base = inputs["mhc_scale"], inputs["mhc_base"]
    pre = torch.sigmoid(logits[:, :4] * scale[0] + base[:4]) + options["mhc_pre_eps"]
    post = (
        torch.sigmoid(logits[:, 4:8] * scale[1] + base[4:8])
        * options["mhc_post_mult_value"]
    )
    comb = torch.softmax((logits[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4), -1)
    comb = comb + options["mhc_sinkhorn_eps"]
    comb = comb / (comb.sum(-2, keepdim=True) + options["mhc_sinkhorn_eps"])
    for _ in range(options["sinkhorn_repeat"] - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + options["mhc_sinkhorn_eps"])
        comb = comb / (comb.sum(-2, keepdim=True) + options["mhc_sinkhorn_eps"])
    collapsed = (residual * pre[:, :, None]).sum(1).bfloat16().float()
    normalized = (
        collapsed
        * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + options["norm_eps"])
        * inputs["norm_weight"].float()
    ).bfloat16()
    return post, comb, normalized


def _assert_relative(got, expected, limit):
    assert torch.isfinite(got).all()
    error = (got.float() - expected.float()).norm() / expected.float().norm().clamp_min(
        1e-6
    )
    assert error.item() < limit


# Every split meets every width and block; each also sees both norm dtypes.
CASES = [
    (
        4096 if (split_index, width_index) == (1, 0) else (1, 17, 257)[width_index],
        hidden,
        splits,
        (96, 160, 288)[(width_index + split_index) % 3],
        (torch.bfloat16, torch.float32)[(width_index + split_index) % 2],
    )
    for split_index, splits in enumerate((1, 2, 4, 8, 16))
    for width_index, hidden in enumerate((4096, 5120, 7168))
]


@pytest.mark.parametrize("rows,hidden,splits,block,norm_dtype", CASES)
def test_fused_norm_matches_separate_and_reference(
    rows, hidden, splits, block, norm_dtype
):
    inputs, options = _case(rows, hidden, splits, norm_dtype)
    options["block_size"] = 0 if (splits, hidden) == (1, 5120) else block
    if splits == 8:
        options.update(
            rms_eps=0.01,
            mhc_pre_eps=0.0003,
            mhc_sinkhorn_eps=0.0002,
            mhc_post_mult_value=1.25,
            sinkhorn_repeat=3,
            norm_eps=0.02,
        )
    originals = {name: value.clone() for name, value in inputs.items()}
    outputs, guards = _outputs(inputs)
    reference = _reference(inputs, options)
    separate_post = torch.empty_like(outputs["post_mix"])
    separate_comb = torch.empty_like(outputs["comb_mix"])
    collapsed = torch.empty_like(outputs["normalized"])
    _mhc_pre_big_fuse_impl(
        separate_post,
        separate_comb,
        collapsed,
        *(
            inputs[name]
            for name in ("dot_mix", "sqrsum", "residual", "mhc_scale", "mhc_base")
        ),
        4 * hidden,
        options["rms_eps"],
        options["mhc_pre_eps"],
        options["mhc_sinkhorn_eps"],
        options["mhc_post_mult_value"],
        options["sinkhorn_repeat"],
        splits,
        128,
    )
    wide = collapsed.float()
    separate_norm = (
        wide
        * torch.rsqrt(wide.square().mean(-1, keepdim=True) + options["norm_eps"])
        * inputs["norm_weight"].float()
    ).bfloat16()

    def run():
        _mhc_pre_big_fuse_with_norm_impl(**outputs, **inputs, **options)

    run()
    torch.cuda.synchronize()
    for got, want, limit in zip(
        outputs.values(), reference, (1e-3, 1e-3, 5e-3), strict=True
    ):
        _assert_relative(got, want, limit)
    torch.testing.assert_close(outputs["post_mix"], separate_post, rtol=0, atol=0)
    torch.testing.assert_close(outputs["comb_mix"], separate_comb, rtol=0, atol=0)
    _assert_relative(outputs["normalized"], separate_norm, 5e-3)
    exact = tuple(value.clone() for value in outputs.values())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    allocated = torch.cuda.memory_allocated()
    for _ in range(5):
        for value in outputs.values():
            value.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        for got, want in zip(outputs.values(), exact, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        for guard in guards:
            assert torch.all(guard[:8] == 17) and torch.all(guard[-8:] == 17)
        for name, value in inputs.items():
            torch.testing.assert_close(value, originals[name], rtol=0, atol=0)


BATCHED_ROWS = (1, 2, 3, 513, 4093, 4096)


@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("splits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("rows", BATCHED_ROWS)
def test_two_tokens_per_cta_match_one_token_bitwise(rows, splits, norm_dtype):
    inputs, options = _case(rows, 4096, splits, norm_dtype)
    results = {}
    for tokens in (1, 2):
        outputs, guards = _outputs(inputs)
        _mhc_pre_big_fuse_with_norm_impl(
            **outputs, **inputs, **options, tokens_per_cta=tokens
        )
        torch.cuda.synchronize()
        # An odd row count leaves the second group of the last CTA idle; it must
        # not store past the final row.
        for guard in guards:
            assert torch.all(guard[:8] == 17) and torch.all(guard[-8:] == 17)
        results[tokens] = {name: value.clone() for name, value in outputs.items()}
    for name in ("post_mix", "comb_mix", "normalized"):
        assert torch.equal(results[1][name], results[2][name]), name
    reference = _reference(inputs, options)
    for got, want, limit in zip(
        results[2].values(), reference, (1e-3, 1e-3, 5e-3), strict=True
    ):
        _assert_relative(got, want, limit)


@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("splits", [1, 4])
def test_two_tokens_per_cta_match_one_token_bitwise_on_special_values(splits, norm_dtype):
    rows = 4093
    inputs, options = _case(rows, 4096, splits, norm_dtype)
    specials = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), -0.0], device="cuda"
    )
    # Distinct rows carry each special value in the residual and in the projections.
    residual = inputs["residual"]
    dot = inputs["dot_mix"].view(splits, rows, 24) if splits > 1 else inputs["dot_mix"][None]
    for index, value in enumerate(specials):
        for row in (index, 2 * index + 1000, rows - 1 - index):
            residual[row, index % 4, (index * 97) % 4096] = value
            dot[:, row, (index * 5) % 24] = value
    results = {}
    for tokens in (1, 2):
        outputs, guards = _outputs(inputs)
        _mhc_pre_big_fuse_with_norm_impl(
            **outputs, **inputs, **options, tokens_per_cta=tokens
        )
        torch.cuda.synchronize()
        for guard in guards:
            assert torch.all(guard[:8] == 17) and torch.all(guard[-8:] == 17)
        results[tokens] = outputs
    for name in ("post_mix", "comb_mix", "normalized"):
        # Compare bit patterns so NaN payloads and signed zeros must agree.
        one, two = results[1][name], results[2][name]
        assert torch.equal(one.contiguous().view(torch.int16 if one.element_size() == 2 else torch.int32),
                           two.contiguous().view(torch.int16 if two.element_size() == 2 else torch.int32)), name
    assert not torch.isfinite(results[1]["normalized"]).all()


@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_two_tokens_per_cta_replay_poisoned_outputs_bitwise(norm_dtype):
    inputs, options = _case(4093, 4096, 2, norm_dtype)
    outputs, _ = _outputs(inputs)
    expected_outputs, _ = _outputs(inputs)
    _mhc_pre_big_fuse_with_norm_impl(**expected_outputs, **inputs, **options)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _mhc_pre_big_fuse_with_norm_impl(
            **outputs, **inputs, **options, tokens_per_cta=2
        )
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        for value in outputs.values():
            value.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        for name in outputs:
            assert torch.equal(outputs[name], expected_outputs[name]), name


def test_two_tokens_per_cta_single_row_matches_reference():
    inputs, options = _case(1, 4096, 1, torch.float32)
    outputs, _ = _outputs(inputs)
    _mhc_pre_big_fuse_with_norm_impl(**outputs, **inputs, **options, tokens_per_cta=2)
    for got, want, limit in zip(
        outputs.values(), _reference(inputs, options), (1e-3, 1e-3, 5e-3), strict=True
    ):
        _assert_relative(got, want, limit)


@pytest.mark.parametrize("boundary", ["rounded_variance", "fp32_norm"])
def test_fused_norm_preserves_rounding_boundaries(boundary):
    inputs, options = _case(3, 4096, 1, torch.float32)
    inputs["dot_mix"].zero_()
    inputs["mhc_base"].zero_()
    inputs["residual"].fill_(1)
    if boundary == "rounded_variance":
        inputs["residual"][:, :, 2048:] = 1.5
        inputs["mhc_base"][:4] = 1.7
        inputs["norm_weight"].fill_(1)
    else:
        inputs["residual"][:, :, 2048:] = 2
        inputs["norm_weight"].fill_(1.0038)
        options.update(mhc_pre_eps=1e-20, norm_eps=1e-20)
    outputs, _ = _outputs(inputs)
    expected = _reference(inputs, options)[2]
    _mhc_pre_big_fuse_with_norm_impl(**outputs, **inputs, **options)
    torch.testing.assert_close(outputs["normalized"], expected, rtol=0, atol=0)
    if boundary == "rounded_variance":
        pre = torch.sigmoid(inputs["mhc_base"][:4]) + options["mhc_pre_eps"]
        unrounded = (inputs["residual"].float() * pre[None, :, None]).sum(1)
        incorrect = (
            unrounded.bfloat16().float()
            * torch.rsqrt(
                unrounded.square().mean(-1, keepdim=True) + options["norm_eps"]
            )
        ).bfloat16()
    else:
        incorrect = _reference(
            inputs | {"norm_weight": inputs["norm_weight"].bfloat16()}, options
        )[2]
    assert not torch.equal(expected, incorrect)


@pytest.mark.parametrize(
    "invalid",
    [
        "norm_dtype",
        "norm_shape",
        "norm_stride",
        "normalized_dtype",
        "normalized_shape",
        "residual_stride",
        "residual_alignment",
        "normalized_alignment",
        "dot_shape",
        "square_shape",
        "splits",
        "block",
        "block_overflow",
        "iterations",
        "iterations_overflow",
        "rms_eps",
        "pre_eps",
        "sinkhorn_eps",
        "norm_eps",
        "hidden",
        "normalized_alias",
        "output_alias",
        "tokens_per_cta",
        "tokens_per_cta_block",
        "tokens_per_cta_hidden",
        "tokens_per_cta_norm_alignment",
    ],
)
def test_fused_norm_rejects_invalid_inputs(invalid):
    inputs, options = _case(2, 4096, 4, torch.float32)
    outputs, _ = _outputs(inputs)
    if invalid == "norm_dtype":
        inputs["norm_weight"] = inputs["norm_weight"].half()
    elif invalid == "norm_shape":
        inputs["norm_weight"] = inputs["norm_weight"][:-1]
    elif invalid == "norm_stride":
        inputs["norm_weight"] = torch.empty(8192, device="cuda")[::2]
    elif invalid == "normalized_dtype":
        outputs["normalized"] = outputs["normalized"].float()
    elif invalid == "normalized_shape":
        outputs["normalized"] = outputs["normalized"][:1]
    elif invalid == "residual_stride":
        inputs["residual"] = inputs["residual"].transpose(0, 1)
    elif invalid.endswith("alignment") and not invalid.startswith("tokens"):
        name = invalid.removesuffix("_alignment")
        target = inputs if name == "residual" else outputs
        original = target[name]
        target[name] = torch.empty(
            original.numel() + 1, device="cuda", dtype=original.dtype
        )[1:].view_as(original)
    elif invalid == "dot_shape":
        inputs["dot_mix"] = inputs["dot_mix"][:, :, :23].contiguous()
    elif invalid == "square_shape":
        inputs["sqrsum"] = inputs["sqrsum"][:, :1].contiguous()
    elif invalid == "normalized_alias":
        outputs["normalized"] = inputs["residual"].view(-1)[:8192].view(2, 4096)
    elif invalid == "output_alias":
        outputs["post_mix"] = outputs["comb_mix"].view(-1)[:8].view(2, 4)
    elif invalid == "tokens_per_cta_hidden":
        inputs, options = _case(2, 5120, 4, torch.float32)
        outputs, _ = _outputs(inputs)
        options["tokens_per_cta"] = 2
    elif invalid == "tokens_per_cta_norm_alignment":
        original = inputs["norm_weight"]
        inputs["norm_weight"] = torch.empty(
            original.numel() + 1, device="cuda", dtype=original.dtype
        )[1:].view_as(original)
        options["tokens_per_cta"] = 2
    elif invalid == "hidden":
        inputs["residual"] = inputs["residual"][:, :, :2048].contiguous()
        inputs["norm_weight"] = inputs["norm_weight"][:2048]
        outputs["normalized"] = outputs["normalized"][:, :2048].contiguous()
    else:
        key, value = {
            "splits": ("num_splits", 3),
            "block": ("block_size", 128),
            "block_overflow": ("block_size", 2**32 + 96),
            "iterations": ("sinkhorn_repeat", 0),
            "iterations_overflow": ("sinkhorn_repeat", 2**32 + 1),
            "rms_eps": ("rms_eps", 0),
            "pre_eps": ("mhc_pre_eps", 0),
            "sinkhorn_eps": ("mhc_sinkhorn_eps", 0),
            "norm_eps": ("norm_eps", 0),
            "tokens_per_cta": ("tokens_per_cta", 3),
            "tokens_per_cta_block": ("tokens_per_cta", 2),
        }[invalid]
        if invalid == "tokens_per_cta_block":
            options["block_size"] = 96
        options[key] = value
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        _mhc_pre_big_fuse_with_norm_impl(**outputs, **inputs, **options)
