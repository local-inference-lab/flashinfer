"""Regression tests for EXL3 roofline launch compilation in CPU-only compiler workers."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from b12x.moe._shared.kernels import exl3_roofline_launch as launch_mod


def _stub_compile_environment(monkeypatch, *, cuda_initialized: bool, lut_sentinel):
    """Stub CuTe/Triton compilation so compile_exl3_roofline_launches runs on CPU."""

    class _FakeCarrier:
        def __init__(self, tag: str):
            self.tag = tag

        def carriers(self):
            return (self,)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev=None: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _ordinal: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: cuda_initialized)
    monkeypatch.setattr(launch_mod, "get_num_sm", lambda _dev: 188)
    monkeypatch.setattr(launch_mod, "current_cuda_stream", lambda: 0)
    monkeypatch.setattr(
        launch_mod,
        "b12x_compile",
        lambda _fn, *args, compile_spec=None, **kwargs: _FakeCarrier(
            getattr(compile_spec, "family", "kernel")
        ),
    )
    monkeypatch.setattr(
        launch_mod,
        "compile_w4a16_route_pack_launches",
        lambda **_kwargs: SimpleNamespace(carriers=lambda: ()),
    )

    lut_calls: list[torch.device] = []

    def _fake_lut(device):
        lut_calls.append(device)
        if not cuda_initialized:
            raise RuntimeError("compiler workers cannot initialize CUDA")
        return lut_sentinel

    monkeypatch.setattr(launch_mod, "lut_e4m3_direct_table", _fake_lut)
    return lut_calls


def test_compile_exl3_roofline_launches_skips_cuda_lut_in_compiler_worker(monkeypatch):
    """CPU-only b12x compiler workers must not touch CUDA to materialize direct_lut."""
    lut_calls = _stub_compile_environment(
        monkeypatch, cuda_initialized=False, lut_sentinel=None
    )
    launches = launch_mod.compile_exl3_roofline_launches(
        capacity_tokens=16,
        hidden_size=1024,
        intermediate_size=256,
        num_experts=16,
        top_k=4,
        trellis_bits=2,
        prefill_mode="mxfp8",
        device=torch.device("cuda", 0),
    )
    assert lut_calls == []
    assert launches.direct_lut is None


def test_compile_exl3_roofline_launches_retains_direct_lut_when_cuda_initialized(
    monkeypatch,
):
    """Parent GPU processes keep pre-materializing direct_lut before graph capture."""
    sentinel = torch.empty(1, dtype=torch.uint8)
    lut_calls = _stub_compile_environment(
        monkeypatch, cuda_initialized=True, lut_sentinel=sentinel
    )
    launches = launch_mod.compile_exl3_roofline_launches(
        capacity_tokens=16,
        hidden_size=1024,
        intermediate_size=256,
        num_experts=16,
        top_k=4,
        trellis_bits=2,
        prefill_mode="w2a16",
        device=torch.device("cuda", 0),
    )
    assert lut_calls == [torch.device("cuda", 0)]
    assert launches.direct_lut is sentinel
