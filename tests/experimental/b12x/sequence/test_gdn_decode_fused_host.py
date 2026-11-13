"""Host dispatch contracts for fused KDA verification without CUDA execution."""

from contextlib import contextmanager
import importlib
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch


def test_support_probe_does_not_import_fused_cutlass_kernel():
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib.metadata
        import sys

        version = importlib.metadata.version
        def without_cutlass(name):
            if name in ("nvidia-cutlass-dsl", "cutlass"):
                raise importlib.metadata.PackageNotFoundError(name)
            return version(name)
        importlib.metadata.version = without_cutlass

        class NoCutlass(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cutlass" or fullname.startswith("cutlass."):
                    raise ModuleNotFoundError("Cutlass unavailable", name=fullname)

        sys.meta_path.insert(0, NoCutlass())
        from b12x.sequence import gdn_decode
        assert gdn_decode.is_supported() is False
        assert "b12x.sequence.gdn_decode._fused_verify" not in sys.modules
        assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)
        assert not any(
            name == "cutlass" or name.startswith("cutlass.") for name in sys.modules
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


class _CudaTensorMetadata:
    """CPU storage exposing the CUDA device metadata consumed by host dispatch."""

    device = torch.device("cuda", 1)

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def view(self, dtype):
        return type(self)(self.tensor.view(dtype))


def _bindings():
    from b12x.sequence.gdn_decode._preparation import _kda_strides

    def tensor(shape, dtype=torch.bfloat16):
        return _CudaTensorMetadata(torch.empty(shape, dtype=dtype, device="cpu"))

    shared = dict(
        query_start_loc=tensor((2,), torch.int32),
        num_accepted_tokens=tensor((1,), torch.int32),
        num_seqs=tensor((1,), torch.int32),
        num_tokens=tensor((1,), torch.int32),
    )
    binding = SimpleNamespace(
        mixed_qkv=tensor((2, 384)),
        raw_g=tensor((2, 1, 128)),
        raw_beta=tensor((2, 1)),
        A_log=tensor((1,), torch.float32),
        dt_bias=tensor((128,), torch.float32),
        recurrent_state=tensor((3, 1, 128, 128), torch.float32),
        output=tensor((2, 1, 128)),
        state_indices=tensor((1, 2), torch.int64),
        z=tensor((2, 1, 128)),
        norm_weight=tensor((128,), torch.float32),
        correction_cache=tensor((3, 1, 2, 128), torch.float32),
        kg_cache=tensor((3, 1, 2, 256)),
        **shared,
    )
    query = SimpleNamespace(
        gate_activation="sigmoid",
        qk_l2norm=True,
        state_dtype="float32",
        recover_speculative_state=True,
        key_heads=1,
        value_heads=1,
        state_index_columns=2,
        null_state_index=None,
        a_log_dtype="float32",
        dt_bias_dtype="float32",
        state_indices_dtype="int64",
        kda_strides=_kda_strides(
            binding.mixed_qkv,
            binding.raw_g,
            binding.raw_beta,
            binding.dt_bias,
            binding.recurrent_state,
            binding.output,
        ),
    )
    binding.plan = SimpleNamespace(
        query=query, layout=SimpleNamespace(recurrent_block_v=32)
    )
    conv = SimpleNamespace(
        x=tensor((2, 384)),
        out=binding.mixed_qkv,
        conv_weight=tensor((384, 4)),
        conv_bias=None,
        state_slot_ids=tensor((1,), torch.int64),
        state_is_fresh=tensor((1,), torch.bool),
        conv_state=tensor((3, 384, 5)),
        plan=SimpleNamespace(
            query=SimpleNamespace(
                mode="decode",
                max_speculative_tokens=1,
                kernel_size=4,
                dilation=1,
                channels=384,
                round_before_silu=True,
                bias_dtype=None,
                input_row_stride=384,
                max_seqs=1,
                state_length=3,
                state_capacity=5,
            )
        ),
        **shared,
    )
    return binding, conv


@pytest.fixture
def host_dispatch(monkeypatch):
    pytest.importorskip("cutlass")
    module = importlib.import_module("b12x.sequence.gdn_decode._fused_verify")
    types = importlib.import_module("b12x.preparation.types")
    monkeypatch.setattr(types, "require_prepared", lambda plan, _op: plan)
    active = SimpleNamespace(device=0, observed=[])

    @contextmanager
    def device_guard(device):
        previous = active.device
        active.device = torch.device(device).index
        try:
            yield
        finally:
            active.device = previous

    def stream():
        active.observed.append(("stream", active.device))
        return active.device * 100 + 7

    monkeypatch.setattr(torch.cuda, "device", device_guard)
    monkeypatch.setattr(module, "current_cuda_stream", stream)
    monkeypatch.setattr(module, "_fake_pointer", lambda dtype: dtype)
    monkeypatch.setattr(module, "_pointer", lambda tensor, dtype: tensor.data_ptr())
    return module, active


@pytest.mark.parametrize(
    ("retain_examples", "dynamic"), [(True, False), (True, True), (False, True)]
)
@pytest.mark.parametrize("fail_launch", [False, True])
def test_prepare_and_run_use_operand_device_and_restore_caller(
    host_dispatch, monkeypatch, retain_examples, dynamic, fail_launch
):
    module, active = host_dispatch
    binding, conv = _bindings()
    launched_pointers = []

    def launch(*args):
        active.observed.append(("launch", active.device))
        assert active.device == 1
        assert args[-1] == 107
        launched_pointers.append(args[0])
        if fail_launch:
            raise RuntimeError("launch failure")

    def compile_kernel(*args, **kwargs):
        active.observed.append(("compile", active.device))
        assert active.device == 1
        assert args[-1] == 107
        return launch

    monkeypatch.setattr(module, "b12x_compile", compile_kernel)
    fused = module.prepare(binding, conv, retain_examples=retain_examples)
    assert active.device == 0
    if not retain_examples:
        assert fused.binding is None
        assert fused.conv is None
    if dynamic:
        binding, conv = _bindings()

    def run():
        return module.run_kda_fused_verify(
            fused,
            **({"binding": binding, "conv": conv} if dynamic else {}),
        )

    if fail_launch:
        with pytest.raises(RuntimeError, match="launch failure"):
            run()
    else:
        assert run() is binding.output
    assert active.device == 0
    assert launched_pointers == [conv.x.data_ptr()]
    assert active.observed == [
        ("stream", 1),
        ("compile", 1),
        ("stream", 1),
        ("launch", 1),
    ]


def test_prepare_restores_caller_device_after_compiler_failure(
    host_dispatch, monkeypatch
):
    module, active = host_dispatch
    binding, conv = _bindings()

    def compile_kernel(*args, **kwargs):
        assert active.device == 1
        assert args[-1] == 107
        raise RuntimeError("compiler failure")

    monkeypatch.setattr(module, "b12x_compile", compile_kernel)
    with pytest.raises(RuntimeError, match="compiler failure"):
        module.prepare(binding, conv)
    assert active.device == 0
