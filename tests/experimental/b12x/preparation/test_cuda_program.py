"""CUDA programs discover without compilation and bind only on the owning rank."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from b12x._lib import cuda_program as cuda
from b12x._lib.compile_plan import (
    compiled_program_available,
    load_programs,
    plan_compilations,
)
from b12x._lib.compile_pool import CompileJob, describe_compilation
from b12x._lib.runtime_control import (
    KernelResolutionFrozenError,
    kernel_resolution_guard,
)


@pytest.fixture
def spec(tmp_path, monkeypatch):
    source = tmp_path / "kernel.cu"
    source.write_text('extern "C" __global__ void entry() {}')
    compiler = tmp_path / "cuda/bin/nvcc"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"compiler")
    host_compiler = tmp_path / "host-cxx"
    host_compiler.write_bytes(b"host compiler")
    includes = tmp_path / "cuda/include"
    includes.mkdir()
    (includes / "header.h").write_text("header")
    monkeypatch.setattr(cuda, "_cache_root", lambda: tmp_path / "cache")
    return cuda.CudaCompileSpec(
        str(source),
        str(compiler),
        "sm_120",
        cuda.CudaKernelSpec("entry", ("ptr", "i32"), (256, 1, 1), 17408, True),
        include_dirs=(str(includes),),
        host_compiler=str(host_compiler),
    )


def fake_compiler(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"\x7fELF" + bytes(64))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(cuda.subprocess, "run", run)
    return commands


def test_compile_job_discovery_is_metadata_only(spec, monkeypatch):
    monkeypatch.setattr(
        cuda.subprocess, "run", lambda *a, **k: pytest.fail("discovery invoked NVCC")
    )
    monkeypatch.setattr(
        cuda, "_current_device_context", lambda: pytest.fail("discovery touched CUDA")
    )
    description = describe_compilation(
        CompileJob.create("b12x._lib.cuda_program:compile_cuda", spec)
    )
    assert description.programs == (spec.program,)
    assert not compiled_program_available(spec.program)


def test_real_artifact_identity_and_integrity(spec, monkeypatch):
    commands = fake_compiler(monkeypatch)
    program = cuda.compile_cuda(spec)
    assert (
        len(commands) == 1
        and "--cubin" in commands[0]
        and "-arch=sm_120" in commands[0]
    )
    assert compiled_program_available(spec.program)
    assert program.__b12x_programs__ == (spec.program,)
    with kernel_resolution_guard("capture"):
        cuda.compile_cuda(spec)
    assert len(commands) == 1
    cubin, manifest, _ = cuda._paths(spec.key)
    assert json.loads(manifest.read_text())["identity"]["kernel"]["argument_types"] == [
        "ptr",
        "i32",
    ]
    cubin.write_bytes(b"corrupt")
    assert not compiled_program_available(spec.program)
    with kernel_resolution_guard("capture"), pytest.raises(KernelResolutionFrozenError):
        cuda.compile_cuda(spec)
    assert len(commands) == 1


def test_source_headers_toolchain_and_abi_change_program_key(spec):
    assert replace(spec).key == spec.key
    assert (
        replace(spec, kernel=replace(spec.kernel, programmatic_serialization=False)).key
        != spec.key
    )
    assert replace(spec, architecture="sm_121").key != spec.key
    assert replace(spec, options=(*spec.options, "--use_fast_math")).key != spec.key
    Path(spec.source).write_text("changed kernel")
    assert replace(spec).key != spec.key
    with pytest.raises(RuntimeError, match="inputs changed"):
        spec.verify_inputs()


def test_header_and_compiler_content_are_part_of_identity(spec):
    header = Path(spec.include_dirs[0]) / "header.h"
    header.write_text("changed header")
    assert replace(spec).key != spec.key
    other = replace(spec)
    Path(spec.nvcc).write_bytes(b"changed compiler")
    assert replace(spec).key != other.key


def test_frozen_unloaded_carrier_fails_before_cuda(spec, monkeypatch):
    with plan_compilations():
        program = cuda.compile_cuda(spec)
    monkeypatch.setattr(
        cuda, "_current_device_context", lambda: pytest.fail("frozen miss touched CUDA")
    )
    with (
        kernel_resolution_guard("native replay"),
        pytest.raises(KernelResolutionFrozenError),
    ):
        load_programs(program)
    with pytest.raises(RuntimeError, match="bind CUDA launch ownership"):
        program(0, 1, grid=(1, 1, 1), stream=0)


def test_loading_is_rank_context_local_and_retained(spec, monkeypatch):
    fake_compiler(monkeypatch)
    program = cuda.compile_cuda(spec)
    created = []

    class Owner:
        def __init__(self, spec, ordinal):
            self.ordinal = ordinal
            self.__b12x_programs__ = (spec.program,)
            created.append(self)

        def local_memory_bytes(self):
            return 64

        def launch(self, args, grid, stream):
            return self.ordinal, args, grid, stream

    context = [(3, 31)]
    monkeypatch.setattr(cuda, "_current_device_context", lambda: context[0])
    monkeypatch.setattr(cuda, "_CudaKernelOwner", Owner)
    load_programs(program)
    load_programs(program)
    assert len(created) == 1
    with kernel_resolution_guard("native replay"):
        assert program(0, 7, grid=(1, 1, 1), stream=17) == (3, (0, 7), (1, 1, 1), 17)
    context[0] = (4, 41)
    load_programs(program)
    assert [owner.ordinal for owner in created] == [3, 4]
    assert program.local_memory_bytes() == 64
    context[0] = (5, 51)
    with (
        kernel_resolution_guard("native replay"),
        pytest.raises(KernelResolutionFrozenError),
    ):
        load_programs(program)


def test_descriptor_rejects_invalid_launch_contract():
    with pytest.raises(ValueError, match="ABI"):
        cuda.CudaKernelSpec("entry", ("object",), (256, 1, 1))
    with pytest.raises(ValueError, match="1024"):
        cuda.CudaKernelSpec("entry", ("ptr",), (1025, 1, 1))
    with pytest.raises(ValueError, match="identifier"):
        cuda.CudaKernelSpec("entry;bad", ("ptr",), (32, 1, 1))
    with pytest.raises(TypeError, match="boolean"):
        cuda.CudaKernelSpec("entry", ("ptr",), (32, 1, 1), programmatic_serialization=1)


def test_native_launch_marshals_flat_abi_and_programmatic_attribute(spec):
    import ctypes

    calls = []

    class Structure:
        pass

    driver = SimpleNamespace(
        CUlaunchAttribute=Structure,
        CUlaunchAttributeValue=Structure,
        CUlaunchConfig=Structure,
        CUlaunchAttributeID=SimpleNamespace(
            CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION=19
        ),
        CUstream=lambda stream: stream,
        cuLaunchKernelEx=lambda *args: calls.append(args) or (0,),
    )
    owner = object.__new__(cuda._CudaKernelOwner)
    owner.spec, owner.driver, owner.kernel = spec, driver, 17
    owner.launch((4096, 7), (3, 1, 1), 23)
    launch, kernel, parameters, extra = calls.pop()
    assert (launch.gridDimX, launch.blockDimX, launch.sharedMemBytes) == (3, 256, 17408)
    assert launch.hStream == 23 and kernel == 17 and extra == 0
    assert parameters == ((4096, 7), (ctypes.c_void_p, ctypes.c_int32))
    assert launch.numAttrs == 1 and launch.attrs[0].id == 19
    assert launch.attrs[0].value.programmaticStreamSerializationAllowed == 1
    for args in [(4096,), (-1, 7), (4096, 2**31), (4096, True)]:
        with pytest.raises(ValueError):
            owner.launch(args, (1, 1, 1), 0)
    for grid in [(0, 1, 1), (1, 65536, 1), (1, 1), (True, 1, 1)]:
        with pytest.raises(ValueError):
            owner.launch((4096, 7), grid, 0)
    assert not calls


def test_equivalent_descriptors_reuse_file_digests_and_detect_changes(
    spec, monkeypatch
):
    reads = []
    original = Path.read_bytes
    cuda._FILE_DIGESTS.clear()
    cuda._DIRECTORY_FILES.clear()

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    assert replace(spec).key == spec.key
    first = list(reads)
    for _ in range(12):
        assert replace(spec).key == spec.key
        spec.verify_inputs()
    assert reads == first
    assert len(spec.identity) < 8192
    header = Path(spec.include_dirs[0]) / "header.h"
    stat = header.stat()
    header.write_bytes(b"HEADER")
    import os

    os.utime(header, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert replace(spec).key != spec.key
    assert reads == [*first, header]
    another = header.parent / "additional.h"
    another.write_bytes(b"additional")
    assert replace(spec).key != spec.key
    assert another in reads
    another.unlink()
    assert replace(spec).key != spec.key


def test_host_compiler_is_explicit_in_command_and_cache_identity(
    spec, monkeypatch, tmp_path
):
    compiler = tmp_path / "host-cxx"
    compiler.write_bytes(b"host C++ frontend")
    explicit = replace(spec, host_compiler=str(compiler))
    assert explicit.key != spec.key
    commands = fake_compiler(monkeypatch)
    cuda.compile_cuda(explicit)
    assert "-ccbin=" + str(compiler) in commands[0]
    compiler.write_bytes(b"changed host frontend")
    assert replace(explicit).key != explicit.key


def test_compile_only_launch_records_without_loading_or_launching(spec, monkeypatch):
    from b12x._lib.compile_plan import compile_only_launches

    with plan_compilations():
        program = cuda.compile_cuda(spec)
    monkeypatch.setattr(
        cuda,
        "_current_device_context",
        lambda: pytest.fail("metadata launch touched CUDA"),
    )
    with compile_only_launches() as observed:
        assert program(0, 7, grid=(1, 1, 1), stream=0) is program
    assert observed == {spec.program}
    assert not program._owners
