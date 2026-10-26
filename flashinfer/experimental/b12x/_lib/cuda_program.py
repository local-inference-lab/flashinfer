"""NVCC cubins with metadata discovery and owning-rank CUDA launch ownership."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from threading import RLock

from filelock import FileLock

from .cache_integrity import atomic_write_bytes
from .compile_plan import (
    ProgramKey,
    planning,
    record_program,
    compile_only_launches_enabled,
)
from .runtime_control import raise_if_kernel_resolution_frozen

_ABI_TYPES = {
    "ptr": ctypes.c_void_p,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "u64": ctypes.c_uint64,
    "f32": ctypes.c_float,
}
_CACHE_SCHEMA = 1
_COMPILE_COUNT = 0
_LOCK = RLock()
_FILE_DIGESTS = {}
_DIRECTORY_FILES = {}
_DIRECTORY_DIGESTS = {}


def _stat_identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _file_digest(path):
    # Match the checkpoint loader's identity policy: writes and replacements
    # invalidate even when a writer preserves size and modification time.
    identity = _stat_identity(path)
    with _LOCK:
        cached = _FILE_DIGESTS.get(path)
        if cached is not None and cached[0] == identity:
            return cached[1]
        data = path.read_bytes()
        if _stat_identity(path) != identity:
            raise RuntimeError("CUDA compilation input changed while fingerprinting")
        digest = _digest(data)
        _FILE_DIGESTS[path] = (identity, digest)
        return digest


def _directory_files(root):
    with _LOCK:
        cached = _DIRECTORY_FILES.get(root)
        if cached is not None:
            directories, files = cached
            try:
                if all(
                    _stat_identity(path) == identity for path, identity in directories
                ):
                    return files
            except OSError:
                pass
        paths = tuple(sorted(root.rglob("*")))
        directories = tuple(
            (path, _stat_identity(path)) for path in (root, *paths) if path.is_dir()
        )
        files = tuple(path for path in paths if path.is_file())
        _DIRECTORY_FILES[root] = (directories, files)
        return files


def _directory_digest(root):
    files = _directory_files(root)
    digests = tuple(_file_digest(path) for path in files)
    with _LOCK:
        cached = _DIRECTORY_DIGESTS.get(root)
        if cached is not None and files is cached[0] and digests == cached[1]:
            return cached[2]
        # Relative names are constructed only when the file list or contents
        # change, so descriptors carry a bounded digest rather than an SDK index.
        prefix = len(str(root)) + 1
        digest = _digest(
            _json(
                [
                    (str(path)[prefix:], value)
                    for path, value in zip(files, digests, strict=True)
                ]
            )
        )
        _DIRECTORY_DIGESTS[root] = (files, digests, digest)
        return digest


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _file_identity(path: Path):
    return path.name, _file_digest(path)


@dataclass(frozen=True)
class CudaKernelSpec:
    """One flat CUDA entry ABI and its explicit launch attributes.

    Pointer arguments carry device addresses; scalar arguments are passed by
    value. Programmatic serialization is a launch dependency, not a numerical
    implementation selector. Grid dimensions are supplied by the prepared op.
    """

    symbol: str
    argument_types: tuple[str, ...]
    block: tuple[int, int, int]
    dynamic_smem_bytes: int = 0
    programmatic_serialization: bool = False

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.symbol):
            raise ValueError("CUDA entry symbol must be an identifier")
        if not self.argument_types or any(
            kind not in _ABI_TYPES for kind in self.argument_types
        ):
            raise ValueError(
                "CUDA entry ABI requires supported pointer/scalar argument types"
            )
        if (
            len(self.block) != 3
            or any(type(x) is not int or x <= 0 for x in self.block)
            or self.block[0] * self.block[1] * self.block[2] > 1024
        ):
            raise ValueError(
                "CUDA blocks require three positive dimensions and at most 1024 threads"
            )
        if type(self.dynamic_smem_bytes) is not int or self.dynamic_smem_bytes < 0:
            raise ValueError("dynamic shared memory must be a nonnegative byte extent")
        if type(self.programmatic_serialization) is not bool:
            raise TypeError("programmatic serialization must be boolean")

    def facts(self):
        return dict(
            symbol=self.symbol,
            argument_types=self.argument_types,
            block=self.block,
            dynamic_smem_bytes=self.dynamic_smem_bytes,
            programmatic_serialization=self.programmatic_serialization,
        )


@dataclass(frozen=True)
class CudaCompileSpec:
    """Device-only source, explicit toolchain and content-addressed entry contract.

    Source/dependency contents and all files in declared include directories are
    fingerprinted through stat-invalidated process caches. Include inventories
    stay in those caches; descriptors carry only their bounded digests.
    No compiler or CUDA API runs
    while constructing this descriptor. Operational absolute paths are excluded
    from its program key, allowing identical installations to share artifacts.
    """

    source: str
    nvcc: str
    architecture: str
    kernel: CudaKernelSpec
    options: tuple[str, ...] = ("-O3", "--std=c++20")
    include_dirs: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    host_compiler: str | None = None
    key: str = field(init=False)
    identity: bytes = field(init=False, repr=False)

    def __post_init__(self):
        if self.host_compiler is None:
            compiler = shutil.which("c++")
            if compiler is None:
                raise ValueError(
                    "CUDA compilation requires an explicit host C++ compiler"
                )
            object.__setattr__(self, "host_compiler", compiler)
        if not re.fullmatch(r"sm_[0-9]+[af]?", self.architecture):
            raise ValueError(
                "CUDA compilation requires an explicit real SM architecture"
            )
        # These are controlled by the descriptor and cannot be overridden through options.
        if any(
            x
            in (
                "-o",
                "--output-file",
                "-cubin",
                "--cubin",
                "-arch",
                "--gpu-architecture",
                "-ccbin",
                "--compiler-bindir",
            )
            or x.startswith(
                (
                    "-arch=",
                    "--gpu-architecture=",
                    "--output-file=",
                    "-ccbin=",
                    "--compiler-bindir=",
                )
            )
            for x in self.options
        ):
            raise ValueError(
                "CUDA output, architecture and host compiler belong in the compile descriptor"
            )
        identity = _json(self._facts())
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "key", _digest(identity))

    def _facts(self):
        source = Path(self.source)
        nvcc = Path(self.nvcc).resolve(strict=True)
        if not source.is_file() or not nvcc.is_file():
            raise ValueError("CUDA source and NVCC compiler must be files")
        toolkit = nvcc.parent.parent
        host = Path(self.host_compiler).resolve(strict=True)
        tool_files = [nvcc, host]
        # GCC keeps its language frontend outside the driver executable.
        match = re.fullmatch(r"(.+)-g\+\+-(\d+)", host.name)
        if match:
            target, version = match.groups()
            prefix = host.parent.parent
            for relative in (
                f"libexec/gcc/{target}/{version}/cc1plus",
                f"lib/gcc/{target}/{version}/cc1plus",
            ):
                frontend = prefix / relative
                if frontend.is_file():
                    tool_files.append(frontend)
        for relative in (
            "bin/ptxas",
            "bin/nvcc.profile",
            "nvvm/bin/cicc",
            "version.json",
        ):
            path = toolkit / relative
            if path.is_file():
                tool_files.append(path)
        for directory in (toolkit / "nvvm/libdevice", toolkit / "nvvm/lib64"):
            if directory.is_dir():
                tool_files.extend(_directory_files(directory))
        includes = []
        for directory in self.include_dirs:
            root = Path(directory).resolve(strict=True)
            if not root.is_dir():
                raise ValueError("CUDA include roots must be directories")
            includes.append(_directory_digest(root))
        return dict(
            schema=_CACHE_SCHEMA,
            source=_file_identity(source),
            dependencies=[_file_identity(Path(path)) for path in self.dependencies],
            toolchain=[_file_identity(path) for path in tool_files],
            includes=includes,
            architecture=self.architecture,
            options=self.options,
            kernel=self.kernel.facts(),
        )

    def verify_inputs(self):
        if _json(self._facts()) != self.identity:
            raise RuntimeError(
                "CUDA compilation inputs changed after program discovery"
            )

    @property
    def program(self):
        return ProgramKey("cuda", self.key, self.kernel.symbol)


def _cache_root():
    from .compiler import _cute_compile_cache_dir

    return _cute_compile_cache_dir() / "cuda"


def _paths(key):
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("invalid CUDA artifact key")
    root = _cache_root() / key[:2]
    return root / (key + ".cubin"), root / (key + ".json"), root / (key + ".lock")


def cuda_program_available(key):
    cubin, manifest, _ = _paths(key)
    try:
        record = json.loads(manifest.read_text())
        data = cubin.read_bytes()
        return (
            record["schema"] == _CACHE_SCHEMA
            and record["key"] == key
            and record["cubin_bytes"] == len(data)
            and record["cubin_sha256"] == _digest(data)
            and data.startswith(b"\x7fELF")
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _compile(spec):
    global _COMPILE_COUNT
    if cuda_program_available(spec.key):
        return
    raise_if_kernel_resolution_frozen("CUDA cubin compilation", cache_key=spec.key)
    cubin, manifest, lock = _paths(spec.key)
    cubin.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock):
        if cuda_program_available(spec.key):
            return
        spec.verify_inputs()
        with tempfile.TemporaryDirectory(
            dir=cubin.parent, prefix=".cuda-"
        ) as temporary:
            output = Path(temporary) / "kernel.cubin"
            command = [
                spec.nvcc,
                "--cubin",
                "-arch=" + spec.architecture,
                "-ccbin=" + spec.host_compiler,
                *spec.options,
                *(flag for path in spec.include_dirs for flag in ("-I", path)),
                spec.source,
                "-o",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(
                    f"CUDA compilation failed for {spec.kernel.symbol}:\n{result.stderr}"
                )
            data = output.read_bytes()
            if not data.startswith(b"\x7fELF"):
                raise RuntimeError("NVCC did not produce a CUDA ELF cubin")
            record = dict(
                schema=_CACHE_SCHEMA,
                key=spec.key,
                identity=json.loads(spec.identity),
                cubin_bytes=len(data),
                cubin_sha256=_digest(data),
            )
            atomic_write_bytes(cubin, data)
            atomic_write_bytes(manifest, _json(record))
            _COMPILE_COUNT += 1


def cuda_compile_count():
    return _COMPILE_COUNT


def _check(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f"CUDA driver operation failed: {error}")
    return values[0] if len(values) == 1 else values


def _current_device_context():
    from cuda.bindings import driver, runtime

    ordinal = int(_check(runtime.cudaGetDevice()))
    context = _check(driver.cuCtxGetCurrent())
    if not int(context):
        raise RuntimeError(
            "CUDA program loading requires the owning rank's current context"
        )
    return ordinal, int(context)


class _CudaKernelOwner:
    def __init__(self, spec, ordinal):
        from cuda.bindings import driver

        self.driver, self.spec, self.ordinal = driver, spec, ordinal
        self.__b12x_programs__ = (spec.program,)
        cubin, _, _ = _paths(spec.key)
        self.library = _check(
            driver.cuLibraryLoadFromFile(str(cubin).encode(), [], [], 0, [], [], 0)
        )
        try:
            self.kernel = _check(
                driver.cuLibraryGetKernel(self.library, spec.kernel.symbol.encode())
            )
            if spec.kernel.dynamic_smem_bytes > 48 * 1024:
                _check(
                    driver.cuKernelSetAttribute(
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        spec.kernel.dynamic_smem_bytes,
                        self.kernel,
                        driver.CUdevice(ordinal),
                    )
                )
        except BaseException:
            self.close()
            raise

    def local_memory_bytes(self):
        d = self.driver
        return int(
            _check(
                d.cuKernelGetAttribute(
                    d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                    self.kernel,
                    d.CUdevice(self.ordinal),
                )
            )
        )

    def launch(self, args, grid, stream):
        descriptor = self.spec.kernel
        if len(args) != len(descriptor.argument_types):
            raise ValueError("CUDA launch operands differ from the declared flat ABI")
        if (
            len(grid) != 3
            or any(type(x) is not int or x <= 0 for x in grid)
            or grid[0] > 2**31 - 1
            or grid[1] > 65535
            or grid[2] > 65535
        ):
            raise ValueError("CUDA launch grid exceeds supported positive dimensions")
        values = []
        for kind, value in zip(descriptor.argument_types, args, strict=True):
            if kind == "ptr":
                if hasattr(value, "data_ptr"):
                    if value.device.type != "cuda":
                        raise ValueError("CUDA pointer operands require CUDA tensors")
                    value = value.data_ptr()
                if type(value) is not int or not 0 <= value < 2**64:
                    raise ValueError("CUDA pointer operand requires a device address")
            elif kind in ("i32", "i64", "u64"):
                bits = 32 if kind == "i32" else 64
                low, high = (
                    (0, 2**64)
                    if kind == "u64"
                    else (-(2 ** (bits - 1)), 2 ** (bits - 1))
                )
                if type(value) is not int or not low <= value < high:
                    raise ValueError(f"CUDA scalar operand exceeds {kind}")
            elif not isinstance(value, (float, int)):
                raise TypeError("CUDA floating scalar requires a number")
            values.append(value)
        d = self.driver
        attributes = []
        if descriptor.programmatic_serialization:
            attribute = d.CUlaunchAttribute()
            attribute.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            value = d.CUlaunchAttributeValue()
            value.programmaticStreamSerializationAllowed = 1
            attribute.value = value
            attributes.append(attribute)
        config = d.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = grid
        config.blockDimX, config.blockDimY, config.blockDimZ = descriptor.block
        config.sharedMemBytes = descriptor.dynamic_smem_bytes
        config.hStream = d.CUstream(stream)
        config.attrs, config.numAttrs = attributes, len(attributes)
        _check(
            d.cuLaunchKernelEx(
                config,
                self.kernel,
                (
                    tuple(values),
                    tuple(_ABI_TYPES[kind] for kind in descriptor.argument_types),
                ),
                0,
            )
        )

    def close(self):
        if getattr(self, "library", None) is not None:
            library, self.library = self.library, None
            _check(self.driver.cuLibraryUnload(library))

    def __del__(self):
        # The plan/session retains this owner until native graph replay drains.
        try:
            self.close()
        except Exception:
            pass


class CudaProgram:
    """A discoverable cubin program whose executable belongs to each rank context."""

    def __init__(self, spec):
        self.spec = spec
        self.__b12x_programs__ = (spec.program,)
        self._owners = {}

    def _load(self):
        if planning() or compile_only_launches_enabled():
            raise RuntimeError("CUDA executable loading is not metadata discovery")
        if not self._owners:
            raise_if_kernel_resolution_frozen(
                "CUDA library load", cache_key=self.spec.key
            )
        key = _current_device_context()
        with _LOCK:
            owner = self._owners.get(key)
            if owner is None:
                raise_if_kernel_resolution_frozen(
                    "CUDA library load", cache_key=self.spec.key
                )
                if not cuda_program_available(self.spec.key):
                    raise RuntimeError(
                        "CUDA cubin was not compiled before owning-rank load"
                    )
                owner = _CudaKernelOwner(self.spec, key[0])
                self._owners[key] = owner
            from .compile_plan import _RESIDENT_PROGRAMS

            _RESIDENT_PROGRAMS[self.spec.program] = owner
            record_program(self.spec.program, owner)
        return owner

    def __call__(self, *args, grid, stream):
        if compile_only_launches_enabled():
            record_program(self.spec.program, self)
            return self
        if not self._owners:
            raise RuntimeError(
                "load_programs must bind CUDA launch ownership before execution"
            )
        key = _current_device_context()
        owner = self._owners.get(key)
        if owner is None:
            raise RuntimeError(
                "load_programs must bind CUDA launch ownership before execution"
            )
        record_program(self.spec.program, owner)
        return owner.launch(args, grid, stream)

    def local_memory_bytes(self):
        return max(
            (owner.local_memory_bytes() for owner in self._owners.values()), default=0
        )


def compile_cuda(spec):
    """Discover or compile one device artifact; never load a CUDA library here."""
    if not isinstance(spec, CudaCompileSpec):
        raise TypeError("CUDA compilation requires CudaCompileSpec")
    record_program(spec.program)
    if not planning():
        spec.verify_inputs()
        _compile(spec)
    return CudaProgram(spec)
