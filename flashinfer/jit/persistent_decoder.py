# SPDX-License-Identifier: Apache-2.0
"""NVCC module and source identity for the tensor-parallel persistent decoder."""

import hashlib
import json
import os
import shlex
import subprocess
from functools import cache
from pathlib import Path

from . import env as jit_env
from .core import gen_jit_spec

CUDA_FLAGS = ("-std=c++20", "-lineinfo")


@cache
def toolchain_identity():
    """Fingerprint the actual NVCC, its host compiler and the link compiler."""
    from .cpp_ext import build_cuda_cflags, get_cuda_path

    flags = build_cuda_cflags([], list(CUDA_FLAGS))
    host = None
    for index, flag in enumerate(flags):
        if flag in ("-ccbin", "--compiler-bindir"):
            host = flags[index + 1]
        elif flag.startswith(("-ccbin=", "--compiler-bindir=")):
            host = flag.split("=", 1)[1]
    host = host or os.environ.get("NVCC_CCBIN") or ("cl" if os.name == "nt" else "gcc")
    if Path(host).is_dir():
        host = str(Path(host) / ("cl.exe" if os.name == "nt" else "gcc"))
    commands = [
        shlex.split(
            os.environ.get("FLASHINFER_NVCC", str(Path(get_cuda_path()) / "bin/nvcc"))
        ),
        shlex.split(host),
        shlex.split(os.environ.get("CXX", "cl" if os.name == "nt" else "c++")),
    ]
    payload = dict(
        cuda_flags=flags,
        commands=commands,
        versions=[
            subprocess.check_output([*command, "--version"]).decode()
            for command in commands
        ],
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@cache
def source_identity():
    """Identify every compiled decoder input and its numerical compiler flags."""
    sources = {
        "persistent_decoder.cu": jit_env.FLASHINFER_CSRC_DIR / "persistent_decoder.cu",
        **{
            name: jit_env.FLASHINFER_INCLUDE_DIR
            / "flashinfer/experimental/persistent_decoder"
            / name
            for name in ("decoder.cuh", "common.cuh", "fp8_rows.cuh")
        },
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"cuda_flags": CUDA_FLAGS, "use_fast_math": False}, sort_keys=True
        ).encode()
    )
    for name, path in sorted(sources.items()):
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def gen_persistent_decoder_module():
    return gen_jit_spec(
        "persistent_decoder_sm120_"
        + source_identity()[:16]
        + "_"
        + toolchain_identity()[:8],
        [jit_env.FLASHINFER_CSRC_DIR / "persistent_decoder.cu"],
        extra_cuda_cflags=list(CUDA_FLAGS),
        use_fast_math=False,
    )
