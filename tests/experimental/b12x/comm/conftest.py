"""Compilation of the CPU-only proxy drivers in this directory.

Each ``test_<name>.py`` here runs ``_<name>.c``, which links the production
``_roce_proxy.c`` against an in-memory verbs provider. A C compiler and the
libibverbs headers are needed; no RDMA device and no GPU are.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "flashinfer/experimental/b12x/comm/roce/_roce_proxy.c"
)


@pytest.fixture(scope="module")
def proxy_driver(request, tmp_path_factory) -> Path:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("the CPU proxy checks require a C compiler")
    name = Path(request.path).stem.removeprefix("test_")
    driver = tmp_path_factory.mktemp(name) / name
    subprocess.run(
        [
            compiler,
            "-std=gnu11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            f'-DROCE_PROXY_SOURCE="{SOURCE}"',
            str(Path(request.path).with_name(f"_{name}.c")),
            "-libverbs",
            "-o",
            str(driver),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return driver
