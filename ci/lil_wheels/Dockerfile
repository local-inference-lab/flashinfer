# syntax=docker/dockerfile:1.7

ARG BUILDER_IMAGE
FROM ${BUILDER_IMAGE} AS build

ARG FLASHINFER_LOCAL_VERSION
ARG FLASHINFER_SOURCE_COMMIT
ARG FLASHINFER_SOURCE_DATE_EPOCH
ARG MAX_JOBS
ARG NVCC_THREADS

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.0a \
    FLASHINFER_CUDA_ARCH_LIST=12.0f \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    FLASHINFER_ENABLE_SM90=0 \
    BUILD_NVEP=0 \
    BUILD_NCCL_EP=0 \
    BUILD_NIXL_EP=0

RUN /opt/venv/bin/python - <<'PY'
import importlib.metadata
import torch

assert torch.__version__ == "2.13.0"
assert torch.version.cuda == "13.3"
assert torch._C._GLIBCXX_USE_CXX11_ABI
expected = {
    "pip": "26.2.1",
    "setuptools": "80.10.2",
    "packaging": "26.2",
    "wheel": "0.47.0",
    "build": "1.5.0",
    "tqdm": "4.69.1",
    "ninja": "1.13.0",
    "requests": "2.34.2",
    "numpy": "2.1.0",
    "nvidia-ml-py": "13.610.43",
    "apache-tvm-ffi": "0.1.11",
    "nvidia-cutlass-dsl": "4.6.2",
    "nvidia-cutlass-dsl-libs-cu13": "4.6.2",
}
for package, version in expected.items():
    assert importlib.metadata.version(package) == version, package
PY

COPY . /src/flashinfer
WORKDIR /src/flashinfer

# Docker build contexts intentionally omit Git internals.  Supplying the exact
# source revision here keeps package diagnostics useful without making cache
# keys depend on mutable repository metadata.
RUN FLASHINFER_LOCAL_VERSION="${FLASHINFER_LOCAL_VERSION}" \
    FLASHINFER_SOURCE_COMMIT="${FLASHINFER_SOURCE_COMMIT}" \
    /opt/venv/bin/python ci/lil_wheels/write_build_metadata.py

RUN --mount=type=cache,id=lil-flashinfer-pip-cu133-py312,target=/root/.cache/pip,sharing=locked \
    mkdir -p /wheelhouse \
 && FLASHINFER_LOCAL_VERSION="${FLASHINFER_LOCAL_VERSION}" \
    SOURCE_DATE_EPOCH="${FLASHINFER_SOURCE_DATE_EPOCH}" \
    MAX_JOBS="${MAX_JOBS}" \
    FLASHINFER_NVCC_THREADS="${NVCC_THREADS}" \
    NVCC_THREADS="${NVCC_THREADS}" \
    /opt/venv/bin/python -m pip wheel --no-build-isolation --no-deps \
      --wheel-dir /wheelhouse .

RUN --mount=type=cache,id=lil-flashinfer-aot-cu133-sm120f,target=/src/flashinfer/build,sharing=locked \
    --mount=type=cache,id=lil-flashinfer-pip-cu133-py312,target=/root/.cache/pip,sharing=locked \
    FLASHINFER_LOCAL_VERSION="${FLASHINFER_LOCAL_VERSION}" \
    SOURCE_DATE_EPOCH="${FLASHINFER_SOURCE_DATE_EPOCH}" \
    MAX_JOBS="${MAX_JOBS}" \
    FLASHINFER_NVCC_THREADS="${NVCC_THREADS}" \
    NVCC_THREADS="${NVCC_THREADS}" \
    /opt/venv/bin/python -m pip wheel --no-build-isolation --no-deps \
      --wheel-dir /wheelhouse ./flashinfer-jit-cache \
 && test "$(find /wheelhouse -maxdepth 1 -name 'flashinfer_python-*.whl' | wc -l)" -eq 1 \
 && test "$(find /wheelhouse -maxdepth 1 -name 'flashinfer_jit_cache-*.whl' | wc -l)" -eq 1

FROM scratch AS wheels
COPY --from=build /wheelhouse/ /wheels/
