"""Prepared split MLA RMS normalization with BF16 output and MXFP8 query encoding.

The input contains 1280 query features followed by 512 latent-KV features.
Each part uses FP32 RMS statistics and its checkpoint affine vector, then
writes BF16. Query encoding consumes that rounded BF16 result, using E4M3
values and per-32 UE8M0 scales with min_amax=1e-4. Caller-owned scale padding
must be initialized to 127 before preparation; every logical scale is written.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mla_norm_mxfp8",
    group="quantization",
    api_style="planned",
    entry_points=("Query", "Config", "plan", "run", "is_supported"),
    dtypes=("bf16", "fp32"),
    recipes=("mxfp8",),
    provenance=Provenance(
        repo="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017",
        commit="1bae0e1088993b9b82de08f59de5be5a34e42170",
        paths=("inference/model.py", "inference/kernel.py"),
    ),
    test_path="tests/experimental/b12x/quantization/test_mla_norm_mxfp8.py",
    since="0.7.1",
    notes="Triton split RMSNorm and BF16-seam query encoding with caller-owned outputs.",
)
if TYPE_CHECKING:
    from .api import Query, Config, plan, run, is_supported
install_lazy_api(globals(), META)
