"""Prepared exact row selection with caller-provided candidate IDs."""

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="row_topk",
    group="attention",
    api_style="planned",
    entry_points=("RowTopKQuery", "plan", "bind", "topk", "is_supported"),
    dtypes=("fp32",),
    requires=("cutlass",),
    provenance=Provenance(
        repo="https://github.com/flashinfer-ai/flashinfer",
        commit="f7e67305a216f12c04ea53af40b4e77e9082ed20",
        paths=("flashinfer/experimental/b12x/attention/dsa_indexer/tiled_topk.py",),
    ),
    test_path="tests/experimental/b12x/attention/test_row_topk.py",
    notes="Exact dense row selection with prepared launchers and no replay scratch.",
)
install_lazy_api(globals(), META)
