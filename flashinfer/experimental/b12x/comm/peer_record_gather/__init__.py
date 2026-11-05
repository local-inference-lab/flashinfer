"""Prepared byte-exact gathering from addressable peer record storage."""

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="peer_record_gather",
    group="comm",
    api_style="planned",
    entry_points=("PeerRecordGatherQuery", "plan", "bind", "gather", "is_supported"),
    dtypes=("uint8", "fp8"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/flashinfer-ai/flashinfer",
        commit="f7e67305a216f12c04ea53af40b4e77e9082ed20",
        paths=("flashinfer/experimental/b12x/moe/fused_moe/_preparation.py",),
    ),
    test_path="tests/experimental/b12x/comm/test_peer_record_gather.py",
    notes="Caller-owned peer pointers and byte records; no collective synchronization.",
)
install_lazy_api(globals(), META)
