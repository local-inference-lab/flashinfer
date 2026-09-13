"""Write deterministic package metadata when Git internals are not in Docker."""

from __future__ import annotations

import os
from pathlib import Path


source_root = Path(__file__).resolve().parents[2]
base_version = (source_root / "version.txt").read_text().strip()
local_version = os.environ["FLASHINFER_LOCAL_VERSION"]
source_commit = os.environ["FLASHINFER_SOURCE_COMMIT"]
version = f"{base_version}+{local_version}"

metadata = (
    '"""Build metadata for a source-addressed FlashInfer wheel."""\n'
    f'__version__ = "{version}"\n'
)
(source_root / "flashinfer" / "_build_meta.py").write_text(
    metadata + f'__git_commit__ = "{source_commit}"\n'
)

jit_metadata = (
    '"""Build metadata for a source-addressed FlashInfer JIT-cache wheel."""\n'
    f'__version__ = "{version}"\n'
    f'__git_version__ = "{source_commit}"\n'
)
jit_metadata_path = (
    source_root
    / "flashinfer-jit-cache"
    / "flashinfer_jit_cache"
    / "_build_meta.py"
)
jit_metadata_path.write_text(jit_metadata)
