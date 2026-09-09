"""Import compatibility for the canonical :mod:`flashinfer.b12x` package.

Only explicitly requested legacy names are resolved. Source modules always load
under their canonical names, so registries, caches, classes and compiler state
are shared across both import spellings.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical_spec):
        self._canonical_spec = canonical_spec

    def create_module(self, spec):
        # Let importlib initialize a disposable alias, not the canonical module:
        # module_from_spec would otherwise overwrite the canonical __spec__.
        return None

    def exec_module(self, module):
        canonical = importlib.import_module(self._canonical_spec.name)
        sys.modules[module.__name__] = canonical

    def get_code(self, fullname):
        # runpy (including `python -m b12x.tools.<command>`) asks for code
        # without importing the command first. Delegate using the loader's
        # canonical name to avoid SourceFileLoader's name-mismatch check.
        loader = self._canonical_spec.loader
        if loader is None or not hasattr(loader, "get_code"):
            return None
        return loader.get_code(self._canonical_spec.name)


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("b12x."):
            return None
        canonical_name = "flashinfer." + fullname
        canonical_spec = importlib.util.find_spec(canonical_name)
        if canonical_spec is None:
            # Falling through would let PathFinder execute canonical files
            # again under a legacy name via the shared package __path__.
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(canonical_spec),
            origin=canonical_spec.origin,
            is_package=canonical_spec.submodule_search_locations is not None,
        )


_canonical = importlib.import_module("flashinfer.b12x")
sys.meta_path.insert(0, _AliasFinder())
sys.modules[__name__] = _canonical
