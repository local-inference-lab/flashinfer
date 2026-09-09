"""Exercise the real compatibility importer with isolated on-disk packages.

A tiny canonical package keeps import-order/failure tests independent of GPU
availability and, unlike patching sys.modules, exercises Python's real loaders.
Installed-package and command smoke checks also need the actual b12x runtime.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture
def alias_package(tmp_path):
    files = {
        "flashinfer/__init__.py": "",
        "flashinfer/b12x/__init__.py": "executions = {}\n",
        "flashinfer/b12x/nested/__init__.py": "",
        "flashinfer/b12x/nested/leaf.py": """
            import flashinfer.b12x as root
            root.executions[__name__] = root.executions.get(__name__, 0) + 1
            class Token:
                pass
            singleton = Token()
        """,
        "flashinfer/b12x/unrelated.py": "raise AssertionError('eager import')\n",
        "flashinfer/b12x/broken.py": """
            import flashinfer.b12x as root
            if not getattr(root, 'allow_import', False):
                import _b12x_compat_missing_dependency
            result = 42
        """,
        "flashinfer/b12x/nested/command.py": """
            from .leaf import singleton
            from flashinfer.b12x.nested.leaf import singleton as canonical
            assert singleton is canonical
            if __name__ == '__main__':
                print('command ran')
        """,
    }
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    # Only flashinfer is replaced. b12x resolves to the production compatibility
    # package in the checkout, through the normal Python import path.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path(__file__).resolve().parents[1]))
    )
    return tmp_path, env


def _run(alias_package, source):
    cwd, env = alias_package
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("first", ["b12x", "flashinfer.b12x"])
def test_nested_import_identity_and_canonical_metadata(alias_package, first):
    _run(
        alias_package,
        f"""
        import importlib
        import pickle
        import sys

        first = importlib.import_module({first!r} + '.nested.leaf')
        metadata = (first.__name__, first.__package__, first.__spec__, first.__loader__)
        other = importlib.import_module(
            {"flashinfer.b12x" if first == "b12x" else "b12x"!r} + '.nested.leaf'
        )
        import b12x.nested.leaf
        import flashinfer.b12x.nested.leaf
        from b12x.nested import leaf
        from b12x.nested.leaf import singleton, Token

        assert first is other is leaf
        assert b12x is flashinfer.b12x
        assert b12x.nested is flashinfer.b12x.nested
        assert singleton is first.singleton
        assert Token is first.Token
        assert metadata == (first.__name__, first.__package__, first.__spec__, first.__loader__)
        assert first.__name__ == 'flashinfer.b12x.nested.leaf'
        assert first.__package__ == 'flashinfer.b12x.nested'
        assert first.__spec__.name == first.__name__
        assert first.__loader__.name == first.__name__
        assert first.__spec__.loader is first.__loader__
        assert pickle.loads(pickle.dumps(Token)) is Token
        assert b12x.executions == {{'flashinfer.b12x.nested.leaf': 1}}
        assert sys.modules['b12x.nested.leaf'] is sys.modules['flashinfer.b12x.nested.leaf']
        """,
    )


def test_imports_remain_lazy_and_failed_imports_can_retry(alias_package):
    _run(
        alias_package,
        """
        import importlib
        import sys
        import b12x

        assert 'flashinfer.b12x.nested' not in sys.modules
        assert 'flashinfer.b12x.unrelated' not in sys.modules
        spec = importlib.util.find_spec('b12x.nested.leaf')
        assert spec is not None
        assert 'flashinfer.b12x.nested.leaf' not in sys.modules
        assert 'b12x.nested.leaf' not in sys.modules

        try:
            importlib.import_module('b12x.broken')
        except ModuleNotFoundError as error:
            assert error.name == '_b12x_compat_missing_dependency'
        else:
            raise AssertionError('missing dependency was swallowed')
        assert 'b12x.broken' not in sys.modules
        assert 'flashinfer.b12x.broken' not in sys.modules

        b12x.allow_import = True
        legacy = importlib.import_module('b12x.broken')
        assert legacy.result == 42
        assert legacy is importlib.import_module('flashinfer.b12x.broken')
        try:
            importlib.import_module('b12x.does_not_exist')
        except ModuleNotFoundError as error:
            assert error.name == 'b12x.does_not_exist'
        else:
            raise AssertionError('missing module was accepted')
        assert 'flashinfer.b12x.unrelated' not in sys.modules
        """,
    )


def test_legacy_module_command_supports_relative_imports(alias_package):
    cwd, env = alias_package
    result = subprocess.run(
        [sys.executable, "-m", "b12x.nested.command"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "command ran\n"
