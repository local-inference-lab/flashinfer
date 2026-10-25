"""CPU-only tests for publication and reuse of CuTe cache objects.

Load the stdlib helper directly and execute the real compiler entry points with
mock CUDA boundaries. This exercises reuse and recompile decisions without
initializing B12X, CUTLASS, Torch, or a GPU.
"""

from __future__ import annotations


import ast
from collections import namedtuple
from contextlib import contextmanager, nullcontext, suppress
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from threading import RLock
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from filelock import FileLock


LIB = Path(__file__).resolve().parents[4] / "flashinfer/experimental/b12x/_lib"
spec = importlib.util.spec_from_file_location(
    "cache_integrity", LIB / "cache_integrity.py"
)
integrity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integrity)


def load_functions(path, names, namespace):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), "exec"),
        namespace,
    )


class CompileCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        payload = ("test-compile-payload",)
        self.key = hashlib.sha256(repr(payload).encode()).hexdigest()
        self.obj = self.root / self.key[:2] / (self.key + ".o")
        self.obj.parent.mkdir()
        self.data = b"compiled object fixture"
        self.compiled = SimpleNamespace(dump_to_object=lambda _: self.data)
        self.compile_kernel = Mock(return_value=self.compiled)
        self.loaded_paths = []

        def load(path):
            # The real loader parses the object bytes; reject anything but the
            # fixture so a corrupt file surfaces at load time.
            self.loaded_paths.append(path)
            if Path(path).read_bytes() != self.data:
                raise RuntimeError("unloadable object")
            return SimpleNamespace(**{"b12x_cute_" + self.key: self.compiled})

        self.loader = Mock(side_effect=load)
        self.modules = {
            name: ModuleType(name)
            for name in (
                "b12x",
                "b12x._lib",
                "b12x._lib.compiler",
                "b12x._lib.compile_plan",
                "b12x._lib.runtime_control",
                "cutlass",
                "cutlass.cute",
                "cutlass.base_dsl",
                "cutlass.base_dsl.export",
                "cutlass.base_dsl.export.external_binary_module",
            )
        }
        self.modules["cutlass"].cute = self.modules["cutlass.cute"]
        self.modules["cutlass.cute"].compile = object()
        self.modules[
            "cutlass.base_dsl.export.external_binary_module"
        ].ExternalBinaryModule = self.loader
        self.modules[
            "b12x._lib.runtime_control"
        ].raise_if_kernel_resolution_frozen = Mock()
        self.program_type = namedtuple("ProgramKey", "dialect key name")
        plan = self.modules["b12x._lib.compile_plan"]
        plan.ProgramKey = self.program_type
        plan.DeferredCuTeKernel = object
        plan.planning = lambda: False
        plan.record_program = Mock()
        plan.tag_compiled = lambda compiled, program: compiled
        self.module = self.modules["b12x._lib.compiler"]
        self.namespace = self.module.__dict__
        self.memory = {}
        self.namespace.update(
            __package__="b12x._lib",
            Path=Path,
            os=os,
            json=json,
            hashlib=hashlib,
            contextmanager=contextmanager,
            FileLock=FileLock,
            nullcontext=nullcontext,
            suppress=suppress,
            atomic_write_bytes=integrity.atomic_write_bytes,
            _mkdir_durable=integrity._mkdir_durable,
            _cute_compile_cache_dir=lambda: self.root,
            _compile_memory_cache_key=lambda *args: "memory-key",
            _compile_disk_cache_payload=lambda *args: payload,
            _cute_compile_disk_cache_enabled_for_payload=lambda *args: True,
            _memory_cache_get=self.memory.get,
            _memory_cache_put=self.memory.__setitem__,
            _cute_compile_post_engine_start_log_enabled=lambda: False,
            _cute_compile_log_enabled=lambda: False,
            _compile_target_name=lambda func: "test.kernel",
            _call_cute_compile=self.compile_kernel,
            _MEMORY_CACHE_LOCK=RLock(),
            _DISK_CACHE_HITS=0,
            _COMPILE_MISSES=0,
            _OFFLINE_CUTE_NO_JIT=False,
            _install_race_fuzz=lambda: None,
            resolve_compile_options=lambda fn, *_args: fn,
        )
        load_functions(
            LIB / "compiler.py",
            (
                "_cache_prefix",
                "_cache_object_path",
                "_cache_lock_path",
                "_valid_cute_compile_cache",
                "_disk_cache_key_lock",
                "_load_cute_compile_from_disk",
                "_store_cute_compile_to_disk",
                "compile",
            ),
            self.namespace,
        )
        plan.__dict__.update(
            __package__="b12x._lib", _RESIDENT_PROGRAMS=set(), Path=Path
        )
        load_functions(
            LIB / "compile_plan.py", ("compiled_program_available",), plan.__dict__
        )
        self.available = plan.compiled_program_available
        self.program = self.program_type("cute", self.key, "test.kernel")
        modules_patch = patch.dict(sys.modules, self.modules)
        modules_patch.start()
        self.addCleanup(modules_patch.stop)

    def write_object(self):
        self.obj.write_bytes(self.data)

    def test_published_object_is_reused_without_compilation(self):
        self.write_object()
        self.assertTrue(self.available(self.program))
        self.assertIs(self.module.compile(object()), self.compiled)
        self.compile_kernel.assert_not_called()

    def test_object_is_loaded_in_place_without_copies_or_hashing(self):
        self.write_object()
        with patch.object(integrity, "atomic_write_bytes") as publish:
            self.assertIs(
                self.module._load_cute_compile_from_disk(self.key), self.compiled
            )
        self.assertEqual(self.loaded_paths, [str(self.obj)])
        self.assertEqual(self.obj.read_bytes(), self.data)
        self.assertEqual(list(self.obj.parent.iterdir()), [self.obj])
        publish.assert_not_called()

    def test_missing_object_is_unavailable_and_compiled_once(self):
        self.assertFalse(self.available(self.program))
        self.assertIsNone(self.module._load_cute_compile_from_disk(self.key))
        self.assertIs(self.module.compile(object()), self.compiled)
        self.compile_kernel.assert_called_once()
        self.assertTrue(self.available(self.program))
        self.assertEqual(self.obj.read_bytes(), self.data)
        self.assertEqual(list(self.obj.parent.glob("*.o")), [self.obj])

    def test_unloadable_object_is_recompiled_and_replaced(self):
        for name, corrupt in (
            ("empty", lambda: self.obj.write_bytes(b"")),
            ("truncated", lambda: self.obj.write_bytes(self.data[:3])),
            ("same_size", lambda: self.obj.write_bytes(b"x" * len(self.data))),
        ):
            with self.subTest(name=name):
                corrupt()
                self.memory.clear()
                self.compile_kernel.reset_mock()
                self.assertIsNone(self.module._load_cute_compile_from_disk(self.key))
                self.assertIs(self.module.compile(object()), self.compiled)
                self.compile_kernel.assert_called_once()
                self.assertEqual(self.obj.read_bytes(), self.data)

    def test_interrupted_publication_leaves_no_object_and_spares_neighbors(self):
        neighbor = self.obj.parent / "other.o"
        neighbor.write_bytes(b"keep this")
        with (
            patch.object(os, "fsync", side_effect=OSError("disk failure")),
            self.assertRaises(OSError),
        ):
            self.module._store_cute_compile_to_disk(
                self.key, self.compiled, cache_payload=(1,), func=object()
            )
        self.assertFalse(self.available(self.program))
        self.assertEqual(list(self.obj.parent.glob("*.tmp")), [])
        self.module.compile(object())
        self.assertTrue(self.available(self.program))
        self.assertEqual(neighbor.read_bytes(), b"keep this")

    def test_compiler_reloads_after_another_writer_repairs_the_entry(self):
        self.obj.write_bytes(b"corrupt")

        @contextmanager
        def repaired_by_other_writer(key):
            self.write_object()
            yield

        with patch.object(
            self.module, "_disk_cache_key_lock", repaired_by_other_writer
        ):
            self.assertIs(self.module.compile(object()), self.compiled)
        self.compile_kernel.assert_not_called()
        self.assertEqual(len(self.loaded_paths), 2)

    def test_publication_fsyncs_file_before_rename_then_directory(self):
        events = []
        real_fsync, real_replace = os.fsync, os.replace
        real_sync_directory = integrity._fsync_directory

        def fsync(fd):
            if not stat.S_ISDIR(os.fstat(fd).st_mode):
                events.append("fsync")
            real_fsync(fd)

        def replace(source, target):
            events.append("replace")
            real_replace(source, target)

        def sync_directory(path):
            events.append("directory")
            real_sync_directory(path)

        with (
            patch.object(os, "fsync", fsync),
            patch.object(os, "replace", replace),
            patch.object(integrity, "_fsync_directory", sync_directory),
        ):
            self.module._store_cute_compile_to_disk(
                self.key,
                self.compiled,
                cache_payload=(1,),
                func=object(),
            )
        self.assertEqual(events, ["fsync", "replace", "directory"])
        self.assertTrue(self.available(self.program))
        self.assertEqual(list(self.obj.parent.iterdir()), [self.obj])

    def test_failure_before_rename_preserves_previous_file_and_cleans_temporary(self):
        self.write_object()
        with (
            patch.object(os, "fsync", side_effect=OSError("disk failure")),
            self.assertRaises(OSError),
        ):
            integrity.atomic_write_bytes(self.obj, b"replacement")
        self.assertEqual(self.obj.read_bytes(), self.data)
        self.assertEqual(list(self.obj.parent.glob("*.tmp")), [])

    def test_new_cache_directory_entries_are_synced(self):
        target = self.root / "new" / "shard" / "object.o"
        with patch.object(
            integrity, "_fsync_directory", wraps=integrity._fsync_directory
        ) as sync:
            integrity.atomic_write_bytes(target, self.data)
        self.assertEqual(
            [call.args[0] for call in sync.call_args_list],
            [self.root, self.root / "new", self.root / "new/shard"],
        )

    def test_preparation_metadata_is_not_scanned(self):
        selection = self.root / "preparation" / "selection.json"
        selection.parent.mkdir()
        selection.write_text('{"identity":{},"records":{}}')
        self.write_object()
        self.assertTrue(self.available(self.program))
        self.module.compile(object())
        self.assertEqual(selection.read_text(), '{"identity":{},"records":{}}')


if __name__ == "__main__":
    unittest.main()
