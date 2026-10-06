"""No-model configuration, cache, route and snapshot regression tests."""

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_sam31_persistence_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
config = importlib.import_module(f"{PACKAGE}.director.sam31_config")
backend = importlib.import_module(f"{PACKAGE}.director.sam31")


class Response:
    def __init__(self, data, status=200):
        self.body = json.dumps(data).encode()
        self.status = status


# No HTTP server or aiohttp installation required for route logic tests.
web = types.SimpleNamespace(Response=Response, json_response=Response)
aiohttp_stub = types.ModuleType("aiohttp")
aiohttp_stub.web = web
with patch.dict(sys.modules, {"aiohttp": aiohttp_stub}):
    routes = importlib.import_module(f"{PACKAGE}.director.sam31_routes")
# Keep the bridge's lazy import on the same isolated registry.
sys.modules[routes.__name__] = routes
bridge = importlib.import_module(f"{PACKAGE}.nodes.director_sam31")


class ConfigTests(unittest.TestCase):
    def test_defaults_bounds_and_roundtrip(self):
        default = config.normalize_config()
        self.assertFalse(default["enabled"])
        self.assertEqual(config.normalize_config(json.dumps(default)), default)
        value = config.normalize_config({
            "enabled": "false", "threshold": float("nan"),
            "max_objects": 1000, "detect_interval": -1, "long_edge": 9000,
        })
        self.assertFalse(value["enabled"])
        self.assertEqual(value["threshold"], 0.5)
        self.assertEqual(value["max_objects"], 16)
        self.assertEqual(value["detect_interval"], 1)
        self.assertEqual(value["long_edge"], 1024)
        self.assertEqual(list(config.sam31_widget_inputs()), ["sam31_config"])

    def test_versions_group_selection_and_identity(self):
        for raw in ([], {"version": 2}, "invalid json"):
            with self.assertRaises((ValueError, TypeError)):
                config.normalize_config(raw)
        group = config.normalize_group({"prompt": " dog ", "selected": [2, 2, 0, True, -1, 99]})
        self.assertEqual(group["prompt"], "dog")
        self.assertEqual(group["selected"], [0, 2])
        first = {"source": {"digest": "a", "range": [0, 3]}, "model": "one"}
        self.assertEqual(config.identity_key(first), config.identity_key({
            "model": "one", "source": {"range": [0, 3], "digest": "a"},
        }))
        for changed in (
            {"source": {"digest": "b", "range": [0, 3]}, "model": "one"},
            {"source": {"digest": "a", "range": [1, 4]}, "model": "one"},
            {**first, "model": "two"},
        ):
            self.assertNotEqual(config.identity_key(first), config.identity_key(changed))

    def test_missing_native_dependency_reports_reason(self):
        folders = types.ModuleType("folder_paths")
        folders.get_filename_list = lambda _: ["sam3.1_multiplex_fp16.safetensors"]
        with patch.dict(sys.modules, {"folder_paths": folders}), \
                patch.object(backend, "native_backend", side_effect=ImportError("missing native SAM")):
            status = backend.capabilities()
        self.assertFalse(status["available"])
        self.assertIn("missing native SAM", status["reason"])
        self.assertFalse(status["points_boxes"])
        self.assertFalse(status["cancel"])


class CacheTests(unittest.TestCase):
    def setUp(self):
        parent = Path(r"C:\Users\s\AppData\Local\Temp\kilo")
        self.directory = Path(tempfile.mkdtemp(
            prefix="sam31-regression-", dir=str(parent) if parent.is_dir() else None,
        ))
        folders = types.ModuleType("folder_paths")
        folders.get_temp_directory = lambda: str(self.directory)
        self.module_patch = patch.dict(sys.modules, {"folder_paths": folders})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.identity = {
            "base": {"implementation_revision": backend.CACHE_IMPLEMENTATION_REVISION},
            "action": "detect",
        }
        self.key = config.identity_key(self.identity)
        binary = np.array([[[1, 0, 1, 0, 1, 0, 1, 0, 1]]], dtype=np.uint8)
        self.packed = np.packbits(binary, axis=-1, bitorder="little")
        self.meta = dict(
            version=1, key=self.key, identity=self.identity, frames=1,
            objects=[{"index": 0}], mask_width=9, packed_shape=[1, 2],
        )

    def test_roundtrip_reuses_valid_cache_without_rewrite(self):
        backend.write_cache(self.key, self.meta, [self.packed])
        meta, packed = backend.read_packed_frame(self.key, 0)
        self.assertEqual(meta, self.meta)
        np.testing.assert_array_equal(packed, self.packed)
        before = backend.cache_path(self.key).read_bytes()
        with patch.object(np, "save", side_effect=AssertionError("valid cache rewritten")):
            backend.write_cache(self.key, self.meta, [self.packed])
        self.assertEqual(backend.cache_path(self.key).read_bytes(), before)
        self.assertEqual(np.unpackbits(packed, axis=-1, bitorder="little")[..., :9].tolist(),
                         [[[1, 0, 1, 0, 1, 0, 1, 0, 1]]])

    def test_failed_temporary_write_does_not_block_recovery(self):
        with patch.object(np, "save", side_effect=OSError("injected write failure")):
            with self.assertRaisesRegex(OSError, "injected"):
                backend.write_cache(self.key, self.meta, [self.packed])
        self.assertFalse(backend.cache_path(self.key).exists())
        pending = list((self.directory / "director_sam31").glob("*.pending"))
        self.assertTrue(pending)
        backend.write_cache(self.key, self.meta, [self.packed])
        self.assertEqual(backend.read_metadata(self.key), self.meta)
        self.assertTrue(all(path.exists() for path in pending))

    def test_damaged_cache_is_atomically_replaced(self):
        destination = backend.cache_path(self.key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"damaged zip")
        with patch.object(backend.os, "replace", wraps=backend.os.replace) as replace:
            backend.write_cache(self.key, self.meta, [self.packed])
        replace.assert_called_once()
        self.assertEqual(Path(replace.call_args.args[1]), destination)
        self.assertEqual(backend.read_metadata(self.key), self.meta)
        with self.assertRaises(ValueError):
            backend.read_packed_frame(self.key, 1)
        with self.assertRaises(ValueError):
            backend.cache_path("../escape")


class RouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        routes._JOBS.clear()
        self.server = types.SimpleNamespace(routes=[], prompt_queue=None)
        server_module = types.ModuleType("server")
        server_module.PromptServer = types.SimpleNamespace(instance=self.server)
        self.server_patch = patch.dict(sys.modules, {"server": server_module})
        self.server_patch.start()
        self.addCleanup(self.server_patch.stop)

    async def test_native_handler_owns_five_and_six_field_queue_layouts(self):
        for width in (5, 6):
            routes._JOBS.clear()
            captured = {}

            async def native(request):
                payload = await request.json()
                captured["headers"] = request.headers
                captured["payload"] = payload
                captured["queue"] = (0, "native-id", payload["prompt"], {}, ["sam31"])
                if width == 6:
                    captured["queue"] += ({"sensitive": "not returned"},)
                return Response({"prompt_id": "native-id"})

            self.server.routes = [types.SimpleNamespace(
                method="POST", path="/prompt", handler=native,
            )]
            request = types.SimpleNamespace(
                headers={"test": "preserved"}, json=self.async_body,
            )
            with patch.object(backend, "capabilities", return_value={"available": True}), \
                    patch.object(backend, "prepare_request", return_value={"enabled": True}), \
                    patch.object(backend, "run_request") as gpu:
                response = await routes.sam31_submit(request)
            data = json.loads(response.body)
            self.assertEqual(response.status, 200)
            self.assertEqual(data["prompt_id"], "native-id")
            self.assertEqual(len(captured["queue"]), width)
            self.assertEqual(captured["headers"], request.headers)
            self.assertEqual(captured["payload"]["prompt"]["sam31"]["class_type"], "DirectorSAM31Job")
            self.assertEqual(routes._JOBS[data["job_id"]]["prompt_id"], "native-id")
            self.assertEqual(routes._JOBS[data["job_id"]]["status"], "queued")
            gpu.assert_not_called()

    async def async_body(self):
        return {"config": {}, "timeline": {}, "group_id": "stable"}

    async def test_native_validation_rejection_marks_submission_failed(self):
        async def reject(request):
            return Response({"error": "validation rejected"}, 400)
        self.server.routes = [types.SimpleNamespace(method="POST", path="/prompt", handler=reject)]
        with patch.object(backend, "capabilities", return_value={"available": True}), \
                patch.object(backend, "prepare_request", return_value={"enabled": True}):
            response = await routes.sam31_submit(types.SimpleNamespace(json=self.async_body))
        self.assertEqual(response.status, 400)
        self.assertEqual(next(iter(routes._JOBS.values()))["status"], "error")
        self.assertNotIn("spec", next(iter(routes._JOBS.values())))

    async def test_queue_snapshot_cannot_overwrite_completion_or_start(self):
        for observed, updated in (("running", "complete"), ("queued", "running")):
            job = {"status": observed, "prompt_id": "p", "spec": {}}
            routes._JOBS["j"] = job
            def snapshot():
                with routes._LOCK:
                    job["status"] = updated
                return [], []
            self.server.prompt_queue = types.SimpleNamespace(get_current_queue=snapshot)
            response = await routes.sam31_job_status(types.SimpleNamespace(query={"job_id": "j"}))
            self.assertEqual(json.loads(response.body)["status"], updated)
            self.assertNotIn("error", job)

    async def test_removed_queue_item_is_reported_without_exposing_spec(self):
        routes._JOBS["j"] = {"status": "queued", "prompt_id": "p", "spec": {"private": True}}
        self.server.prompt_queue = types.SimpleNamespace(get_current_queue=lambda: ([], []))
        response = await routes.sam31_job_status(types.SimpleNamespace(query={"job_id": "j"}))
        data = json.loads(response.body)
        self.assertEqual(data["status"], "error")
        self.assertNotIn("spec", data)

    async def test_internal_node_executes_only_the_prepared_job(self):
        self.assertIs(importlib.import_module(routes.__name__), routes)
        routes._JOBS["j"] = {"status": "queued", "prompt_id": "p", "spec": {"key": "prepared"}}
        with patch.object(backend, "run_request", return_value={"key": "result"}) as run:
            self.assertEqual(bridge.DirectorSAM31Job().execute("j"), ("j",))
        run.assert_called_once_with({"key": "prepared"})
        self.assertEqual(routes._JOBS["j"]["status"], "complete")
        self.assertNotIn("spec", routes._JOBS["j"])
        with self.assertRaises(ValueError):
            bridge.DirectorSAM31Job().execute("j")


class PackTests(unittest.TestCase):
    def test_server_pack_whitelist_and_import_preserve_sam(self):
        spec = importlib.util.spec_from_file_location(
            "_sam_pack_fixture", ROOT / "tests" / "test_timeline_pack_regressions.py",
        )
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        pack = fixture.pack
        data = fixture.timeline()
        data["segments"][0]["id"] = "stable"
        data["segments"][0]["sam31"] = {
            "version": 1, "prompt": "person", "selected": [0],
            "result": {"key": "a" * 64, "action": "detect", "frames": 1, "objects": [{"index": 0}]},
        }
        value = json.dumps(config.normalize_config({"enabled": True, "target_group": "stable"}))
        root = ROOT / "mock_sam_pack"
        documents = {}
        def save(path, payload):
            documents[str(path)] = copy.deepcopy(payload)
        with patch.object(pack.tempfile, "mkdtemp", return_value=str(root)), \
                patch.object(pack, "_pack_export_root", return_value=root), \
                patch.object(pack, "_purge_pack_exports"), \
                patch.object(pack, "_write_json", side_effect=save), \
                patch.object(Path, "mkdir"), patch.object(Path, "rglob", return_value=[]), \
                patch.object(Path, "stat", return_value=types.SimpleNamespace(st_size=123)), \
                patch.object(pack.zipfile, "ZipFile"), patch.object(pack.shutil, "rmtree"):
            pack.build_export_pack(data, {"sam31_config": value})
        self.assertEqual(documents[str(root / "pack.json")]["widgets"]["sam31_config"], value)
        with patch.object(Path, "is_file", autospec=True, side_effect=lambda p: str(p) in documents), \
                patch.object(Path, "is_dir", return_value=False), \
                patch.object(pack, "_read_json", side_effect=lambda p: copy.deepcopy(documents[str(p)])), \
                patch.object(pack, "h3_input_path", return_value=root / "input"), \
                patch.object(pack, "_copy_tree_media"):
            restored = pack.import_extracted_pack(root)
        self.assertEqual(restored["widgets"]["sam31_config"], value)
        self.assertEqual(restored["timeline"]["segments"][0]["sam31"], data["segments"][0]["sam31"])


if __name__ == "__main__":
    unittest.main()
