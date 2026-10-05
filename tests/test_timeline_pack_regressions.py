"""CPU-only timeline/pack compatibility checks; run this file directly."""

import copy
import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_timeline_pack_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package

folder_paths = types.ModuleType("folder_paths")
folder_paths.get_input_directory = lambda: str(ROOT)
folder_paths.get_output_directory = lambda: str(ROOT)
folder_paths.get_temp_directory = lambda: str(ROOT)
sys.modules.setdefault("folder_paths", folder_paths)
comfy = types.ModuleType("comfy")
comfy.__path__ = []
utils = types.ModuleType("comfy.utils")
utils.common_upscale = lambda tensor, width, height, *_: torch.nn.functional.interpolate(
    tensor, size=(height, width), mode="nearest"
)
comfy.utils = utils
sys.modules.setdefault("comfy", comfy)
sys.modules.setdefault("comfy.utils", utils)
try:
    import aiohttp
except ModuleNotFoundError:
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.SimpleNamespace()
    sys.modules["aiohttp"] = aiohttp

gen = importlib.import_module(f"{PACKAGE}.director.gen_timeline")
planner = importlib.import_module(f"{PACKAGE}.director.plan")
pack = importlib.import_module(f"{PACKAGE}.director.pack")


def timeline(task="mixed"):
    return {
        "version": 5, "timelineMode": "prompt_batch", "editMode": "segment",
        "global": {"taskType": task, "prompt": "shared scene"},
        "output": {"mode": "fixed", "width": 864, "height": 480},
        "segments": [
            {"taskType": "t2v", "frameCount": 5, "prompt": ""},
            {"taskType": "t2v", "frameCount": 5, "prompt": "local scene"},
        ],
    }


def build(data):
    return planner.build_director_plan(
        json.dumps(data), global_task_type=data["global"]["taskType"],
        global_prompt="widget fallback", total_frames=5, frame_rate=24,
        width=864, height=480, ref_max_size=864, load_media=False,
    )


class TimelineCompatibilityTests(unittest.TestCase):
    def test_legacy_empty_local_falls_back_without_concatenating_nonempty(self):
        for task in ("t2v", "mixed"):
            with self.subTest(task=task):
                data = timeline(task)
                before = copy.deepcopy(data)
                result = build(data)
                self.assertEqual([s.prompt for s in result.segments],
                                 ["shared scene", "local scene"])
                self.assertEqual(data, before)
                self.assertNotIn("commonEnabled", result.raw["global"])
                self.assertNotIn("t2vCommon", result.raw["global"])

    def test_whitespace_local_and_widget_fallback(self):
        data = timeline()
        data["segments"][0]["prompt"] = "  \n "
        data["global"]["prompt"] = ""
        self.assertEqual(build(data).segments[0].prompt, "widget fallback")

    def test_generation_mode_is_segment_not_legacy_video_global(self):
        data = timeline()
        data["editMode"] = "global"
        result = build(data)
        self.assertEqual(result.edit_mode, "segment")
        self.assertTrue(all(not s.use_global for s in result.segments))
        self.assertEqual((result.width, result.height), (864, 480))

    def test_source_aspect_is_unchanged_by_prompt_fallback(self):
        data = timeline()
        data["output"].update(aspectRatio="\u4e0e\u539f\u89c6\u9891\u4e00\u81f4", megapixels=0.4)
        data["segments"] = [{"taskType": "i2v", "frameCount": 5, "prompt": "",
                             "genImage": {"imageFile": "portrait.png", "width": 90, "height": 160}}]
        result = build(data)
        self.assertEqual(result.segments[0].prompt, "shared scene")
        self.assertEqual((result.segments[0].output_width, result.segments[0].output_height),
                         (480, 864))


class PackCompatibilityTests(unittest.TestCase):
    def test_full_timeline_roundtrip_preserves_metadata_without_weights(self):
        data = timeline()
        data["global"]["loras"] = [{"name": "not_packaged.safetensors", "strength": 0.7}]
        data["segments"][0]["loras"] = [{"name": "segment.safetensors", "strength": 0.4}]
        before = copy.deepcopy(data)
        root = ROOT / "mock_pack"
        documents = {}
        def save(path, payload):
            documents[str(path)] = copy.deepcopy(payload)
        with patch.object(pack.tempfile, "mkdtemp", return_value=str(root)), \
                patch.object(pack, "_pack_export_root", return_value=root), \
                patch.object(pack, "_purge_pack_exports"), \
                patch.object(pack, "_write_json", side_effect=save), \
                patch.object(Path, "mkdir"), \
                patch.object(Path, "rglob", return_value=[]), \
                patch.object(Path, "stat", return_value=types.SimpleNamespace(st_size=123)), \
                patch.object(pack.zipfile, "ZipFile") as archive, \
                patch.object(pack.shutil, "rmtree"), \
                patch.object(pack, "_copy_file", side_effect=AssertionError("weight/media copy")):
            result = pack.build_export_pack(data)
        self.assertEqual(result["fileCount"], 0)
        archive.return_value.__enter__.return_value.write.assert_not_called()
        with patch.object(Path, "is_file", autospec=True,
                          side_effect=lambda path: str(path) in documents), \
                patch.object(Path, "is_dir", return_value=False), \
                patch.object(pack, "_read_json", side_effect=lambda path: copy.deepcopy(documents[str(path)])), \
                patch.object(pack, "h3_input_path", return_value=root / "input"), \
                patch.object(pack, "_copy_tree_media"):
            imported = pack.import_extracted_pack(root)["timeline"]
        self.assertEqual(imported["global"]["loras"], data["global"]["loras"])
        self.assertEqual(imported["segments"][0]["loras"], data["segments"][0]["loras"])
        self.assertEqual(imported["editMode"], "segment")
        self.assertEqual(imported["output"], data["output"])
        self.assertEqual(data, before)

    def test_missing_timeline_is_rejected_in_this_baseline(self):
        metadata = {"format": pack.PACK_FORMAT, "formatVersion": pack.PACK_VERSION}
        with patch.object(pack, "_read_json", return_value=metadata), \
                patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(ValueError, "missing pack.json"):
                pack.import_extracted_pack(ROOT)
        # Isolate header existence from the absent timeline file.
        with patch.object(pack, "_read_json", return_value=metadata), \
                patch.object(Path, "is_file", side_effect=[True, False]):
            with self.assertRaisesRegex(ValueError, "missing timeline.json"):
                pack.import_extracted_pack(ROOT)

    def test_old_pack_does_not_acquire_lora_or_edit_mode_defaults(self):
        data = timeline()
        del data["editMode"]
        header = {"format": pack.PACK_FORMAT, "formatVersion": pack.PACK_VERSION}
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(pack, "_read_json", side_effect=[header, data]), \
                patch.object(pack, "_fill_empty_prompts_from_pack_files"), \
                patch.object(pack, "_copy_tree_media"), \
                patch.object(pack, "_collect_missing_media"):
            restored = pack.import_extracted_pack(ROOT)["timeline"]
        self.assertNotIn("loras", restored["global"])
        self.assertNotIn("editMode", restored)


if __name__ == "__main__":
    unittest.main()
