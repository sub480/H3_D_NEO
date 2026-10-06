"""Pixel-space SAM/V2V regression tests; no GPU, models, service or files."""

import ast
import copy
import importlib
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_sam31_composite_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
utils = types.ModuleType("comfy.utils")


def upscale(tensor, width, height, mode, crop="disabled"):
    if crop == "center":
        h, w = tensor.shape[-2:]
        aspect = width / height
        if w / h > aspect:
            new_w = round(h * aspect)
            tensor = tensor[..., :, (w - new_w) // 2:(w + new_w) // 2]
        else:
            new_h = round(w / aspect)
            tensor = tensor[..., (h - new_h) // 2:(h + new_h) // 2, :]
    return F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)


utils.common_upscale = upscale
comfy = types.ModuleType("comfy")
comfy.__path__ = []
sys.modules.setdefault("comfy", comfy)
sys.modules.setdefault("comfy.utils", utils)
sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
sam = importlib.import_module(f"{PACKAGE}.director.sam31")
composite = importlib.import_module(f"{PACKAGE}.director.sam31_composite")
config = importlib.import_module(f"{PACKAGE}.director.sam31_config")
NS = types.SimpleNamespace


def fixture(count=5, height=4, width=8, fit="stretch"):
    source = dict(
        kind="video", fps=24, range=[0, count], size=[width, height],
        entries=[[0, index] for index in range(count)],
        assets=[dict(path="mock.mp4", digest="content", width=width, height=height)],
    )
    identity = dict(action="track", base=dict(
        version=1, implementation_revision=sam.CACHE_IMPLEMENTATION_REVISION,
        group_id="g", source=source, model=dict(name="sam3.1.safetensors"),
        prompt="shirt", parameters={},
    ))
    key = config.identity_key(identity)
    meta = dict(version=1, key=key, identity=identity, frames=count,
                objects=[dict(index=0), dict(index=1)],
                mask_width=width, packed_shape=[height, (width + 7) // 8])
    state = dict(version=1, prompt="shirt", selected=[0],
                 result=dict(key=key, action="track"), composite=dict(enabled=True, feather=0))
    seg = NS(sam31_group=dict(id="g", taskType="v2v", sam31=state),
             task_key="v2v", source_clip=torch.zeros(count, height, width, 3),
             frame_count=count, source_frame_count=count,
             use_source_resolution=fit == "stretch", video_fit=fit, sam31_composite=None)
    plan = NS(sam31=dict(enabled=True), continuity_enabled=False, frame_rate=24, raw={})
    spec = dict(identity=identity)
    return plan, seg, meta, spec


def zip_masks(meta, masks):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("metadata.json", json.dumps(meta))
        for index, mask in enumerate(masks):
            packed = np.packbits(mask, axis=-1, bitorder="little")
            frame = io.BytesIO()
            np.save(frame, packed, allow_pickle=False)
            archive.writestr(f"frames/{index}.npy", frame.getvalue())
    return buffer.getvalue()


class CompositeTests(unittest.TestCase):
    def prepared(self, **kwargs):
        plan, seg, meta, spec = fixture(**kwargs)
        with (patch.object(sam, "prepare_request", return_value=spec),
              patch.object(sam, "read_metadata", return_value=meta),
              patch.object(sam, "read_packed_frame"),
              patch.object(sam, "cache_path", return_value="fake.zip"),
              patch.object(sam, "file_digest", return_value="artifact-digest")):
            composite.prepare(plan, seg)
        return plan, seg, meta

    def run_apply(self, seg, meta, masks, generated=None, original=None, transforms=()):
        data = zip_masks(meta, masks)
        if generated is None:
            generated = torch.ones(meta["frames"], *masks.shape[-2:], 3)
        if original is None:
            original = torch.zeros_like(generated)
        with (patch.object(sam, "cache_path", side_effect=lambda key: io.BytesIO(data)),
              patch.object(sam, "file_digest", return_value="artifact-digest"),
              patch.object(sam, "verify_assets")):
            return composite.apply(generated, original, seg, transforms=transforms)

    def test_off_default_no_sam_io(self):
        plan, seg, _, _ = fixture()
        seg.sam31_group["sam31"]["composite"]["enabled"] = False
        with patch.object(sam, "prepare_request", side_effect=AssertionError("must bypass")):
            composite.prepare(plan, seg)
        self.assertIsNone(seg.sam31_composite)
        self.assertEqual(composite.fingerprint(seg), {})
        self.assertEqual(composite.fingerprint(seg, first_pass=True), {})
        self.assertFalse(composite.execution_required({"segments": []}))
        self.assertFalse(composite.execution_required("broken json"))
        self.assertFalse(composite.execution_required(json.dumps({"segments": [seg.sam31_group]})))
        seg.sam31_group["sam31"]["composite"]["enabled"] = True
        self.assertTrue(composite.execution_required(json.dumps({"segments": [seg.sam31_group]})))

    def test_invalid_mode_continuity_missing_selection_stop_before_sampling(self):
        for change in ("disabled", "task", "continuity", "detect", "selected"):
            with self.subTest(change=change):
                plan, seg, _, _ = fixture()
                if change == "disabled": plan.sam31["enabled"] = False
                if change == "task": seg.task_key = "t2v"
                if change == "continuity": plan.continuity_enabled = True
                if change == "detect": seg.sam31_group["sam31"]["result"]["action"] = "detect"
                if change == "selected": seg.sam31_group["sam31"]["selected"] = []
                with self.assertRaises(ValueError), patch.object(sam, "prepare_request") as request:
                    composite.prepare(plan, seg)
                request.assert_not_called()

    def test_preflight_stale_range_frame_count_and_object(self):
        for change in ("source", "count", "object", "loaded"):
            with self.subTest(change=change):
                plan, seg, meta, spec = fixture()
                meta = copy.deepcopy(meta)
                if change == "source": meta["identity"]["base"]["source"]["range"] = [1, 6]
                if change == "count": seg.frame_count = 6
                if change == "object": seg.sam31_group["sam31"]["selected"] = [15]
                if change == "loaded": seg.source_clip = None
                with (patch.object(sam, "prepare_request", return_value=spec),
                      patch.object(sam, "read_metadata", return_value=meta),
                      patch.object(sam, "read_packed_frame"), self.assertRaises(ValueError)):
                    composite.prepare(plan, seg)

    def test_masks_union_and_empty_frame_preserve_original(self):
        _, seg, meta = self.prepared()
        seg.sam31_composite["selected"] = [0, 1]
        masks = np.zeros((5, 2, 4, 8), dtype=np.uint8)
        masks[0, 0, :, 0:2] = 1
        masks[0, 1, :, 6:8] = 1
        result = self.run_apply(seg, meta, masks)
        self.assertTrue(torch.all(result[0, :, 0:2] == 1))
        self.assertTrue(torch.all(result[0, :, 6:8] == 1))
        self.assertTrue(torch.all(result[0, :, 2:6] == 0))
        self.assertTrue(torch.all(result[1:] == 0))

    def test_unselected_object_and_fractional_blend(self):
        _, seg, meta = self.prepared()
        seg.sam31_composite["settings"]["blend"] = 0.25
        masks = np.ones((5, 2, 4, 8), dtype=np.uint8)
        masks[:, 0, :, 4:] = 0
        original = torch.full((5, 4, 8, 3), 0.2)
        result = self.run_apply(seg, meta, masks, original=original)
        torch.testing.assert_close(result[..., :4, :], torch.full((5, 4, 4, 3), 0.4))
        self.assertTrue(torch.equal(result[..., 4:, :], original[..., 4:, :]))

    def test_grow_and_feather_have_bounded_support(self):
        _, seg, meta = self.prepared(height=16, width=24)
        settings = seg.sam31_composite["settings"]
        settings.update(grow=1, feather=1)
        masks = np.zeros((5, 2, 16, 24), dtype=np.uint8)
        masks[:, 0, 6:10, 10:14] = 1
        result = self.run_apply(seg, meta, masks)
        self.assertGreater(float(result[0, 5, 9, 0]), 0)
        self.assertTrue(torch.all(result[:, :3] == 0))
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)

    def test_contain_mask_uses_zero_letterbox(self):
        _, seg, meta = self.prepared(height=4, width=8, fit="contain")
        seg.sam31_composite["canvas"] = [8, 8]
        masks = np.ones((5, 2, 4, 8), dtype=np.uint8)
        original = torch.full((5, 8, 8, 3), 0.5)
        result = self.run_apply(seg, meta, masks, generated=torch.ones_like(original), original=original)
        self.assertTrue(torch.all(result[:, :2] == 0.5))
        self.assertTrue(torch.all(result[:, 2:6] == 1))
        self.assertTrue(torch.all(result[:, 6:] == 0.5))

    def test_no_time_resize_and_corrupt_cache_fail(self):
        _, seg, meta = self.prepared()
        masks = np.zeros((5, 2, 4, 8), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "帧数或画幅不同"):
            self.run_apply(seg, meta, masks, generated=torch.zeros(4, 4, 8, 3))
        changed = copy.deepcopy(meta)
        changed["objects"][0]["score"] = 0.2
        with self.assertRaisesRegex(ValueError, "缓存发生变化"):
            self.run_apply(seg, changed, masks)

    def test_source_decoder_uses_native_size_not_analysis_resolution(self):
        _, seg, meta = self.prepared()
        meta["identity"]["base"]["source"]["size"] = [4, 2]
        calls = []
        def decode(descriptor, indices):
            calls.append((descriptor["size"], indices))
            return torch.full((1, 4, 8, 3), indices[0] / 10)
        with (patch.object(sam, "decode_source", side_effect=decode),
              patch.object(sam, "verify_assets")):
            original = composite.source_frames(seg)
        self.assertEqual([size for size, _ in calls], [[8, 4]] * 5)
        self.assertEqual([indices for _, indices in calls], [[i] for i in range(5)])
        torch.testing.assert_close(original[4], torch.full((4, 8, 3), 0.4))

    def test_mask_controls_invalidate_first_pass_but_paste_controls_do_not(self):
        _, seg, _ = self.prepared()
        first = copy.deepcopy(composite.fingerprint(seg, first_pass=True))
        final = copy.deepcopy(composite.fingerprint(seg))
        seg.sam31_composite["selected"] = [1]
        seg.sam31_composite["settings"]["grow"] = 10
        self.assertNotEqual(composite.fingerprint(seg, first_pass=True), first)
        self.assertNotEqual(composite.fingerprint(seg), final)
        first = copy.deepcopy(composite.fingerprint(seg, first_pass=True))
        seg.sam31_composite["settings"].update(feather=8, blend=0.5)
        self.assertEqual(composite.fingerprint(seg, first_pass=True), first)
        seg.sam31_composite["meta"]["identity"]["base"]["source"]["assets"][0]["digest"] = "changed"
        self.assertNotEqual(composite.fingerprint(seg, first_pass=True), first)

    def test_executor_stages_final_composite_before_cache_export_and_preview(self):
        source = (ROOT / "director/executor_core.py").read_text(encoding="utf-8-sig")
        ast.parse(source)
        self.assertLess(source.index("prepare_sam31_composite(plan, segment)"),
                        source.index("prune_segment_cache(node_id"))
        start = source.index("chunk = apply_sam31_composite")
        self.assertLess(source.index("chunk, face_note ="), start)
        self.assertLess(start, source.index("save_segment_cache(", start))
        self.assertLess(start, source.index("maybe_export_segment_mp4s(", start))
        self.assertIn("not (_first_pass_phase and sam31_context is not None)", source)
        self.assertIn("preview_pixels = chunk if sam31_context", source)
        self.assertIn("sam31_group=seg_data", (ROOT / "director/gen_timeline.py").read_text(encoding="utf-8-sig"))

    def test_masked_edit_preflight_before_any_cache_or_sampling(self):
        code = (ROOT / "director/executor_core.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(code)
        execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "execute_director_plan_core")
        guard = next(node for node in execute.body if isinstance(node, ast.For)
                     and "prepare_sam31_composite" in ast.unparse(node))
        masked = types.ModuleType(f"{PACKAGE}.director.sam31_masked")
        masked.preflight = MagicMock(side_effect=ValueError("incompatible masked edit"))
        plan, seg, _, _ = fixture()
        seg.index = 0
        scope = {
            "__package__": f"{PACKAGE}.director", "all_segments": [seg],
            "sam31_run_indices": {0}, "plan": plan,
            "prepare_sam31_composite": lambda p, s: setattr(s, "sam31_composite", {"canvas": [8, 4]}),
            "_model_for_segment": lambda *args: "model", "model": "model", "model_r2v": None,
        }
        with patch.dict(sys.modules, {masked.__name__: masked}):
            statement = compile(ast.Module(body=[guard], type_ignores=[]), "guard", "exec")
            with self.assertRaisesRegex(ValueError, "incompatible masked edit"):
                exec(statement, scope)
        masked.preflight.assert_called_once_with(plan, seg, "model")

    def test_cache_status_reports_unchecked_not_false_mismatch(self):
        cache = importlib.import_module(f"{PACKAGE}.director.segment_cache")
        plan, seg, _, _ = fixture()
        seg.index, seg.timeline_index = 0, 0
        plan.segments, plan.run_indices = [seg], None
        path = MagicMock()
        path.is_file.return_value = True
        path.read_text.return_value = json.dumps({"seed": 1, "sam31_source": {"verified": True}})
        root = MagicMock()
        root.__truediv__.return_value = path
        with (patch.object(cache, "h3_output_path", return_value=root),
              patch.object(cache, "_first_pass_paths", return_value={"latent": path}),
              patch.object(cache, "first_pass_cache_fingerprint", return_value={"seed": 1})):
            result = cache.inspect_first_pass_cache("node", plan)
        self.assertEqual(result["segments"][0]["status"], "unchecked")
        self.assertEqual(result["diff_keys"], [])
        self.assertFalse(result["matches"])
        self.assertIn("实际运行时", result["segments"][0]["error"])


if __name__ == "__main__":
    unittest.main()
