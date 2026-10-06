"""Run with python -B tests/test_sam31_backend.py; no models or service."""

import copy
import importlib
import json
from pathlib import Path
import sys
import types
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_sam31_backend_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
folder_paths = types.ModuleType("folder_paths")
sys.modules.setdefault("folder_paths", folder_paths)
comfy = types.ModuleType("comfy")
comfy.__path__ = []
sys.modules.setdefault("comfy", comfy)
utils = types.ModuleType("comfy.utils")
utils.common_upscale = lambda tensor, width, height, *_: torch.nn.functional.interpolate(
    tensor, size=(height, width), mode="nearest",
)
sys.modules.setdefault("comfy.utils", utils)
management = types.ModuleType("comfy.model_management")
management.cleanup_models_gc = lambda: None
supported = types.ModuleType("comfy.supported_models")
supported.SAM31 = type("SAM31", (), {})
sam = importlib.import_module(f"{PACKAGE}.director.sam31")
video = importlib.import_module(f"{PACKAGE}.lib.video_io")
NS = types.SimpleNamespace


class Capture:
    def __init__(self, failures=(), seek_ok=True):
        self.failures = set(failures)
        self.seek_ok = seek_ok
        self.pos = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {1: 24, 2: 8, 3: 4, 4: 3}[prop]

    def set(self, prop, pos):
        self.pos = pos
        return self.seek_ok

    def read(self):
        pos = self.pos
        self.pos += 1
        if pos in self.failures:
            return False, None
        return True, np.full((8, 4, 3), pos + 10, dtype=np.uint8)

    def release(self):
        self.released = True


def cv_backend(captures, failures=(), seek_ok=True):
    def open_capture(path):
        cap = Capture(failures, seek_ok)
        captures.append(cap)
        return cap

    return NS(
        VideoCapture=open_capture,
        CAP_PROP_FPS=1, CAP_PROP_FRAME_WIDTH=2,
        CAP_PROP_FRAME_HEIGHT=3, CAP_PROP_FRAME_COUNT=4,
        CAP_PROP_POS_FRAMES=5, COLOR_BGR2RGB=6,
        INTER_AREA=7, ROTATE_90_CLOCKWISE=8,
        cvtColor=lambda frame, _: frame,
        rotate=lambda frame, _: np.rot90(frame, -1),
        resize=lambda frame, size, **_: np.full(
            (size[1], size[0], 3), frame[0, 0, 0], dtype=np.uint8,
        ),
    )


class SAMBackendTests(unittest.TestCase):
    def test_failed_frame_default_fallback_but_strict_rejects(self):
        captures = []
        with patch.object(video, "_require_cv2", return_value=cv_backend(captures, {1})):
            result = video.load_video_resampled(
                "mock.mp4", 24, [0, 1, 2],
                storage_width=4, storage_height=8, hold_past_eof=False,
            )
            self.assertEqual(result.shape[0], 3)
            self.assertTrue(torch.equal(result[0], result[1]))
            with self.assertRaisesRegex(ValueError, "Cannot decode source frame 1"):
                video.load_video_resampled(
                    "mock.mp4", 24, [0, 1, 2],
                    storage_width=4, storage_height=8,
                    hold_past_eof=False, strict=True,
                )
        self.assertTrue(all(cap.released for cap in captures))

    def test_strict_eof_and_seek_failure_release_capture(self):
        for frames, seek_ok, message in (
            ([0, 3], True, "exceeds video EOF"),
            ([0], False, "Cannot seek source frame"),
        ):
            with self.subTest(message=message):
                captures = []
                with patch.object(
                    video, "_require_cv2",
                    return_value=cv_backend(captures, seek_ok=seek_ok),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        video.load_video_resampled(
                            "mock.mp4", 24, frames, strict=True,
                            hold_past_eof=False,
                        )
                self.assertTrue(captures[-1].released)

    def test_display_descriptor_and_strict_sam_decode_align(self):
        clip = dict(videoFile="mock.mp4", sourceFrameCount=3)
        segment = dict(taskType="v2v", sourceVideo=dict(
            mediaKind="video", video=clip, videoClips=[clip],
            totalFrames=3, rangeStart=0, rangeEnd=3,
        ))
        captures = []
        with (
            patch.object(video, "_require_cv2", return_value=cv_backend(captures)),
            patch.object(video, "probe_video_file", return_value=dict(
                width=8, height=4, duration=3 / 24,
            )),
            patch.object(sam, "input_path", return_value=ROOT / "mock.mp4"),
            patch.object(sam, "file_digest", return_value="mock-digest"),
        ):
            source = sam.source_descriptor(segment, 24, 768)
            self.assertEqual(source["size"], [4, 8])
            self.assertEqual(
                (source["assets"][0]["width"], source["assets"][0]["height"]),
                (4, 8),
            )
            images = sam.decode_source(source, [0, 1, 2])
            self.assertEqual(tuple(images.shape), (3, 8, 4, 3))
        self.assertTrue(all(cap.released for cap in captures))

    def test_preserves_multi_cond_and_global_topk_mask_box_alignment(self):
        multi = [
            dict(cond=torch.tensor([1]), attention_mask=None, max_detections=2),
            dict(cond=torch.tensor([2]), attention_mask=None, max_detections=2),
        ]
        conditioning = [[multi[0]["cond"], dict(sam3_multi_cond=multi)]]
        clip = NS(
            tokenize=lambda text: text,
            encode_from_tokens_scheduled=lambda tokens: conditioning,
        )
        masks = torch.zeros(3, 2, 8)
        for i in range(3):
            masks[i, :, i] = 1
        boxes = [dict(
            x=i, y=0, width=1, height=2, score=score,
        ) for i, score in enumerate((0.4, 0.9, 0.8))]

        def detect(**kwargs):
            self.assertIs(kwargs["conditioning"][0][1]["sam3_multi_cond"], multi)
            self.assertEqual(len(multi), 2)
            return NS(result=(masks, [boxes]))

        model = NS(model=NS(model_config=supported.SAM31()))
        loader = NS(load_checkpoint_guess_config=lambda *args, **kwargs: (
            model, clip, None, None,
        ))
        native = NS(SAM3_Detect=NS(execute=detect))
        source = dict(
            kind="image", size=[8, 2], range=[0, 1], entries=[[0, 0]],
            assets=[dict(width=8, height=2)],
        )
        identity = dict(action="detect", base=dict(
            implementation_revision=sam.CACHE_IMPLEMENTATION_REVISION,
            source=source, model=dict(path="mock.safetensors"),
            prompt="person, dog", parameters=dict(threshold=0.5, max_objects=2),
        ))
        saved = {}

        def write_cache(key, meta, frames):
            saved.update(meta=meta, packed=np.stack(list(frames)))

        with (
            patch.dict(sys.modules, {
                "comfy.model_management": management,
                "comfy.supported_models": supported,
            }),
            patch.object(sam, "native_backend", return_value=(native, loader)),
            patch.object(sam, "verify_assets"),
            patch.object(sam, "_valid_cache", return_value=None),
            patch.object(sam, "decode_source", return_value=torch.zeros(1, 2, 8, 3)),
            patch.object(sam, "write_cache", side_effect=write_cache),
        ):
            result = sam.run_request(dict(
                key=sam.identity_key(identity), identity=identity,
            ))
        self.assertEqual([o["score"] for o in result["objects"]], [0.9, 0.8])
        self.assertEqual([o["box"]["x"] for o in result["objects"]], [1, 2])
        unpacked = np.unpackbits(saved["packed"], axis=-1, bitorder="little")
        np.testing.assert_array_equal(unpacked[0], masks[[1, 2]].numpy())

    def run_mock_tracking(self, count, selected=True):
        refs, cleanup_alive = [], []

        def track(**kwargs):
            packed = torch.zeros(1, count, 2, 1, dtype=torch.uint8)
            refs.append(weakref.ref(packed))
            return NS(result=(dict(packed_masks=packed, scores=[]),))

        loader = NS(load_checkpoint_guess_config=lambda *a, **kw: (
            NS(model=NS(model_config=supported.SAM31())),
            NS(tokenize=lambda text: text,
               encode_from_tokens_scheduled=lambda tokens: [[torch.zeros(1), {}]]),
            None, None,
        ))
        identity = dict(
            action="track_selected" if selected else "track",
            base=dict(
                implementation_revision=sam.CACHE_IMPLEMENTATION_REVISION,
                source=dict(kind="video", size=[8, 2], range=[0, 1],
                            entries=[[0, 0]], assets=[dict(width=8, height=2)]),
                model=dict(path="mock"), prompt="person",
                parameters=dict(threshold=0.5, max_objects=4, detect_interval=1),
            ),
        )
        if selected:
            identity["initial"] = dict(key="mock", indices=[0, 1])
        with (
            patch.dict(sys.modules, {
                "comfy.model_management": management,
                "comfy.supported_models": supported,
            }),
            patch.object(management, "cleanup_models_gc",
                         side_effect=lambda: cleanup_alive.append(
                             refs[0]() is not None if refs else False)),
            patch.object(sam, "native_backend", return_value=(
                NS(SAM3_VideoTrack=NS(execute=track)), loader,
            )),
            patch.object(sam, "verify_assets"),
            patch.object(sam, "_valid_cache", return_value=None),
            patch.object(sam, "decode_source", return_value=torch.zeros(1, 2, 8, 3)),
            patch.object(sam, "read_packed_frame", return_value=(
                dict(objects=[{}, {}], mask_width=8),
                np.zeros((2, 2, 1), dtype=np.uint8),
            )),
            patch.object(sam, "write_cache",
                         side_effect=lambda key, meta, frames: list(frames)) as write,
        ):
            try:
                result = sam.run_request(dict(
                    key=sam.identity_key(identity), identity=identity,
                ))
            except RuntimeError:
                write.assert_not_called()
                raise
        return result, cleanup_alive

    def test_selected_count_must_match_before_mapping_and_cache(self):
        for count in (0, 1, 3):
            with self.subTest(count=count):
                with self.assertRaisesRegex(RuntimeError, "selected object count"):
                    self.run_mock_tracking(count)
        result, alive = self.run_mock_tracking(2)
        self.assertEqual(
            [obj["detection_index"] for obj in result["objects"]], [0, 1],
        )
        self.assertEqual(alive, [False])

    def test_tracking_result_released_before_cleanup(self):
        result, alive = self.run_mock_tracking(1, selected=False)
        self.assertEqual(len(result["objects"]), 1)
        self.assertEqual(alive, [False])

    def test_revision_changes_key_and_rejects_old_metadata(self):
        body = dict(
            group_id="g",
            timeline=dict(frameRate=24, segments=[
                dict(id="g", sam31=dict(prompt="person")),
            ]),
        )
        with (
            patch.object(sam, "source_descriptor", return_value=dict(
                kind="image", entries=[[0, 0]], size=[8, 2],
            )),
            patch.object(sam, "model_identity", return_value=dict(name="mock")),
        ):
            spec = sam.prepare_request(body)
        old = copy.deepcopy(spec["identity"])
        old["base"].pop("implementation_revision")
        self.assertNotEqual(spec["key"], sam.identity_key(old))
        meta = dict(
            version=1, key=spec["key"], identity=spec["identity"],
            objects=[], packed_shape=[2, 1], mask_width=8, frames=1,
        )
        archive = NS(read=lambda name: json.dumps(meta).encode())
        self.assertEqual(sam._metadata(archive, spec["key"])["frames"], 1)
        meta.update(key=sam.identity_key(old), identity=old)
        with self.assertRaisesRegex(ValueError, "implementation changed"):
            sam._metadata(archive, meta["key"])


if __name__ == "__main__":
    unittest.main()
