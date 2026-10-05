"""CPU-only detection diagnostics; no detector weights or ComfyUI required."""

import ast
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_face_diagnostics_track", ROOT / "director/face_refine/track.py"
)
track = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(track)


def detection(boxes):
    xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    result = types.SimpleNamespace(boxes=xyxy)
    # Ultralytics exposes both len(boxes) and boxes.xyxy.
    class Boxes:
        def __len__(self):
            return len(xyxy)

        def __init__(self):
            self.xyxy = xyxy

    result.boxes = Boxes()
    return [result]


class FaceDiagnosticsTests(unittest.TestCase):
    def run_track(self, predictions):
        detector = Mock()
        detector.predict.side_effect = predictions
        frames = torch.zeros((len(predictions), 32, 32, 3))
        with patch.object(track, "load_detector", return_value=detector):
            return track.track_and_crop(
                frames, {"canvas_width": 32, "canvas_height": 32, "crop_factor": 1}
            )

    def test_all_detection_failures_report_fault_not_no_face(self):
        with self.assertLogs(track.log, level="WARNING"):
            crops, transform, note = self.run_track(
                [RuntimeError("backend unavailable"), RuntimeError("backend unavailable")]
            )
        self.assertIsNone(crops)
        self.assertIsNone(transform)
        self.assertIn("detector failed on all 2 frames", note)
        self.assertIn("RuntimeError: backend unavailable", note)
        self.assertNotIn("降低 confidence", note)

    def test_successful_empty_detections_keep_existing_no_face_behavior(self):
        with self.assertRaisesRegex(ValueError, "未检测到人脸"):
            self.run_track([detection([]), detection([])])

    def test_partial_failures_do_not_discard_a_valid_track(self):
        with self.assertLogs(track.log, level="WARNING"):
            crops, transform, note = self.run_track(
                [RuntimeError("one frame failed"), detection([[8, 8, 24, 24]])]
            )
        self.assertEqual(tuple(crops.shape), (2, 32, 32, 3))
        self.assertEqual(transform["detected"], [False, True])
        self.assertIn("detection failed on 1 frames", note)

    def test_detector_load_errors_are_not_swallowed(self):
        with patch.object(track, "load_detector", side_effect=ImportError("missing dependency")):
            with self.assertRaisesRegex(ImportError, "missing dependency"):
                track.track_and_crop(torch.zeros((1, 32, 32, 3)), {})

    def test_runtime_skipped_detection_returns_original_without_sampling(self):
        tree = ast.parse((ROOT / "director/face_refine/runtime.py").read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "apply_segment_face_refine"
        )
        # Keep the complete production body, supplying ComfyUI dependencies as spies.
        function.body = [node for node in function.body if not isinstance(node, ast.ImportFrom)]
        conditioning = Mock(side_effect=AssertionError("conditioning must not run"))
        inject = Mock(side_effect=AssertionError("injection must not run"))
        sample = Mock(side_effect=AssertionError("sampling must not run"))
        namespace = {
            "torch": torch,
            "Any": object,
            "track_and_crop": track.track_and_crop,
            "run_minimax_conditioning": conditioning,
            "inject_video_latent": inject,
            "sample_single_stage": sample,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "runtime.py", "exec"), namespace)
        frames = torch.rand((2, 32, 32, 3))
        detector = Mock()
        detector.predict.side_effect = RuntimeError("backend unavailable")
        with patch.object(track, "load_detector", return_value=detector):
            with self.assertLogs(track.log, level="WARNING"):
                output, note = namespace["apply_segment_face_refine"](
                    frames=frames, plan=None, seg=None, pack={}, model=None, vae=None,
                    audio_vae=None, clip=None, seed=0, cfg=1, shift_video=12, shift_audio=3,
                )
        self.assertTrue(torch.equal(output, frames))
        self.assertIn("detector failed on all 2 frames", note)
        self.assertIn("original frames retained", note)
        conditioning.assert_not_called()
        inject.assert_not_called()
        sample.assert_not_called()


if __name__ == "__main__":
    unittest.main()
