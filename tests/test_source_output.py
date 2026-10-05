"""Verify source output uses selected source clips, without ComfyUI or video IO."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
import unittest
from unittest.mock import Mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "_fit_source_clip_to_plan", "_segment_source_for_output",
    "_pad_source_last_frame", "build_source_images_output",
}
tree = ast.parse((ROOT / "nodes/director_common.py").read_text(encoding="utf-8-sig"))
namespace = {
    "torch": torch,
    "load_timeline_segment": Mock(side_effect=AssertionError("must use the group's selected clip")),
}
for path, name in (("director/frame_align.py", "pad_or_trim_frames"),
                   ("director/frame_align.py", "minimax_floor_frame_count"),
                   ("lib/image_prep.py", "cat_frames_variable_size")):
    helper_tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    helper = next(n for n in helper_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), path, "exec"), namespace)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in NAMES], type_ignores=[]), "director_common.py", "exec"), namespace)
build_source = namespace["build_source_images_output"]


def segment(task, frames):
    return SimpleNamespace(task_key=task, source_clip=frames, use_source_resolution=True,
                           frame_count=len(frames), start_frame=0, end_frame=len(frames))


def plan(segments, indices=None):
    return SimpleNamespace(segments=segments, run_indices=indices, output_mode="fixed",
                           total_frames=sum(s.frame_count for s in segments),
                           width=2, height=2, raw={})


class SourceOutputTests(unittest.TestCase):
    def test_video_decoder_samples_the_cut_range_on_a_24_fps_clock(self):
        video_tree = ast.parse((ROOT / "lib/video_io.py").read_text(encoding="utf-8-sig"))
        decoder = next(n for n in video_tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "load_video_resampled")
        for native_fps, first, last in ((24, 48, 71), (30, 60, 89), (60, 120, 178)):
            with self.subTest(native_fps=native_fps):
                cap = Mock()
                cap.isOpened.return_value = True
                cap.get.side_effect = lambda key: {1: native_fps, 2: 2, 3: 2, 4: 1000}[key]
                cap.read.return_value = (True, np.zeros((2, 2, 3), dtype=np.uint8))
                cv2 = SimpleNamespace(VideoCapture=lambda _: cap, CAP_PROP_FPS=1,
                                      CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3,
                                      CAP_PROP_FRAME_COUNT=4, CAP_PROP_POS_FRAMES=5,
                                      COLOR_BGR2RGB=6, cvtColor=lambda image, _: image)
                scope = {"torch": torch, "np": np, "Sequence": Sequence,
                         "_require_cv2": lambda: cv2,
                         "_resolve_load_dimensions": lambda *_args, **_kwargs: (2, 2, False)}
                exec(compile(ast.Module(body=[decoder], type_ignores=[]), "video_io.py", "exec"), scope)
                # Logical [48,72) is the one-second source interval beginning at 2s.
                frames = scope["load_video_resampled"]("unused.mp4", 24, list(range(48, 72)))
                self.assertEqual(len(frames), 24)
                seeks = [call.args[1] for call in cap.set.call_args_list]
                self.assertEqual((seeks[0], seeks[-1]), (first, last))
                self.assertEqual(seeks, [round(i / 24 * native_fps) for i in range(48, 72)])
                cap.release.assert_called_once()

    def test_h3_alignment_changes_frame_count_not_the_24_fps_clock(self):
        align = namespace["minimax_floor_frame_count"]
        self.assertEqual(align(48), 39)
        self.assertEqual(align(39), 39)
        self.assertEqual(39 / 24, 1.625)

    def test_v2v_and_rv2v_output_the_selected_clip_not_generated_images(self):
        original = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1).expand(-1, 2, 2, 3) / 10
        selected = original[2:5]
        generated = torch.full_like(selected, 0.9)
        for task in ("v2v", "rv2v"):
            with self.subTest(task=task):
                result = build_source(plan([segment(task, selected)]), [generated], split_outputs=True)
                self.assertEqual(len(result), 1)
                self.assertTrue(torch.equal(result[0], selected))
                self.assertFalse(torch.equal(result[0], generated))

    def test_run_selection_uses_the_corresponding_group_and_pads_its_last_frame(self):
        a = torch.full((3, 2, 2, 3), 0.1)
        b = torch.full((2, 2, 2, 3), 0.7)
        p = plan([segment("v2v", a), segment("rv2v", b)], [1])
        result = build_source(p, [torch.zeros((4, 2, 2, 3))], split_outputs=True)
        self.assertEqual(tuple(result[0].shape), (4, 2, 2, 3))
        self.assertTrue(torch.equal(result[0], b[-1:].repeat(4, 1, 1, 1)))

    def test_combined_output_preserves_source_order_and_export_lengths(self):
        a = torch.full((4, 2, 2, 3), 0.1)
        b = torch.full((4, 2, 2, 3), 0.7)
        p = plan([segment("v2v", a), segment("rv2v", b)])
        result = build_source(p, [torch.zeros((5, 2, 2, 3))], split_outputs=False,
                              segment_frame_counts=[2, 3])
        expected = torch.cat([a[:2], b[:3]])
        self.assertTrue(torch.equal(result[0], expected))

    def test_combined_run_selection_excludes_unselected_sources(self):
        clips = [torch.full((4, 2, 2, 3), value) for value in (0.1, 0.4, 0.7)]
        p = plan([segment("v2v", clip) for clip in clips], [0, 2])
        result = build_source(p, [torch.zeros((5, 2, 2, 3))], split_outputs=False,
                              segment_frame_counts=[2, 3])
        self.assertTrue(torch.equal(result[0], torch.cat([clips[0][:2], clips[2][:3]])))

    def test_non_source_group_is_placeholder_not_generated_or_another_source(self):
        clip = torch.full((3, 2, 2, 3), 0.7)
        non_source = SimpleNamespace(task_key="t2v", source_clip=None,
                                     use_source_resolution=True, frame_count=2)
        p = plan([non_source, segment("rv2v", clip)])
        result = build_source(p, [torch.zeros((5, 2, 2, 3))], split_outputs=False,
                              segment_frame_counts=[2, 3])
        self.assertTrue(torch.equal(result[0], torch.cat([torch.full((2, 2, 2, 3), 0.5), clip])))


if __name__ == "__main__":
    unittest.main()
