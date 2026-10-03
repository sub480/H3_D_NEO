"""Run with python tests/test_source_aspect.py; no ComfyUI install or GPU required."""

import ast
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = '_source_aspect_tests'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package

# Only ComfyUI infrastructure is stubbed; planning and tensor fits are real.
folder_paths = types.ModuleType('folder_paths')
folder_paths.get_input_directory = lambda: str(ROOT)
folder_paths.get_output_directory = lambda: str(ROOT)
sys.modules.setdefault('folder_paths', folder_paths)
comfy = types.ModuleType('comfy')
comfy.__path__ = []
utils = types.ModuleType('comfy.utils')
utils.common_upscale = lambda tensor, width, height, *_: torch.nn.functional.interpolate(
    tensor, size=(height, width), mode='nearest'
)
comfy.utils = utils
sys.modules.setdefault('comfy', comfy)
sys.modules.setdefault('comfy.utils', utils)

gen = importlib.import_module(f'{PACKAGE}.director.gen_timeline')
prep = importlib.import_module(f'{PACKAGE}.lib.image_prep')
runtime = importlib.import_module(f'{PACKAGE}.director.segment_runtime')
refine = importlib.import_module(f'{PACKAGE}.director.refine_sampling')
cache = importlib.import_module(f'{PACKAGE}.director.segment_cache')
video_io = importlib.import_module(f'{PACKAGE}.lib.video_io')
contract = importlib.import_module(f'{PACKAGE}.director.prompt_contract')


def segment(width=1920, height=1080, task='v2v', resolution='target'):
    clip = dict(videoFile='source.mp4', width=width, height=height, sourceFrameCount=5)
    return dict(taskType=task, frameCount=5, prompt='test', videoResolution=resolution,
                sourceVideo=dict(mediaKind='video', totalFrames=5, video=copy.deepcopy(clip),
                                 videoClips=[clip]))


def timeline(segments, mp=0.4):
    return dict(timelineMode='prompt_batch', global_={}, frameRate=24,
                output=dict(mode='fixed', width=864, height=480,
                            aspectRatio='与原视频一致', megapixels=mp, continuityEnabled=False),
                segments=segments)


def build(data, load_media=True):
    return gen.build_gen_director_plan(data, global_task_type='mixed', global_prompt='',
                                       total_frames=5, frame_rate=24, width=864, height=480,
                                       ref_max_size=864, load_media=load_media)


def fake_decode(data, start, end):
    clip = data['videoClips'][0]
    return torch.zeros((end - start, clip['storageHeight'], clip['storageWidth'], 3))


class SourceAspectTests(unittest.TestCase):
    def test_image_modes_and_fl2v_priority(self):
        first = dict(imageFile='first.png', width=160, height=90)
        last = dict(imageFile='last.png', width=90, height=160)
        cases = [
            (dict(taskType='i2v', genImage=first, videoResolution='source'), (864, 480)),
            (dict(taskType='i2v', genImage=last), (480, 864)),
            (dict(taskType='fl2v', startImage=first), (864, 480)),
            (dict(taskType='fl2v', endImage=last), (480, 864)),
            (dict(taskType='fl2v', startImage=first, endImage=last), (864, 480)),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                raw.update(frameCount=5, prompt='test')
                data = timeline([raw])
                def decode(ref):
                    return torch.zeros((1, ref['height'], ref['width'], 3))
                with patch.object(gen, '_load_gen_image_tensor', side_effect=decode):
                    plan = build(data)
                seg = plan.segments[0]
                self.assertEqual((seg.output_width, seg.output_height), expected)
                if seg.source_clip is not None:
                    self.assertEqual(tuple(seg.source_clip.shape[1:3]), expected[::-1])
                for ref in seg.refs:
                    self.assertEqual(tuple(ref.tensor.shape[1:3]), expected[::-1])
                lazy = build(data, False)
                self.assertEqual(cache._segment_identity_fingerprint(seg, plan),
                                 cache._segment_identity_fingerprint(lazy.segments[0], lazy))
                follow = dict(aspect_ratio='跟随导演台', megapixels=0.8)
                w, h = refine._resolve_refine_canvas(plan, follow, seg)
                self.assertEqual(w < h, expected[0] < expected[1])
                normal = copy.deepcopy(data)
                normal['output']['aspectRatio'] = '16:9 (宽屏)'
                self.assertEqual(build(normal, False).segments[0].output_width, 0)

    def test_image_headers_without_pixel_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            Image.new('RGB', (90, 160)).save(Path(directory) / 'portrait.png')
            cases = [dict(taskType='i2v', imageFile='portrait.png'),
                     dict(taskType='fl2v', startImage='portrait.png'),
                     dict(taskType='fl2v', endImage=dict(imageFile='portrait.png'))]
            with patch.object(folder_paths, 'get_input_directory', return_value=directory), \
                    patch.object(Image.Image, 'load', side_effect=AssertionError('pixel decode')):
                for raw in cases:
                    raw.update(frameCount=5, prompt='test')
                    seg = build(timeline([raw]), False).segments[0]
                    self.assertEqual((seg.output_width, seg.output_height), (480, 864))
                raw = dict(taskType='fl2v', startImage=dict(imageFile='missing.png'),
                           endImage=dict(imageFile='portrait.png', width=90, height=160))
                self.assertIsNone(gen._source_aspect_canvas(raw, timeline([])['output'], load_media=False))

    def test_mp_grid_and_nonpreset_ratio(self):
        for width, height in [(1920, 1080), (1080, 1920), (1234, 567), (1, 10000)]:
            for mp in [0.1, 0.4, 1, 16]:
                w, h = gen._source_aspect_canvas(segment(width, height), timeline([], mp)['output'],
                                                 load_media=False)
                self.assertEqual(w % 32, 0)
                self.assertEqual(h % 32, 0)
                self.assertGreaterEqual(min(w, h), 32)
                if width / height > 0.1:
                    self.assertLess(abs(w * h / (mp * 1024 * 1024) - 1), 0.13)
                    self.assertLess(abs(w / h / (width / height) - 1), 0.1)

    def test_groups_keep_distinct_canvas_and_nonvideo_fallback(self):
        data = timeline([segment(), segment(1080, 1920, 'rv2v'),
                         dict(taskType='t2v', frameCount=5, prompt='blank')])
        with patch.object(video_io, 'load_timeline_segment', side_effect=fake_decode) as decode:
            plan = build(data)
        self.assertEqual((plan.width, plan.height), (864, 480))
        self.assertEqual([(s.output_width, s.output_height) for s in plan.segments],
                         [(864, 480), (480, 864), (0, 0)])
        self.assertEqual(tuple(plan.segments[1].source_clip.shape), (5, 864, 480, 3))
        self.assertEqual(decode.call_count, 2)
        # Planning without media (cache/selection paths) resolves the same targets.
        lazy = build(data, False)
        self.assertEqual([(s.output_width, s.output_height) for s in lazy.segments],
                         [(s.output_width, s.output_height) for s in plan.segments])

    def test_old_source_resolution_wins(self):
        data = timeline([segment(320, 640, resolution='source')], mp=16)
        with patch.object(video_io, 'load_timeline_segment', side_effect=fake_decode):
            plan = build(data)
        seg = plan.segments[0]
        self.assertTrue(seg.use_source_resolution)
        self.assertEqual((seg.output_width, seg.output_height), (0, 0))
        self.assertEqual(tuple(seg.source_clip.shape[1:3]), (640, 320))
        self.assertEqual(tuple(runtime.source_passthrough_chunk(plan, seg).shape[1:3]), (640, 320))

    def test_selected_clip_range_and_missing_metadata(self):
        seg = segment()
        portrait = segment(1080, 1920)['sourceVideo']['videoClips'][0]
        seg['sourceVideo']['videoClips'].append(portrait)
        seg['sourceVideo']['rangeStart'] = 5
        self.assertEqual(gen._source_aspect_canvas(seg, timeline([])['output'], load_media=False),
                         (480, 864))
        seg['sourceVideo']['video']['frameMap'] = [dict(clip=1, frame=0)]
        seg['sourceVideo']['rangeStart'] = 0
        self.assertEqual(gen._source_aspect_canvas(seg, timeline([])['output'], load_media=False),
                         (480, 864))
        portrait.pop('width')
        portrait.pop('height')
        with patch.object(video_io, 'probe_video_clip', return_value=dict(width=1080, height=1920)) as probe:
            self.assertEqual(gen._source_aspect_canvas(seg, timeline([])['output'], load_media=True),
                             (480, 864))
            probe.assert_called_once_with(portrait)

    def test_source_images_and_normal_presets_unchanged(self):
        seg = segment()
        seg['sourceVideo']['mediaKind'] = 'image'
        self.assertIsNone(gen._source_aspect_canvas(seg, timeline([])['output'], load_media=False))
        output = dict(timeline([])['output'], aspectRatio='16:9 (宽屏)')
        self.assertIsNone(gen._source_aspect_canvas(segment(), output, load_media=False))

    def test_execution_does_not_reset_to_global_canvas(self):
        # Isolate the executor preparation block from sampler/GPU infrastructure.
        source = (ROOT / 'director/executor_core.py').read_text(encoding='utf-8')
        start = source.index('        target_len = max(1, int(seg.frame_count')
        end = source.index('        num_frames = minimax_align_frame_count(target_len)', start)
        with patch.object(video_io, 'load_timeline_segment', side_effect=fake_decode):
            plan = build(timeline([segment(1080, 1920)]))
        seg = plan.segments[0]
        scope = dict(seg=seg, plan=plan, fit_frames_to_canvas=prep.fit_frames_to_canvas,
                     fit_video_long_edge=prep.fit_video_long_edge,
                     resolve_segment_raw_clip=runtime.resolve_segment_raw_clip)
        exec(textwrap.dedent(source[start:end]), scope)
        self.assertEqual(tuple(scope['clip_frames'].shape[1:3]), (864, 480))
        self.assertEqual(tuple(runtime.source_passthrough_chunk(plan, seg).shape[1:3]), (864, 480))
        # Source-image export uses its own helper, not the sampler path.
        tree = ast.parse((ROOT / 'nodes/director_common.py').read_text(encoding='utf-8'))
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_fit_source_clip_to_plan')
        scope.update(torch=torch)
        exec(compile(ast.Module(body=[func], type_ignores=[]), 'director_common.py', 'exec'), scope)
        fitted = scope['_fit_source_clip_to_plan'](plan, seg.source_clip, seg)
        self.assertEqual(tuple(fitted.shape[1:3]), (864, 480))

    def test_follow_director_refine_and_cache_invalidation(self):
        plan = build(timeline([segment(1080, 1920)]), False)
        seg = plan.segments[0]
        follow = dict(aspect_ratio='跟随导演台', megapixels=0.8, target_width=1216, target_height=672)
        w, h = refine._resolve_refine_canvas(plan, follow, seg)
        self.assertLess(w, h)
        explicit = dict(follow, aspect_ratio='16:9 (宽屏)')
        self.assertEqual(refine._resolve_refine_canvas(plan, explicit, seg), (1216, 672))
        first = cache._segment_identity_fingerprint(seg, plan)
        bigger = build(timeline([segment(1080, 1920)], 1), False)
        second = cache._segment_identity_fingerprint(bigger.segments[0], bigger)
        self.assertNotEqual(first['source_aspect_canvas'], second['source_aspect_canvas'])

    def test_director_prompt_roundtrip_keeps_aspect_and_media(self):
        data = timeline([segment(1080, 1920)], mp=1)
        payload = dict(schema='minimax-h3-director-prompt/v1',
                       groups=[dict(prompt='edit', taskType='v2v', durationSec=5)],
                       settings=dict(aspectRatio='与原视频一致'))
        restored = json.loads(contract.director_prompt_to_timeline(
            json.dumps(payload), base_timeline_data=json.dumps(data)))
        self.assertEqual(restored['output']['aspectRatio'], '与原视频一致')
        self.assertEqual(restored['output']['width'], 864)
        self.assertEqual(restored['output']['height'], 480)
        self.assertEqual(restored['output']['megapixels'], 1)
        self.assertEqual(restored['segments'][0]['sourceVideo'], data['segments'][0]['sourceVideo'])


if __name__ == '__main__':
    unittest.main()
