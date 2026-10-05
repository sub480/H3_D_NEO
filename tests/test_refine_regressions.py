"""CPU-only: python tests/test_refine_regressions.py (optional COMFYUI_PATH)."""

import ast
from contextlib import ExitStack
import importlib
import math
import os
from pathlib import Path
import sys
import types
import unittest
import uuid
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = '_refine_regression_tests'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
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
refine = importlib.import_module(f'{PACKAGE}.director.refine_sampling')
pack = importlib.import_module(f'{PACKAGE}.director.refine_pack')
context = importlib.import_module(f'{PACKAGE}.director.h3_context_patches')
motion = importlib.import_module(f'{PACKAGE}.director.h3_motion_context')
CTX = context.CTX_FRAME_KEY


def conditioning_set_values(conditioning, values, append=False):
    return [[tensor, dict(meta, **values)] for tensor, meta in conditioning]


helpers = types.ModuleType('node_helpers')
helpers.conditioning_set_values = conditioning_set_values


def keyframe(index, value=0, marked=False):
    out = dict(resolved_frame_index=index, latent=torch.full((1, 24, 1, 2, 2), value))
    if marked:
        out[CTX] = index
        out['resolved_frame_index'] = 0
    return out


class RefineRegressions(unittest.TestCase):
    def test_marked_tail_replaced_in_place_without_prefix_or_duplicates(self):
        head, interior, tail = keyframe(0, 1, True), keyframe(1, 2, True), keyframe(8, 3, True)
        rebuilt = {0: torch.ones_like(head['latent']) * 9, 8: torch.ones_like(tail['latent']) * 8}
        existing = [head, interior, tail, keyframe(8)]
        out = refine._overwrite_endpoint_keyframes(existing, rebuilt)
        self.assertEqual(len(out), 3)
        self.assertIs(out[0], head)
        self.assertIs(out[1], interior)
        self.assertIs(out[2]['latent'], rebuilt[8])
        self.assertEqual(out[2][CTX], 8)
        self.assertEqual(out[2]['resolved_frame_index'], 0)
        self.assertIsNot(tail['latent'], rebuilt[8])
        again = refine._overwrite_endpoint_keyframes(out, rebuilt)
        self.assertEqual(len(again), 3)
        for existing in ([keyframe(0), head, interior, tail],
                         [head, keyframe(0), interior, tail]):
            out = refine._overwrite_endpoint_keyframes(existing, rebuilt)
            self.assertEqual(len(out), 3)
            self.assertIs(out[0], head)
            self.assertIs(out[1], interior)
        out = refine._overwrite_endpoint_keyframes([keyframe(8), head, interior, tail], rebuilt)
        self.assertEqual(len(out), 3)
        self.assertIs(out[0], head)
        self.assertIs(out[1], interior)
        self.assertEqual(out[2]['resolved_frame_index'], tail['resolved_frame_index'])
        self.assertIs(out[2]['latent'], rebuilt[8])

    def test_new_tail_is_marked_and_interior_resolved_zero_is_not_head(self):
        interior = keyframe(1, 2, True)
        rebuilt = {0: torch.zeros(1), 8: torch.ones(1)}
        out = refine._overwrite_endpoint_keyframes([interior], rebuilt)
        self.assertIs(out[0], interior)
        self.assertEqual([kf[CTX] for kf in out], [1, 0, 8])
        self.assertIs(out[2]['latent'], rebuilt[8])

    def test_stock_endpoints_keep_stock_indices_and_metadata(self):
        first, last = keyframe(0), keyframe(8)
        last['custom'] = 'preserved'
        rebuilt = {0: torch.zeros(1), 8: torch.ones(1)}
        for existing in ([], [first, last], [first, last, keyframe(8)]):
            out = refine._overwrite_endpoint_keyframes(existing, rebuilt)
            self.assertEqual([kf['resolved_frame_index'] for kf in out], [0, 8])
            self.assertTrue(all(CTX not in kf for kf in out))
            self.assertIs(out[0]['latent'], rebuilt[0])
            self.assertIs(out[1]['latent'], rebuilt[8])
        self.assertEqual(out[1]['custom'], 'preserved')

    def test_rebuild_encodes_target_canvas_and_keeps_continuity_head(self):
        calls = []
        def encode(images):
            calls.append(tuple(images.shape))
            return torch.full((1, 24, 1, 4, 6), len(calls))
        head, tail = keyframe(0, 1, True), keyframe(8, 3, True)
        positive = [[torch.zeros(1, 3, 2), dict(minimax_keyframes=[head, tail])]]
        with patch.dict(sys.modules, {'node_helpers': helpers}):
            out, changed = refine._rebuild_second_pass_keyframes(
                positive, vae=types.SimpleNamespace(encode=encode),
                video_latent={'samples': torch.zeros(1, 24, 3, 4, 6)},
                first_frame=torch.zeros(1, 32, 48, 3), last_frame=torch.zeros(1, 48, 32, 3),
            )
        self.assertTrue(changed)
        self.assertEqual(calls, [(1, 64, 96, 3), (1, 64, 96, 3)])
        kfs = out[0][1]['minimax_keyframes']
        self.assertEqual(len(kfs), 2)
        self.assertIs(kfs[0]['latent'], head['latent'])
        self.assertEqual(kfs[1][CTX], 8)
        self.assertEqual(tuple(kfs[1]['latent'].shape[-2:]), (4, 6))
        self.assertIs(positive[0][1]['minimax_keyframes'][1], tail)

    def test_offset_seed_wraps_and_normal_inherit_are_unchanged(self):
        for seed in [0, 42, 2**64 - 3, 2**64 - 1]:
            for index in [0, 1, 3, 9998]:
                self.assertEqual(pack.refine_seed_for({'seed_mode': 'offset'}, seed, index),
                                 (seed + 1 + index) % (2**64))
                self.assertEqual(pack.refine_seed_for({'seed_mode': 'inherit'}, seed, index), seed)
        self.assertEqual(pack.refine_seed_for({'seed_mode': 'offset'}, 42, -1), 43)

    def test_multi_pass_sampler_receives_wrapped_seeds(self):
        seeds = []
        def sample(**kwargs):
            seeds.append(kwargs['seed'])
            return kwargs['latent']
        plan = types.SimpleNamespace(refine=dict(enabled=True, mode='refine', passes=3,
                                                seed_mode='offset', tile_enabled=False))
        seg = types.SimpleNamespace(task_key='t2v', index=0)
        with patch.object(refine, 'sample_single_stage', side_effect=sample), \
                patch.object(refine, 'resolve_refine_sigmas', return_value=(0.5, 0.0)):
            _, note = refine.apply_segment_refine(
                plan, seg, samples={'samples': torch.zeros(1)}, model=object(), vae=None,
                positive=[], negative=[], seed=2**64 - 1, cfg=1, first_steps=1,
                sampler_name='euler', scheduler='simple', shift_video=12, shift_audio=3,
            )
        self.assertNotIn('FAILED', note)
        self.assertEqual(seeds, [0, 1, 2])

    def test_upscale_repin_restores_fl2v_tail_at_sampler_entry(self):
        latent_upscale = importlib.import_module(f'{PACKAGE}.director.h3_latent_upscale')
        for method in ('h3_latent', 'lanczos'):
            for pin_source, pin_frames, frame_count in (
                (source, pin, count) for source in ('previous_latent', 'current_prefix', 'pixel_fallback')
                for pin, count in ((5, 22), (22, 39))
            ):
                with self.subTest(method=method, pin_source=pin_source, pin_frames=pin_frames), ExitStack() as stack:
                    latent_t = motion.steps_for_frames(frame_count)
                    prefix_t = motion.steps_for_frames(pin_frames)
                    encoded_images = []
                    def encode(images):
                        encoded_images.append(tuple(images.shape))
                        steps = motion.steps_for_frames(int(images.shape[0]))
                        self.assertIsNotNone(steps)
                        return torch.full((1, 24, steps, images.shape[1] // 16,
                                           images.shape[2] // 16), float(images.mean()))
                    vae = types.SimpleNamespace(encode=encode)
                    video = torch.full((1, 24, latent_t, 2, 3), 0.25)
                    audio = torch.zeros(1, 2, 32, 5)
                    previous = {'samples': (torch.full((1, 24, prefix_t, 4, 6), 0.6), audio)}
                    frames = torch.full((frame_count, 32, 48, 3), 0.25)
                    last = torch.full((1, 48, 32, 3), 0.8)
                    refs = [dict(kind='audio', ref_audio_t=4, audio_latent=audio,
                                 **{context.CTX_AUDIO_END_KEY: 5})]
                    positive = [[torch.zeros(1, 3, 2), dict(
                        minimax_keyframes=[keyframe(i, marked=True) for i in
                                           motion.step_offsets(prefix_t) + [frame_count - 1]],
                        minimax_refs=refs, minimax_frame_count=frame_count)]]
                    plan = types.SimpleNamespace(width=48, height=32, refine=dict(
                        enabled=True, mode='upscale', upscale_method=method, passes=2,
                        tile_enabled=False, aspect_ratio=pack.FOLLOW_DIRECTOR_ASPECT,
                        target_width=96, target_height=64))
                    seg = types.SimpleNamespace(task_key='fl2v', index=0,
                                                output_width=0, output_height=0)
                    captured = []
                    def sample(**kwargs):
                        captured.append(kwargs)
                        return kwargs['latent']
                    def upscale(latent, **kwargs):
                        return {'samples': latent['samples'].repeat_interleave(2, -2).repeat_interleave(2, -1)}
                    stack.enter_context(patch.dict(sys.modules, {'node_helpers': helpers}))
                    stack.enter_context(patch.object(motion, 'ensure_layout_patch', return_value=True))
                    stack.enter_context(patch.object(motion, 'ensure_payload_patch', return_value=True))
                    actual_pin = stack.enter_context(patch.object(motion, 'apply_motion_context',
                                                                 wraps=motion.apply_motion_context))
                    stack.enter_context(patch.object(refine, '_unload_sampling_model'))
                    stack.enter_context(patch.object(refine, '_split_av', side_effect=lambda work:
                        ({'samples': work['samples'][0]}, {'samples': work['samples'][1]})))
                    stack.enter_context(patch.object(refine, '_join_av', side_effect=lambda v, a, work:
                        dict(work, samples=(v['samples'], a['samples']))))
                    stack.enter_context(patch.object(refine, '_encode_video',
                                                     side_effect=lambda vae, images: {'samples': vae.encode(images)}))
                    stack.enter_context(patch.object(latent_upscale, 'upscale_h3_video_latent', side_effect=upscale))
                    stack.enter_context(patch.object(refine, 'sample_single_stage', side_effect=sample))
                    stack.enter_context(patch.object(refine, 'resolve_refine_sigmas', return_value=(0.5, 0.0)))
                    if pin_source == 'pixel_fallback':
                        stack.enter_context(patch.object(motion, 'slice_av_prefix', side_effect=ValueError('fallback')))
                    result, note = refine.apply_segment_refine(
                        plan, seg, samples={'samples': (video, audio)}, model=object(), vae=vae,
                        positive=positive, negative=[], seed=42, cfg=1, first_steps=1,
                        sampler_name='euler', scheduler='simple', shift_video=12, shift_audio=3,
                        first_pass_images=frames, trim_frames=pin_frames,
                        first_frame=torch.full((1, 32, 48, 3), 0.9), last_frame=last,
                        prev_refine_av=previous if pin_source == 'previous_latent' else None,
                    )
                    self.assertNotIn('FAILED', note)
                    self.assertIn(f're-pin {pin_frames}f', note)
                    self.assertEqual(actual_pin.call_count, 1)
                    self.assertEqual(len(captured), 2)
                    for call in captured:
                        meta = call['positive'][0][1]
                        kfs = meta['minimax_keyframes']
                        self.assertEqual([kf[CTX] for kf in kfs],
                                         motion.step_offsets(prefix_t) + [frame_count - 1])
                        self.assertEqual(meta['minimax_frame_count'], frame_count)
                        self.assertIs(meta['minimax_refs'], refs)
                        for kf in kfs:
                            self.assertEqual(tuple(kf['latent'].shape), (1, 24, 1, 4, 6))
                        expected = 0.6 if pin_source == 'previous_latent' else 0.25
                        for kf in kfs[:-1]:
                            self.assertTrue(torch.allclose(kf['latent'], torch.full_like(kf['latent'], expected)))
                        self.assertTrue(torch.allclose(kfs[-1]['latent'], torch.full_like(kfs[-1]['latent'], 0.8)))
                        # Real time rewriting: four audio-reference steps shift the target origin.
                        layout = types.SimpleNamespace(
                            segments=[(i, i + 1, 'cond') for i in range(len(kfs))]
                                     + [(len(kfs), len(kfs) + 1, 'video')],
                            position_ids=torch.zeros(len(kfs) + 1, 3, dtype=torch.float64))
                        layout.position_ids[-1, 0] = 7
                        mm = types.SimpleNamespace(FRAME_RESCALE=motion.FRAME_RESCALE,
                            _video_t_spans=lambda n: [motion.FRAME_RESCALE * motion.FRAME_PER_TOKEN[k % 5]
                                                     for k in range(n)])
                        with patch.object(context, '_mm', return_value=mm):
                            context._rewrite_keyframe_times(layout, 3, latent_t, frame_count, kfs, refs)
                        self.assertAlmostEqual(float(layout.position_ids[0, 0]), 7)
                        self.assertAlmostEqual(float(layout.position_ids[1, 0]), 7 + motion.FRAME_RESCALE, places=5)
                        self.assertAlmostEqual(float(layout.position_ids[-2, 0]),
                                               7 + motion.FRAME_RESCALE * (frame_count - 1))
                    self.assertEqual(tuple(result['samples'][0].shape), (1, 24, latent_t, 4, 6))
                    self.assertEqual(encoded_images[-1], (1, 64, 96, 3))
                    self.assertEqual([kf[CTX] for kf in positive[0][1]['minimax_keyframes']],
                                     motion.step_offsets(prefix_t) + [frame_count - 1])


def load_ast(path, names, scope):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if getattr(node, 'name', None) in names]
    if len(nodes) != len(names):
        raise AssertionError(f'Missing CPU contract definitions in {path}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)


COMFY_ROOT = Path(os.environ.get('COMFYUI_PATH', r'D:\ComfyUI\ComfyUI'))


@unittest.skipUnless((COMFY_ROOT / 'comfy/model_base.py').is_file(), 'ComfyUI CPU contracts not installed')
class InstalledComfyContracts(unittest.TestCase):
    def test_fresh_payload_layout_replaces_stale_model_conds_and_preserves_refs(self):
        # Execute the installed CPU definitions, not a GPU model or generation.
        scope = dict(torch=torch, math=math, FRAME_PER_TOKEN=motion.FRAME_PER_TOKEN,
                     FRAME_RESCALE=motion.FRAME_RESCALE)
        load_ast(COMFY_ROOT / 'comfy/ldm/minimax/model.py',
                 {'_axis_from_sqrt_area', '_frame_grid', '_video_t_spans', '_video_t_grid',
                  '_ref_t_span', '_audio_grid', '_video_grid', 'PackedLayout'}, scope)
        mm = types.SimpleNamespace(**scope)
        original = mm.PackedLayout.__init__
        class Base:
            def extra_conds(self, **kwargs):
                return {}
        tree = ast.parse((COMFY_ROOT / 'comfy/model_base.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MiniMaxH3')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'extra_conds')
        cls.bases = [ast.Name(id='Base', ctx=ast.Load())]
        cls.body = [method]
        fake_comfy = types.SimpleNamespace(
            conds=types.SimpleNamespace(CONDConstant=lambda value: types.SimpleNamespace(cond=value),
                                        CONDRegular=lambda value: types.SimpleNamespace(cond=value)),
            ldm=types.SimpleNamespace(minimax=types.SimpleNamespace(model=mm)),
        )
        scope.update(Base=Base, comfy=fake_comfy, uuid=uuid)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                     'model_base.py', 'exec'), scope)
        model = scope['MiniMaxH3']()
        model.diffusion_model = types.SimpleNamespace(preprocess_text_embeds=lambda value: value)
        model.get_dtype_inference = lambda: torch.float32
        model.audio_scale = lambda: 1.0
        load_ast(COMFY_ROOT / 'comfy/samplers.py', {'encode_model_conds'}, scope)
        load_ast(COMFY_ROOT / 'comfy/sampler_helpers.py', {'convert_cond'}, scope)
        ref_video = torch.zeros(1, 24, 1, 2, 2)
        ref_audio = torch.zeros(1, 2, 32, 3)
        refs = [dict(kind='audio', ref_audio_t=3, audio_latent=ref_audio),
                dict(kind='image', latent_h=2, latent_w=2, latent=ref_video)]
        refs[0][context.CTX_AUDIO_END_KEY] = 2
        for existing in ([keyframe(0, marked=True), keyframe(1, marked=True)],
                         [keyframe(0, marked=True), keyframe(1, marked=True), keyframe(8, marked=True)]):
            kfs = refine._overwrite_endpoint_keyframes(existing, {8: torch.ones(1, 24, 1, 4, 6)})
            stale = types.SimpleNamespace(cond={'layout': object(), 'keyframes': existing})
            positive = [[torch.zeros(1, 3, 2), dict(minimax_keyframes=kfs, minimax_refs=refs,
                         model_conds={'minimax_payload': stale})]]
            with patch.object(context, '_mm', return_value=mm), \
                    patch.object(context, '_layout_orig', original), \
                    patch.object(context, '_payload_orig', lambda self, **kw: model.extra_conds(**kw)):
                mm.PackedLayout.__init__ = context._director_layout_init
                try:
                    out = scope['encode_model_conds'](
                        lambda **kw: context._director_extra_conds(model, **kw),
                        scope['convert_cond'](positive), torch.zeros(1, 1, 1, 1), 'cpu', 'positive',
                        latent_shapes=[(1, 24, 3, 4, 6), (1, 2, 32, 5)],
                    )
                finally:
                    mm.PackedLayout.__init__ = original
            payload = out[0]['model_conds']['minimax_payload'].cond
            self.assertIs(payload['keyframes'], kfs)
            self.assertEqual(payload['layout'].signature, (3, 3, 4, 6, 5))
            self.assertIs(payload['refs'], refs)
            self.assertIs(payload['cond_video_latents'][-1], ref_video)
            self.assertIs(payload['cond_audio_latents'][0], ref_audio)
            self.assertIs(payload['cond_video_latents'][-2], kfs[-1]['latent'])
            layout = payload['layout']
            origin = context._target_origin(layout)
            times = [float(layout.position_ids[a, 0]) for a, _, kind in layout.segments if kind == 'cond']
            self.assertAlmostEqual(origin, 7.0)
            self.assertAlmostEqual(times[-1], origin + motion.FRAME_RESCALE * 8)
            self.assertEqual(len(times), len(kfs))
            audio_start = next(a for a, _, kind in layout.segments if kind == 'ref_audio')
            self.assertAlmostEqual(float(layout.position_ids[audio_start, 0]),
                                   origin + motion.FRAME_RESCALE * 2 - 3)
            self.assertIs(positive[0][1]['model_conds']['minimax_payload'], stale)


if __name__ == '__main__':
    unittest.main()
