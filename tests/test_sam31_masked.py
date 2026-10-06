"""CPU tensors and mocks only: no GPU, service, weights or model loading."""

import ast
import copy
import importlib
import io
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import test_sam31_composite as fixture

masked = importlib.import_module(f"{fixture.PACKAGE}.director.sam31_masked")
NS = types.SimpleNamespace


class Nested:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def unbind(self):
        return self.tensors


class MaskedTests(unittest.TestCase):
    def test_legacy_source_tags_rebound_without_touching_picture_audio_tags(self):
        for tag in ("<Video 1>", "<Video1>", "<video 1>"):
            text = f"Edit {tag}: change the shirt to white; use <Picture 1> and <Audio 1>."
            self.assertEqual(masked.source_prompt(text),
                             "Edit the source video: change the shirt to white; use <Picture 1> and <Audio 1>.")
        self.assertEqual(masked.source_prompt("Change the shirt to white."), "Change the shirt to white.")
        with self.assertRaisesRegex(ValueError, "additional Video reference"):
            masked.source_prompt("Copy <Video 2>.")

    def setUp(self):
        self.nested_module = types.ModuleType("comfy.nested_tensor")
        self.nested_module.NestedTensor = Nested
        self.video = torch.zeros(1, 24, 2, 4, 4)
        self.audio = torch.zeros(1, 32, 2, 8)
        self.latent = {"samples": Nested((self.video, self.audio)), "extra": "kept"}
        self.source = torch.rand(5, 64, 64, 3)
        self.mask = torch.zeros(5, 64, 64)
        self.mask[1, 0, 0] = 1
        self.encoded = torch.full_like(self.video, 0.7)
        self.vae = NS(encode=MagicMock(return_value=self.encoded))

    def initialize(self, **kwargs):
        with patch.dict(sys.modules, {"comfy.nested_tensor": self.nested_module}):
            return masked.initialize(self.latent, self.source, self.mask, self.vae, **kwargs)

    def test_causal_groups_cycle_and_all_frames(self):
        groups = masked.causal_groups(39, 12)
        self.assertEqual(groups[:7], [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 18), (18, 22)])
        self.assertEqual(groups[-1], (35, 39))
        self.assertEqual([n for a, b in groups for n in range(a, b)], list(range(39)))
        with self.assertRaises(ValueError):
            masked.causal_groups(22, 6)

    def test_one_pixel_survives_16x_and_2x2_max_reduction(self):
        mask = masked.latent_mask(self.mask, self.video.shape)
        self.assertEqual(tuple(mask.shape), (1, 1, 2, 4, 4))
        self.assertEqual(float(mask[:, :, 0].sum()), 0)
        self.assertEqual(float(mask[:, :, 1].sum()), 4)
        self.assertTrue(torch.all(mask[:, :, 1, :2, :2] == 1))
        self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})

    def test_grow_uses_hard_whole_cells(self):
        mask = masked.latent_mask(self.mask, self.video.shape, grow=1)
        self.assertEqual(float(mask[:, :, 1].sum()), 16)
        self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})

    def test_bad_time_canvas_batch_are_rejected(self):
        for pixels, shape in ((self.mask[:4], self.video.shape), (self.mask[:, :32], self.video.shape), (self.mask, (2, 24, 2, 4, 4))):
            with self.assertRaises(ValueError):
                masked.latent_mask(pixels, shape)

    def test_encoded_source_is_actual_target_and_nested_masks_match(self):
        out = self.initialize()
        self.vae.encode.assert_called_once()
        self.assertTrue(torch.equal(self.vae.encode.call_args.args[0], self.source))
        self.assertIs(out["samples"].unbind()[0], self.encoded)
        self.assertIs(out["samples"].unbind()[1], self.audio)
        video_mask, audio_mask = out["noise_mask"].unbind()
        self.assertEqual(video_mask.shape, self.video.shape)
        self.assertEqual(audio_mask.shape, self.audio.shape)
        self.assertTrue(torch.all(audio_mask == 1))
        self.assertEqual(out["extra"], "kept")
        self.assertIs(self.latent["samples"].unbind()[0], self.video)

    def test_source_and_mute_do_not_regenerate_audio(self):
        for mode in ("source", "mute"):
            out = self.initialize(audio_mode=mode)
            self.assertTrue(torch.all(out["noise_mask"].unbind()[1] == 0))

    def test_initialized_mask_survives_real_single_stage_sampling_entry(self):
        latent = self.initialize()
        sampling = importlib.import_module(f"{fixture.PACKAGE}.director.core_sampling")
        custom = types.ModuleType("comfy_extras.nodes_custom_sampler")
        h3 = types.ModuleType("comfy_extras.nodes_minimax_h3")
        model, shifted, guider = object(), object(), object()
        h3.MiniMaxH3SigmaShift = NS(execute=MagicMock(return_value=(shifted,)))
        custom.BasicScheduler = NS(execute=MagicMock(return_value=(torch.tensor([1.0, 0.0]),)))
        custom.BasicGuider = NS(execute=MagicMock(return_value=(guider,)))
        custom.CFGGuider = NS(execute=MagicMock(side_effect=AssertionError("unexpected CFG")))
        custom.KSamplerSelect = NS(execute=MagicMock(return_value=("sampler",)))
        custom.RandomNoise = NS(execute=MagicMock(return_value=("noise",)))
        custom.SamplerCustomAdvanced = NS(execute=MagicMock(return_value=(latent,)))
        with patch.dict(sys.modules, {custom.__name__: custom, h3.__name__: h3}):
            sampled = sampling.sample_single_stage(model=model, positive=["positive"], negative=[], latent=latent,
                                                  seed=1, cfg=1, steps=2, sampler_name="euler", scheduler="simple")
        self.assertIs(sampled, latent)
        self.assertIs(custom.BasicGuider.execute.call_args.args[0], shifted)
        self.assertIs(custom.SamplerCustomAdvanced.execute.call_args.args[1], guider)
        self.assertIs(custom.SamplerCustomAdvanced.execute.call_args.args[4], latent)
        self.assertIs(custom.SamplerCustomAdvanced.execute.call_args.args[4]["noise_mask"], latent["noise_mask"])

    def test_firstpass_combines_revision2_mask_fingerprint_before_cache_hit(self):
        _, seg, _ = fixture.CompositeTests().prepared()
        cache = importlib.import_module(f"{fixture.PACKAGE}.director.segment_cache")
        plan = NS(raw={}, selflift=None, semantic_bridge=None)
        with (patch.object(cache, "_segment_identity_fingerprint", return_value={}),
              patch.object(cache, "resolve_segment_seed", return_value=1)):
            first = copy.deepcopy(cache.first_pass_cache_fingerprint(seg, plan))
            self.assertEqual(first["sam31_source"]["revision"], 2)
            seg.sam31_composite["settings"]["grow"] = 32
            changed = cache.first_pass_cache_fingerprint(seg, plan)
            self.assertNotEqual(first, changed)
        executor = (fixture.ROOT / "director/executor_core.py").read_text(encoding="utf-8-sig")
        self.assertIn("if sam31_context is not None and not skip_first_sample:", executor)
        self.assertIn('samples = pre_cache["av_latent"]', executor)

    def test_source_pcm_encoded_only_for_conditioning_not_replaced(self):
        pcm = {"waveform": torch.ones(1, 2, 6400), "sample_rate": 32000}
        original = pcm["waveform"].clone()
        encoded = torch.full((1, 32, 2, 7), 0.3)
        audio_module = types.ModuleType("comfy_extras.nodes_audio")
        encode = MagicMock(return_value=({"samples": encoded},))
        audio_module.VAEEncodeAudio = NS(execute=encode)
        with patch.dict(sys.modules, {"comfy_extras.nodes_audio": audio_module}):
            out = self.initialize(audio_mode="source", audio_vae="audio_vae", source_audio=pcm)
        encode.assert_called_once_with("audio_vae", pcm)
        audio = out["samples"].unbind()[1]
        self.assertTrue(torch.equal(audio[..., :7], encoded))
        self.assertTrue(torch.all(audio[..., 7:] == 0))
        self.assertTrue(torch.equal(pcm["waveform"], original))

    def test_wrong_encoded_shape_and_flat_latent_fail(self):
        self.vae.encode.return_value = torch.zeros(1, 24, 3, 4, 4)
        with self.assertRaisesRegex(ValueError, "encoded source"):
            self.initialize()
        self.latent["samples"] = self.video
        with self.assertRaisesRegex(ValueError, "nested"):
            self.initialize()

    def test_pixel_masks_union_selected_objects_and_ignore_paste_controls(self):
        _, seg, meta = fixture.CompositeTests().prepared(height=64, width=64)
        seg.sam31_composite["selected"] = [0, 1]
        seg.sam31_composite["settings"].update(feather=64, blend=0.1)
        masks = np.zeros((5, 2, 64, 64), dtype=np.uint8)
        masks[1, 0, 0, 0] = 1
        masks[1, 1, 63, 63] = 1
        data = fixture.zip_masks(meta, masks)
        with (patch.object(fixture.sam, "cache_path", side_effect=lambda key: io.BytesIO(data)),
              patch.object(fixture.sam, "file_digest", return_value="artifact-digest"),
              patch.object(fixture.sam, "verify_assets")):
            pixels = masked.pixel_masks(seg, self.source)
        self.assertEqual(pixels.dtype, torch.bool)
        self.assertEqual(int(pixels.sum()), 2)
        self.assertEqual(float(masked.latent_mask(pixels, self.video.shape).sum()), 8)

    def test_generation_inputs_invalidate_fingerprint(self):
        _, seg, _ = fixture.CompositeTests().prepared()
        original = copy.deepcopy(fixture.composite.fingerprint(seg, first_pass=True))
        for key, value in (("key", "other"), ("artifact", "changed"), ("selected", [1]), ("audio_mode", "source")):
            current = copy.deepcopy(seg)
            current.sam31_composite[key] = value
            self.assertNotEqual(fixture.composite.fingerprint(current, first_pass=True), original)

    def test_preflight_native_path_and_incompatible_combinations(self):
        plan_module = types.ModuleType(f"{fixture.PACKAGE}.director.plan")
        plan_module.resolve_segment_pass_mode = lambda seg: seg.pass_mode
        lift_module = types.ModuleType(f"{fixture.PACKAGE}.director.selflift.pack")
        lift_module.selflift_enabled = lambda plan: bool(plan.selflift)
        def forward(x=None, denoise_mask=None, audio_denoise_mask=None):
            pass
        base = NS(diffusion_model=NS(_forward=forward), _denoise_mask_values=lambda: None, _token_grid_masks=lambda: None)
        model = NS(model=base)
        plan = NS(selflift=None, face_refine={"enabled": False})
        seg = NS(pass_mode="first")
        with patch.dict(sys.modules, {plan_module.__name__: plan_module, lift_module.__name__: lift_module}):
            masked.preflight(plan, seg, model)
            # A stale per-group label must not be confused with an active pass.
            for inactive in (None, {}, {"enabled": False}):
                p, s = copy.deepcopy(plan), copy.deepcopy(seg)
                p.refine = inactive
                s.pass_mode = "second"
                with self.subTest(inactive=inactive):
                    masked.preflight(p, s, model)
            p = copy.deepcopy(plan)
            p.refine = {"enabled": True}
            masked.preflight(p, seg, model)  # first-only group, global Refine on
            for mode in ("second", "selflift", "face", "native"):
                p, s, m = copy.deepcopy(plan), copy.deepcopy(seg), copy.deepcopy(model)
                if mode == "second":
                    s.pass_mode = "second"
                    p.refine = {"enabled": True}
                if mode == "selflift": p.selflift = {"enabled": True}
                if mode == "face": p.face_refine["enabled"] = True
                if mode == "native": m.model.diffusion_model._forward = lambda x: x
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    masked.preflight(p, s, m)

    def test_source_reference_removed_only_for_masked_edit(self):
        tree = ast.parse((fixture.ROOT / "director/executor_core.py").read_text(encoding="utf-8-sig"))
        build = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_minimax_inputs")
        scope = {"DirectorPlan": object, "torch": torch,
                 "ref_image_long_preset_px": lambda *args: 0, "resolve_ref_image_size": lambda *args: None,
                 "refs_to_kwargs_for_context": lambda *args: {}, "ref_audios_to_dict": lambda *args, **kw: None}
        exec(compile(ast.Module(body=[build], type_ignores=[]), "inputs", "exec"), scope)
        for task in ("v2v", "rv2v"):
            seg = NS(task_key=task, index=0, refs=[], ref_audios=[], sam31_composite=None)
            ordinary = scope["_build_minimax_inputs"](NS(), seg, clip_frames=self.source, ctx_w=64, ctx_h=64, prev_tail=None)
            self.assertIs(ordinary[3]["ref_video_0"], self.source)
            seg.sam31_composite = {"revision": 2}
            edited = scope["_build_minimax_inputs"](NS(), seg, clip_frames=self.source, ctx_w=64, ctx_h=64, prev_tail=None)
            self.assertIsNone(edited[3])


if __name__ == "__main__":
    unittest.main()
