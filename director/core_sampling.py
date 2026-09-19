"""Single-stage sampling via official MiniMax H3 custom-sampler nodes.

Matches ``video_minimax_h3_r2v.json``:
MiniMaxH3SigmaShift → BasicScheduler → BasicGuider (or CFGGuider) →
KSamplerSelect → RandomNoise → SamplerCustomAdvanced.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.core_sampling")

PhaseCallback = Callable[[str, float], None]
StepPreviewCallback = Callable[[int, int, Any], None]
StepStateCallback = Callable[[int, int, Any, Any], None]


def _unpack_node_output(out):
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"Unexpected node output type: {type(out)!r}")


def _use_basic_guider(cfg: float, negative) -> bool:
    """Official r2v template uses BasicGuider (no CFG)."""
    if negative:
        return False
    return abs(float(cfg) - 1.0) < 1e-6


class ShiftedModelCache:
    """Reuse one SigmaShift clone per parent model for one Director execution."""

    def __init__(self) -> None:
        self._items: dict[tuple[int, float, float], Any] = {}

    def get(self, model, shift_video: float, shift_audio: float):
        key = (id(model), float(shift_video), float(shift_audio))
        cached = self._items.get(key)
        if cached is not None:
            return cached
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

        shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
        model_use = _unpack_node_output(shifted)[0]
        self._items[key] = model_use
        return model_use

    def holds(self, model) -> bool:
        return any(item is model for item in self._items.values())

    def clear(self) -> None:
        self._items.clear()


def sample_single_stage(
    *,
    model,
    positive,
    negative,
    latent,
    seed: int,
    cfg: float,
    steps: int,
    sampler_name: str,
    scheduler: str,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    on_phase: PhaseCallback | None = None,
    on_step_preview: StepPreviewCallback | None = None,
    preview_every: int = 1,
    denoise: float = 1.0,
    phase_name: str = "sample",
    sigmas=None,
    apply_shift: bool = True,
    tile=None,
    after_shift=None,
    shift_cache: ShiftedModelCache | None = None,
    on_step_state: StepStateCallback | None = None,
    zero_noise: bool = False,
    noise_override=None,
):
    import torch
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        BasicScheduler,
        CFGGuider,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    def notify(phase: str, value: float) -> None:
        if on_phase:
            on_phase(phase, value)

    notify(phase_name, 0)
    model_use = model
    if apply_shift:
        if shift_cache is not None:
            model_use = shift_cache.get(model, shift_video, shift_audio)
        else:
            shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
            model_use = _unpack_node_output(shifted)[0]

    if sigmas is not None:
        if torch.is_tensor(sigmas):
            sigma_t = sigmas.detach().float().cpu().reshape(-1)
        else:
            sigma_t = torch.tensor([float(x) for x in sigmas], dtype=torch.float32)
    else:
        denoise_use = float(max(0.0, min(1.0, denoise)))
        sigma_out = BasicScheduler.execute(
            model_use, str(scheduler), int(steps), denoise_use
        )
        sigma_t = _unpack_node_output(sigma_out)[0]

    if callable(after_shift):
        remasked = after_shift(model_use, latent, sigma_t)
        if remasked is not None:
            model_use = remasked

    sampler_obj = _unpack_node_output(KSamplerSelect.execute(str(sampler_name)))[0]
    if noise_override is not None:
        class _FixedNoise:
            def __init__(self, seed_value):
                self.seed = int(seed_value)

            def generate_noise(self, input_latent): return noise_override
        noise_obj = _FixedNoise(seed)
    elif zero_noise:
        class _ZeroNoise:
            def __init__(self, seed_value):
                self.seed = int(seed_value)

            def generate_noise(self, input_latent):
                import torch
                samples = input_latent["samples"] if isinstance(input_latent, dict) else input_latent
                if torch.is_tensor(samples): return torch.zeros_like(samples)
                return type(samples)(tuple(torch.zeros_like(x) for x in samples.unbind()))
        noise_obj = _ZeroNoise(seed)
    else:
        noise_obj = _unpack_node_output(RandomNoise.execute(int(seed)))[0]

    neg = negative if negative else []
    if _use_basic_guider(cfg, neg):
        guider = _unpack_node_output(BasicGuider.execute(model_use, positive))[0]
    else:
        guider = _unpack_node_output(
            CFGGuider.execute(model_use, positive, neg, float(cfg))
        )[0]

    def _run_official() -> dict:
        sampled = SamplerCustomAdvanced.execute(
            noise_obj, guider, sampler_obj, sigma_t, latent
        )
        return _unpack_node_output(sampled)[0]

    tile_cfg = tile if isinstance(tile, dict) and int(tile.get("n_tiles") or 1) > 1 else None

    def _run_sample() -> dict:
        if not tile_cfg:
            return _run_official()
        from .h3_tiled_sampler import sample_h3_tiled

        return sample_h3_tiled(
            noise=noise_obj,
            guider=guider,
            sampler=sampler_obj,
            sigmas=sigma_t,
            latent=latent,
            n_tiles=int(tile_cfg.get("n_tiles") or 2),
            tile_axis=str(tile_cfg.get("tile_axis") or "auto"),
            tile_overlap=int(tile_cfg.get("tile_overlap") or 8),
            max_size_for_no_tile=int(tile_cfg.get("max_size_for_no_tile") or 64),
            refine_seams=bool(tile_cfg.get("refine_seams", True)),
            refine_steps=int(tile_cfg.get("refine_steps") or 8),
            on_step_preview=on_step_preview,
            preview_every=preview_every,
        )

    orig_sample = guider.sample if (on_step_preview is not None or on_step_state is not None) and not tile_cfg else None
    if orig_sample is not None:
        every = max(1, int(preview_every))

        def sample_wrapped(noise, latent_image, sampler, sigmas_in, **kwargs):
            inner_cb = kwargs.get("callback")

            def callback(step, x0, x, total_steps):
                if on_step_state is not None:
                    on_step_state(int(step), int(total_steps), x0, x)
                try:
                    last = max(0, int(total_steps) - 1)
                    if int(preview_every) < 0:
                        show = step >= last
                    else:
                        show = step % every == 0 or step >= last
                    if show:
                        on_step_preview(int(step), int(total_steps), x0)
                except Exception as exc:
                    log.debug("Step preview callback skipped: %s", exc)
                if inner_cb is not None:
                    try:
                        inner_cb(step, x0, x, total_steps)
                    except Exception as exc:
                        log.debug("Comfy preview callback skipped: %s", exc)

            kwargs["callback"] = callback
            return orig_sample(noise, latent_image, sampler, sigmas_in, **kwargs)

        guider.sample = sample_wrapped
    try:
        out = _run_sample()
    finally:
        if orig_sample is not None:
            guider.sample = orig_sample
        if callable(after_shift):
            try:
                from .h3_latent_continue import uninstall_continue_prefix_remask

                uninstall_continue_prefix_remask(model_use)
            except Exception as exc:
                log.debug("Prefix remask uninstall skipped: %s", exc)
        guider = None
        noise_obj = None
        sampler_obj = None
        if model_use is not model and (shift_cache is None or not shift_cache.holds(model_use)):
            model_use = None

    notify(phase_name, 1)
    return out
