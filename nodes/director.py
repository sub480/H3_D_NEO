"""MiniMax H3 Director — timeline UI + official MiniMax H3 AV execution."""

from __future__ import annotations

import hashlib
import json

import comfy.samplers

from ..director.executor_core import execute_director_plan_core
from ..director.refine_pack import pack_director_builtin_refine
from .director_common import (
    finalize_director_outputs,
    prepare_director_plan,
    timeline_required_inputs,
    director_perf_inputs,
)
from .director_refine import (
    director_face_refine_widget_inputs,
    director_refine_widget_inputs,
    director_selflift_widget_inputs,
)

_CATEGORY = "H3_D_NEO"

_DEFAULT_GLOBAL_PROMPT = "A cinematic scene with natural motion and synchronized ambience"


def _widget_bool(value, default: bool = False) -> bool:
    if value is True or value == 1:
        return True
    if value is False or value == 0:
        return False
    if value is None:
        return default
    text = str(value).strip().lower()
    return text in {"1", "true", "on", "yes"}


_DIRECTOR_LINKED_INPUTS = frozenset({
    "model",
    "model_r2v",
    "video_vae",
    "audio_vae",
    "clip",
    "refine_model",
    "refine_model_r2v",
    "upscale_model",
})


def _director_is_changed_value(value):
    """Convert ordinary Director inputs into a stable, compact cache value.

    ``IS_CHANGED`` replaces ComfyUI's default input hashing.  Keep this
    intentionally limited to JSON-like values: linked model/latent objects are
    invalidated by their upstream nodes, while their repr often contains a
    process-specific address and would make every queue a cache miss.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _director_is_changed_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_director_is_changed_value(item) for item in value]
        return sorted(items, key=lambda item: repr(item)) if isinstance(value, (set, frozenset)) else items
    # External group packs are normally dataclasses/simple objects.  Include
    # their data without hashing model/tensor payloads embedded in them.
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "data": _director_is_changed_value(attrs),
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _director_input_signature(kwargs: dict, pre_cache_signature: str) -> str:
    inputs = {
        key: _director_is_changed_value(value)
        for key, value in kwargs.items()
        if key not in _DIRECTOR_LINKED_INPUTS
    }
    payload = json.dumps(
        {"inputs": inputs, "pre_cache": pre_cache_signature},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _live_tae_vae_choices():
    try:
        from ..director.tae_preview import list_vae_approx_names
        return ["auto", *list_vae_approx_names()]
    except Exception:
        return ["auto"]


def director_timeline_required_inputs() -> dict:
    """Timeline widgets — defaults aligned with official MiniMax H3 workflow templates."""
    inputs = timeline_required_inputs()
    combo_options, combo_meta = inputs["task_type"]

    gp_meta = dict(inputs["global_prompt"][1])
    gp_meta["default"] = _DEFAULT_GLOBAL_PROMPT
    gp_meta["tooltip"] = (
        "User prompt — sent directly to MiniMaxH3ImageToVideo / ReferenceToVideo. "
        "r2v: <Picture 1>. v2v: source-timeline edit (<Video 1>). "
        "rv2v: source timeline + reference images (<Video 1> + <Picture N>)."
    )

    frames_meta = dict(inputs["total_frames"][1])
    frames_meta["default"] = 124
    frames_meta["tooltip"] = (
        "Frame count at 24 fps; snapped to MiniMax 17k+5 grid (124 ≈ 5s)."
    )

    return {
        **inputs,
        "task_type": (combo_options, combo_meta),
        "global_prompt": ("STRING", gp_meta),
        "total_frames": ("INT", frames_meta),
    }


class H3_D_NEO:
    """In-node timeline Director using ComfyUI official MiniMax H3 pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "MiniMax H3 UNET (UNETLoader). "
                            "t2v / i2v / fl2v 以及混合模式中的这三类组用 ImageToVideo（fl2va）。"
                            "r2v / v2v / rv2v 仍接此口（ref2va）。"
                            "混合模式的 r2v / v2v / rv2v 组请另接可选 model_r2v。"
                        ),
                    },
                ),
                "video_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 video VAE (minimax_h3_video_vae)."},
                ),
                "audio_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 audio VAE (minimax_h3_audio_vae). Required for r2v / v2v / rv2v."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "CLIPLoader type=minimax (qwen3vl)."},
                ),
                **director_timeline_required_inputs(),
            },
            "optional": {
                "model_r2v": (
                    "MODEL",
                    {
                        "tooltip": (
                            "可选 ReferenceToVideo UNET（ref2va）。"
                            "混合模式中 r2v / v2v / rv2v 组使用此模型；不接则回退到 model。"
                            "纯 r2v / v2v / rv2v 仍只用上面的 model 口。"
                        ),
                    },
                ),
                "lora_trigger_words": (
                    "STRING",
                    {
                        "forceInput": True,
                        "default": "",
                        "tooltip": (
                            "可选。LoRA 触发词。r2v 追加到文末 style_tags 块；"
                            "其它模式拼到每组提示词最前面（mh3turbo, <原提示词>）。"
                            "不接或为空则不改提示词。"
                        ),
                    },
                ),
                "lora_trigger_words_r2v": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "可选。仅供 r2v / v2v / rv2v 使用的 LoRA 触发词。"
                            "连接时覆盖 lora_trigger_words；未连接时回退到 lora_trigger_words。"
                        ),
                    },
                ),
                "director_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "可选。接收 h3-director-export 输出的 "
                            "minimax-h3-director-prompt/v1 JSON，自动编排分组。"
                            "未连接时完全使用导演台现有时间线；连接后保留同序号组的已有素材槽位。"
                        ),
                    },
                ),
                "refine_model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "可选二采 UNET。不接则用导演台主模型。"
                            "适合一采挂 Turbo LoRA、二采卸掉或换另一套。"
                        ),
                    },
                ),
                "refine_model_r2v": (
                    "MODEL",
                    {
                        "tooltip": (
                            "可选 r2v / v2v / rv2v 二采 UNET。"
                            "连接时优先于 refine_model；未连接时回退 refine_model，"
                            "再回退当前段的一采模型。"
                        ),
                    },
                ),
                "upscale_model": (
                    "UPSCALE_MODEL",
                    {
                        "tooltip": (
                            "可选像素放大模型（RealESRGAN 等）。"
                            "仅 mode=upscale 且 upscale_method=lanczos 时使用。"
                        ),
                    },
                ),
                "bd_grp_advanced": ("BDGROUP", {"default": "高级采样"}),
                "steps": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 200,
                        "tooltip": "一采步数（官方模板 25）。",
                    },
                ),
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": "res_multistep",
                        "tooltip": "Official template: KSamplerSelect res_multistep.",
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": "simple",
                        "tooltip": "一采调度器（官方模板 simple）。",
                    },
                ),
                "shift_video": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "MiniMaxH3SigmaShift shift_video."},
                ),
                "shift_audio": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "MiniMaxH3SigmaShift shift_audio."},
                ),
                "live_tae_vae": (
                    _live_tae_vae_choices(),
                    {
                        "default": "auto",
                        "tooltip": (
                            "实时预览 TinyVAE / taeh3（扫描 models/vae_approx）。"
                            "auto=自动选用 minimax-h3/taeh3。"
                        ),
                    },
                ),
                **director_perf_inputs(),
                **director_refine_widget_inputs(),
                **director_selflift_widget_inputs(),
                **director_face_refine_widget_inputs(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **_kwargs):
        if input_types is not None:
            expected = {
                "model": "MODEL",
                "model_r2v": "MODEL",
                "video_vae": "VAE",
                "audio_vae": "VAE",
                "clip": "CLIP",
            }
            for name, want in expected.items():
                got = input_types.get(name)
                if got is not None and got != want:
                    return f"{name}: expected {want}, linked node returns {got}."
            got_refine_model = input_types.get("refine_model")
            if got_refine_model is not None and got_refine_model != "MODEL":
                return f"refine_model: expected MODEL, linked node returns {got_refine_model}."
            got_refine_model_r2v = input_types.get("refine_model_r2v")
            if got_refine_model_r2v is not None and got_refine_model_r2v != "MODEL":
                return f"refine_model_r2v: expected MODEL, linked node returns {got_refine_model_r2v}."
        return True

    @classmethod
    def IS_CHANGED(cls, unique_id=None, **kwargs):
        # Fingerprint both execution inputs and .pre cache files.  The old
        # implementation discarded kwargs and therefore returned the same
        # value when only timeline_data/runSelection changed; ComfyUI then
        # reused the previous node output, making preview and exported MP4
        # appear to come from the previous run.
        from ..director.segment_cache import first_pass_cache_disk_signature

        pre_cache_signature = first_pass_cache_disk_signature(unique_id)
        return _director_input_signature(kwargs, pre_cache_signature)

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "IMAGE", "IMAGE", "BOOLEAN", "IMAGE")
    RETURN_NAMES = ("images", "audio", "frame_count", "source_images", "images_pre_refine", "refine_enabled", "images_pre_face_refine")
    OUTPUT_IS_LIST = (True, True, False, True, True, False, True)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "H3_D_NEO: MiniMaxH3ImageToVideo / ReferenceToVideo conditioning, "
        "single-stage KSampler + MiniMaxH3SigmaShift, LTXVSeparateAVLatent decode. "
        "Supports t2v / i2v / fl2v / mixed / r2v / v2v / rv2v. "
        "Built-in 二采 group runs a second sample / upscale. "
        "images_pre_refine is the first-pass video before refine. "
        "refine_enabled is the 二采 checkbox. "
        "Defaults: 0.4MP 16:9 (864×480), 5s / 124 frames @ 24 fps."
    )

    def execute(
        self,
        model,
        video_vae,
        audio_vae,
        clip,
        task_type,
        global_prompt,
        frame_rate,
        width,
        height,
        ref_max_size,
        total_frames,
        timeline_data,
        unique_id=None,
        model_r2v=None,
        director_prompt=None,
        lora_trigger_words="",
        lora_trigger_words_r2v=None,
        refine_model=None,
        refine_model_r2v=None,
        upscale_model=None,
        steps=25,
        sampler="res_multistep",
        scheduler="simple",
        cfg=1.0,
        seed=0,
        shift_video=12.0,
        shift_audio=3.0,
        export_source_images=False,
        live_tae_vae="auto",
        **kwargs,
    ):
        refine_enabled = bool(kwargs.get("refine_enable", False))
        face_refine = {
            "enabled": _widget_bool(kwargs.get("face_refine_enable", False)),
            "detector": kwargs.get("face_refine_detector"),
            "confidence": kwargs.get("face_refine_confidence"),
            "crop_factor": kwargs.get("face_refine_crop_factor"),
            "canvas_width": kwargs.get("face_refine_canvas_width"),
            "canvas_height": kwargs.get("face_refine_canvas_height"),
            "canvas_mode": kwargs.get("face_refine_canvas_mode"),
            "select": kwargs.get("face_refine_select"),
            "denoise": kwargs.get("face_refine_denoise"),
            "steps": kwargs.get("face_refine_steps"),
            "sampler": kwargs.get("face_refine_sampler"),
            "scheduler": kwargs.get("face_refine_scheduler"),
            "seed_mode": kwargs.get("face_refine_seed_mode"),
            "paste_region": kwargs.get("face_refine_paste_region"),
            "mask_dilation": kwargs.get("face_refine_mask_dilation"),
            "feather": kwargs.get("face_refine_feather"),
            "colour_match": kwargs.get("face_refine_colour_match"),
            "blend": kwargs.get("face_refine_blend"),
        }
        clear_vram_before_face_refine = _widget_bool(kwargs.get("clear_vram_before_face_refine", False))
        clear_vram_before_refine = _widget_bool(kwargs.get("clear_vram_before_refine", False))
        export_pre_face_refine = _widget_bool(kwargs.get("export_pre_face_refine", False))
        selflift = None
        if _widget_bool(kwargs.get("selflift_enable", False)):
            from ..director.selflift.pack import pack_selflift
            selflift = pack_selflift(
                split_mode=kwargs.get("selflift_split_mode", "highres_steps"),
                highres_steps=kwargs.get("selflift_highres_steps", 2),
                transition_step=kwargs.get("selflift_transition_step", 6),
                lowres_scale=kwargs.get("selflift_lowres_scale", 0.5),
                latent_upscale_model=kwargs.get("selflift_latent_upscale_model"),
                sampler_mode=kwargs.get("selflift_sampler_mode", "euler"),
                native_low_carry=_widget_bool(kwargs.get("selflift_native_low_carry"), True),
                rho=kwargs.get("selflift_rho", 0.0),
                w_min=kwargs.get("selflift_w_min", 0.5),
                w_max=kwargs.get("selflift_w_max", 1.0),
                latent_upsample=kwargs.get("selflift_latent_upsample", "bilinear"),
                enable_latent_chunking=_widget_bool(kwargs.get("selflift_enable_latent_chunking")),
                enable_tiling=_widget_bool(kwargs.get("selflift_enable_tiling")),
                tile_count=kwargs.get("selflift_tile_count", 2),
                tile_overlap=kwargs.get("selflift_tile_overlap", 128),
            )
        refine = pack_director_builtin_refine(
            enabled=refine_enabled,
            refine_model=refine_model,
            refine_model_r2v=refine_model_r2v,
            upscale_model=upscale_model,
            **kwargs,
        )
        del kwargs

        plan = prepare_director_plan(
            timeline_data=timeline_data,
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
            unique_id=unique_id,
            director_prompt=director_prompt,
            selflift=selflift,
            face_refine=face_refine,
            refine=refine,
            lora_trigger_words=lora_trigger_words,
            lora_trigger_words_r2v=lora_trigger_words_r2v,
        )
        raw = getattr(plan, "raw", None)
        if isinstance(raw, dict):
            vae_name = "" if str(live_tae_vae or "").strip() in ("", "auto") else str(live_tae_vae).strip()
            raw["liveTaeVae"] = vae_name

        try:
            combined, segment_outputs, segment_audios, export_frame_counts, pre_combined, pre_segments, pre_face_combined, pre_face_segments = (
                execute_director_plan_core(
                    plan,
                    node_id=unique_id,
                    model=model,
                    model_r2v=model_r2v,
                    vae=video_vae,
                    audio_vae=audio_vae,
                    clip=clip,
                    cfg=cfg,
                    seed=seed,
                    steps=steps,
                    sampler=sampler,
                    scheduler=scheduler,
                    shift_video=shift_video,
                    shift_audio=shift_audio,
                    clear_vram_between_segments=True,
                    clear_vram_before_refine=clear_vram_before_refine,
                    clear_vram_before_face_refine=clear_vram_before_face_refine,
                    export_pre_face_refine=export_pre_face_refine,
                )
            )

            finalized = finalize_director_outputs(
                plan,
                combined,
                segment_outputs,
                export_source_images=export_source_images,
                segment_audios=segment_audios,
                segment_frame_counts=export_frame_counts,
                pre_refine_combined=pre_combined,
                pre_refine_segments=pre_segments,
                pre_face_combined=pre_face_combined,
                pre_face_segments=pre_face_segments,
                export_pre_face_refine=export_pre_face_refine,
            )
            # Keep this order aligned with RETURN_NAMES/OUTPUT_IS_LIST:
            # pre_face output is the final slot, after the scalar refine flag.
            return (*finalized[:5], refine_enabled, finalized[5])
        finally:
            # Full source/reference PCM is execution-scoped.
            cache = getattr(plan, "audio_decode_cache", None)
            if isinstance(cache, dict):
                cache.clear()
            for item in getattr(plan, "global_ref_audios", None) or []:
                if getattr(item, "audio_path", ""):
                    item.audio = None
