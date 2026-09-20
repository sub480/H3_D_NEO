"""Shared helpers for the MiniMax H3 Director timeline node."""

from __future__ import annotations

import json
import logging

import torch
from comfy_execution.graph_utils import ExecutionBlocker

from ..director.audio_export import (
    AUDIO_MODE_GENERATE,
    build_director_audio_outputs,
    resolve_audio_mode,
)
from ..director.frame_align import H3_FPS, pad_or_trim_frames
from ..director.gen_timeline import is_prompt_batch_timeline, is_video_batch_task_key
from ..director.plan import (
    apply_lora_trigger_words,
    apply_lora_trigger_words_r2v,
    build_director_plan,
    count_all_timeline_segments,
    count_timeline_segments,
    plan_summary,
)
from ..director.progress import report_director_planning
from ..director.segment_mp4_export import released_output_slots
from ..lib.image_prep import cat_frames_variable_size, fit_frames_to_canvas, fit_video_long_edge
from ..lib.video_io import load_timeline_segment
from ..lib.task_prompts import task_type_combo_options

log = logging.getLogger("ComfyUI-MiniMaxH3-Director")


def timeline_required_inputs() -> dict:
    """Timeline + prompt widgets shared by Director nodes."""
    combo_options, combo_meta = task_type_combo_options()
    return {
        "task_type": (combo_options, combo_meta),
        "global_prompt": (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "tooltip": "Synced from in-node UI (global mode).",
            },
        ),
        "bd_grp_sample": ("BDGROUP", {"default": "采样设置"}),
        "cfg": (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "CFG for KSampler."},
        ),
        "seed": (
            "INT",
            {
                "default": 42,
                "min": 0,
                "max": 0xFFFFFFFFFFFFFFFF,
                "control_after_generate": True,
                "tooltip": "Random seed for sampling.",
            },
        ),
        "frame_rate": (
            "FLOAT",
            {
                "default": H3_FPS,
                "min": H3_FPS,
                "max": H3_FPS,
                "step": 0.01,
                "tooltip": "Locked to 24 fps (MiniMax H3 training / audio clock).",
            },
        ),
        "width": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
        "height": ("INT", {"default": 480, "min": 32, "max": 8192, "step": 32}),
        "ref_max_size": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
        "total_frames": (
            "INT",
            {
                "default": 124,
                "min": 5,
                "max": 100000,
                "tooltip": "Timeline total frames (fl2v = sum of shots). Per-shot generation still capped near 512.",
            },
        ),
        "timeline_data": (
            "STRING",
            {"default": "", "multiline": True, "tooltip": "Internal — video, segments, refs (populated by UI)."},
        ),
    }


def director_perf_inputs() -> dict:
    """Hidden Director extras. Segment VRAM clear is always on."""
    return {
        "export_source_images": (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "将时间轴原片解码到独立的 source_images 输出口；"
                    "需将 source_images 另接预览/合成节点才能查看，不会改变主 images。"
                    "默认关以节省内存。"
                ),
            },
        ),
    }


def default_timeline_json(
    *,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
) -> str:
    return json.dumps(
        {
            "version": 5,
            "timelineMode": "prompt_batch",
            "editMode": "segment",
            "totalFrames": total_frames,
            "frameRate": frame_rate,
            "width": width,
            "height": height,
            "refMaxSize": ref_max_size,
            "output": {
                "mode": "fixed",
                "longEdge": ref_max_size,
                "width": width,
                "height": height,
                "maxExportFrames": 0,
                "exportMode": "all",
                "audioMode": "generate",
                "refImageSize": "match",
            },
            "videoClips": [],
            "video": {
                "fileName": "",
                "videoFile": "",
                "subfolder": "",
                "type": "input",
                "frames": [],
                "frameMap": [],
            },
            "global": {"taskType": "mixed", "prompt": global_prompt, "refs": [], "referenceVideo": {}, "continuousReference": False},
            "segments": [
                {
                    "id": "s0",
                    "start": 0,
                    "length": total_frames,
                    "prompt": "",
                    "taskType": "",
                    "refs": [],
                    "referenceVideo": {},
                }
            ],
        },
        ensure_ascii=False,
    )


def prepare_director_plan(
    *,
    timeline_data: str,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
    unique_id: str | None,
    director_prompt=None,
    selflift=None,
    semantic_bridge=None,
    face_refine=None,
    refine=None,
    lora_trigger_words=None,
    lora_trigger_words_r2v=None,
):
    frame_rate = H3_FPS
    if not timeline_data or not timeline_data.strip():
        timeline_data = default_timeline_json(
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )

    report_director_planning(
        unique_id,
        count_timeline_segments(timeline_data),
        timeline_segment_total=count_all_timeline_segments(timeline_data),
    )

    plan = build_director_plan(
        timeline_data,
        global_task_type=task_type,
        global_prompt=global_prompt,
        total_frames=total_frames,
        frame_rate=frame_rate,
        width=width,
        height=height,
        ref_max_size=ref_max_size,
    )
    from ..director.selflift.pack import normalize_selflift_pack
    plan.selflift = normalize_selflift_pack(selflift)
    from ..director.semantic_bridge import normalize_semantic_bridge_pack
    plan.semantic_bridge = normalize_semantic_bridge_pack(semantic_bridge)
    from ..director.face_refine.pack import normalize_face_refine_pack
    plan.face_refine = normalize_face_refine_pack(face_refine)
    plan = _attach_refine(plan, refine)
    plan = apply_lora_trigger_words(plan, lora_trigger_words)
    plan = apply_lora_trigger_words_r2v(plan, lora_trigger_words_r2v)
    log.info(plan_summary(plan).replace("\n", " | "))
    return plan


def _attach_refine(plan, refine):
    from ..director.refine_pack import normalize_refine_pack

    plan.refine = normalize_refine_pack(
        refine,
        base_width=int(getattr(plan, "width", 0) or 0),
        base_height=int(getattr(plan, "height", 0) or 0),
    )
    return plan


def _fit_source_clip_to_plan(plan, raw_clip: torch.Tensor, seg=None) -> torch.Tensor:
    if plan.output_mode == "fixed":
        mode = "contain"
        if seg is not None and not bool(getattr(seg, "use_source_resolution", False)):
            mode = getattr(seg, "video_fit", "contain") or "contain"
        return fit_frames_to_canvas(raw_clip, plan.width, plan.height, mode)
    return fit_video_long_edge(raw_clip, plan.ref_max_size)


def _segment_source_for_output(plan, seg) -> torch.Tensor:
    if seg.source_clip is not None and int(seg.source_clip.shape[0]) > 0:
        return seg.source_clip
    if seg.task_key in {"t2v", "r2v"}:
        return torch.full(
            (1, int(plan.height), int(plan.width), 3),
            0.5,
            dtype=torch.float32,
        )
    return load_timeline_segment(plan.raw, seg.start_frame, seg.end_frame)


def _pad_source_last_frame(frames: torch.Tensor, target_len: int) -> torch.Tensor:
    missing = max(0, int(target_len) - int(frames.shape[0]))
    if missing <= 0 or int(frames.shape[0]) <= 0:
        return frames
    return torch.cat([frames, frames[-1:].repeat(missing, 1, 1, 1)], dim=0)


def build_source_images_output(
    plan,
    images_out: list[torch.Tensor],
    *,
    split_outputs: bool,
    segment_frame_counts: list[int] | None = None,
) -> list[torch.Tensor]:
    if split_outputs:
        chunks: list[torch.Tensor] = []
        # images_out is run-order (选择运行), not full timeline order.
        segs = plan.segments
        run_idx = getattr(plan, "run_indices", None)
        if run_idx is not None:
            segs = [
                plan.segments[i]
                for i in sorted(run_idx)
                if 0 <= i < len(plan.segments)
            ]
        for seg, generated in zip(segs, images_out):
            target_len = int(generated.shape[0])
            raw = _segment_source_for_output(plan, seg)
            fitted = _fit_source_clip_to_plan(plan, raw, seg)
            fitted = _pad_source_last_frame(fitted, target_len)
            chunks.append(pad_or_trim_frames(fitted, target_len).cpu().float())
        return chunks

    target_len = int(images_out[0].shape[0]) if images_out else int(plan.total_frames or 0)
    if plan.run_indices is not None:
        chunks: list[torch.Tensor] = []
        for pos, index in enumerate(sorted(plan.run_indices)):
            seg = plan.segments[index]
            raw = _segment_source_for_output(plan, seg)
            fitted = _fit_source_clip_to_plan(plan, raw, seg)
            chunk_len = (
                int(segment_frame_counts[pos])
                if segment_frame_counts is not None and pos < len(segment_frame_counts)
                else int(seg.frame_count)
            )
            fitted = _pad_source_last_frame(fitted, chunk_len)
            chunks.append(pad_or_trim_frames(fitted, chunk_len).cpu().float())
        return [pad_or_trim_frames(cat_frames_variable_size(chunks), target_len)]
    if any(seg.source_clip is not None for seg in plan.segments):
        chunks = []
        for pos, seg in enumerate(plan.segments):
            raw = _segment_source_for_output(plan, seg)
            fitted = _fit_source_clip_to_plan(plan, raw, seg)
            chunk_len = (
                int(segment_frame_counts[pos])
                if segment_frame_counts is not None and pos < len(segment_frame_counts)
                else int(seg.frame_count)
            )
            chunks.append(_pad_source_last_frame(fitted, chunk_len).cpu().float())
        return [pad_or_trim_frames(cat_frames_variable_size(chunks), target_len)]
    raw = load_timeline_segment(plan.raw, 0, target_len)
    fitted = _fit_source_clip_to_plan(plan, raw)
    return [pad_or_trim_frames(fitted, target_len).cpu().float()]


def _empty_source_images_for(images_out: list[torch.Tensor]) -> list[torch.Tensor]:
    if not images_out:
        return [torch.full((1, 1, 1, 3), 0.5)]
    placeholders: list[torch.Tensor] = []
    for img in images_out:
        if isinstance(img, torch.Tensor) and img.ndim == 4:
            h, w, c = int(img.shape[1]), int(img.shape[2]), int(img.shape[3])
        else:
            h, w, c = 1, 1, 3
        placeholders.append(torch.full((1, h, w, c), 0.5))
    return placeholders


def _pad_even_hw_image(image: torch.Tensor) -> torch.Tensor:
    """Pad IMAGE batches for yuv420p encoders, which require even dimensions."""
    frames, height, width, channels = (int(value) for value in image.shape)
    even_height = max(2, height + height % 2)
    even_width = max(2, width + width % 2)
    if even_height == height and even_width == width:
        return image
    padded = image.new_zeros((frames, even_height, even_width, channels))
    padded[:, :height, :width].copy_(image)
    return padded


def _ensure_nonempty_image_batches(images_out: list[torch.Tensor], *, label: str) -> list[torch.Tensor]:
    fixed: list[torch.Tensor] = []
    for i, img in enumerate(images_out):
        if not isinstance(img, torch.Tensor) or img.ndim != 4:
            raise ValueError(f"Director {label}[{i}] is not a valid IMAGE tensor.")
        if int(img.shape[0]) <= 0:
            h, w, c = int(img.shape[1]), int(img.shape[2]), int(img.shape[3])
            log.warning("Director %s[%d] has 0 frames; emitting 1-frame placeholder.", label, i)
            img = torch.full((1, max(2, h), max(2, w), max(1, c)), 0.5)
        fixed.append(_pad_even_hw_image(img))
    return fixed


def _layout_image_batches(
    plan,
    combined,
    segment_outputs,
    *,
    export_segments: bool,
    is_batch: bool,
    video_batch: bool,
) -> tuple[list[torch.Tensor], int]:
    if export_segments or (is_batch and not video_batch):
        images_out = segment_outputs
        frame_count = sum(int(s.shape[0]) for s in segment_outputs)
        return images_out, frame_count
    if getattr(plan, "continuity_enabled", False) and getattr(
        plan, "continuity_keep_tail", False
    ):
        combined = combined.cpu().float()
    else:
        combined = pad_or_trim_frames(combined, plan.total_frames).cpu().float()
    return [combined], int(combined.shape[0])


def finalize_director_outputs(
    plan,
    combined,
    segment_outputs,
    *,
    export_source_images: bool = False,
    segment_audios: list | None = None,
    segment_frame_counts: list[int] | None = None,
    pre_refine_combined=None,
    pre_refine_segments: list | None = None,
    pre_face_combined=None,
    pre_face_segments: list | None = None,
    export_pre_face_refine: bool = False,
    block_final_images: bool = False,
):
    is_batch = is_prompt_batch_timeline(plan.raw, plan.global_task_key)
    export_segments = plan.export_mode == "segments"
    video_batch = is_video_batch_task_key(plan.global_task_key)
    split_layout = export_segments or (is_batch and not video_batch)

    images_out, frame_count = _layout_image_batches(
        plan,
        combined,
        segment_outputs,
        export_segments=export_segments,
        is_batch=is_batch,
        video_batch=video_batch,
    )
    if segment_frame_counts:
        frame_count = int(sum(int(n) for n in segment_frame_counts))
    released_slots: list[int] = []
    if export_segments and len(segment_outputs) > 1:
        released_slots = released_output_slots(segment_outputs, segment_frame_counts)

    pre_segs = pre_refine_segments if pre_refine_segments else segment_outputs
    pre_comb = pre_refine_combined if pre_refine_combined is not None else combined
    share_pre = pre_comb is combined and (
        pre_segs is segment_outputs
        or (
            len(pre_segs) == len(segment_outputs)
            and all(a is b for a, b in zip(pre_segs, segment_outputs))
        )
    )
    if share_pre:
        pre_refine_out = images_out
    else:
        try:
            pre_refine_out, _ = _layout_image_batches(
                plan,
                pre_comb,
                pre_segs,
                export_segments=export_segments,
                is_batch=is_batch,
                video_batch=video_batch,
            )
        except Exception as exc:
            log.warning("images_pre_refine layout failed: %s", exc)
            pre_refine_out = images_out

    pre_face_out = None
    from ..director.face_refine.pack import face_refine_enabled
    if export_pre_face_refine and face_refine_enabled(plan):
        face_segs = pre_face_segments if pre_face_segments else segment_outputs
        face_comb = pre_face_combined if pre_face_combined is not None else combined
        try:
            pre_face_out, _ = _layout_image_batches(
                plan,
                face_comb,
                face_segs,
                export_segments=export_segments,
                is_batch=is_batch,
                video_batch=video_batch,
            )
        except Exception as exc:
            log.warning("images_pre_face_refine layout failed: %s", exc)
            pre_face_out = images_out

    if export_segments:
        released_slots = sorted(
            set(released_slots) | set(released_output_slots(pre_refine_out, segment_frame_counts))
        )
    if released_slots:
        released = set(released_slots)
        keep = [index for index in range(len(images_out)) if index not in released]
        if keep:
            images_out = [images_out[index] for index in keep]
            kept_pre_refine = [
                pre_refine_out[index]
                for index in keep
                if index < len(pre_refine_out)
            ]
            if kept_pre_refine:
                pre_refine_out = kept_pre_refine
            if pre_face_out is not None:
                kept_pre_face = [
                    pre_face_out[index]
                    for index in keep
                    if index < len(pre_face_out)
                ]
                if kept_pre_face:
                    pre_face_out = kept_pre_face
            if segment_audios:
                segment_audios = [
                    segment_audios[index]
                    for index in keep
                    if index < len(segment_audios)
                ]
            if segment_frame_counts:
                segment_frame_counts = [
                    segment_frame_counts[index]
                    for index in keep
                    if index < len(segment_frame_counts)
                ]

    split_for_audio = split_layout
    audio_frame_end = frame_count if not split_for_audio else None
    audio_mode = resolve_audio_mode(plan)
    use_generated = audio_mode == AUDIO_MODE_GENERATE
    # Prefer caller-provided export lengths (post continuity trim); else match IMAGE batches.
    # Source audio needs the same lengths so rv2v clips follow the emitted picture,
    # rather than the pre-trim plan duration.
    if segment_frame_counts is None and split_for_audio:
        segment_frame_counts = [int(s.shape[0]) for s in images_out]
    audio_out, _ = build_director_audio_outputs(
        plan,
        images_out,
        export_segments=split_for_audio,
        output_frame_end=audio_frame_end,
        segment_audios=segment_audios if use_generated else None,
        segment_frame_counts=segment_frame_counts,
        audio_mode=audio_mode,
    )

    split_source_outputs = export_segments or (is_batch and not video_batch)
    if export_source_images:
        try:
            source_images_out = build_source_images_output(
                plan,
                images_out,
                split_outputs=split_source_outputs,
                segment_frame_counts=segment_frame_counts,
            )
        except Exception as exc:
            log.warning("Source images output failed: %s", exc)
            # Never disguise generated frames as the source comparison. A neutral
            # placeholder makes the failure visible while preserving the expensive run.
            source_images_out = _empty_source_images_for(images_out)
    else:
        source_images_out = _empty_source_images_for(images_out)

    images_out = _ensure_nonempty_image_batches(images_out, label="images")
    source_images_out = _ensure_nonempty_image_batches(source_images_out, label="source_images")
    pre_refine_out = _ensure_nonempty_image_batches(pre_refine_out, label="images_pre_refine")
    if pre_face_out is None:
        pre_face_out = ExecutionBlocker(None)
    else:
        pre_face_out = _ensure_nonempty_image_batches(pre_face_out, label="images_pre_face_refine")

    if block_final_images:
        images_out = ExecutionBlocker(None)
    return images_out, audio_out, frame_count, source_images_out, pre_refine_out, pre_face_out
