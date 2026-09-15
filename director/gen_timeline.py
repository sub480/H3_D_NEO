"""MiniMax H3 Director 鈥?generation timeline (t2i / t2v / i2i / i2v) plan building."""

from __future__ import annotations

import hashlib
import logging

import torch

from ..lib.image_prep import (
    assert_minimax_canvas,
    cat_frames_variable_size,
    fit_canvas,
    fit_video_long_edge,
    resolve_output_dimensions,
)
from ..lib.task_prompts import resolve_task_key
from .frame_align import H3_FPS, minimax_floor_frame_count

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.gen")

GEN_BLANK_KEYS = frozenset({"t2v", "r2v"})
GEN_IMAGE_KEYS = frozenset({"i2v"})
FL2V_KEYS = frozenset({"fl2v"})
MIXED_KEY = "mixed"
MIXED_GROUP_KEYS = frozenset({"t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"})
IMAGE_BATCH_KEYS = frozenset()

MIN_GEN_FRAMES = 1
MIN_GEN_VIDEO_FRAMES = 4


def is_gen_task_key(task_key: str) -> bool:
    return task_key == MIXED_KEY


def is_gen_timeline(timeline: dict, task_key: str) -> bool:
    mode = str(timeline.get("timelineMode") or "").lower()
    return mode == "prompt_batch" and task_key == MIXED_KEY


def is_prompt_batch_timeline(timeline: dict, task_key: str) -> bool:
    mode = str(timeline.get("timelineMode") or "").lower()
    return mode == "prompt_batch" and task_key == MIXED_KEY


def is_image_batch_timeline(timeline: dict, task_key: str) -> bool:
    return is_prompt_batch_timeline(timeline, task_key)


def is_video_batch_task_key(task_key: str) -> bool:
    return task_key == MIXED_KEY


def is_mixed_task_key(task_key: str) -> bool:
    return task_key == MIXED_KEY


def resolve_mixed_segment_task_key(seg_data: dict | None, global_key: str) -> str:
    """Mixed groups store their own type; empty / unknown → t2v."""
    if global_key != MIXED_KEY:
        return global_key
    raw = ""
    if isinstance(seg_data, dict):
        raw = seg_data.get("taskType") or ""
    key = resolve_task_key(raw) if str(raw).strip() else "t2v"
    return key if key in MIXED_GROUP_KEYS else "t2v"


def gen_submode(timeline: dict, task_key: str) -> str:
    del timeline, task_key
    return "gen_blank"


def _min_frames_for_task(task_key: str) -> int:
    if task_key in IMAGE_BATCH_KEYS or task_key in ("t2i", "i2i"):
        return MIN_GEN_FRAMES
    if task_key in ("t2v", "i2v", "r2v", "fl2v", MIXED_KEY):
        return MIN_GEN_VIDEO_FRAMES
    return MIN_GEN_VIDEO_FRAMES


def _segment_frame_count(raw: dict, *, default: int, task_key: str) -> int:
    fc = int(raw.get("frameCount") or raw.get("length") or default)
    return max(_min_frames_for_task(task_key), fc)


def _mixed_source_range(raw: dict) -> tuple[int, int, int] | None:
    source = raw.get("sourceVideo") or {}
    if not isinstance(source, dict):
        return None
    video = source.get("video") if isinstance(source.get("video"), dict) else source
    clips = source.get("videoClips") if isinstance(source.get("videoClips"), list) else []
    total_candidates = (
        len(video.get("frameMap") or []),
        source.get("totalFrames"),
        video.get("sourceFrameCount"),
        sum(max(0, int(clip.get("sourceFrameCount") or 0)) for clip in clips),
    )
    total = next((int(value) for value in total_candidates if value), 0)
    if total <= 0:
        return None
    start = max(0, min(total, int(source.get("rangeStart") or 0)))
    raw_end = source.get("rangeEnd")
    end = total if raw_end is None else max(start, min(total, int(raw_end)))
    return start, end, minimax_floor_frame_count(end - start)


def _gen_segment_ranges(
    segments: list[dict],
    *,
    default_frame_count: int,
    task_key: str,
) -> list[tuple[int, int, dict]]:
    ranges: list[tuple[int, int, dict]] = []
    start = 0
    for index, raw in enumerate(segments):
        fc = _segment_frame_count(raw, default=default_frame_count, task_key=task_key)
        seg_task_key = resolve_mixed_segment_task_key(raw, task_key)
        source_range = _mixed_source_range(raw) if seg_task_key in {"v2v", "rv2v"} else None
        if source_range is not None:
            range_start, range_end, aligned_count = source_range
            selected_count = range_end - range_start
            if aligned_count <= 0:
                raise ValueError(
                    f"Mixed {seg_task_key} group #{index + 1} source range has {selected_count} frame(s); "
                    "MiniMax H3 requires at least 5 source frames."
                )
            if aligned_count != selected_count:
                log.warning(
                    "Mixed %s group #%d source range aligned down: start=%d, frames=%d -> %d "
                    "(cropped %d tail frame(s))",
                    seg_task_key,
                    index + 1,
                    range_start,
                    selected_count,
                    aligned_count,
                    selected_count - aligned_count,
                )
            fc = max(fc, aligned_count)
        ranges.append((start, start + fc, raw))
        start += fc
    if not ranges:
        fc = max(_min_frames_for_task(task_key), default_frame_count)
        ranges.append((0, fc, {}))
    return ranges


def _resolve_gen_image_ref(
    seg_data: dict,
    *,
    edit_mode: str,
    global_block: dict,
) -> dict | None:
    if edit_mode == "segment":
        img = seg_data.get("genImage") or {}
        if img.get("imageFile") or img.get("imageB64"):
            return img
        if seg_data.get("imageFile"):
            return {"imageFile": seg_data["imageFile"]}
        return None
    img = global_block.get("genImage") or {}
    if img.get("imageFile") or img.get("imageB64"):
        return img
    if global_block.get("imageFile"):
        return {"imageFile": global_block["imageFile"]}
    return None


def _image_ref_identity(ref: dict | None) -> str:
    if not isinstance(ref, dict):
        return ""
    file_name = str(ref.get("imageFile") or ref.get("fileName") or "").replace("\\", "/").strip()
    if file_name:
        return file_name
    image_b64 = str(ref.get("imageB64") or "")
    return f"inline:{hashlib.sha256(image_b64.encode('utf-8')).hexdigest()}" if image_b64 else ""


def _segment_source_media_identity(
    seg_data: dict,
    *,
    task_key: str,
    edit_mode: str,
    global_block: dict,
) -> tuple[str, ...]:
    if task_key == "i2v":
        identity = _image_ref_identity(
            _resolve_gen_image_ref(seg_data, edit_mode=edit_mode, global_block=global_block)
        )
        return (f"first:{identity}",) if identity else ()
    if task_key == "fl2v":
        identities = (
            f"first:{_image_ref_identity(seg_data.get('startImage') or seg_data.get('genImage'))}",
            f"last:{_image_ref_identity(seg_data.get('endImage'))}",
        )
        return tuple(identity for identity in identities if not identity.endswith(":"))
    if task_key in {"v2v", "rv2v"}:
        source = seg_data.get("sourceVideo") or {}
        video = source.get("video") if isinstance(source.get("video"), dict) else source
        clips = source.get("videoClips") if isinstance(source.get("videoClips"), list) else []
        identities = [
            str(clip.get("videoFile") or clip.get("fileName") or "").replace("\\", "/").strip()
            for clip in clips
            if isinstance(clip, dict)
        ]
        identity = "|".join(item for item in identities if item)
        if not identity:
            identity = str(video.get("videoFile") or video.get("fileName") or "").replace("\\", "/").strip()
        frame_map = video.get("frameMap") or []
        map_digest = hashlib.sha256(
            repr(frame_map).encode("utf-8")
        ).hexdigest() if frame_map else "full"
        range_start = source.get("rangeStart")
        range_end = source.get("rangeEnd")
        return (f"source-video:{identity}:{map_digest}:{range_start}:{range_end}",) if identity else ()
    return ()


def _mixed_source_video_timeline(
    seg_data: dict,
    timeline: dict,
    *,
    target_width: int,
    target_height: int,
) -> dict | None:
    source = seg_data.get("sourceVideo") or {}
    if not isinstance(source, dict):
        return None
    video = source.get("video") if isinstance(source.get("video"), dict) else source
    clips = source.get("videoClips") if isinstance(source.get("videoClips"), list) else []
    if not clips and (video.get("videoFile") or video.get("fileName")):
        clips = [dict(video)]
    if not clips:
        return None
    total_candidates = (
        len(video.get("frameMap") or []),
        source.get("totalFrames"),
        video.get("sourceFrameCount"),
        sum(max(0, int(clip.get("sourceFrameCount") or 0)) for clip in clips),
    )
    total = next((int(value) for value in total_candidates if value), 0)
    if total <= 0:
        return None
    range_start = max(0, min(total, int(source.get("rangeStart") or 0)))
    raw_range_end = source.get("rangeEnd")
    range_end = total if raw_range_end is None else max(range_start, min(total, int(raw_range_end)))
    aligned_count = minimax_floor_frame_count(range_end - range_start)
    if aligned_count <= 0:
        return None
    range_end = range_start + aligned_count
    selected_frame_map = None
    if range_start > 0 or range_end < total:
        full_frame_map = list(video.get("frameMap") or [])
        if not full_frame_map:
            full_frame_map = [
                {"clip": clip_index, "frame": frame_index}
                for clip_index, clip in enumerate(clips)
                for frame_index in range(max(0, int(clip.get("sourceFrameCount") or 0)))
            ]
        selected_frame_map = full_frame_map[range_start:range_end]
        total = len(selected_frame_map)
        if total <= 0:
            return None
    use_source_resolution = str(seg_data.get("videoResolution") or "target") == "source"
    resolved_clips = []
    for clip in clips:
        resolved = dict(clip)
        clip_width = int(resolved.get("width") or target_width)
        clip_height = int(resolved.get("height") or target_height)
        if use_source_resolution:
            clip_width, clip_height, _, _ = resolve_output_dimensions(
                clip_width,
                clip_height,
                mode="long_edge",
                long_edge=max(clip_width, clip_height),
            )
        else:
            clip_width, clip_height = target_width, target_height
        resolved["storageWidth"] = clip_width
        resolved["storageHeight"] = clip_height
        resolved_clips.append(resolved)
    resolved_video = dict(video)
    if selected_frame_map is not None:
        resolved_video["frameMap"] = selected_frame_map
        resolved_video["deletedSourceRanges"] = []
    primary = resolved_clips[0]
    resolved_video["storageWidth"] = primary["storageWidth"]
    resolved_video["storageHeight"] = primary["storageHeight"]
    return {
        "frameRate": timeline.get("frameRate") or H3_FPS,
        "refMaxSize": timeline.get("refMaxSize"),
        "output": dict(timeline.get("output") or {}),
        "totalFrames": total,
        "video": resolved_video,
        "videoClips": resolved_clips,
    }


def _load_gen_image_tensor(ref: dict) -> torch.Tensor:
    from .plan import load_reference_tensor

    tensor = load_reference_tensor(ref)
    if tensor is None:
        raise ValueError("Generation segment image could not be loaded.")
    return tensor


def _build_i2v_source_clip(
    img: torch.Tensor,
    _frame_count: int,
    *,
    width: int,
    height: int,
    output_mode: str,
    ref_max_size: int,
) -> torch.Tensor:
    """Use the source image as a one-frame source-video context."""
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if output_mode == "fixed":
        return fit_canvas(img, width, height)
    return fit_video_long_edge(img, ref_max_size)


def _resolve_gen_image_source_dims(
    segment_ranges: list[tuple[int, int, dict]],
    global_block: dict,
    output_block: dict,
) -> tuple[int, int]:
    sw = int(global_block.get("sourceWidth") or output_block.get("sourceWidth") or 0)
    sh = int(global_block.get("sourceHeight") or output_block.get("sourceHeight") or 0)
    if sw > 0 and sh > 0:
        return sw, sh
    for _start, _end, seg_data in segment_ranges:
        gi = seg_data.get("genImage") or {}
        sw = int(gi.get("width") or 0)
        sh = int(gi.get("height") or 0)
        if sw > 0 and sh > 0:
            return sw, sh
    return 0, 0


def _build_gen_source_clips(
    ranges: list[tuple[int, int, dict]],
    *,
    task_key: str,
    submode: str,
    edit_mode: str,
    global_block: dict,
    height: int,
    width: int,
    output_mode: str,
    ref_max_size: int,
    active_indices: frozenset[int] | None = None,
) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    for index, (_start, end, seg_data) in enumerate(ranges):
        frame_count = end - _start
        if frame_count <= 0:
            continue
        if active_indices is not None and index not in active_indices:
            chunks.append(torch.empty((0, 16, 16, 3), dtype=torch.float32))
            continue
        if submode == "gen_blank":
            # t2v/r2v duration lives on the segment range. Do not allocate
            # (N,H,W,3) gray canvases — they are unused as first_frame / <Video 1>
            # and concatenating them into source_video OOMs on long jobs.
            continue
        ref = _resolve_gen_image_ref(seg_data, edit_mode=edit_mode, global_block=global_block)
        if ref is None:
            seg_idx = len(chunks) + 1
            raise ValueError(
                f"Segment #{seg_idx} has no source image. "
                "Upload an image in the generation timeline (global or per-segment)."
            )
        img = _load_gen_image_tensor(ref)
        if task_key == "i2v":
            clip = _build_i2v_source_clip(
                img,
                frame_count,
                width=width,
                height=height,
                output_mode=output_mode,
                ref_max_size=ref_max_size,
            )
        else:
            clip = img.repeat(frame_count, 1, 1, 1)
            if output_mode == "fixed":
                clip = fit_canvas(clip, width, height)
            else:
                clip = fit_video_long_edge(clip, ref_max_size)
        chunks.append(clip)
    if not any(int(chunk.shape[0]) > 0 for chunk in chunks):
        if submode == "gen_blank":
            return []
        raise ValueError("Generation timeline has no frames.")
    return chunks


def _build_gen_source_video(
    ranges: list[tuple[int, int, dict]],
    *,
    task_key: str,
    submode: str,
    edit_mode: str,
    global_block: dict,
    height: int,
    width: int,
    output_mode: str,
    ref_max_size: int,
) -> torch.Tensor:
    if submode == "gen_blank":
        return torch.full((max(1, len(ranges)), 16, 16, 3), 0.5, dtype=torch.float32)
    return cat_frames_variable_size(
        _build_gen_source_clips(
            ranges,
            task_key=task_key,
            submode=submode,
            edit_mode=edit_mode,
            global_block=global_block,
            height=height,
            width=width,
            output_mode=output_mode,
            ref_max_size=ref_max_size,
        )
    )


def build_gen_director_plan(
    timeline: dict,
    *,
    global_task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
    load_media: bool = True,
):
    """Build DirectorPlan for generation timeline modes (lazy import avoids cycles)."""
    from .plan import (
        DirectorPlan,
        SegmentPlan,
        _load_ref_audios,
        _load_ref_videos,
        _load_refs,
        _ref_audio_metadata,
        _ref_metadata,
        _ref_video_metadata,
        _parse_run_selection,
        _resolve_export_mode,
        resolve_ref_image_size,
        resolve_segment_pass_mode,
        segment_seed_settings_from_data,
        segment_ref_audios_for_context,
        segment_refs_for_context,
    )
    from ..lib.video_io import load_timeline_segment

    global_block = timeline.get("global") or {}
    edit_mode = "segment"
    task_type = "mixed"
    task_key = MIXED_KEY

    submode = gen_submode(timeline, task_key)
    prompt = global_block.get("prompt") or global_prompt or ""
    global_refs = (
        _load_refs(global_block.get("refs") or [])
        if load_media else _ref_metadata(global_block.get("refs") or [])
    )
    shared_ref_audios = (
        (
            _load_ref_audios(global_block.get("refAudios") or [])
            if load_media else _ref_audio_metadata(global_block.get("refAudios") or [])
        )
        if edit_mode == "global"
        else []
    )

    output_block = timeline.get("output") or {}
    gen_block = timeline.get("gen") or {}
    default_fc = int(gen_block.get("defaultFrameCount") or total_frames or 81)

    segment_ranges = _gen_segment_ranges(
        timeline.get("segments") or [],
        default_frame_count=default_fc,
        task_key=task_key,
    )
    run_indices = _parse_run_selection(timeline, len(segment_ranges))

    if submode == "gen_blank":
        out_mode = "fixed"
        fw = int(output_block.get("width") or timeline.get("width") or width or 0)
        fh = int(output_block.get("height") or timeline.get("height") or height or 0)
        if fw < 32 or fh < 32:
            raise ValueError(
                "t2i / t2v / r2i / r2v require fixed output width and height "
                "(≥32, multiples of 32 for MiniMax H3). "
                "Set width and height in the generation timeline output panel."
            )
        out_w, out_h, ref_max, _ = resolve_output_dimensions(
            fw,
            fh,
            mode="fixed",
            long_edge=ref_max_size,
            fixed_width=fw,
            fixed_height=fh,
        )
    else:
        out_mode = str(output_block.get("mode") or "long_edge").lower()
        if out_mode not in ("fixed", "long_edge"):
            out_mode = "long_edge"
        src_w, src_h = _resolve_gen_image_source_dims(segment_ranges, global_block, output_block)
        out_w, out_h, ref_max, out_mode = resolve_output_dimensions(
            src_w or int(width or 832),
            src_h or int(height or 480),
            mode=out_mode,
            long_edge=int(output_block.get("longEdge") or ref_max_size or 848),
            fixed_width=int(output_block.get("width") or timeline.get("width") or width),
            fixed_height=int(output_block.get("height") or timeline.get("height") or height),
        )

    assert_minimax_canvas(out_w, out_h)

    export_mode = _resolve_export_mode(output_block)
    # Image prompt-batch (t2i/i2i/r2i) always merges to images list; video batch (t2v/i2v/r2v) respects export mode.
    if is_prompt_batch_timeline(timeline, task_key) and not is_video_batch_task_key(task_key):
        export_mode = "all"

    if submode == "gen_blank":
        source_clips = []
        # Index-only placeholder. Spatial size comes from plan.width/height;
        # t2v/r2v segments keep source_clip=None so the executor does not
        # treat these frames as a real source clip.
        source_video = torch.full(
            (max(1, len(segment_ranges)), 16, 16, 3), 0.5, dtype=torch.float32
        )
    else:
        source_clips = _build_gen_source_clips(
            segment_ranges,
            task_key=task_key,
            submode=submode,
            edit_mode=edit_mode,
            global_block=global_block,
            height=out_h,
            width=out_w,
            output_mode=out_mode,
            ref_max_size=ref_max,
            active_indices=run_indices,
        )
        attach_source_clips = is_prompt_batch_timeline(timeline, task_key) and task_key in ("i2i", "i2v")
        if attach_source_clips:
            # Placeholder timeline index only — spatial data comes from each segment's source_clip.
            source_video = torch.full((len(source_clips), 16, 16, 3), 0.5, dtype=torch.float32)
        else:
            source_video = cat_frames_variable_size(
                [clip for clip in source_clips if int(clip.shape[0]) > 0]
            )

    from .segment_continuity import (
        resolve_segment_continuity_force_prev_cache,
        resolve_segment_continuity_from_prev,
    )

    segments: list[SegmentPlan] = []
    for idx, (start, end, seg_data) in enumerate(segment_ranges):
        seg_seed_mode, seg_seed = segment_seed_settings_from_data(seg_data)
        is_selected = run_indices is None or idx in run_indices
        if edit_mode == "global":
            seg_prompt = prompt
            seg_task = task_type
            seg_refs = list(global_refs)
            use_global = True
            seg_negative = ""
        else:
            use_global = False
            if task_key == MIXED_KEY:
                seg_task_key_preview = resolve_mixed_segment_task_key(seg_data, task_key)
                seg_task = seg_task_key_preview
            else:
                seg_task = seg_data.get("taskType") or task_type
                seg_task_key_preview = resolve_task_key(seg_task)
            local_prompt = (seg_data.get("prompt") or "").strip()
            seg_prompt = local_prompt or prompt
            seg_refs = (
                _load_refs(seg_data.get("refs") or [])
                if load_media and is_selected
                else _ref_metadata(seg_data.get("refs") or [])
            )
            seg_negative = (
                (seg_data.get("negativePrompt") or "").strip()
            )

        seg_task_key = (
            resolve_mixed_segment_task_key(seg_data, task_key)
            if task_key == MIXED_KEY
            else resolve_task_key(seg_task)
        )
        source_media_identity = _segment_source_media_identity(
            seg_data if isinstance(seg_data, dict) else {},
            task_key=seg_task_key,
            edit_mode=edit_mode,
            global_block=global_block,
        )
        mixed_fl2v_refs = None
        mixed_fl2v_source = None
        mixed_i2v_source = None
        mixed_video_source = None
        source_audio_timeline = None
        if load_media and is_selected and task_key == MIXED_KEY and seg_task_key == "fl2v":
            from .fl2v_timeline import load_fl2v_segment_media

            mixed_fl2v_refs, mixed_fl2v_source = load_fl2v_segment_media(
                seg_data if isinstance(seg_data, dict) else {},
                width=out_w,
                height=out_h,
                output_mode=out_mode,
                ref_max_size=ref_max,
                frame_count=max(1, int(end) - int(start)),
            )
        elif load_media and is_selected and task_key == MIXED_KEY and seg_task_key == "i2v":
            img_ref = _resolve_gen_image_ref(
                seg_data if isinstance(seg_data, dict) else {},
                edit_mode="segment",
                global_block=global_block,
            )
            if img_ref is None:
                raise ValueError(
                    f"混合模式组 #{idx + 1} 是图生视频(i2v)，但没有源图。"
                    "请在该组上传首帧，或把类型改回文生视频。"
                )
            mixed_i2v_source = _build_i2v_source_clip(
                _load_gen_image_tensor(img_ref),
                max(1, int(end) - int(start)),
                width=out_w,
                height=out_h,
                output_mode=out_mode,
                ref_max_size=ref_max,
            )
        elif load_media and is_selected and task_key == MIXED_KEY and seg_task_key in {"v2v", "rv2v"}:
            source_timeline = _mixed_source_video_timeline(
                seg_data,
                timeline,
                target_width=out_w,
                target_height=out_h,
            )
            if source_timeline is None:
                raise ValueError(
                    f"混合模式组 #{idx + 1} 是 {seg_task_key}，但没有源视频。"
                    "请在该组上传源视频，或选择其他组类型。"
                )
            source_audio_timeline = source_timeline
            source_count = int(source_timeline["totalFrames"])
            requested_count = max(1, int(end) - int(start))
            mixed_video_source = load_timeline_segment(
                source_timeline,
                0,
                min(source_count, requested_count),
            )
            if str(seg_data.get("videoResolution") or "target") != "source":
                mixed_video_source = fit_canvas(mixed_video_source, out_w, out_h)
        if seg_task_key == "i2v" and seg_refs:
            log.info(
                "i2v segment #%d: ignoring %d reference image(s); using source video context only",
                idx + 1,
                len(seg_refs),
            )
        if mixed_fl2v_refs is not None:
            seg_refs = mixed_fl2v_refs
        else:
            seg_refs = segment_refs_for_context(seg_task_key, seg_refs)
        seg_ref_audios = []
        seg_ref_videos = []
        if edit_mode == "global":
            seg_ref_audios = segment_ref_audios_for_context(
                seg_task_key,
                list(shared_ref_audios),
            )
        else:
            local_audios = segment_ref_audios_for_context(
                seg_task_key,
                (
                    _load_ref_audios(seg_data.get("refAudios") or [])
                    if load_media and is_selected
                    else _ref_audio_metadata(seg_data.get("refAudios") or [])
                ),
            )
            seg_ref_audios = local_audios
            if seg_task_key == "r2v":
                seg_len = max(5, int(end) - int(start))
                raw_vids = list(seg_data.get("refVideos") or [])
                seg_ref_videos = (
                    _load_ref_videos(raw_vids, timeline, seg_len)
                    if load_media and is_selected else _ref_video_metadata(raw_vids)
                )
        if seg_task_key in ("r2v", "r2i") and not seg_refs and not seg_ref_videos and not seg_ref_audios:
            log.warning(
                "gen segment #%d task=%s has no reference media — will behave like "
                "t2v/t2i. Upload per-group 图片/音频/视频.",
                idx + 1,
                seg_task_key,
            )
        if task_key == MIXED_KEY:
            if seg_task_key == "i2v":
                seg_source = mixed_i2v_source
            elif seg_task_key == "fl2v":
                seg_source = mixed_fl2v_source
            elif seg_task_key in {"v2v", "rv2v"}:
                seg_source = mixed_video_source
            else:
                seg_source = None
        elif submode == "gen_blank" or seg_task_key in GEN_BLANK_KEYS:
            seg_source = None
        else:
            seg_source = source_clips[idx].clone() if idx < len(source_clips) else None
        use_source_resolution = seg_task_key in {"v2v", "rv2v"} and (
            str(seg_data.get("videoResolution") or "target") == "source"
        )
        source_range = (
            _mixed_source_range(seg_data)
            if seg_task_key in {"v2v", "rv2v"}
            else None
        )

        segments.append(
            SegmentPlan(
                index=idx,
                start_frame=start,
                end_frame=end,
                prompt=seg_prompt,
                task_type=seg_task,
                task_key=seg_task_key,
                use_global=use_global,
                refs=seg_refs,
                ref_audios=seg_ref_audios,
                ref_videos=seg_ref_videos,
                negative_prompt=seg_negative,
                source_clip=seg_source,
                source_frame_count=source_range[2] if source_range is not None else 0,
                use_source_resolution=use_source_resolution,
                source_audio_timeline=source_audio_timeline,
                source_media_identity=source_media_identity,
                continuity_from_prev=resolve_segment_continuity_from_prev(
                    seg_data if isinstance(seg_data, dict) else {},
                    segment_index=idx,
                ),
                continuity_force_prev_cache=resolve_segment_continuity_force_prev_cache(
                    seg_data if isinstance(seg_data, dict) else {},
                    segment_index=idx,
                ),
                ref_image_size=resolve_ref_image_size(
                    seg_data if isinstance(seg_data, dict) else {},
                    timeline,
                ),
                pass_mode=resolve_segment_pass_mode(
                    seg_data if isinstance(seg_data, dict) else {},
                ),
                force_resample=bool(
                    (seg_data or {}).get("forceResample", False)
                    if isinstance(seg_data, dict) else False
                ),
                seed_mode=seg_seed_mode,
                seed=seg_seed,
            )
        )

    total = int(segment_ranges[-1][1]) if segment_ranges else int(source_video.shape[0])
    raw = dict(timeline)
    raw["frameRate"] = H3_FPS
    raw["timelineMode"] = "prompt_batch"
    raw["editMode"] = "segment"
    raw_global = dict(raw.get("global") or {})
    raw_global["taskType"] = MIXED_KEY
    raw["global"] = raw_global
    src_w, src_h = _resolve_gen_image_source_dims(segment_ranges, global_block, output_block)

    from .segment_continuity import (
        resolve_continuity_keep_tail,
        resolve_continuity_mode,
        resolve_continuity_redraw,
        resolve_continuity_settings,
    )

    continuity_enabled, continuity_overlap = resolve_continuity_settings(
        timeline, segment_count=len(segments)
    )

    return DirectorPlan(
        frame_rate=float(H3_FPS),
        total_frames=total,
        width=out_w,
        height=out_h,
        ref_max_size=ref_max,
        output_mode=out_mode,
        source_width=int(src_w or out_w),
        source_height=int(src_h or out_h),
        global_task_type=task_type,
        global_task_key=task_key,
        global_prompt=prompt,
        global_refs=global_refs,
        source_video=source_video,
        segments=segments,
        edit_mode=edit_mode,
        raw=raw,
        export_mode=export_mode,
        run_indices=run_indices,
        continuity_enabled=continuity_enabled,
        continuity_overlap_frames=continuity_overlap,
        continuity_mode=resolve_continuity_mode(timeline),
        continuity_redraw=resolve_continuity_redraw(timeline),
        continuity_keep_tail=resolve_continuity_keep_tail(timeline),
        global_ref_audios=shared_ref_audios,
    )
