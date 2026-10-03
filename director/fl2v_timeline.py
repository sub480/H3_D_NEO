"""First/last-frame helpers shared by mixed Director groups."""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..lib.image_prep import fit_canvas, fit_video_long_edge
from .frame_align import minimax_align_frame_count

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.fl2v")

MIN_FL2V_FRAMES = 5
DEFAULT_FL2V_DURATION_SEC = 5.0
# Match node ``negative_prompt`` default in director_common / director.py.
DEFAULT_FL2V_NEGATIVE = "bad video"


def _duration_to_minimax_frames(seconds: float, fps: float) -> int:
    frame_count = max(MIN_FL2V_FRAMES, int(round(max(0.1, float(seconds)) * float(fps))))
    return minimax_align_frame_count(frame_count)


def _image_ref_from_raw(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        path = raw.strip()
        return {"imageFile": path, "imageB64": "", "width": 0, "height": 0} if path else None
    if not isinstance(raw, dict):
        return None
    image_file = str(raw.get("imageFile") or "").strip()
    image_b64 = str(raw.get("imageB64") or "").strip()
    if not image_file and not image_b64:
        return None
    try:
        width, height = int(raw.get("width") or 0), int(raw.get("height") or 0)
    except (TypeError, ValueError, OverflowError):
        width, height = 0, 0
    return {
        "imageFile": image_file,
        "imageB64": image_b64,
        "width": max(0, width),
        "height": max(0, height),
    }


# Hard locks for every fl2v shot (re-applied after PE). Community cue words:
# MiniMax H3 locks first/last via MiniMaxH3ImageToVideo keyframe latents.
# Prompt text reinforces continuity; avoid Bernini image0/image1 tokens.
FLF_PROMPT_PREFIX = (
    "完全保持首尾帧。"
    "视频第一帧必须与给定首帧画面一致，最后一帧必须与给定尾帧画面一致；"
    "首尾帧是硬锁定关键帧，不是软参考，禁止改动首尾画面、主体外观与机位。"
)
I2V_PROMPT_PREFIX = (
    "完全保持首帧。"
    "视频第一帧必须与给定首帧画面一致；首帧是硬锁定关键帧，不是软参考，禁止改动首帧画面与主体外观。"
)
L2V_PROMPT_PREFIX = (
    "完全保持尾帧。"
    "视频最后一帧必须与给定尾帧画面一致；尾帧是硬锁定关键帧，不是软参考，禁止改动尾帧画面与主体外观。"
)
FLF_PROMPT_SUFFIX = (
    "完全保持首尾帧：开头锁定首帧，结尾锁定尾帧。"
)
I2V_PROMPT_SUFFIX = (
    "完全保持首帧：开头锁定首帧。"
)
L2V_PROMPT_SUFFIX = (
    "完全保持尾帧：结尾锁定尾帧。"
)


def _strip_fl2v_wraps(text: str) -> str:
    changed = True
    while changed and text:
        changed = False
        for p in (FLF_PROMPT_PREFIX, I2V_PROMPT_PREFIX, L2V_PROMPT_PREFIX):
            if text.startswith(p):
                text = text[len(p) :].strip()
                changed = True
        for s in (FLF_PROMPT_SUFFIX, I2V_PROMPT_SUFFIX, L2V_PROMPT_SUFFIX):
            if text.endswith(s):
                text = text[: -len(s)].strip()
                changed = True
    return text


def _sanitize_fl2v_body(text: str) -> str:
    """Rewrite soft「参考」wording that weakens first/last-frame locking."""
    if not text:
        return text
    replacements = (
        ("reference image0", "image0"),
        ("reference image1", "image1"),
        ("Reference image0", "image0"),
        ("Reference image1", "image1"),
        ("参考图 image0", "image0"),
        ("参考图 image1", "image1"),
        ("参考图image0", "image0"),
        ("参考图image1", "image1"),
        ("参考 image0", "image0"),
        ("参考 image1", "image1"),
        ("参考image0", "image0"),
        ("参考image1", "image1"),
        ("以image0为参考", "完全按照image0"),
        ("以image1为参考", "完全保持image1"),
        ("把image0当作参考", "完全按照image0"),
        ("把image1当作参考", "完全保持image1"),
        ("image0的构图", "image0的画面"),
        ("image1的构图", "image1的画面"),
        ("首尾构图", "首尾画面"),
        ("首帧构图", "首帧画面"),
    )
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    # Drop duplicated lock lines already present in the body (prefix/suffix will re-add).
    for marker in (
        "完全保持首尾帧。",
        "完全保持首帧。",
        "视频开始完全按照image0的画面，不修改，视频结束完全保持image1的画面。",
        "视频开始完全按照image0的画面，不修改，视频结束完全保持image1。",
        "视频开始完全按照image0的构图，不修改，视频结束完全保持image1。",
        "视频开始完全按照image0的画面，不修改。",
        "视频开始完全按照image0的构图，不修改。",
        "视频结束完全保持image1的画面。",
        "视频结束完全保持image1。",
    ):
        if out.startswith(marker):
            out = out[len(marker) :].strip()
    return out.strip()


def fl2v_prompt_body_only(prompt: str) -> str:
    """Strip hard-lock wraps / PE duplicates; keep only the motion body for UI storage."""
    text = _sanitize_fl2v_body(_strip_fl2v_wraps((prompt or "").strip()))
    if text.startswith("中间过程："):
        text = text[len("中间过程：") :].strip()
    return text


def reinforce_fl2v_prompt(
    prompt: str,
    *,
    has_end_frame: bool,
    has_start_frame: bool = True,
) -> str:
    """Ensure first/last-frame hard constraints wrap the (possibly PE-enhanced) prompt.

    Hard locks are placed *before* the motion body so they survive token truncation.
    Supports start-only, end-only, start+end, and neither (plain text-to-video).
    """
    text = fl2v_prompt_body_only(prompt)
    if not has_start_frame and not has_end_frame:
        return text
    if has_start_frame and has_end_frame:
        prefix, suffix = FLF_PROMPT_PREFIX, FLF_PROMPT_SUFFIX
    elif has_end_frame:
        prefix, suffix = L2V_PROMPT_PREFIX, L2V_PROMPT_SUFFIX
    else:
        prefix, suffix = I2V_PROMPT_PREFIX, I2V_PROMPT_SUFFIX
    if text:
        return f"{prefix}{suffix}中间过程：{text}"
    return f"{prefix}{suffix}"


def _load_image_ref(ref: dict) -> torch.Tensor:
    from .gen_timeline import _load_gen_image_tensor

    return _load_gen_image_tensor(ref)


def _fit_image(
    img: torch.Tensor,
    *,
    width: int,
    height: int,
    output_mode: str,
    ref_max_size: int,
) -> torch.Tensor:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if output_mode == "fixed":
        return fit_canvas(img, width, height)
    return fit_video_long_edge(img, ref_max_size)


def _build_fl2v_endpoint_source(
    start_img: torch.Tensor,
    end_img: torch.Tensor | None,
    frame_count: int,
) -> torch.Tensor:
    """Build source video with temporal endpoints for Bernini context encoding.

    Prompt/refs alone are soft. Encoding a full-length source whose first frame is
    image0 and last frame is image1 (middle held at image0) gives the model a much
    stronger first/last-frame signal — especially on long later shots.
    """
    if start_img.ndim == 3:
        start_img = start_img.unsqueeze(0)
    n = minimax_align_frame_count(max(MIN_FL2V_FRAMES, int(frame_count)))
    # Hold start through the body (i2v-like), snap last frame to end when present.
    clip = start_img[:1].expand(n, -1, -1, -1).contiguous().clone()
    if end_img is not None:
        if end_img.ndim == 3:
            end_img = end_img.unsqueeze(0)
        # long_edge fit never upscales: different source aspects → different H×W
        # (e.g. start 848×640, end 688×512). Match end to the start canvas.
        h, w = int(clip.shape[1]), int(clip.shape[2])
        if int(end_img.shape[1]) != h or int(end_img.shape[2]) != w:
            log.info(
                "fl2v: fitting end frame %dx%d → %dx%d to match start canvas",
                int(end_img.shape[2]),
                int(end_img.shape[1]),
                w,
                h,
            )
            end_img = fit_canvas(end_img[:1], w, h)
        clip[-1:] = end_img[:1].to(device=clip.device, dtype=clip.dtype)
    return clip


def load_fl2v_segment_media(
    seg_data: dict | None,
    *,
    width: int,
    height: int,
    output_mode: str,
    ref_max_size: int,
    frame_count: int,
) -> tuple[list, torch.Tensor | None]:
    """Load start/end images from a mixed-mode (or batch) segment into fl2v refs.

    Returns (refs, source_clip). refs use index 0=start, 1=end. source_clip is
    only built when both endpoints exist (same as the dedicated fl2v planner).
    """
    from .plan import SegmentRef

    raw = seg_data if isinstance(seg_data, dict) else {}
    start = _image_ref_from_raw(raw.get("startImage"))
    end = _image_ref_from_raw(raw.get("endImage"))
    start_img = None
    if start is not None:
        start_img = _fit_image(
            _load_image_ref(start),
            width=width,
            height=height,
            output_mode=output_mode,
            ref_max_size=ref_max_size,
        )
    end_img = None
    if end is not None:
        end_img = _fit_image(
            _load_image_ref(end),
            width=width,
            height=height,
            output_mode=output_mode,
            ref_max_size=ref_max_size,
        )
    start_img, end_img = _unify_fl2v_pair_canvas(start_img, end_img)
    refs: list[SegmentRef] = []
    if start_img is not None:
        refs.append(SegmentRef(index=0, tensor=start_img[:1].clone()))
    if end_img is not None:
        refs.append(SegmentRef(index=1, tensor=end_img[:1].clone()))
    source_clip = None
    if start_img is not None and end_img is not None:
        source_clip = _build_fl2v_endpoint_source(start_img, end_img, frame_count)
    return refs, source_clip


def _unify_fl2v_pair_canvas(
    start_img: torch.Tensor | None,
    end_img: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Ensure image0/image1 share one HxW after independent long_edge fits."""
    if start_img is not None and start_img.ndim == 3:
        start_img = start_img.unsqueeze(0)
    if end_img is not None and end_img.ndim == 3:
        end_img = end_img.unsqueeze(0)
    if start_img is None or end_img is None:
        return start_img, end_img
    h, w = int(start_img.shape[1]), int(start_img.shape[2])
    if int(end_img.shape[1]) != h or int(end_img.shape[2]) != w:
        end_img = fit_canvas(end_img[:1], w, h)
    return start_img, end_img
