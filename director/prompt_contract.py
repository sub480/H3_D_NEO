"""Convert prompt-agent JSON into a MiniMax H3 Director timeline."""

from __future__ import annotations

import copy
import json
import math
import re


SCHEMA_ID = "minimax-h3-director-prompt/v1"
_TASK_ALIASES = {
    "t2v": "t2v",
    "t2va": "t2v",
    "i2v": "i2v",
    "i2va": "i2v",
    "fl2v": "fl2v",
    "fl2va": "fl2v",
    "l2v": "fl2v",
    "l2va": "fl2v",
    "r2v": "r2v",
    "ref2v": "r2v",
    "ref2va": "r2v",
    "v2v": "v2v",
    "rv2v": "rv2v",
}
_MEDIA_KEYS = (
    "refs",
    "refAudios",
    "refVideos",
    "referenceVideo",
    "genImage",
    "imageFile",
    "startImage",
    "endImage",
    "sourceVideo",
    "videoResolution",
    "audioMode",
    "refImageSize",
)
_CONTEXT_CHOICES = (5, 22, 39, 56)
_ASPECT_RATIOS = {
    "1:1": (1, 1),
    "2:3": (2, 3),
    "3:2": (3, 2),
    "3:4": (3, 4),
    "4:3": (4, 3),
    "9:16": (9, 16),
    "16:9": (16, 9),
    "21:9": (21, 9),
}
_DEFAULT_MEGAPIXELS = 0.4
_DEFAULT_CANVAS_MULTIPLE = 32
_DEFAULT_GROUP_NAME = re.compile(
    r"^(?:组|提示词组|素材组|group|prompt group|asset group)\s*\d+$",
    re.IGNORECASE,
)


def _aligned_frames(duration_seconds: int) -> int:
    frames = max(5, int(math.ceil(duration_seconds * 24.0)))
    return frames + ((5 - frames) % 17)


def _aspect_resolution(
    raw_aspect_ratio: object,
    megapixels: object,
    multiple: object,
) -> tuple[str, int, int, float, int]:
    aspect_ratio = str(raw_aspect_ratio or "").strip().split(" ", 1)[0]
    ratio = _ASPECT_RATIOS.get(aspect_ratio)
    if ratio is None:
        supported = ", ".join(_ASPECT_RATIOS)
        raise ValueError(
            f"director_prompt settings.aspectRatio must be one of {supported}."
        )
    try:
        resolved_megapixels = float(megapixels)
    except (TypeError, ValueError):
        resolved_megapixels = _DEFAULT_MEGAPIXELS
    if not math.isfinite(resolved_megapixels) or resolved_megapixels <= 0:
        resolved_megapixels = _DEFAULT_MEGAPIXELS
    try:
        resolved_multiple = max(8, int(multiple))
    except (TypeError, ValueError):
        resolved_multiple = _DEFAULT_CANVAS_MULTIPLE
    width_ratio, height_ratio = ratio
    scale = math.sqrt(
        resolved_megapixels * 1024 * 1024 / (width_ratio * height_ratio)
    )
    width_steps = math.floor(width_ratio * scale / resolved_multiple + 0.5)
    height_steps = math.floor(height_ratio * scale / resolved_multiple + 0.5)
    width = max(resolved_multiple, width_steps * resolved_multiple)
    height = max(resolved_multiple, height_steps * resolved_multiple)
    return aspect_ratio, width, height, resolved_megapixels, resolved_multiple


def _group_ui_name(raw: object) -> str:
    if not isinstance(raw, str):
        return ""
    name = raw.strip()
    if not name or _DEFAULT_GROUP_NAME.fullmatch(name):
        return ""
    return name


def _task_key(raw: object, group_number: int) -> str:
    key = str(raw or "t2v").strip().lower()
    task = _TASK_ALIASES.get(key)
    if task is None:
        supported = ", ".join(sorted(_TASK_ALIASES))
        raise ValueError(
            f"director_prompt group #{group_number}: unsupported taskType {raw!r}; "
            f"use one of {supported}."
        )
    return task


def _base_timeline(raw: str | None) -> dict:
    if not raw or not str(raw).strip():
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _base_run_selection(base: dict, group_count: int) -> tuple[bool, list[int]]:
    enabled = bool(base.get("runSelectEnabled"))
    raw = base.get("runSelection")
    if not isinstance(raw, list):
        return enabled, []
    selected = set()
    for value in raw:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < group_count:
            selected.add(index)
    return enabled, sorted(selected)


def director_prompt_to_timeline(
    director_prompt: str,
    *,
    base_timeline_data: str = "",
    default_width: int = 864,
    default_height: int = 480,
    default_ref_max_size: int = 864,
) -> str:
    """Validate a v1 prompt payload and return canonical timeline JSON.

    Existing media assignments are retained by group index. The incoming
    contract controls prompts, task types, durations, group names, aspect
    ratio, pass mode and continuity.
    """
    try:
        payload = json.loads(str(director_prompt or ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid director_prompt JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("director_prompt must be a JSON object.")
    if payload.get("schema") != SCHEMA_ID:
        raise ValueError(f"director_prompt schema must be {SCHEMA_ID!r}.")

    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("director_prompt groups must be a non-empty array.")

    base = _base_timeline(base_timeline_data)
    base_segments = base.get("segments") if isinstance(base.get("segments"), list) else []
    segments = []
    task_keys = []
    start = 0
    for index, group in enumerate(groups):
        number = index + 1
        if not isinstance(group, dict):
            raise ValueError(f"director_prompt group #{number} must be an object.")
        prompt = group.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"director_prompt group #{number} requires a non-empty prompt.")
        duration = group.get("durationSec", 5)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(f"director_prompt group #{number} durationSec must be a number.")
        if not float(duration).is_integer() or not 4 <= int(duration) <= 15:
            raise ValueError(
                f"director_prompt group #{number} durationSec must be an integer from 4 to 15."
            )
        task = _task_key(group.get("taskType"), number)
        frame_count = _aligned_frames(int(duration))
        existing = base_segments[index] if index < len(base_segments) and isinstance(base_segments[index], dict) else {}
        segment = {
            "id": str(group.get("id") or existing.get("id") or f"prompt-{number}"),
            "start": start,
            "length": frame_count,
            "frameCount": frame_count,
            "durationSec": int(duration),
            "prompt": prompt.strip(),
            "negativePrompt": str(group.get("negativePrompt") or "").strip(),
            "taskType": task,
            "continuityFromPrev": bool(group.get("continuityFromPrev", index > 0)),
            "continuityForcePrevCache": bool(group.get("continuityForcePrevCache", False)),
            "passMode": "first" if group.get("passMode") == "first" else "second",
            "uiGroupName": _group_ui_name(group.get("name")),
        }
        for key in _MEDIA_KEYS:
            if key in existing:
                segment[key] = copy.deepcopy(existing[key])
        segments.append(segment)
        task_keys.append(task)
        start += frame_count

    settings = payload.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError("director_prompt settings must be an object when provided.")
    context_frames = int(settings.get("continuityOverlapFrames", 22))
    if context_frames not in _CONTEXT_CHOICES:
        raise ValueError(
            "director_prompt settings.continuityOverlapFrames must be 5, 22, 39 or 56."
        )
    continuity_mode = str(settings.get("continuityMode") or "guide").strip().lower()
    if continuity_mode not in {"guide", "continue"}:
        raise ValueError("director_prompt settings.continuityMode must be guide or continue.")
    base_output = base.get("output") if isinstance(base.get("output"), dict) else {}
    export_mode = str(
        settings.get("exportMode") or base_output.get("exportMode") or "segments"
    ).strip().lower()
    if export_mode not in {"all", "segments"}:
        raise ValueError("director_prompt settings.exportMode must be all or segments.")

    aspect_ratio = None
    output_width = int(base_output.get("width") or default_width)
    output_height = int(base_output.get("height") or default_height)
    output_megapixels = base_output.get("megapixels", _DEFAULT_MEGAPIXELS)
    output_multiple = base_output.get("multiple", _DEFAULT_CANVAS_MULTIPLE)
    if settings.get("aspectRatio") is not None:
        (
            aspect_ratio,
            output_width,
            output_height,
            output_megapixels,
            output_multiple,
        ) = _aspect_resolution(
            settings.get("aspectRatio"), output_megapixels, output_multiple
        )

    timeline = copy.deepcopy(base)
    run_select_enabled, run_selection = _base_run_selection(base, len(segments))
    output = timeline.get("output") if isinstance(timeline.get("output"), dict) else {}
    output.update(
        {
            "mode": output.get("mode") or "fixed",
            "width": output_width,
            "height": output_height,
            "longEdge": int(output.get("longEdge") or default_ref_max_size),
            "exportMode": export_mode,
            "continuityEnabled": bool(settings.get("continuityEnabled", True)),
            "continuityOverlapFrames": context_frames,
            "continuityMode": continuity_mode,
            "continuityRedraw": float(settings.get("continuityRedraw", 0.10)),
            "continuityKeepTail": bool(settings.get("continuityKeepTail", False)),
            "audioMode": str(settings.get("audioMode") or output.get("audioMode") or "generate"),
        }
    )
    if aspect_ratio is not None:
        output["aspectRatio"] = aspect_ratio
        output["megapixels"] = output_megapixels
        output["multiple"] = output_multiple
    global_block = timeline.get("global") if isinstance(timeline.get("global"), dict) else {}
    global_block["taskType"] = "mixed"
    global_block["prompt"] = ""
    global_block.setdefault("refs", [])
    global_block.setdefault("refAudios", [])
    global_block.setdefault("refVideos", [])

    timeline.update(
        {
            "version": 5,
            "editMode": "segment",
            "timelineMode": "prompt_batch",
            "frameRate": 24,
            "totalFrames": start,
            "width": int(output["width"]),
            "height": int(output["height"]),
            "refMaxSize": int(output["longEdge"]),
            "global": global_block,
            "output": output,
            "segments": segments,
            "runSelectEnabled": run_select_enabled,
            "runSelection": run_selection,
        }
    )
    return json.dumps(timeline, ensure_ascii=False, separators=(",", ":"))
