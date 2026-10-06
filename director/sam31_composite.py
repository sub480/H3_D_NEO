"""SAM source-latent edit validation and final pixel-space paste.

Independent alpha blending implementation. Related public workflows:
ComfyUI ImageCompositeMasked and drozbay/MaskVidExperiments Subject Uncrop.
No third-party node installation or GPL implementation is copied here.
"""

from __future__ import annotations

import copy
import json
import zipfile

from .sam31_config import normalize_group

COMPOSITE_REVISION = 2


def execution_required(raw):
    """External mask/source files must be revalidated on every enabled run."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return False
    if not isinstance(raw, dict):
        return False
    return any(
        isinstance(group, dict)
        and isinstance(group.get("sam31"), dict)
        and normalize_group(group["sam31"])["composite"]["enabled"]
        for group in raw.get("segments", [])
    )


def enabled(seg):
    group = getattr(seg, "sam31_group", None)
    return isinstance(group, dict) and normalize_group(group.get("sam31"))["composite"]["enabled"]


def prepare(plan, seg):
    """Fail closed before any H3 sampling or cached final output can be used."""
    seg.sam31_composite = None
    if not isinstance(getattr(seg, "sam31_group", None), dict):
        groups = (getattr(plan, "raw", None) or {}).get("segments") or []
        index = getattr(seg, "timeline_index", getattr(seg, "index", -1))
        if 0 <= index < len(groups) and isinstance(groups[index], dict):
            seg.sam31_group = groups[index]
    if not enabled(seg):
        return
    from .sam31 import prepare_request, read_metadata, read_packed_frame, cache_path, file_digest
    config = getattr(plan, "sam31", None) or {}
    if not config.get("enabled"):
        raise ValueError("SAM3.1 蒙版合成：请先启用 SAM3.1 分析工具。")
    if seg.task_key not in {"v2v", "rv2v"}:
        raise ValueError("SAM3.1 蒙版合成仅支持有源视频的 V2V / RV2V 组。")
    if getattr(plan, "continuity_enabled", False):
        raise ValueError("SAM3.1 蒙版合成：请关闭段间引导，以保持源视频逐帧对应。")
    group = seg.sam31_group
    state = normalize_group(group.get("sam31"))
    result = state["result"] or {}
    if result.get("action") not in {"track", "track_selected"} or not state["selected"]:
        raise ValueError("SAM3.1 蒙版合成：先完成视频跟踪并勾选至少一个对象；首帧分割不能替代逐帧蒙版。")
    spec = prepare_request({
        "timeline": {"frameRate": plan.frame_rate, "segments": [group]},
        "group_id": str(group.get("id") or ""), "config": config, "action": "track",
    })
    meta = read_metadata(result.get("key"))
    if (meta["identity"]["base"] != spec["identity"]["base"]
            or meta["identity"]["action"] != result.get("action")):
        raise ValueError("SAM3.1 蒙版合成：跟踪结果已过期，请按当前素材、描述和分析参数重新跟踪。")
    source = meta["identity"]["base"]["source"]
    if source["kind"] != "video" or meta["frames"] != len(source["entries"]):
        raise ValueError("SAM3.1 蒙版合成：缓存不是完整的源视频跟踪结果。")
    count = int(seg.frame_count)
    if count != meta["frames"] or int(seg.source_frame_count) != count:
        raise ValueError("SAM3.1 蒙版合成：组帧数、源范围与蒙版帧数必须一致；请使用 17k+5 帧范围后重新跟踪。")
    if any(index >= len(meta["objects"]) for index in state["selected"]):
        raise ValueError("SAM3.1 蒙版合成：所选对象不属于当前跟踪缓存。")
    # Check every packed frame without materializing a whole float mask video.
    for index in range(count):
        read_packed_frame(meta["key"], index)
    if seg.source_clip is None or int(seg.source_clip.shape[0]) != count:
        raise ValueError("SAM3.1 蒙版合成：源视频加载范围与跟踪范围不一致。")
    from .audio_export import resolve_audio_mode
    seg.sam31_composite = {
        "revision": COMPOSITE_REVISION, "key": meta["key"], "meta": meta,
        "selected": state["selected"], "settings": state["composite"],
        "canvas": [int(seg.source_clip.shape[2]), int(seg.source_clip.shape[1])],
        "fit": "stretch" if seg.use_source_resolution else seg.video_fit,
        "artifact": file_digest(cache_path(meta["key"])),
        "audio_mode": resolve_audio_mode(plan),
    }


def fingerprint(seg, *, first_pass=False):
    context = getattr(seg, "sam31_composite", None)
    if context is None:
        return {}
    base = context["meta"]["identity"]["base"]
    # Generation geometry belongs to first pass; feather/blend belong to paste.
    if first_pass:
        source = {key: value for key, value in base["source"].items() if key != "size"}
        return {"sam31_source": {"revision": COMPOSITE_REVISION, "source": source,
                                 "canvas": context["canvas"], "fit": context["fit"],
                                 "key": context["key"], "artifact": context["artifact"],
                                 "selected": context["selected"], "grow": context["settings"]["grow"],
                                 "audio_mode": context.get("audio_mode", "generate")}}
    return {"sam31_composite": {
        "revision": COMPOSITE_REVISION, "key": context["key"],
        "artifact": context["artifact"],
        "selected": context["selected"], "settings": context["settings"],
        "canvas": context["canvas"], "fit": context["fit"],
        "audio_mode": context.get("audio_mode", "generate"),
    }}


def _fit(frame, width, height, mode, *, mask=False):
    import torch.nn.functional as F
    from ..lib.image_prep import fit_frames_to_canvas, fit_contain
    if mode == "stretch":
        return F.interpolate(frame.movedim(-1, 1), size=(height, width),
                             mode="bilinear", align_corners=False).movedim(1, -1)
    if mask and mode == "contain":
        return fit_contain(frame, width, height, fill=0.0)
    return fit_frames_to_canvas(frame, width, height, mode)


def source_frames(seg):
    """Decode strictly with the same transform later used for its mask.

    One native-resolution frame at a time keeps transient decoding bounded;
    the final source tensor is the tensor H3 already needs as its reference.
    """
    import torch
    from .sam31 import decode_source, verify_assets
    context = seg.sam31_composite
    base = context["meta"]["identity"]["base"]
    verify_assets(base)
    source = base["source"]
    width, height = context["canvas"]
    output = torch.empty(len(source["entries"]), height, width, 3)
    for index, (clip, _) in enumerate(source["entries"]):
        descriptor = copy.deepcopy(source)
        asset = source["assets"][clip]
        descriptor["size"] = [asset["width"], asset["height"]]
        frame = decode_source(descriptor, [index])
        output[index].copy_(_fit(frame, width, height, context["fit"])[0])
    verify_assets(base)
    return output


def apply(generated, source, seg, *, transforms=()):
    """Union selected objects and blend one frame at a time. No time resize."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .sam31 import cache_path, _metadata, _packed, verify_assets, file_digest
    context = seg.sam31_composite
    base = context["meta"]["identity"]["base"]
    verify_assets(base)
    if file_digest(cache_path(context["key"])) != context["artifact"]:
        raise ValueError("SAM3.1 蒙版合成：运行期间蒙版缓存内容发生变化。")
    if generated.shape != source.shape or generated.shape[0] != context["meta"]["frames"]:
        raise ValueError("SAM3.1 蒙版合成：生成结果与源视频帧数或画幅不同，不能自动拉伸时间/重复蒙版。")
    settings = context["settings"]
    output = torch.empty_like(generated, device="cpu", dtype=torch.float32)
    width, height = context["canvas"]
    with zipfile.ZipFile(cache_path(context["key"])) as archive:
        meta = _metadata(archive, context["key"])
        if meta != context["meta"]:
            raise ValueError("SAM3.1 蒙版合成：运行期间跟踪缓存发生变化。")
        for index in range(meta["frames"]):
            packed = _packed(archive, meta, index)[context["selected"]]
            union = np.bitwise_or.reduce(packed, axis=0)
            binary = np.unpackbits(union, axis=-1, bitorder="little")[..., :meta["mask_width"]]
            alpha = torch.from_numpy(binary.copy()).float()[None, ..., None].expand(-1, -1, -1, 3)
            # Tracker raster is square: restore display geometry first.
            asset = base["source"]["assets"][base["source"]["entries"][index][0]]
            alpha = _fit(alpha, asset["width"], asset["height"], "stretch", mask=True)
            alpha = _fit(alpha, width, height, context["fit"], mask=True)
            for op, w, h in transforms:
                alpha = _fit(alpha, w, h, op, mask=True)
            if tuple(alpha.shape[1:3]) != tuple(source.shape[1:3]):
                raise ValueError("SAM3.1 蒙版合成：蒙版空间变换与源视频不同。")
            alpha = alpha[..., :1].movedim(-1, 1).clamp(0, 1)
            grow, feather = settings["grow"], settings["feather"]
            if grow:
                alpha = F.max_pool2d(alpha, 2 * grow + 1, stride=1, padding=grow)
            if feather:
                alpha = F.avg_pool2d(F.pad(alpha, (feather,) * 4, mode="replicate"),
                                     2 * feather + 1, stride=1)
            alpha = alpha[0].movedim(0, -1) * settings["blend"]
            original = source[index].detach().cpu().float()
            edited = generated[index].detach().cpu().float()
            output[index] = original * (1 - alpha) + edited * alpha
    verify_assets(base)
    if file_digest(cache_path(context["key"])) != context["artifact"]:
        raise ValueError("SAM3.1 蒙版合成：运行期间蒙版缓存内容发生变化。")
    return output
