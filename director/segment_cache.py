"""Disk cache for MiniMax H3 Director segment decode outputs (partial re-run + merge).

Cache is best-effort: write failures (cloud RO mounts, same-name overwrite
blocks, full disks) must never abort the main generation run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable

import torch

from .h3_motion_context import (
    CONTINUITY_PIPELINE_ID,
    CONTINUITY_TASK_KEYS,
    trim_context_prefix,
    trim_export_tail,
)
from .plan import (
    continuity_predecessor_index,
    DirectorPlan,
    SegmentPlan,
    normalize_segment_seed_mode,
    resolve_lora_trigger_words,
    resolve_ref_image_size,
    resolve_segment_seed,
)
from .output_layout import SEGMENT_CACHE_DIR_NAME, h3_output_path

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache")

SOURCE_VIDEO_FP_KEY = "source_video"
_PROMPT_REF_TAG_RE = {
    "image": re.compile(r"<\s*picture\s+(\d+)\s*>", re.IGNORECASE),
    "video": re.compile(r"<\s*video\s+(\d+)\s*>", re.IGNORECASE),
    "audio": re.compile(r"<\s*audio\s+(\d+)\s*>", re.IGNORECASE),
}


def _prompt_ref_slots(prompt: str, kind: str) -> set[int]:
    """Return 0-based reference slots explicitly used by the prompt."""
    pattern = _PROMPT_REF_TAG_RE[kind]
    return {int(match) - 1 for match in pattern.findall(str(prompt or "")) if int(match) > 0}


def source_video_identity(plan: DirectorPlan) -> list[str]:
    """Stable source-clip identity: relative path + size + mtime (overwrite-safe)."""
    cached = getattr(plan, "_source_video_identity_cache", None)
    if isinstance(cached, list):
        return cached
    from ..lib.video_io import resolve_video_path, video_clips_from_timeline

    clips = video_clips_from_timeline((plan.raw or {}) if plan is not None else {})
    tokens: list[str] = []
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        rel = str(clip.get("videoFile") or clip.get("fileName") or "").strip().replace("\\", "/")
        if not rel:
            continue
        try:
            path = resolve_video_path(clip)
            st = os.stat(path)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
            tokens.append(f"{rel}:{st.st_size}:{mtime_ns}")
        except Exception:
            tokens.append(f"{rel}:missing")
    setattr(plan, "_source_video_identity_cache", tokens)
    return tokens


def source_identity_changed(stored: Any, expected: dict[str, Any]) -> bool:
    """True when the current plan has a source video that does not match cache meta.

    Gen timelines (no source clips) never count as a source change, so stale
    fill/continuity still work after pipeline-only fingerprint churn.
    """
    exp = expected.get(SOURCE_VIDEO_FP_KEY) or []
    if not exp:
        return False
    if not isinstance(stored, dict) or SOURCE_VIDEO_FP_KEY not in stored:
        return True
    return stored.get(SOURCE_VIDEO_FP_KEY) != exp


def _reject_source_stale(
    stored: Any,
    expected: dict[str, Any],
    *,
    seg_index: int,
    quiet: bool = False,
) -> bool:
    if not source_identity_changed(stored, expected):
        return False
    if not quiet:
        log.info(
            "Segment %d cache is from a different source video; ignoring stale render.",
            seg_index + 1,
        )
    return True


def _cache_root(node_id: str) -> Path | None:
    try:
        root = h3_output_path(SEGMENT_CACHE_DIR_NAME, str(node_id))
        root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError as exc:
        log.warning("Segment cache dir unavailable (%s); cache disabled for this run.", exc)
        return None


def _previous_run_segment(seg: SegmentPlan, plan: DirectorPlan) -> SegmentPlan | None:
    segments = list(getattr(plan, "segments", None) or [])
    previous_index = continuity_predecessor_index(plan, seg)
    if previous_index is None:
        return None
    return segments[previous_index] if 0 <= previous_index < len(segments) else None


def _segment_uses_motion_context(seg: SegmentPlan, plan: DirectorPlan) -> bool:
    """True when this segment's first-pass sample pins the previous AV tail.

    Segment 0 never pins. Adding a later group or toggling「段间引导」must not
    bust group 1's first-pass cache. Inline of ``is_continuity_active`` — do not
    import ``segment_continuity`` here (circular).
    """
    if not bool(getattr(plan, "continuity_enabled", False)):
        return False
    if _previous_run_segment(seg, plan) is None:
        return False
    if not bool(getattr(seg, "continuity_from_prev", True)):
        return False
    task = str(getattr(seg, "task_key", "") or "")
    return task in CONTINUITY_TASK_KEYS


def _segment_identity_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Identity that affects first-pass sampling (no Refine settings)."""
    prompt = str(getattr(seg, "prompt", "") or "")
    task_key = str(getattr(seg, "task_key", "") or "")
    used_images = _prompt_ref_slots(prompt, "image")
    used_videos = _prompt_ref_slots(prompt, "video")
    used_audios = _prompt_ref_slots(prompt, "audio")
    # Prompt-batch FL2V loads start/end keyframes through
    # ``load_fl2v_segment_media``.  Those temporary SegmentRef objects carry
    # the pixel tensors but intentionally have no ``image_file``; treating
    # them as ordinary ``<Picture N>`` references stamps values such as
    # ``img0:`` into the execution cache.  The cache-status path does not
    # decode keyframes and therefore sees no ordinary refs, producing a false
    # ``参考图片`` mismatch after every successful run.  FL2V endpoint
    # identity is already covered by ``source_media`` below.
    ref_files = sorted(
        f"img{ref.index}:{(getattr(ref, 'image_file', '') or '')}"
        for ref in seg.refs
        if int(getattr(ref, "index", -1)) in used_images
        and task_key != "fl2v"
    )
    ref_audio_files = sorted(
        f"aud{getattr(a, 'index', i)}:{(getattr(a, 'audio_file', '') or '')}"
        for i, a in enumerate(getattr(seg, "ref_audios", None) or [])
        if int(getattr(a, "index", i)) in used_audios
    )
    ref_video_files = sorted(
        f"vid{getattr(v, 'index', i)}:{(getattr(v, 'video_file', '') or '')}"
        for i, v in enumerate(getattr(seg, "ref_videos", None) or [])
        if int(getattr(v, "index", i)) in used_videos
    )
    ref_video_file = (
        seg.reference_video_meta.get("videoFile")
        or seg.reference_video_meta.get("fileName")
        or ""
    ).strip()
    uses_mc = _segment_uses_motion_context(seg, plan)
    continuity_mode = str(getattr(plan, "continuity_mode", "guide") or "guide")
    if uses_mc and continuity_mode == "continue":
        from .h3_latent_continue import CONTINUE_PIPELINE_ID, clamp_seam_min_mask

        continuity_pipeline = CONTINUE_PIPELINE_ID
        continuity_redraw = round(
            clamp_seam_min_mask(getattr(plan, "continuity_redraw", 0.10)), 2
        )
    else:
        continuity_pipeline = CONTINUITY_PIPELINE_ID
        continuity_redraw = 0
    payload = {
        "index": seg.index,
        "start": seg.start_frame,
        "end": seg.end_frame,
        "prompt": seg.prompt,
        "lora_trigger_words": resolve_lora_trigger_words(plan, seg.task_key),
        "negative": seg.negative_prompt,
        "task_key": task_key,
        "width": plan.width,
        "height": plan.height,
        "frame_rate": float(getattr(plan, "frame_rate", 24) or 24),
        "output_mode": plan.output_mode,
        "ref_max": plan.ref_max_size,
        "ref_image_size": resolve_ref_image_size(seg, plan),
        "refs": ref_files,
        "ref_audios": ref_audio_files,
        "ref_videos": ref_video_files,
        "ref_video": ref_video_file,
        "ref_video_start": seg.reference_video_start_frame,
        "source_media": list(getattr(seg, "source_media_identity", ()) or ()),
        "video_edit_export": (
            "source_range_v1" if seg.task_key in {"v2v", "rv2v"} else ""
        ),
        SOURCE_VIDEO_FP_KEY: source_video_identity(plan),
        "continuity": uses_mc,
        "continuity_overlap": (
            int(plan.continuity_overlap_frames or 0) if uses_mc else 0
        ),
        "continuity_from_prev": uses_mc,
        "continuity_mode": continuity_mode if uses_mc else "off",
        "continuity_redraw": continuity_redraw,
        "continuity_pipeline": continuity_pipeline,
    }
    if uses_mc:
        payload["continuity_keep_tail"] = bool(
            getattr(plan, "continuity_keep_tail", False)
        )
    return payload


def first_pass_cache_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Exact-match key for first-pass AV latent. Refine knobs are excluded."""
    fp = _segment_identity_fingerprint(seg, plan)
    fp.update({
        "kind": "first_pass",
        "seed": resolve_segment_seed(seg, getattr(plan, "sample_seed", 0)),
        "cfg": round(float(getattr(plan, "sample_cfg", 1.0) or 1.0), 6),
        "sampler": str(getattr(plan, "sample_sampler", "") or ""),
        "shift_video": round(float(getattr(plan, "sample_shift_video", 12.0) or 12.0), 6),
        "shift_audio": round(float(getattr(plan, "sample_shift_audio", 3.0) or 3.0), 6),
        "steps": int(getattr(plan, "sample_steps", 25) or 25),
        "scheduler": str(getattr(plan, "sample_scheduler", "") or ""),
    })
    if fp.get("continuity"):
        previous = _previous_run_segment(seg, plan)
        if previous is not None:
            fp["continuity_predecessor"] = first_pass_cache_fingerprint(previous, plan)
    return fp


def segment_cache_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Stable identity for a segment — cache invalidates when edit params change."""
    fp = _segment_identity_fingerprint(seg, plan)
    if fp.get("continuity"):
        previous = _previous_run_segment(seg, plan)
        if previous is not None:
            fp["continuity_predecessor"] = first_pass_cache_fingerprint(previous, plan)
    from .refine_pack import refine_fingerprint

    fp.update(refine_fingerprint(plan))
    return fp


def _safe_unlink(path: Path) -> bool:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        return True
    except OSError:
        return False


def _recycle_file(path: Path) -> bool:
    """Move a user-requested cache deletion to the OS recycle bin."""
    try:
        from send2trash import send2trash
    except ImportError:
        send2trash = None
    try:
        if path.is_file() or path.is_symlink():
            if send2trash is not None:
                send2trash(str(path))
            elif os.name == "nt":
                import ctypes
                from ctypes import wintypes

                class _SHFILEOPSTRUCTW(ctypes.Structure):
                    _fields_ = [
                        ("hwnd", wintypes.HWND),
                        ("wFunc", wintypes.UINT),
                        ("pFrom", wintypes.LPCWSTR),
                        ("pTo", wintypes.LPCWSTR),
                        ("fFlags", wintypes.WORD),
                        ("fAnyOperationsAborted", wintypes.BOOL),
                        ("hNameMappings", wintypes.LPVOID),
                        ("lpszProgressTitle", wintypes.LPCWSTR),
                    ]

                operation = _SHFILEOPSTRUCTW(
                    None, 3, str(path) + "\0\0", None, 0x0040 | 0x0010, False, None, None
                )
                result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
                if result != 0 or operation.fAnyOperationsAborted:
                    return False
            else:
                raise RuntimeError("清理缓存需要安装 send2trash，以便将文件移入回收站。")
        return True
    except OSError:
        return False


def _atomic_publish(tmp: Path, dest: Path) -> None:
    """Move ``tmp`` 鈫?``dest``, tolerating clouds that block same-name overwrite."""
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    # Some cloud mounts reject overwrite of an existing name 鈥?remove then rename.
    _safe_unlink(dest)
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    try:
        tmp.rename(dest)
        return
    except OSError:
        # Last resort: keep the unique temp as the published file name is blocked.
        # Caller may still fail if even create-new is denied.
        raise


def _write_via_temp(dest: Path, write_fn: Callable[[Path], None]) -> None:
    """Write to a unique temp name in the same folder, then publish to ``dest``."""
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_fn(tmp)
        _atomic_publish(tmp, dest)
    finally:
        _safe_unlink(tmp)


def _audio_payload_to_cpu(audio: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize export AUDIO dict for disk cache (waveform on CPU)."""
    if not isinstance(audio, dict):
        return None
    wave = audio.get("waveform")
    if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
        return None
    sr = int(audio.get("sample_rate") or 0) or 32000
    return {
        "waveform": wave.detach().cpu().contiguous(),
        "sample_rate": sr,
    }


def _frames_to_disk(tensor: torch.Tensor) -> torch.Tensor:
    """Store pixel frames as uint8 [0,255]. Export is 8-bit anyway; float32 is 4× larger."""
    x = tensor.detach().cpu()
    if x.dtype == torch.uint8:
        return x.contiguous()
    return x.float().clamp(0, 1).mul(255).round().clamp(0, 255).to(torch.uint8).contiguous()


def _frames_from_disk(loaded: Any) -> torch.Tensor | None:
    """Restore uint8 cache to float32 [0,1]; pass through legacy float caches."""
    if not isinstance(loaded, torch.Tensor):
        return None
    if loaded.dtype == torch.uint8:
        return loaded.float().div(255.0)
    return loaded.float()


def save_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    tensor: torch.Tensor,
    *,
    av_latent: dict | None = None,
    handoff: dict[str, Any] | None = None,
    audio: dict[str, Any] | None = None,
    replace_audio: bool = True,
) -> None:
    """Persist a segment tensor (+ optional AV latent / export audio). Never raises.

    ``replace_audio``:
      - True (default): write ``audio`` when present, otherwise delete stale audio.pt
        (fresh sample with mute/empty decode).
      - False: write ``audio`` when present, otherwise **keep** existing audio.pt
        (phase-align trim re-save must not wipe a prior audio cache).
    """
    if not node_id:
        return
    root = _cache_root(node_id)
    if root is None:
        return
    fp = segment_cache_fingerprint(seg, plan)
    idx = seg.index
    pt_path = root / f"seg_{idx:04d}.pt"
    meta_path = root / f"seg_{idx:04d}.meta.json"
    latent_path = root / f"seg_{idx:04d}.av.pt"
    handoff_path = root / f"seg_{idx:04d}.handoff.json"
    audio_path = root / f"seg_{idx:04d}.audio.pt"
    try:
        payload = _frames_to_disk(tensor)
        _write_via_temp(pt_path, lambda p: torch.save(payload, p))
        text = json.dumps(fp, ensure_ascii=False, sort_keys=True)
        _write_via_temp(
            meta_path,
            lambda p: p.write_text(text, encoding="utf-8"),
        )
        if av_latent is not None and isinstance(av_latent, dict) and "samples" in av_latent:
            cpu_latent = _av_latent_to_cpu(av_latent)
            _write_via_temp(latent_path, lambda p: torch.save(cpu_latent, p))
        if handoff:
            _write_via_temp(
                handoff_path,
                lambda p: p.write_text(
                    json.dumps(handoff, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                ),
            )
        audio_cpu = _audio_payload_to_cpu(audio)
        if audio_cpu is not None:
            _write_via_temp(audio_path, lambda p: torch.save(audio_cpu, p))
        elif replace_audio:
            # Fresh sample with no waveform — drop stale audio from an older run.
            _safe_unlink(audio_path)
        log.debug(
            "Cached segment %d for node %s (%d frames%s%s)",
            idx + 1,
            node_id,
            int(tensor.shape[0]),
            ", +av_latent" if av_latent is not None else "",
            ", +audio" if audio_cpu is not None else (
                ", keep-audio" if not replace_audio else ""
            ),
        )
    except Exception as exc:
        # Xiangong / similar: RO mount or same-name write → skip cache, keep run alive.
        log.warning(
            "Segment %d cache write skipped (%s). Generation continues without disk cache.",
            idx + 1,
            exc,
        )
        for stray in root.glob(f".seg_{idx:04d}.*"):
            _safe_unlink(stray)


def _fingerprint_diff_keys(stored: Any, expected: dict[str, Any]) -> list[str]:
    if not isinstance(stored, dict):
        return ["<invalid-meta>"]
    keys = sorted(set(stored) | set(expected))
    return [k for k in keys if stored.get(k) != expected.get(k)]


def _media_file_digest(value: Any) -> str | None:
    """Return a content identity for a fingerprint media reference.

    Snapshot restore copies media into a new input-pack directory, so the
    path changes even though the referenced bytes do not.  Keep the digest
    lookup here (rather than in the UI) so old caches remain usable after a
    restore.
    """
    raw = str(value or "").strip()
    if not raw:
        return None
    # Reference lists are prefixed with img0:/aud0:/vid0:.
    rel = raw.split(":", 1)[1] if ":" in raw else raw
    try:
        from .pack import resolve_media_path

        path = resolve_media_path(rel)
        if path is None:
            return None
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"{stat.st_size}:{digest.hexdigest()}"
    except (OSError, ValueError):
        return None


def _media_values_equivalent(stored: Any, expected: Any) -> bool:
    if stored == expected:
        return True
    if isinstance(stored, list) and isinstance(expected, list):
        if len(stored) != len(expected):
            return False
        return all(_media_values_equivalent(a, b) for a, b in zip(stored, expected))
    if isinstance(stored, str) and isinstance(expected, str):
        stored_digest = _media_file_digest(stored)
        expected_digest = _media_file_digest(expected)
        return bool(stored_digest and stored_digest == expected_digest)
    return False


def _align_cache_fingerprint(stored: Any, expected: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Normalize old first-pass meta that baked LoRA trigger into ``prompt``."""
    if not isinstance(stored, dict):
        return stored, expected
    stored_cmp = dict(stored)
    expected_cmp = dict(expected)
    if "lora_trigger_words" not in stored_cmp:
        from .plan import strip_lora_trigger_prefix

        trigger = str(expected_cmp.get("lora_trigger_words") or "").strip()
        raw = stored_cmp.get("prompt") or ""
        stripped = strip_lora_trigger_prefix(raw, trigger)
        if trigger and stripped != raw:
            stored_cmp["prompt"] = stripped
            stored_cmp["lora_trigger_words"] = trigger
        else:
            stored_cmp["lora_trigger_words"] = ""
    # Older caches stamped plan-level「段间引导」onto every segment, including
    # the first one (which never pins). When this segment does not use motion
    # context, ignore that drift so adding a later group keeps group 1's cache.
    if not expected_cmp.get("continuity"):
        for key in ("continuity", "continuity_overlap", "continuity_from_prev"):
            if key in expected_cmp:
                stored_cmp[key] = expected_cmp[key]
    # Restoring a snapshot rewrites media paths to a fresh input-pack id.
    # Treat byte-identical references as the same input while retaining the
    # original path in newly written metadata.
    for key in ("refs", "ref_audios", "ref_videos", "ref_video", "source_media"):
        if key in stored_cmp and key in expected_cmp and _media_values_equivalent(
            stored_cmp[key], expected_cmp[key]
        ):
            stored_cmp[key] = expected_cmp[key]
    return stored_cmp, expected_cmp


def _cache_fingerprint_matches(stored: Any, expected: dict[str, Any]) -> bool:
    stored_cmp, expected_cmp = _align_cache_fingerprint(stored, expected)
    return isinstance(stored_cmp, dict) and stored_cmp == expected_cmp


def inspect_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    require_audio: bool = False,
) -> dict[str, Any]:
    """Return a non-loading, strict cache diagnosis for final export.

    ``status`` is one of ``exact cache hit``, ``stale rejected`` or
    ``missing``.  The diagnosis deliberately never falls back to stale data;
    callers use it to gate ``export=all`` before assembling a timeline.
    """
    result: dict[str, Any] = {
        "segment": int(seg.index) + 1,
        "status": "missing",
        "audio_status": "missing" if require_audio else "not required",
        "diff_keys": [],
    }
    if not node_id:
        return result
    root = h3_output_path(SEGMENT_CACHE_DIR_NAME, str(node_id))
    tensor_path = root / f"seg_{seg.index:04d}.pt"
    meta_path = root / f"seg_{seg.index:04d}.meta.json"
    audio_path = root / f"seg_{seg.index:04d}.audio.pt"
    if not tensor_path.is_file() or not meta_path.is_file():
        return result
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = segment_cache_fingerprint(seg, plan)
        stored_cmp, expected_cmp = _align_cache_fingerprint(stored, expected)
        if stored_cmp != expected_cmp:
            result["status"] = "stale rejected"
            result["diff_keys"] = _fingerprint_diff_keys(stored_cmp, expected_cmp)
            log.info(
                "Segment %d stale cache rejected for final export (diff=%s).",
                seg.index + 1, result["diff_keys"][:8],
            )
            return result
        result["status"] = "exact cache hit"
        if require_audio:
            result["audio_status"] = "exact cache hit" if audio_path.is_file() else "missing"
        return result
    except Exception as exc:
        result["status"] = "stale rejected"
        result["diff_keys"] = ["<invalid-meta>"]
        log.info("Segment %d cache metadata rejected for final export: %s", seg.index + 1, exc)
        return result


def load_segment_handoff_meta(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> dict[str, Any] | None:
    """Load trim/export handoff metadata (fingerprint must match unless ``allow_stale``)."""
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.meta.json"
    handoff_path = root / f"seg_{idx:04d}.handoff.json"
    if not meta_path.is_file() or not handoff_path.is_file():
        return None
    try:
        expected = segment_cache_fingerprint(seg, plan)
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if not _cache_fingerprint_matches(stored, expected):
            if _reject_source_stale(stored, expected, seg_index=idx, quiet=True) or not allow_stale:
                return None
        data = json.loads(handoff_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _av_latent_to_cpu(av_latent: dict) -> dict:
    samples = av_latent["samples"]
    if hasattr(samples, "unbind"):
        parts = [p.detach().cpu().contiguous() for p in samples.unbind()]
        try:
            import comfy.nested_tensor

            samples_cpu = comfy.nested_tensor.NestedTensor(tuple(parts))
        except Exception:
            samples_cpu = tuple(parts)
    elif isinstance(samples, (tuple, list)):
        samples_cpu = tuple(p.detach().cpu().contiguous() for p in samples)
    elif torch.is_tensor(samples):
        samples_cpu = samples.detach().cpu().contiguous()
    else:
        samples_cpu = samples
    out = {"samples": samples_cpu}
    for key, value in av_latent.items():
        if key == "samples":
            continue
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().contiguous()
        else:
            out[key] = value
    return out


def load_first_pass_av_latent(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> dict | None:
    """Load the cached first-pass AV latent while still rejecting source drift."""
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.pre.meta.json"
    latent_path = root / f"seg_{idx:04d}.pre.av.pt"
    if not latent_path.is_file():
        return None
    try:
        if meta_path.is_file():
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = first_pass_cache_fingerprint(seg, plan)
            if not _cache_fingerprint_matches(stored, expected):
                if _reject_source_stale(stored, expected, seg_index=idx, quiet=True):
                    return None
                if not allow_stale:
                    return None
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "samples" not in payload:
            return None
        return payload
    except Exception as exc:
        log.debug("Segment %d first-pass AV latent skipped: %s", idx + 1, exc)
        return None


def load_segment_av_latent(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> dict | None:
    """Load cached AV latent for continuity handoff (fingerprint must match unless stale-ok)."""
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.meta.json"
    latent_path = root / f"seg_{idx:04d}.av.pt"
    if not meta_path.is_file() or not latent_path.is_file():
        return None
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = segment_cache_fingerprint(seg, plan)
        if not _cache_fingerprint_matches(stored, expected):
            if _reject_source_stale(stored, expected, seg_index=idx, quiet=True) or not allow_stale:
                return None
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "samples" not in payload:
            return None
        return payload
    except Exception as exc:
        log.warning("Failed to load segment %d AV latent cache: %s", idx + 1, exc)
        return None


def _fingerprint_matches(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> bool:
    if not node_id:
        return False
    root = _cache_root(node_id)
    if root is None:
        return False
    meta_path = root / f"seg_{seg.index:04d}.meta.json"
    tensor_path = root / f"seg_{seg.index:04d}.pt"
    if not meta_path.is_file():
        return False
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = segment_cache_fingerprint(seg, plan)
        if _cache_fingerprint_matches(stored, expected):
            return True
        if _reject_source_stale(stored, expected, seg_index=seg.index, quiet=True):
            return False
        return bool(allow_stale and tensor_path.is_file())
    except Exception:
        return False


def load_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> torch.Tensor | None:
    """Load cached segment frames.

    ``allow_stale=True``: used for「选择运行」+「全部导出」fill of unselected
    segments. Prefer the last render on disk over blank/gray source placeholders
    when the fingerprint drifted (pipeline bump, minor plan churn). A different
    source video is never treated as usable stale — callers then passthrough
    the current clip (v2v/rv2v) or skip (gen timelines).
    """
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.meta.json"
    tensor_path = root / f"seg_{idx:04d}.pt"
    if not tensor_path.is_file():
        return None
    try:
        expected = segment_cache_fingerprint(seg, plan)
        if meta_path.is_file():
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            stored_cmp, expected_cmp = _align_cache_fingerprint(stored, expected)
            if stored_cmp != expected_cmp:
                if _reject_source_stale(stored, expected, seg_index=idx):
                    return None
                diff = _fingerprint_diff_keys(stored_cmp, expected_cmp)
                if not allow_stale:
                    log.info(
                        "Segment %d cache stale (diff=%s); re-run this segment to refresh.",
                        idx + 1,
                        diff[:8],
                    )
                    return None
                log.warning(
                    "Segment %d: using stale cache for export fill (diff=%s).",
                    idx + 1,
                    diff[:8],
                )
        elif not allow_stale:
            log.info(
                "Segment %d cache missing meta; re-run this segment to refresh.",
                idx + 1,
            )
            return None
        else:
            log.warning(
                "Segment %d: using cache without meta for export fill.",
                idx + 1,
            )
        return _frames_from_disk(
            torch.load(tensor_path, map_location="cpu", weights_only=True)
        )
    except Exception as exc:
        log.warning("Failed to load segment %d cache: %s", idx + 1, exc)
        return None


def load_segment_audio(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> dict[str, Any] | None:
    """Load cached export audio for a segment (same fingerprint policy as video)."""
    if not node_id or not _fingerprint_matches(
        node_id, seg, plan, allow_stale=allow_stale
    ):
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    audio_path = root / f"seg_{seg.index:04d}.audio.pt"
    if not audio_path.is_file():
        return None
    try:
        payload = torch.load(audio_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return None
        wave = payload.get("waveform")
        if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
            return None
        sr = int(payload.get("sample_rate") or 0) or 32000
        return {"waveform": wave.contiguous(), "sample_rate": sr}
    except Exception as exc:
        log.warning("Failed to load segment %d audio cache: %s", seg.index + 1, exc)
        return None


def save_first_pass_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    av_latent: dict | None = None,
    frames: torch.Tensor | None = None,
    handoff: dict[str, Any] | None = None,
) -> None:
    """Persist first-pass AV latent for confirm-then-refine. Never raises."""
    if not node_id:
        return
    if av_latent is None or not isinstance(av_latent, dict) or "samples" not in av_latent:
        return
    root = _cache_root(node_id)
    if root is None:
        return
    fp = first_pass_cache_fingerprint(seg, plan)
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.pre.meta.json"
    latent_path = root / f"seg_{idx:04d}.pre.av.pt"
    frames_path = root / f"seg_{idx:04d}.pre.pt"
    handoff_path = root / f"seg_{idx:04d}.pre.handoff.json"
    try:
        cpu_latent = _av_latent_to_cpu(av_latent)
        _write_via_temp(latent_path, lambda p: torch.save(cpu_latent, p))
        text = json.dumps(fp, ensure_ascii=False, sort_keys=True)
        _write_via_temp(meta_path, lambda p: p.write_text(text, encoding="utf-8"))
        if handoff:
            _write_via_temp(
                handoff_path,
                lambda p: p.write_text(
                    json.dumps(handoff, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                ),
            )
        if isinstance(frames, torch.Tensor) and frames.numel() > 0:
            payload = _frames_to_disk(frames)
            _write_via_temp(frames_path, lambda p: torch.save(payload, p))
        log.debug(
            "Cached first-pass segment %d for node %s (seed=%s)",
            idx + 1,
            node_id,
            fp.get("seed"),
        )
    except Exception as exc:
        log.warning(
            "Segment %d first-pass cache write skipped (%s).",
            idx + 1,
            exc,
        )
        for stray in root.glob(f".seg_{idx:04d}.pre.*"):
            _safe_unlink(stray)


def _trim_stale_first_pass_frames(
    frames: torch.Tensor,
    *,
    plan: DirectorPlan,
    handoff: dict[str, Any] | None,
    match_len: int | None,
) -> torch.Tensor | None:
    """Match in-memory first-pass export: drop context prefix, then crop length."""
    fps = float(getattr(plan, "frame_rate", 24) or 24)
    trim_frames = int((handoff or {}).get("trim_frames") or 0)
    export_len = int((handoff or {}).get("export_frames") or 0)
    if trim_frames > 0:
        if int(frames.shape[0]) <= trim_frames:
            return None
        frames, _ = trim_context_prefix(
            frames, None, trim_frames, fps=fps, match_tail=True
        )
    if export_len > 0 and int(frames.shape[0]) > export_len:
        frames = frames[:export_len]
    want = int(match_len or 0)
    extra = int(frames.shape[0]) - want if want > 0 else 0
    if extra > 0:
        frames, _ = trim_export_tail(frames, None, extra, fps=fps)
    return frames


def load_first_pass_frames_stale(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    match_len: int | None = None,
) -> torch.Tensor | None:
    """Load ``.pre.pt`` frames for unselected-segment pre-refine fill.

    Stale-tolerant counterpart of :func:`load_first_pass_cache`: fingerprint
    drift (different seed, sampling-knob churn) does NOT invalidate the fill,
    so「选择运行」re-roll previews merge all-first-pass frames instead of
    mixing a fresh first pass with cached refined renders. A different source
    video still rejects (same rule as the final-cache fill). Never raises.

    Disk ``.pre.pt`` is written before export trim; this reapplies
    ``.pre.handoff.json`` (context prefix + export length) and optionally
    matches the final-cache frame count after later phase-align tail trims.
    """
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    frames_path = root / f"seg_{idx:04d}.pre.pt"
    meta_path = root / f"seg_{idx:04d}.pre.meta.json"
    handoff_path = root / f"seg_{idx:04d}.pre.handoff.json"
    if not frames_path.is_file():
        return None
    try:
        if meta_path.is_file():
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = first_pass_cache_fingerprint(seg, plan)
            if _reject_source_stale(stored, expected, seg_index=idx, quiet=True):
                return None
        loaded = torch.load(frames_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, torch.Tensor) or loaded.numel() <= 0:
            return None
        frames = _frames_from_disk(loaded)
        if frames is None:
            return None
        handoff = None
        if handoff_path.is_file():
            try:
                data = json.loads(handoff_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    handoff = data
            except Exception:
                handoff = None
        return _trim_stale_first_pass_frames(
            frames, plan=plan, handoff=handoff, match_len=match_len
        )
    except Exception as exc:
        log.debug("Segment %d first-pass stale frames skipped: %s", idx + 1, exc)
    return None


def load_first_pass_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    load_frames: bool = True,
) -> dict[str, Any] | None:
    """Load first-pass cache only on exact fingerprint match. Never stale."""
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.pre.meta.json"
    latent_path = root / f"seg_{idx:04d}.pre.av.pt"
    frames_path = root / f"seg_{idx:04d}.pre.pt"
    handoff_path = root / f"seg_{idx:04d}.pre.handoff.json"
    if not meta_path.is_file() or not latent_path.is_file():
        return None
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = first_pass_cache_fingerprint(seg, plan)
        stored_cmp, expected_cmp = _align_cache_fingerprint(stored, expected)
        if not isinstance(stored, dict) or stored_cmp != expected_cmp:
            if isinstance(stored, dict) and _reject_source_stale(
                stored, expected, seg_index=idx, quiet=True,
            ):
                return None
            diff = _fingerprint_diff_keys(stored_cmp, expected_cmp) if isinstance(stored_cmp, dict) else ["<invalid-meta>"]
            log.info(
                "Segment %d first-pass cache miss (diff=%s); will sample first pass.",
                idx + 1,
                diff[:8],
            )
            return None
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "samples" not in payload:
            return None
        frames = None
        if load_frames and frames_path.is_file():
            try:
                loaded = torch.load(frames_path, map_location="cpu", weights_only=True)
                if isinstance(loaded, torch.Tensor) and loaded.numel() > 0:
                    frames = _frames_from_disk(loaded)
            except Exception as exc:
                log.debug("Segment %d first-pass frames skipped: %s", idx + 1, exc)
        handoff: dict[str, Any] = {}
        if handoff_path.is_file():
            try:
                data = json.loads(handoff_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    handoff = data
            except Exception:
                handoff = {}
        return {"av_latent": payload, "frames": frames, "handoff": handoff}
    except Exception as exc:
        log.warning("Failed to load segment %d first-pass cache: %s", idx + 1, exc)
        return None


_SEG_CACHE_FILE_RE = re.compile(r"^seg_(\d+)\.")


def prune_segment_cache(node_id: str | None, valid_indices) -> None:
    """Remove ``seg_XXXX.*`` files whose index is no longer on the timeline.

    Does not create the cache dir. Uses all current segment indices (not
    「选择运行」), so unselected slots keep merge/export fill. Never raises.
    """
    if not node_id:
        return
    try:
        root = h3_output_path(SEGMENT_CACHE_DIR_NAME, str(node_id))
        if not root.is_dir():
            return
        valid = {int(i) for i in valid_indices}
        removed = 0
        for path in root.iterdir():
            if not path.is_file():
                continue
            m = _SEG_CACHE_FILE_RE.match(path.name)
            if not m or int(m.group(1)) in valid:
                continue
            if _safe_unlink(path):
                removed += 1
        if removed:
            log.info(
                "Segment cache pruned %d stale file(s) for node %s.", removed, node_id
            )
    except Exception as exc:
        log.debug("Segment cache prune skipped (%s).", exc)


def first_pass_cache_disk_signature(node_id: str | None) -> str:
    """Fingerprint confirm-first-pass ``*.pre.*`` files without creating the cache dir.

    Director ``IS_CHANGED`` cannot see the linked Refine pack (ComfyUI only
    forwards widgets). These ``.pre`` files are written only by the confirmation
    hold, so a second Queue observes a new signature and continues into refine.
    """
    if not node_id:
        return ""
    root = h3_output_path(SEGMENT_CACHE_DIR_NAME, str(node_id))
    if not root.is_dir():
        return ""
    parts: list[str] = []
    try:
        for path in sorted(root.glob("seg_*.pre.*")):
            try:
                st = path.stat()
            except OSError:
                continue
            parts.append(f"{path.name}:{int(st.st_mtime_ns)}:{int(st.st_size)}")
    except OSError:
        return ""
    return "|".join(parts)


def inspect_first_pass_cache(
    node_id: str | None,
    plan: DirectorPlan,
    *,
    ui_index: int | None = None,
) -> dict[str, Any]:
    """Inspect first-pass cache files without loading their tensor payloads.

    Unselected「选择运行」slots are reported without touching their cache files.
    ``final_cached_count`` is file presence only (``seg_XXXX.pt``), not a
    fingerprint match — Refine knobs are not on this status request.
    """
    current_seed = int(getattr(plan, "sample_seed", 0) or 0)
    result: dict[str, Any] = {
        "exists": False,
        "matches": False,
        "current_seed": current_seed,
        "cached_seeds": [],
        "segment_total": 0,
        "cached_count": 0,
        "matched_count": 0,
        "selected_total": 0,
        "selected_cached": 0,
        "selected_matched": 0,
        "final_cached_count": 0,
        "diff_keys": [],
        "segments": [],
    }
    if not node_id:
        return result

    root = h3_output_path(SEGMENT_CACHE_DIR_NAME, str(node_id))
    all_segments = list(getattr(plan, "segments", None) or [])
    run_indices = getattr(plan, "run_indices", None)
    selected_set = frozenset(run_indices) if run_indices is not None else None
    result["segment_total"] = len(all_segments)

    cached_seeds: set[int] = set()
    all_diffs: set[str] = set()
    rows: list[dict[str, Any]] = []
    final_cached = 0
    for seg in all_segments:
        if ui_index is not None and int(seg.timeline_index) != int(ui_index):
            continue
        is_selected = selected_set is None or int(seg.index) in selected_set
        idx = int(seg.index)
        if not is_selected:
            rows.append(
                {
                    "segment": idx + 1,
                    "index": idx,
                    "ui_index": int(seg.timeline_index),
                    "exists": False,
                    "matches": False,
                    "status": "unchecked",
                    "selected": False,
                    "cached_seed": None,
                    "diff_keys": [],
                    "error": "",
                }
            )
            continue
        meta_path = root / f"seg_{idx:04d}.pre.meta.json"
        latent_path = root / f"seg_{idx:04d}.pre.av.pt"
        meta_exists = meta_path.is_file()
        latent_exists = latent_path.is_file()
        cache_exists = meta_exists and latent_exists
        if (root / f"seg_{idx:04d}.pt").is_file():
            final_cached += 1
        stored: Any = None
        read_error = ""
        if meta_exists:
            try:
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                read_error = str(exc)

        expected = first_pass_cache_fingerprint(seg, plan)
        stored_cmp, expected_cmp = _align_cache_fingerprint(stored, expected)
        matches = bool(cache_exists and isinstance(stored, dict) and stored_cmp == expected_cmp)
        diff = (
            _fingerprint_diff_keys(stored_cmp, expected_cmp)
            if isinstance(stored_cmp, dict)
            else (["<invalid-meta>"] if meta_exists else ["<missing-cache>"])
        )
        if normalize_segment_seed_mode(getattr(seg, "seed_mode", "inherit")) == "random":
            matches = False
            if "seed" not in diff:
                diff.append("seed")
        if not cache_exists:
            status = "missing"
        elif matches:
            status = "valid"
        else:
            status = "mismatch"
        cached_seed = stored.get("seed") if isinstance(stored, dict) else None
        try:
            if cached_seed is not None:
                cached_seed = int(cached_seed)
                cached_seeds.add(cached_seed)
        except (TypeError, ValueError):
            cached_seed = None
        all_diffs.update(diff)
        rows.append(
            {
                "segment": idx + 1,
                "index": idx,
                "ui_index": int(seg.timeline_index),
                "exists": cache_exists,
                "matches": matches,
                "status": status,
                "selected": is_selected,
                "cached_seed": cached_seed,
                "diff_keys": diff,
                "error": read_error,
            }
        )

    selected_rows = [row for row in rows if row["selected"]]
    cached_count = sum(1 for row in selected_rows if row["exists"])
    matched_count = sum(1 for row in selected_rows if row["matches"])
    selected_total = len(selected_rows) if selected_set is not None else len(rows)
    result.update(
        {
            "exists": cached_count > 0,
            "matches": selected_total > 0 and matched_count == selected_total,
            "cached_seeds": sorted(cached_seeds),
            "cached_count": cached_count,
            "matched_count": matched_count,
            "selected_total": selected_total,
            "selected_cached": sum(1 for row in selected_rows if row["exists"]),
            "selected_matched": sum(1 for row in selected_rows if row["matches"]),
            "final_cached_count": final_cached,
            "diff_keys": sorted(all_diffs),
            "segments": rows,
        }
    )
    return result


def clear_segment_cache(
    node_id: str | None,
    kind: str = "final",
    segment_index: int | None = None,
) -> int:
    """Delete cached segment files for this Director node.

    ``kind``:
      - ``first_pass``: only ``seg_XXXX.pre.*`` (一采)
      - ``final``: everything except ``.pre.*`` (成片 / 二采，含 ``.audio.pt``)
      - ``all``: both

    ``segment_index``: 0-based plan index; omit to clear every segment.

    Never creates the cache dir. Returns the number of files removed.
    """
    if not node_id:
        return 0
    if kind not in {"first_pass", "final", "all"}:
        raise ValueError("kind must be first_pass, final or all")
    if segment_index is not None:
        try:
            segment_index = int(segment_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("segment_index must be an integer") from exc
        if segment_index < 0:
            raise ValueError("segment_index must be >= 0")
    root = h3_output_path(SEGMENT_CACHE_DIR_NAME, str(node_id))
    if not root.is_dir():
        return 0
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0
    prefix = f"seg_{segment_index:04d}." if segment_index is not None else None
    removed = 0
    for path in entries:
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        if prefix is not None and not path.name.startswith(prefix):
            continue
        is_pre = ".pre." in path.name
        if kind == "first_pass" and not is_pre:
            continue
        if kind == "final" and is_pre:
            continue
        if _recycle_file(path):
            removed += 1
    if removed:
        if segment_index is None:
            log.info("Cleared %s cache for node %s (%d file(s)).", kind, node_id, removed)
        else:
            log.info(
                "Cleared %s cache for node %s segment %d (%d file(s)).",
                kind,
                node_id,
                segment_index + 1,
                removed,
            )
    return removed
