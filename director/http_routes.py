"""HTTP routes for MiniMax H3 Director (chunked video upload)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid

import folder_paths
from aiohttp import web
from server import PromptServer

from .frame_align import H3_FPS
from .output_layout import INPUT_REFERENCES_DIR_NAME, INPUT_UPLOADS_DIR_NAME, h3_input_path

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director")

CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_upload_chunks")
REF_AUDIO_CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_ref_audio_chunks")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".mts", ".ts"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
_WIN_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WIN_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)
_SAFE_EXT = re.compile(r"\.[A-Za-z0-9]{1,8}$")
_ROUTES_REGISTERED = False


def _safe_basename(name: str) -> str:
    """Keep CJK names; only strip path pieces and Windows-illegal characters."""
    base = os.path.basename(str(name or "upload.bin").replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if not _SAFE_EXT.fullmatch(ext):
        ext = ".bin"
    if ext == ".jpeg":
        ext = ".jpg"
    stem = _WIN_ILLEGAL.sub("_", stem).rstrip(" .")[:80]
    if not stem or _WIN_RESERVED.match(stem):
        stem = "upload"
    return f"{stem}{ext}"


def _get_media_exts(kind: str) -> set[str]:
    kind = str(kind or "").strip().lower()
    if kind == "image":
        return IMAGE_EXTS
    if kind == "video":
        return VIDEO_EXTS
    if kind == "audio":
        return AUDIO_EXTS
    if kind == "reference_audio":
        return AUDIO_EXTS | VIDEO_EXTS
    raise ValueError("kind must be image, video, audio or reference_audio")


def _peek_image_size(path: str) -> tuple[int, int]:
    """Read width/height from the image header without decoding pixels."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            return int(w or 0), int(h or 0)
    except Exception:
        return 0, 0


def _list_input_media(kind: str) -> list[dict]:
    input_dir = str(h3_input_path())
    os.makedirs(input_dir, exist_ok=True)
    exts = _get_media_exts(kind)
    peek_video = None
    if kind == "video":
        from ..lib.video_io import peek_video_size as peek_video
    items: list[dict] = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue
            abs_path = os.path.join(root, name)
            try:
                stat = os.stat(abs_path)
            except OSError:
                continue
            try:
                rel_path = os.path.relpath(abs_path, input_dir).replace("\\", "/")
            except ValueError:
                continue
            if rel_path.startswith(".."):
                continue
            subfolder = os.path.dirname(rel_path).replace("\\", "/")
            if subfolder == ".":
                subfolder = ""
            width, height = (0, 0)
            if ext in IMAGE_EXTS:
                width, height = _peek_image_size(abs_path)
            elif peek_video is not None and ext in VIDEO_EXTS:
                try:
                    width, height = peek_video(abs_path)
                except Exception:
                    width, height = 0, 0
            items.append(
                {
                    "name": name,
                    "fileName": name,
                    "relPath": rel_path,
                    "subfolder": subfolder,
                    "type": "input",
                    "modified": float(stat.st_mtime),
                    "width": width,
                    "height": height,
                    "mediaKind": "video" if ext in VIDEO_EXTS else (
                        "audio" if ext in AUDIO_EXTS else "image"
                    ),
                }
            )
    items.sort(key=lambda item: (-item["modified"], item["relPath"]))
    return items


async def minimax_upload_video_chunk(request):
    try:
        post = await request.post()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid upload: {exc}")

    upload_id = str(post.get("upload_id") or "").strip()
    filename = _safe_basename(post.get("filename"))
    chunk_field = post.get("chunk")
    if not upload_id or chunk_field is None:
        return web.Response(status=400, text="Missing upload_id or chunk.")

    if os.path.splitext(filename)[1].lower() not in VIDEO_EXTS:
        return web.Response(status=400, text="Unsupported video format.")

    if ".." in upload_id or "/" in upload_id or "\\" in upload_id:
        return web.Response(status=400, text="Invalid upload_id.")

    try:
        chunk_index = int(post.get("chunk_index", 0))
        total_chunks = int(post.get("total_chunks", 1))
    except (TypeError, ValueError):
        return web.Response(status=400, text="Invalid chunk index.")

    if total_chunks < 1 or chunk_index < 0 or chunk_index >= total_chunks:
        return web.Response(status=400, text="Chunk index out of range.")

    session_dir = os.path.join(CHUNK_ROOT, upload_id)
    os.makedirs(session_dir, exist_ok=True)
    part_path = os.path.join(session_dir, f"{chunk_index:06d}.part")

    with open(part_path, "wb") as out:
        while True:
            block = chunk_field.file.read(1024 * 1024)
            if not block:
                break
            out.write(block)

    if chunk_index + 1 < total_chunks:
        return web.json_response({"status": "ok", "chunk_index": chunk_index})

    input_dir = str(h3_input_path(INPUT_UPLOADS_DIR_NAME))
    os.makedirs(input_dir, exist_ok=True)
    out_path = os.path.join(input_dir, filename)
    if os.path.exists(out_path):
        stem, ext = os.path.splitext(filename)
        for n in range(1, 1000):
            candidate = f"{stem}_{n}{ext}"
            candidate_path = os.path.join(input_dir, candidate)
            if not os.path.exists(candidate_path):
                out_path = candidate_path
                filename = candidate
                break

    with open(out_path, "wb") as out:
        for i in range(total_chunks):
            part = os.path.join(session_dir, f"{i:06d}.part")
            if not os.path.isfile(part):
                shutil.rmtree(session_dir, ignore_errors=True)
                return web.Response(status=400, text=f"Missing chunk {i}.")
            with open(part, "rb") as src:
                shutil.copyfileobj(src, out)

    shutil.rmtree(session_dir, ignore_errors=True)
    subfolder = f"H3_D_NEO/{INPUT_UPLOADS_DIR_NAME}"
    log.info("MiniMax H3 Director uploaded video to input/%s: %s", subfolder, filename)
    return web.json_response({"name": filename, "subfolder": subfolder, "type": "input"})


def _reference_audio_result(path: str, *, reused: bool, source_kind: str) -> dict:
    name = os.path.basename(path)
    return {
        "name": name,
        "fileName": name,
        "relPath": f"H3_D_NEO/{INPUT_REFERENCES_DIR_NAME}/{name}",
        "subfolder": f"H3_D_NEO/{INPUT_REFERENCES_DIR_NAME}",
        "type": "input",
        "reused": bool(reused),
        "sourceKind": source_kind,
    }


def _files_identical(first: str, second: str) -> bool:
    """Match ComfyUI upload dedupe without assigning content-derived filenames."""
    try:
        if os.path.getsize(first) != os.path.getsize(second):
            return False
        with open(first, "rb") as left, open(second, "rb") as right:
            while True:
                left_block = left.read(4 * 1024 * 1024)
                right_block = right.read(4 * 1024 * 1024)
                if left_block != right_block:
                    return False
                if not left_block:
                    return True
    except OSError:
        return False


def _place_in_input_like_comfy_upload(temp_path: str, filename: str) -> tuple[str, bool]:
    """Use ComfyUI's non-overwrite rule: reuse identical, otherwise append ` (n)`."""
    input_dir = str(h3_input_path(INPUT_REFERENCES_DIR_NAME))
    os.makedirs(input_dir, exist_ok=True)
    filename = _safe_basename(filename)
    stem, ext = os.path.splitext(filename)
    candidate_name = filename
    index = 1
    while True:
        candidate_path = os.path.join(input_dir, candidate_name)
        if not os.path.exists(candidate_path):
            os.replace(temp_path, candidate_path)
            return candidate_path, False
        if _files_identical(candidate_path, temp_path):
            os.remove(temp_path)
            return candidate_path, True
        candidate_name = f"{stem} ({index}){ext}"
        index += 1


def _prepare_reference_audio(source_path: str, display_name: str) -> dict:
    """Extract a video's first audio stream and place it directly in input/."""
    if not os.path.isfile(source_path) or os.path.getsize(source_path) <= 0:
        raise ValueError("Reference audio source is empty or missing.")
    ext = os.path.splitext(display_name or source_path)[1].lower()
    if ext not in VIDEO_EXTS:
        raise ValueError("Selected source is not a supported video.")

    safe_name = _safe_basename(display_name or os.path.basename(source_path))
    safe_stem = os.path.splitext(safe_name)[0] or "reference_audio"
    output_name = f"{safe_stem}.flac"
    output_dir = str(h3_input_path(INPUT_REFERENCES_DIR_NAME))
    os.makedirs(output_dir, exist_ok=True)

    from ..lib.audio_io import _ffmpeg_bin

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is unavailable; cannot extract audio from video.")
    tmp_path = os.path.join(output_dir, f".minimax_ref_audio_{uuid.uuid4().hex}.flac")
    args = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        source_path,
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "flac",
        "-compression_level",
        "5",
        "-y",
        tmp_path,
    ]
    try:
        result = subprocess.run(args, capture_output=True, check=False)
        if result.returncode != 0 or not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) <= 0:
            error = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or "The selected video has no decodable audio stream.")
        output_path, reused = _place_in_input_like_comfy_upload(tmp_path, output_name)
    finally:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
    return _reference_audio_result(output_path, reused=reused, source_kind="video")


async def minimax_extract_reference_audio(request):
    """Extract an existing input video's audio immediately into input/."""
    try:
        body = await request.json()
        video_file = str(body.get("videoFile") or body.get("relPath") or "").strip()
        if not video_file:
            return web.Response(status=400, text="Missing videoFile.")
        from ..lib.video_io import resolve_video_path

        clip = {
            "videoFile": video_file,
            "fileName": str(body.get("fileName") or os.path.basename(video_file)),
            "subfolder": str(body.get("subfolder") or ""),
            "type": str(body.get("type") or "input"),
        }
        source_path = resolve_video_path(clip)
        if os.path.splitext(source_path)[1].lower() not in VIDEO_EXTS:
            return web.Response(status=400, text="Selected source is not a supported video.")
        result = await asyncio.to_thread(
            _prepare_reference_audio,
            source_path,
            clip["fileName"] or os.path.basename(source_path),
        )
        return web.json_response(result)
    except Exception as exc:
        log.warning("MiniMax H3 Director reference audio extraction failed: %s", exc)
        return web.Response(status=400, text=str(exc))


async def minimax_prepare_reference_audio_chunk(request):
    """Receive large local audio/video; store audio or extract video audio into input/."""
    session_dir = ""
    try:
        post = await request.post()
        upload_id = str(post.get("upload_id") or "").strip()
        filename = _safe_basename(post.get("filename"))
        chunk_field = post.get("chunk")
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", upload_id) or chunk_field is None:
            return web.Response(status=400, text="Invalid reference audio upload.")
        try:
            chunk_index = int(post.get("chunk_index", 0))
            total_chunks = int(post.get("total_chunks", 1))
        except (TypeError, ValueError):
            return web.Response(status=400, text="Invalid chunk index.")
        if total_chunks < 1 or chunk_index < 0 or chunk_index >= total_chunks:
            return web.Response(status=400, text="Chunk index out of range.")
        source_ext = os.path.splitext(filename)[1].lower()
        if source_ext not in AUDIO_EXTS | VIDEO_EXTS:
            return web.Response(status=400, text="Unsupported reference audio source format.")

        session_dir = os.path.join(REF_AUDIO_CHUNK_ROOT, upload_id)
        os.makedirs(session_dir, exist_ok=True)
        part_path = os.path.join(session_dir, f"{chunk_index:06d}.part")
        with open(part_path, "wb") as out:
            while True:
                block = chunk_field.file.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        if chunk_index + 1 < total_chunks:
            response = web.json_response({"status": "ok", "chunk_index": chunk_index})
            session_dir = ""
            return response

        source_path = os.path.join(session_dir, filename)
        with open(source_path, "wb") as out:
            for index in range(total_chunks):
                part = os.path.join(session_dir, f"{index:06d}.part")
                if not os.path.isfile(part):
                    raise ValueError(f"Missing chunk {index}.")
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, out)
        if source_ext in AUDIO_EXTS:
            output_path, reused = await asyncio.to_thread(
                _place_in_input_like_comfy_upload,
                source_path,
                filename,
            )
            result = _reference_audio_result(output_path, reused=reused, source_kind="audio")
        else:
            result = await asyncio.to_thread(_prepare_reference_audio, source_path, filename)
        return web.json_response(result)
    except Exception as exc:
        log.warning("MiniMax H3 Director local reference audio preparation failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    finally:
        if session_dir:
            shutil.rmtree(session_dir, ignore_errors=True)


async def minimax_probe_video(request):
    try:
        if request.can_read_body and request.content_type == "application/json":
            body = await request.json()
        else:
            body = dict(request.query)
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid request: {exc}")

    video_file = str(body.get("videoFile") or body.get("video_file") or "").strip()
    if not video_file:
        return web.Response(status=400, text="Missing videoFile.")

    from ..lib.video_io import probe_video_clip

    clip = {
        "videoFile": video_file,
        "fileName": os.path.basename(video_file),
        "subfolder": str(body.get("subfolder") or "").strip(),
        "type": str(body.get("type") or "input").strip() or "input",
    }
    try:
        info = probe_video_clip(clip)
    except Exception as exc:
        log.warning("MiniMax H3 Director video probe failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    return web.json_response(info)


def _export_video_range(clips: list[dict], frame_rate: float, output_path: str) -> None:
    from ..lib.video_export import _ffmpeg_bin
    from ..lib.video_io import resolve_video_path

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is unavailable; cannot export the selected video range.")

    args = [ffmpeg, "-v", "error", "-nostdin"]
    filters: list[str] = []
    for index, clip in enumerate(clips):
        args.extend(["-i", resolve_video_path(clip)])
        start = max(0, int(clip.get("sourceFrameStart") or 0))
        end = max(start + 1, int(clip.get("sourceFrameEnd") or start + 1))
        filters.append(
            f"[{index}:v:0]trim=start={start / frame_rate:.9f}:end={end / frame_rate:.9f},"
            f"setpts=PTS-STARTPTS,fps={frame_rate}[v{index}]"
        )

    labels = "".join(f"[v{index}]" for index in range(len(clips)))
    filters.append(f"{labels}concat=n={len(clips)}:v=1:a=0[outv]")
    args.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[outv]",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y",
        output_path,
    ])
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode != 0 or not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
        error = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(error or "ffmpeg failed to export the selected video range.")


async def minimax_export_video_range(request):
    output_path = ""
    try:
        body = await request.json()
        clips_in = body.get("clips")
        if not isinstance(clips_in, list) or not clips_in:
            return web.Response(status=400, text="Missing clips[].")
        clips = []
        for item in clips_in:
            if not isinstance(item, dict):
                continue
            video_file = str(item.get("videoFile") or item.get("video_file") or "").strip()
            if not video_file:
                continue
            clips.append({
                "videoFile": video_file,
                "fileName": os.path.basename(video_file),
                "subfolder": str(item.get("subfolder") or "").strip(),
                "type": str(item.get("type") or "input").strip() or "input",
                "sourceFrameStart": item.get("sourceFrameStart", item.get("source_frame_start", 0)),
                "sourceFrameEnd": item.get("sourceFrameEnd", item.get("source_frame_end")),
            })
        if not clips:
            return web.Response(status=400, text="Missing clips[].")
        frame_rate = float(body.get("frameRate") or body.get("frame_rate") or H3_FPS)
        if not 0 < frame_rate <= 240:
            return web.Response(status=400, text="Invalid frameRate.")

        fd, output_path = tempfile.mkstemp(prefix="minimax_range_", suffix=".mp4")
        os.close(fd)
        await asyncio.to_thread(_export_video_range, clips, frame_rate, output_path)
        size = os.path.getsize(output_path)
        response = web.StreamResponse(headers={
            "Content-Type": "video/mp4",
            "Content-Disposition": 'attachment; filename="minimax_selected_range.mp4"',
            "Content-Length": str(size),
            "Cache-Control": "no-store",
        })
        await response.prepare(request)
        with open(output_path, "rb") as source:
            while chunk := source.read(1024 * 1024):
                await response.write(chunk)
        await response.write_eof()
        return response
    except (TypeError, ValueError) as exc:
        return web.Response(status=400, text=str(exc))
    except Exception as exc:
        log.warning("MiniMax H3 Director video range export failed: %s", exc)
        return web.Response(status=500, text=str(exc))
    finally:
        if output_path:
            try:
                os.remove(output_path)
            except OSError:
                pass


async def minimax_list_vae_approx(request):
    try:
        from .tae_preview import default_tae_name, list_vae_approx_names
        names = list_vae_approx_names()
        return web.json_response({
            "items": names,
            "default": default_tae_name(),
        })
    except Exception as exc:
        log.warning("MiniMax H3 Director list vae_approx failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


async def minimax_list_drawer_models(request):
    try:
        from .face_refine.pack import detector_choices
        from .h3_latent_upscale import MISSING_MODEL_LABEL, list_h3_latent_upscale_models

        models = [
            name for name in list_h3_latent_upscale_models()
            if name and name != MISSING_MODEL_LABEL
        ]
        return web.json_response({
            "detectors": detector_choices(),
            "latent_upscale_models": models,
        })
    except Exception as exc:
        log.warning("MiniMax H3 Director list drawer models failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


async def minimax_list_input_media(request):
    try:
        kind = str(request.query.get("kind") or "").strip().lower()
        if not kind:
            return web.Response(status=400, text="Missing kind.")
        items = _list_input_media(kind)
    except ValueError as exc:
        return web.Response(status=400, text=str(exc))
    except Exception as exc:
        log.warning("MiniMax H3 Director list input media failed: %s", exc)
        return web.Response(status=500, text=str(exc))
    return web.json_response({"items": items})


async def minimax_first_pass_cache_status(request):
    """Compare stored first-pass metadata with the Director's current inputs."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    try:
        from .plan import (
            apply_lora_trigger_words,
            apply_lora_trigger_words_r2v,
            build_director_plan,
        )
        from .segment_cache import inspect_first_pass_cache

        plan = build_director_plan(
            str(timeline_data),
            global_task_type=str(body.get("task_type") or ""),
            global_prompt=str(body.get("global_prompt") or ""),
            total_frames=int(body.get("total_frames") or 124),
            frame_rate=float(H3_FPS),
            width=int(body.get("width") or 864),
            height=int(body.get("height") or 480),
            ref_max_size=int(body.get("ref_max_size") or 864),
            load_media=False,
        )
        plan = apply_lora_trigger_words(plan, body.get("lora_trigger_words"))
        plan = apply_lora_trigger_words_r2v(plan, body.get("lora_trigger_words_r2v"))
        plan.sample_seed = int(body.get("seed") or 0)
        plan.sample_cfg = float(body.get("cfg") or 1.0)
        plan.sample_steps = int(body.get("steps") or 25)
        plan.sample_sampler = str(body.get("sampler") or "")
        plan.sample_scheduler = str(body.get("scheduler") or "")
        plan.sample_sigmas_linked = bool(body.get("sigmas_linked"))
        plan.sample_shift_video = float(body.get("shift_video") or 12.0)
        plan.sample_shift_audio = float(body.get("shift_audio") or 3.0)
        from .selflift.pack import pack_selflift

        def _flag(value, default=False):
            # Match the Director drawer: only explicit true values enable SelfLift.
            # Leftover combo strings from old widgets_values must not count as on.
            if value is True or value == 1:
                return True
            if value is False or value == 0:
                return False
            if value is None:
                return default
            text = str(value).strip().lower()
            return text in {"1", "true", "on", "yes"}

        if _flag(body.get("selflift_enable")):
            plan.selflift = pack_selflift(
                split_mode=body.get("selflift_split_mode", "highres_steps"),
                highres_steps=body.get("selflift_highres_steps", 2),
                transition_step=body.get("selflift_transition_step", 6),
                lowres_scale=body.get("selflift_lowres_scale", 0.5),
                latent_upscale_model=body.get("selflift_latent_upscale_model"),
                sampler_mode=body.get("selflift_sampler_mode", "euler"),
                native_low_carry=_flag(body.get("selflift_native_low_carry"), True),
                rho=body.get("selflift_rho", 0),
                w_min=body.get("selflift_w_min", 0.5),
                w_max=body.get("selflift_w_max", 1),
                latent_upsample=body.get("selflift_latent_upsample", "bilinear"),
                enable_latent_chunking=_flag(body.get("selflift_enable_latent_chunking")),
                enable_tiling=_flag(body.get("selflift_enable_tiling")),
                tile_count=body.get("selflift_tile_count", 2),
                tile_overlap=body.get("selflift_tile_overlap", 128),
            )
        raw_cache_index = body.get("cache_index")
        cache_index = int(raw_cache_index) if raw_cache_index is not None else None
        return web.json_response(inspect_first_pass_cache(node_id, plan, ui_index=cache_index))
    except Exception as exc:
        log.warning("MiniMax H3 Director first-pass cache inspection failed: %s", exc)
        return web.json_response(
            {"exists": False, "matches": False, "error": str(exc)},
            status=400,
        )


async def minimax_prompt_to_timeline(request):
    """Convert prompt-agent JSON to timeline data without starting execution."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    try:
        from .prompt_contract import director_prompt_to_timeline

        timeline_json = director_prompt_to_timeline(
            str(body.get("director_prompt") or ""),
            base_timeline_data=json.dumps(
                body.get("timeline_data") or {}, ensure_ascii=False
            )
            if isinstance(body.get("timeline_data"), dict)
            else str(body.get("timeline_data") or ""),
            default_width=int(body.get("width") or 864),
            default_height=int(body.get("height") or 480),
            default_ref_max_size=int(body.get("ref_max_size") or 864),
        )
        return web.json_response({"timeline": json.loads(timeline_json)})
    except Exception as exc:
        log.warning("MiniMax H3 Director prompt sync failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=400)


async def minimax_clear_segment_cache(request):
    """Delete first-pass (.pre.*) or final segment cache files."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    kind = str(body.get("kind") or "final").strip().lower()
    if kind not in {"first_pass", "final", "all"}:
        return web.Response(status=400, text="kind must be first_pass, final or all.")

    raw_idx = body.get("index", body.get("segment_index"))
    segment_index = None
    if raw_idx is not None and str(raw_idx).strip() != "":
        try:
            segment_index = int(raw_idx)
        except (TypeError, ValueError):
            return web.Response(status=400, text="Invalid segment index.")
        if segment_index < 0:
            return web.Response(status=400, text="Invalid segment index.")

    try:
        from .segment_cache import clear_segment_cache

        removed = clear_segment_cache(node_id, kind=kind, segment_index=segment_index)
        payload = {"removed": removed, "kind": kind}
        if segment_index is not None:
            payload["index"] = segment_index
        return web.json_response(payload)
    except Exception as exc:
        log.warning("MiniMax H3 Director clear segment cache failed: %s", exc)
        return web.Response(status=500, text=str(exc))


def _register_route(routes, method: str, path: str, handler) -> None:
    if hasattr(routes, "add_route"):
        routes.add_route(method, path, handler)
    elif method == "POST" and hasattr(routes, "post"):
        routes.post(path)(handler)
    elif method == "GET" and hasattr(routes, "get"):
        routes.get(path)(handler)
    else:
        raise AttributeError("Unsupported ComfyUI route table API")


def register_routes() -> bool:
    """Register MiniMax H3 Director HTTP routes on the ComfyUI PromptServer."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True

    server = PromptServer.instance
    if server is None:
        log.warning("MiniMax H3 Director: PromptServer not ready, HTTP routes not registered")
        return False

    routes = server.routes
    _register_route(routes, "POST", "/minimax/director/upload_chunk", minimax_upload_video_chunk)
    _register_route(
        routes,
        "POST",
        "/minimax/director/extract_reference_audio",
        minimax_extract_reference_audio,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director/prepare_reference_audio_chunk",
        minimax_prepare_reference_audio_chunk,
    )
    _register_route(routes, "POST", "/minimax/director/probe_video", minimax_probe_video)
    _register_route(routes, "GET", "/minimax/director/probe_video", minimax_probe_video)
    _register_route(routes, "POST", "/minimax/director/export_video_range", minimax_export_video_range)
    _register_route(routes, "GET", "/minimax/director/list_input_media", minimax_list_input_media)
    _register_route(routes, "GET", "/minimax/director/list_vae_approx", minimax_list_vae_approx)
    _register_route(routes, "GET", "/minimax/director/list_drawer_models", minimax_list_drawer_models)
    _register_route(
        routes,
        "POST",
        "/minimax/director/first_pass_cache_status",
        minimax_first_pass_cache_status,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director/prompt_to_timeline",
        minimax_prompt_to_timeline,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director/clear_segment_cache",
        minimax_clear_segment_cache,
    )
    from .snapshots import (
        minimax_delete_snapshot,
        minimax_duplicate_snapshot,
        minimax_export_snapshot,
        minimax_import_snapshot,
        minimax_list_snapshots,
        minimax_rename_snapshot,
        minimax_restore_snapshot,
        minimax_save_snapshot,
        minimax_update_snapshot,
    )

    _register_route(routes, "GET", "/minimax/director/snapshots", minimax_list_snapshots)
    _register_route(routes, "POST", "/minimax/director/snapshots/save", minimax_save_snapshot)
    _register_route(routes, "POST", "/minimax/director/snapshots/update", minimax_update_snapshot)
    _register_route(routes, "GET", "/minimax/director/snapshots/export", minimax_export_snapshot)
    _register_route(routes, "POST", "/minimax/director/snapshots/import", minimax_import_snapshot)
    _register_route(routes, "POST", "/minimax/director/snapshots/restore", minimax_restore_snapshot)
    _register_route(routes, "POST", "/minimax/director/snapshots/rename", minimax_rename_snapshot)
    _register_route(routes, "POST", "/minimax/director/snapshots/duplicate", minimax_duplicate_snapshot)
    _register_route(routes, "POST", "/minimax/director/snapshots/delete", minimax_delete_snapshot)
    _ROUTES_REGISTERED = True
    log.info("MiniMax H3 Director HTTP routes registered")
    return True
