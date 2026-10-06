"""Lazy native SAM3.1 analysis with independent, per-frame packed-mask caches."""

from __future__ import annotations

import base64
import gc
import hashlib
import importlib
import inspect
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import uuid
import zipfile

from .sam31_config import identity_key, normalize_config, normalize_group

MAX_FRAMES = 256
MAX_IMAGE_BYTES = 384 * 1024 * 1024
CACHE_IMPLEMENTATION_REVISION = 2
KEY_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_RE = re.compile(r"sam3[._-]?1", re.I)
_CACHE_LOCK = threading.RLock()
log = logging.getLogger(__name__)


def native_backend():
    native = importlib.import_module("comfy_extras.nodes_sam3")
    supported = importlib.import_module("comfy.supported_models")
    sd = importlib.import_module("comfy.sd")
    if not hasattr(supported, "SAM31"):
        raise RuntimeError("ComfyUI lacks native SAM3.1 model support.")
    required = {
        "SAM3_Detect": {"model", "image", "conditioning", "threshold", "individual_masks", "refine_iterations"},
        "SAM3_VideoTrack": {"images", "model", "initial_mask", "conditioning", "detection_threshold", "max_objects", "detect_interval"},
    }
    for name, inputs in required.items():
        node = getattr(native, name, None)
        if node is None or not inputs.issubset(inspect.signature(node.execute).parameters):
            raise RuntimeError(f"Missing or incompatible native {name}.")
    for name in ("SAM3_TrackPreview", "SAM3_TrackToMask"):
        if not hasattr(native, name):
            raise RuntimeError(f"Missing native {name}.")
    if not callable(getattr(sd, "load_checkpoint_guess_config", None)):
        raise RuntimeError("ComfyUI checkpoint loader is unavailable.")
    return native, sd


def capabilities():
    result = dict(available=False, models=[], reason="", max_frames=MAX_FRAMES,
                  max_image_bytes=MAX_IMAGE_BYTES, points_boxes=False, cancel=False)
    try:
        import folder_paths
        result["models"] = [
            name for name in folder_paths.get_filename_list("checkpoints")
            if MODEL_RE.search(name.replace("\\", "/").split("/")[-1])
            and name.lower().endswith(".safetensors")
        ]
        native_backend()
        result["available"] = True
        if not result["models"]:
            result["reason"] = "No SAM3.1 safetensors checkpoint in models/checkpoints."
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def file_digest(path):
    before = Path(path).stat()
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while hashing.")
    return digest.hexdigest()


def model_identity(name):
    import folder_paths
    if name not in folder_paths.get_filename_list("checkpoints"):
        raise ValueError("Selected checkpoint is not installed.")
    if not MODEL_RE.search(name.replace("\\", "/").split("/")[-1]) or not name.lower().endswith(".safetensors"):
        raise ValueError("Select a SAM3.1 safetensors checkpoint.")
    path = folder_paths.get_full_path("checkpoints", name)
    if not path or not Path(path).is_file():
        raise ValueError("Checkpoint file is missing.")
    stat = Path(path).stat()
    return dict(name=name, path=str(Path(path).resolve()), size=stat.st_size,
                modified_ns=stat.st_mtime_ns)


def input_path(media, field):
    import folder_paths
    if not isinstance(media, dict) or str(media.get("type") or "input") != "input":
        raise ValueError("SAM requires uploaded ComfyUI input media.")
    name = str(media.get(field) or media.get("fileName") or "").replace("\\", "/")
    if not name:
        raise ValueError("Source media is missing.")
    root = Path(folder_paths.get_input_directory()).resolve()
    folder = str(media.get("subfolder") or "").replace("\\", "/")
    candidates = ([root / folder / Path(name).name] if folder else []) + [root / name]
    for candidate in candidates:
        path = candidate.resolve()
        if not path.is_relative_to(root):
            raise ValueError("Media path escapes ComfyUI input.")
        if path.is_file():
            return path
    raise ValueError(f"Source file not found: {name}")


def analysis_size(width, height, edge):
    width, height = int(width), int(height)
    if min(width, height) <= 0 or width * height > 32 * 1024 * 1024:
        raise ValueError("Invalid or excessively large source dimensions.")
    scale = min(1, edge / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def source_descriptor(segment, fps, edge):
    from PIL import Image, ImageOps
    from ..lib.task_prompts import resolve_task_key
    from ..lib.video_io import (
        probe_video_file, probe_video_display_size,
        resolve_logical_frame_entry, _resolve_load_dimensions,
    )

    task = resolve_task_key(segment.get("taskType") or "t2v")
    source = segment.get("sourceVideo") or {}
    if not isinstance(source, dict):
        raise ValueError("Invalid sourceVideo.")
    image = None
    if task in {"v2v", "rv2v"} and source.get("mediaKind") == "image":
        image = source.get("image")
    elif task == "i2v":
        image = segment.get("genImage")
    elif task == "fl2v":
        first, last = segment.get("startImage"), segment.get("endImage")
        image = first if isinstance(first, dict) and first.get("imageFile") else last
    elif task == "r2v":
        refs = segment.get("refs") or []
        image = refs[0] if refs else None
    if image is not None:
        path = input_path(image, "imageFile")
        with Image.open(path) as im:
            analysis_size(*im.size, edge)
            width, height = ImageOps.exif_transpose(im).size
        return dict(kind="image", fps=fps, range=[0, 1],
                    size=list(analysis_size(width, height, edge)), entries=[[0, 0]],
                    assets=[dict(path=str(path), digest=file_digest(path), width=width, height=height)])
    if task not in {"v2v", "rv2v"} or source.get("mediaKind") == "image":
        raise ValueError("This group has no uploaded SAM source image/video.")
    video = source.get("video") or {}
    clips = source.get("videoClips") or []
    if not clips and isinstance(video, dict) and (video.get("videoFile") or video.get("fileName")):
        clips = [video]
    if not isinstance(video, dict) or not isinstance(clips, list) or not clips:
        raise ValueError("Source video is missing.")
    assets, resolved, sizes = [], [], []
    for clip in clips:
        path = input_path(clip, "videoFile")
        meta = probe_video_file(str(path))
        width, height = probe_video_display_size(str(path))
        duration = float(meta.get("duration") or 0)
        count = int(clip.get("sourceFrameCount") or 0) or int(duration * fps)
        if count <= 0 or not math.isfinite(duration):
            raise ValueError("Cannot determine source duration.")
        analysis_width, analysis_height = analysis_size(width, height, edge)
        out_width, out_height, rotate = _resolve_load_dimensions(
            width, height,
            storage_width=analysis_width, storage_height=analysis_height,
            long_edge=max(analysis_width, analysis_height),
        )
        if rotate:
            raise ValueError("Unexpected rotation when resolving display-oriented SAM source.")
        sizes.append((out_width, out_height))
        resolved.append({**clip, "sourceFrameCount": count})
        assets.append(dict(path=str(path), digest=file_digest(path),
                           width=width, height=height, duration=duration))
    total = len(video.get("frameMap") or []) or int(source.get("totalFrames") or 0) or sum(
        clip["sourceFrameCount"] for clip in resolved
    )
    start = int(source.get("rangeStart") or 0)
    end = total if source.get("rangeEnd") is None else int(source["rangeEnd"])
    if not 0 <= start < end <= total:
        raise ValueError("Invalid source range.")
    if end - start > MAX_FRAMES:
        raise ValueError(f"Trim the source range to at most {MAX_FRAMES} frames.")
    timeline = dict(video=video, videoClips=resolved, totalFrames=total)
    entries = [list(resolve_logical_frame_entry(timeline, i)) for i in range(start, end)]
    used_sizes = set()
    for clip, frame in entries:
        if not 0 <= clip < len(assets) or frame < 0:
            raise ValueError("Invalid source frame mapping.")
        if assets[clip]["duration"] > 0 and frame / fps >= assets[clip]["duration"]:
            raise ValueError("Selected frame exceeds source duration.")
        used_sizes.add(sizes[clip])
    if len(used_sizes) != 1:
        raise ValueError("Selected clips must have one consistent analysis aspect and size.")
    return dict(kind="video", fps=fps, range=[start, end], size=list(used_sizes.pop()),
                entries=entries, assets=assets)


def prepare_request(body):
    config = normalize_config(body.get("config"))
    timeline = body.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError("Missing timeline.")
    group_id = str(body.get("group_id") or "")
    groups = [g for g in timeline.get("segments", [])
              if isinstance(g, dict) and str(g.get("id") or "") == group_id]
    if not group_id or len(groups) != 1:
        raise ValueError("Select a group with a unique stable ID.")
    group = normalize_group(groups[0].get("sam31"))
    if not group["prompt"]:
        raise ValueError("Enter a short object description.")
    fps = float(timeline.get("frameRate") or 24)
    if not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError("Invalid frame rate.")
    action = str(body.get("action") or "detect")
    if action not in {"detect", "track", "track_selected"}:
        raise ValueError("Unknown SAM operation.")
    source = source_descriptor(groups[0], fps, config["long_edge"])
    if action != "detect" and source["kind"] != "video":
        raise ValueError("Tracking requires source video.")
    count = 1 if action == "detect" else len(source["entries"])
    if count * math.prod(source["size"]) * 12 > MAX_IMAGE_BYTES:
        raise ValueError("Decoded images exceed 384 MiB; reduce long edge or trim the range.")
    base = dict(version=1, implementation_revision=CACHE_IMPLEMENTATION_REVISION,
                group_id=group_id, source=source,
                model=model_identity(config["checkpoint"]), prompt=group["prompt"],
                parameters={k: config[k] for k in ("threshold", "max_objects", "detect_interval", "long_edge")})
    identity = dict(base=base, action=action)
    if action == "track_selected":
        expected = identity_key(dict(base=base, action="detect"))
        result = group["result"] or {}
        if result.get("action") != "detect" or result.get("key") != expected:
            raise ValueError("Detect again: source or settings changed.")
        if not group["selected"] or len(group["selected"]) > config["max_objects"]:
            raise ValueError("Select between one and max_objects detected objects.")
        identity["initial"] = dict(key=expected, indices=group["selected"])
    return dict(key=identity_key(identity), identity=identity, enabled=config["enabled"])


def cache_path(key):
    import folder_paths
    if not isinstance(key, str) or not KEY_RE.fullmatch(key):
        raise ValueError("Invalid SAM cache key.")
    return Path(folder_paths.get_temp_directory()) / "director_sam31" / f"{key}.zip"


def _metadata(archive, key):
    meta = json.loads(archive.read("metadata.json"))
    if not isinstance(meta, dict):
        raise ValueError("Invalid SAM cache metadata.")
    identity = meta.get("identity")
    if (not isinstance(identity, dict)
            or not isinstance(identity.get("base"), dict)
            or identity["base"].get("implementation_revision") != CACHE_IMPLEMENTATION_REVISION):
        raise ValueError("SAM cache implementation changed; run SAM again.")
    objects, shape = meta.get("objects"), meta.get("packed_shape")
    if (meta.get("version") != 1 or meta.get("key") != key
            or identity_key(meta.get("identity")) != key
            or not isinstance(objects, list) or len(objects) > 16
            or not isinstance(shape, list) or len(shape) != 2
            or any(type(n) is not int for n in shape)
            or not 1 <= shape[0] <= 2048 or not 1 <= shape[1] <= 256
            or type(meta.get("frames")) is not int or not 1 <= meta["frames"] <= MAX_FRAMES
            or type(meta.get("mask_width")) is not int
            or not (shape[1] - 1) * 8 < meta["mask_width"] <= shape[1] * 8):
        raise ValueError("Invalid SAM cache metadata.")
    return meta


def _packed(archive, meta, frame):
    import numpy as np
    if not 0 <= frame < meta["frames"]:
        raise ValueError("Preview frame is out of range.")
    name = f"frames/{frame}.npy"
    if archive.getinfo(name).file_size > 9 * 1024 * 1024:
        raise ValueError("Oversized packed mask frame.")
    packed = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
    if packed.dtype != np.uint8 or packed.shape != (len(meta["objects"]), *meta["packed_shape"]):
        raise ValueError("Invalid packed mask layout.")
    return packed


def read_metadata(key):
    with zipfile.ZipFile(cache_path(key)) as archive:
        return _metadata(archive, key)


def read_packed_frame(key, frame):
    with zipfile.ZipFile(cache_path(key)) as archive:
        meta = _metadata(archive, key)
        return meta, _packed(archive, meta, int(frame))


def _valid_cache(key, identity):
    try:
        with zipfile.ZipFile(cache_path(key)) as archive:
            meta = _metadata(archive, key)
            if meta["identity"] != identity:
                return None
            for frame in range(meta["frames"]):
                _packed(archive, meta, frame)
            return meta
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, RuntimeError):
        return None


def write_cache(key, metadata, packed_frames):
    import numpy as np
    path = cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{key}.{uuid.uuid4().hex}.pending")
    with _CACHE_LOCK:
        if _valid_cache(key, metadata["identity"]) is not None:
            return
        with temporary.open("xb") as destination:
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                count = 0
                for count, packed in enumerate(packed_frames, 1):
                    buffer = io.BytesIO()
                    np.save(buffer, packed, allow_pickle=False)
                    archive.writestr(f"frames/{count - 1}.npy", buffer.getvalue())
                if count != metadata["frames"]:
                    raise ValueError("Incomplete SAM cache frame sequence.")
                archive.writestr("metadata.json", json.dumps(metadata, allow_nan=False))
            destination.flush()
            os.fsync(destination.fileno())
        with zipfile.ZipFile(temporary) as archive:
            meta = _metadata(archive, key)
            for frame in range(meta["frames"]):
                _packed(archive, meta, frame)
        # Failed temporary writes remain unreferenced. A complete validated ZIP
        # atomically replaces a damaged cache; no unlink/remove operation is used.
        os.replace(temporary, path)


def verify_assets(base):
    for asset in base["source"]["assets"]:
        if file_digest(asset["path"]) != asset["digest"]:
            raise ValueError("Source content changed; SAM result is stale.")
    if model_identity(base["model"]["name"]) != base["model"]:
        raise ValueError("Checkpoint identity changed; SAM result is stale.")


def decode_source(source, indices):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from ..lib.video_io import load_video_resampled
    width, height = source["size"]
    if source["kind"] == "image":
        with Image.open(source["assets"][0]["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            return torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).unsqueeze(0)
    rows = [source["entries"][i] for i in indices]
    output = torch.empty(len(rows), height, width, 3, dtype=torch.float32)
    for clip in sorted({row[0] for row in rows}):
        frames = sorted({row[1] for row in rows if row[0] == clip})
        tensor = load_video_resampled(
            source["assets"][clip]["path"], source["fps"], frames,
            storage_width=width, storage_height=height, long_edge=max(width, height),
            hold_past_eof=False, strict=True,
        )
        if tuple(tensor.shape) != (len(frames), height, width, 3):
            raise ValueError("Decoded source does not match the frame mapping/analysis dimensions.")
        positions = {frame: index for index, frame in enumerate(frames)}
        for index, (which, frame) in enumerate(rows):
            if which == clip:
                output[index].copy_(tensor[positions[frame]])
        del tensor
    return output


def public_result(meta, cached=False):
    source = meta["identity"]["base"]["source"]
    return dict(version=1, key=meta["key"], action=meta["identity"]["action"],
                frames=meta["frames"], objects=meta["objects"], size=source["size"],
                cached=cached, source_kind=source["kind"], source_range=source["range"])


def run_request(spec):
    key, identity = spec["key"], spec["identity"]
    if identity_key(identity) != key:
        raise ValueError("SAM request identity mismatch.")
    base, action = identity["base"], identity["action"]
    if base.get("implementation_revision") != CACHE_IMPLEMENTATION_REVISION:
        raise ValueError("SAM request implementation changed; submit again.")
    verify_assets(base)
    cached = _valid_cache(key, identity)
    if cached is not None:
        return public_result(cached, True)
    import numpy as np
    import torch
    import comfy.model_management as management
    native, sd = native_backend()
    supported = importlib.import_module("comfy.supported_models")
    model = clip = images = conditioning = initial = output = None
    try:
        with torch.inference_mode():
            model, clip, *_ = sd.load_checkpoint_guess_config(
                base["model"]["path"], output_vae=False, output_clip=True,
            )
            if model is None or clip is None or not isinstance(model.model.model_config, supported.SAM31):
                raise RuntimeError("Checkpoint must load as native SAM31 with its text encoder.")
            params = base["parameters"]
            indices = [0] if action == "detect" else list(range(len(base["source"]["entries"])))
            images = decode_source(base["source"], indices)
            if action == "track_selected":
                initial_meta, packed = read_packed_frame(identity["initial"]["key"], 0)
                selected = identity["initial"]["indices"]
                if any(i >= len(initial_meta["objects"]) for i in selected):
                    raise ValueError("Selected detection object is missing.")
                initial = torch.from_numpy(
                    np.unpackbits(packed[selected], axis=-1, bitorder="little")
                    [..., :initial_meta["mask_width"]].copy()
                ).float()
                if tuple(initial.shape[-2:]) != tuple(images.shape[1:3]):
                    raise ValueError("Detection masks do not match the initial tracking frame.")
            else:
                conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(base["prompt"]))
                if not conditioning:
                    raise RuntimeError("SAM text encoder returned no conditioning.")
                if conditioning[0][1].get("sam3_multi_cond") is None:
                    conditioning[0][1]["sam3_multi_cond"] = [{
                        "cond": conditioning[0][0],
                        "attention_mask": conditioning[0][1].get("attention_mask"),
                        "max_detections": params["max_objects"],
                    }]
            objects = []
            if action == "detect":
                output = native.SAM3_Detect.execute(
                    model=model, image=images, conditioning=conditioning,
                    threshold=params["threshold"], individual_masks=True, refine_iterations=2,
                )
                masks, boxes = output.result
                if masks.ndim != 3 or tuple(masks.shape[-2:]) != tuple(images.shape[1:3]):
                    raise RuntimeError("Unsupported native detection mask layout.")
                if (not isinstance(boxes, list) or len(boxes) != 1
                        or not isinstance(boxes[0], list)
                        or len(boxes[0]) != masks.shape[0]):
                    raise RuntimeError("Native detection masks and boxes are not aligned.")
                scores = [float(box["score"]) for box in boxes[0]]
                if any(not math.isfinite(score) for score in scores):
                    raise RuntimeError("Native detection returned non-finite scores.")
                # Native limits are per category; enforce the drawer's global cap.
                order = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True,
                )[:params["max_objects"]]
                masks = masks.index_select(
                    0, torch.tensor(order, dtype=torch.long, device=masks.device),
                )
                boxes = [boxes[0][i] for i in order]
                packed = np.packbits(masks.detach().cpu().numpy() > 0, axis=-1, bitorder="little")[None]
                mask_width = int(images.shape[2])
                asset = base["source"]["assets"][base["source"]["entries"][0][0]]
                sx, sy = asset["width"] / images.shape[2], asset["height"] / images.shape[1]
                objects = [dict(index=i, score=float(box["score"]), box={
                    "x": box["x"] * sx, "y": box["y"] * sy,
                    "width": box["width"] * sx, "height": box["height"] * sy,
                }) for i, box in enumerate(boxes)]
            else:
                output = native.SAM3_VideoTrack.execute(
                    images=images, model=model, initial_mask=initial, conditioning=conditioning,
                    detection_threshold=params["threshold"], max_objects=params["max_objects"],
                    detect_interval=params["detect_interval"],
                )
                data = output.result[0]
                packed = data["packed_masks"]
                if packed is None:
                    packed = np.zeros((len(indices), 0, images.shape[1], (images.shape[2] + 7) // 8), dtype=np.uint8)
                    mask_width = int(images.shape[2])
                else:
                    if packed.ndim != 4 or packed.dtype != torch.uint8:
                        raise RuntimeError("Unsupported native tracking packed-mask layout.")
                    # Native tracker packs its internal mask raster (currently
                    # 1008 wide), not the input image width. No native W padding.
                    mask_width = int(packed.shape[-1] * 8)
                scores = data.get("scores", [])
                if (action == "track_selected"
                        and packed.shape[1] != len(identity["initial"]["indices"])):
                    raise RuntimeError("Native tracker changed the selected object count.")
                for i in range(packed.shape[1]):
                    obj = dict(index=i, score=float(scores[i]) if i < len(scores) else None)
                    if action == "track_selected":
                        obj["detection_index"] = identity["initial"]["indices"][i]
                    objects.append(obj)
            if packed.shape[0] != len(indices) or packed.shape[1] != len(objects) or len(objects) > params["max_objects"]:
                raise RuntimeError("Native result violates frame/object limits.")
            verify_assets(base)
            meta = dict(version=1, key=key, identity=identity, frames=len(indices),
                        objects=objects, mask_width=mask_width, packed_shape=list(packed.shape[-2:]))
            def frames():
                for index in range(len(indices)):
                    frame = packed[index]
                    yield frame.detach().cpu().numpy() if torch.is_tensor(frame) else frame
            write_cache(key, meta, frames())
            return public_result(meta)
    finally:
        model = clip = images = conditioning = initial = output = None
        masks = packed = data = None
        gc.collect()
        try:
            management.cleanup_models_gc()
        except Exception:
            log.warning("SAM model cleanup failed", exc_info=True)


def preview_request(body):
    import numpy as np
    from PIL import Image
    key, frame = str(body.get("key") or ""), int(body.get("frame") or 0)
    meta, packed = read_packed_frame(key, frame)
    action = meta["identity"]["action"]
    current = prepare_request({**body, "action": "track" if action == "track_selected" else action})
    if current["identity"]["base"] != meta["identity"]["base"]:
        raise ValueError("Source, group, text or settings changed; run SAM again.")
    source = meta["identity"]["base"]["source"]
    images = decode_source(source, [0 if action == "detect" else frame])
    image = Image.fromarray((images[0].numpy().clip(0, 1) * 255).astype(np.uint8))
    selected = body.get("selected", list(range(len(meta["objects"]))))
    if not isinstance(selected, list) or any(type(i) is not int or not 0 <= i < len(meta["objects"]) for i in selected):
        raise ValueError("Invalid object selection.")
    opacity = float(body.get("opacity", 0.5))
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("Invalid preview opacity.")
    array = np.asarray(image).astype(np.float32).copy()
    colors = [(31, 160, 224), (255, 140, 32), (64, 196, 96), (220, 64, 96)]
    for index in sorted(set(selected)):
        # Detection has NumPy byte padding; crop to its recorded raster width.
        # Tracking keeps its full native raster, then maps to analysis size.
        mask = np.unpackbits(packed[index], axis=-1, bitorder="little")[..., :meta["mask_width"]]
        mask = Image.fromarray(mask * 255).resize(image.size, Image.Resampling.NEAREST)
        foreground = np.asarray(mask) > 0
        array[foreground] = array[foreground] * (1 - opacity) + np.asarray(colors[index % len(colors)]) * opacity
    buffer = io.BytesIO()
    Image.fromarray(array.clip(0, 255).astype(np.uint8)).save(buffer, format="PNG")
    entry = source["entries"][0 if action == "detect" else frame]
    asset = source["assets"][entry[0]]
    return dict(
        result=public_result(meta, True),
        image="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        mapping=dict(clip=entry[0], source_frame=entry[1], frame_rate=source["fps"],
                     source_size=[asset["width"], asset["height"]], analysis_size=source["size"]),
    )
