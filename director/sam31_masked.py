"""Source-latent SAM editing using ComfyUI's native H3 per-token masks.

Independent integration; no GPL compatibility shim is copied or installed.
Geometry cross-checked against Apache-2.0 diffusers-modular/minimax-h3-inpainting.
"""

from __future__ import annotations

import inspect
import math
import re
import zipfile


def source_prompt(prompt):
    # Legacy source tags are rebound only in the execution text, never saved.
    text = re.sub(r"<\s*video\s*1\s*>", "the source video", prompt or "", flags=re.IGNORECASE)
    if re.search(r"<\s*video\s*\d+\s*>", text, re.IGNORECASE):
        raise ValueError("SAM3.1 source-latent edit has no additional Video reference; describe the edit directly.")
    return text


def preflight(plan, seg, model):
    from .plan import resolve_segment_pass_mode
    from .selflift.pack import selflift_enabled

    if resolve_segment_pass_mode(seg) == "second":
        raise ValueError("SAM3.1 source-latent edit: disable second pass; it bypasses the generation mask.")
    if selflift_enabled(plan) or (getattr(plan, "face_refine", None) or {}).get("enabled"):
        raise ValueError("SAM3.1 source-latent edit: disable SelfLift and FaceRefine.")
    base = getattr(model, "model", None)
    diffusion = getattr(base, "diffusion_model", None)
    forward = getattr(diffusion, "_forward", None)
    parameters = inspect.signature(forward).parameters if callable(forward) else {}
    if (not {"denoise_mask", "audio_denoise_mask"}.issubset(parameters)
            or not callable(getattr(base, "_denoise_mask_values", None))
            or not callable(getattr(base, "_token_grid_masks", None))):
        raise ValueError("SAM3.1 source-latent edit requires native H3 per-token masks (ComfyUI PR15375).")


def causal_groups(frame_count, latent_count):
    if frame_count < 5 or frame_count % 17 != 5 or latent_count != 5 * ((frame_count - 5) // 17) + 2:
        raise ValueError("SAM3.1 source-latent edit: invalid H3 temporal shape; expected 17k+5 frames.")
    groups = []
    start = 0
    for index in range(latent_count):
        end = start + (1 if index % 5 == 0 else 4)
        groups.append((start, end))
        start = end
    return groups


def latent_mask(pixel_mask, video_shape, grow=0):
    import torch
    import torch.nn.functional as F

    batch, _, count, height, width = video_shape
    if batch != 1 or height % 2 or width % 2 or pixel_mask.ndim != 3:
        raise ValueError("SAM3.1 source-latent edit: expected batch 1 and a 2x2-aligned video latent.")
    if tuple(pixel_mask.shape[1:]) != (height * 16, width * 16):
        raise ValueError("SAM3.1 source-latent edit: mask and encoded source canvas differ.")
    groups = causal_groups(int(pixel_mask.shape[0]), count)
    spatial = F.max_pool2d((pixel_mask > 0).float()[:, None], 16, 16)
    temporal = torch.stack([spatial[a:b].amax(dim=0) for a, b in groups])
    cells = F.max_pool2d(temporal, 2, 2)
    radius = math.ceil(max(0, int(grow)) / 32)
    if radius:
        cells = F.max_pool2d(cells, 2 * radius + 1, stride=1, padding=radius)
    grid = cells.repeat_interleave(2, -2).repeat_interleave(2, -1)
    return grid.movedim(0, 1).unsqueeze(0)


def pixel_masks(seg, source, transforms=()):
    import numpy as np
    import torch
    from .sam31 import cache_path, _metadata, _packed, verify_assets, file_digest
    from .sam31_composite import _fit

    context = seg.sam31_composite
    base = context["meta"]["identity"]["base"]
    verify_assets(base)
    if file_digest(cache_path(context["key"])) != context["artifact"]:
        raise ValueError("SAM3.1 source-latent edit: mask artifact changed during execution.")
    masks = []
    width, height = context["canvas"]
    with zipfile.ZipFile(cache_path(context["key"])) as archive:
        meta = _metadata(archive, context["key"])
        if meta != context["meta"]:
            raise ValueError("SAM3.1 source-latent edit: tracking cache changed during execution.")
        for index in range(meta["frames"]):
            packed = _packed(archive, meta, index)[context["selected"]]
            union = np.bitwise_or.reduce(packed, axis=0)
            binary = np.unpackbits(union, axis=-1, bitorder="little")[..., :meta["mask_width"]]
            frame = torch.from_numpy(binary.copy()).float()[None, ..., None].expand(-1, -1, -1, 3)
            asset = base["source"]["assets"][base["source"]["entries"][index][0]]
            frame = _fit(frame, asset["width"], asset["height"], "stretch", mask=True)
            frame = _fit(frame, width, height, context["fit"], mask=True)
            for op, w, h in transforms:
                frame = _fit(frame, w, h, op, mask=True)
            if tuple(frame.shape[1:3]) != tuple(source.shape[1:3]):
                raise ValueError("SAM3.1 source-latent edit: mask and source geometry differ.")
            masks.append(frame[0, ..., 0] > 0)
    verify_assets(base)
    if file_digest(cache_path(context["key"])) != context["artifact"]:
        raise ValueError("SAM3.1 source-latent edit: mask artifact changed during execution.")
    return torch.stack(masks)


def initialize(latent, source, pixel_mask, vae, *, grow=0, audio_mode="generate", audio_vae=None, source_audio=None):
    import torch
    import torch.nn.functional as F
    from comfy.nested_tensor import NestedTensor

    samples = latent.get("samples")
    if not getattr(samples, "is_nested", False):
        raise ValueError("SAM3.1 source-latent edit requires a nested H3 AV latent.")
    streams = list(samples.unbind())
    if len(streams) != 2:
        raise ValueError("SAM3.1 source-latent edit requires exactly video and audio streams.")
    target_video, target_audio = streams
    video = vae.encode(source[..., :3])
    if tuple(video.shape) != tuple(target_video.shape):
        raise ValueError("SAM3.1 source-latent edit: encoded source does not match the H3 target latent.")
    mask = latent_mask(pixel_mask, tuple(video.shape), grow).to(device=video.device, dtype=video.dtype)
    audio = target_audio
    if audio_mode == "source" and source_audio is not None:
        if audio_vae is None:
            raise ValueError("SAM3.1 source-latent edit: source audio conditioning requires audio_vae.")
        from comfy_extras.nodes_audio import VAEEncodeAudio
        out = VAEEncodeAudio.execute(audio_vae, source_audio)
        args = out.args if hasattr(out, "args") else out
        encoded = args[0]["samples"]
        if tuple(encoded.shape[:-1]) != tuple(audio.shape[:-1]):
            raise ValueError("SAM3.1 source-latent edit: encoded source audio has incompatible channels.")
        # Fit conditioning to the 40Hz clock only; original PCM export is unchanged.
        audio = F.pad(encoded[..., :audio.shape[-1]], (0, max(0, audio.shape[-1] - encoded.shape[-1])))
    audio_mask = torch.ones_like(audio) if audio_mode == "generate" else torch.zeros_like(audio)
    result = dict(latent)
    result["samples"] = NestedTensor((video, audio))
    result["noise_mask"] = NestedTensor((mask.expand_as(video).contiguous(), audio_mask))
    return result
