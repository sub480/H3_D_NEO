"""In-repo MiniMax H3 spatial tiled sampler for Director Refine.

Port of the AV / keyframe / R2V-aware tiled second-sample path from
ComfyUI-MiniMax-H3-Block-based-sampler (MIT), itself based on
yichengup/ComfyUI-YCNodes-MiniMax-H3. Director Refine does not depend on
that custom node.

Video is split on H or W; audio is passed through whole. Keyframes that
match the second-pass canvas are cropped per tile. Identity-sized R2V
refs are left intact (they are not spatially aligned with the canvas).

Each tile's DiT RoPE grid is a slice of the full-canvas coordinates so
overlapping regions share the same (h, w) positions. Denoise is step-synced
for every sampler: the sampler walks the full latent, and each model()
evaluates tiles then blends.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.h3_tiled")

TILE_AXES = ("auto", "H", "W")
_PI = 3.14159265
_tile_rope = threading.local()


def _h3_extract(samples, debug=False):
    type_name = type(samples).__name__

    if hasattr(samples, "is_nested") and samples.is_nested:
        try:
            parts = list(samples.unbind())
            video = None
            audio = None
            for part in parts:
                if isinstance(part, torch.Tensor):
                    if video is None:
                        video = part
                    else:
                        audio = part
            if video is not None:
                if debug:
                    log.info(
                        "tiled extract NestedTensor video=%s audio=%s",
                        tuple(video.shape),
                        tuple(audio.shape) if audio is not None else None,
                    )
                return video, audio, {"type": "nested_tensor"}
        except Exception as exc:
            if debug:
                log.info("tiled extract NestedTensor unbind failed: %s", exc)

    if isinstance(samples, torch.Tensor):
        if debug:
            log.info("tiled extract plain tensor %s", tuple(samples.shape))
        return samples, None, {"type": "tensor"}

    if isinstance(samples, (tuple, list)):
        video = None
        audio = None
        for item in samples:
            if isinstance(item, torch.Tensor):
                if video is None:
                    video = item
                else:
                    audio = item
        if video is not None:
            fmt = "tuple" if isinstance(samples, tuple) else "list"
            if debug:
                log.info(
                    "tiled extract %s video=%s audio=%s",
                    fmt,
                    tuple(video.shape),
                    tuple(audio.shape) if audio is not None else None,
                )
            return video, audio, {"type": fmt}
        raise TypeError(f"H3 tiled extract: {type_name} has no video tensor")

    raise TypeError(
        f"H3 tiled extract: unsupported '{type_name}'. "
        "Expected 5D tensor / NestedTensor / (video, audio)."
    )


def _h3_reconstruct(video, audio, format_info):
    fmt = format_info.get("type", "tensor")
    if fmt == "nested_tensor":
        from comfy.nested_tensor import NestedTensor

        parts = [video] + ([audio] if audio is not None else [])
        return NestedTensor(parts)
    if fmt == "tensor":
        return video
    if fmt == "tuple":
        return (video, audio) if audio is not None else (video,)
    if fmt == "list":
        return [video, audio] if audio is not None else [video]
    return (video, audio) if audio is not None else video


def _h3_make_nested(video, audio):
    if audio is None:
        return video
    from comfy.nested_tensor import NestedTensor

    return NestedTensor([video, audio])


def _even(n: int) -> int:
    return (int(n) + 1) // 2 * 2


def _align_down_even(n: int) -> int:
    n = int(n)
    return n - (n % 2)


def _align_up_even(n: int, total: int) -> int:
    n = int(n)
    if n % 2:
        n += 1
    return min(int(total), n)


def _compute_tile_regions(total, n_tiles, overlap):
    """Cover [0, total) with even-aligned overlapping tiles.

    Interior tiles extend ``overlap`` past each core boundary, so adjacent
    tiles overlap by about ``2 * overlap``. First start is 0, last end is
    ``total``. Starts are even so they sit on H3's 2x2 patch grid.
    """
    total = int(total)
    n_tiles = max(1, int(n_tiles))
    overlap = max(0, int(overlap))
    if n_tiles <= 1 or total <= 0:
        return [(0, total)]
    overlap = _align_down_even(overlap)
    core = total / n_tiles
    regions = []
    for i in range(n_tiles):
        core_start = int(round(i * core))
        core_end = int(round((i + 1) * core))
        start = 0 if i == 0 else max(0, core_start - overlap)
        end = total if i == n_tiles - 1 else min(total, core_end + overlap)
        if i > 0:
            start = _align_down_even(start)
        if i < n_tiles - 1:
            end = _align_up_even(end, total)
        if end - start < 2:
            continue
        regions.append((int(start), int(end)))
    if not regions:
        return [(0, total)]
    regions[0] = (0, regions[0][1])
    regions[-1] = (regions[-1][0], total)
    cleaned = []
    for start, end in regions:
        if cleaned and start <= cleaned[-1][0] and end >= cleaned[-1][1]:
            cleaned[-1] = (start, end)
        elif not cleaned or start > cleaned[-1][0]:
            cleaned.append((start, end))
    if cleaned[-1][1] != total:
        cleaned[-1] = (cleaned[-1][0], total)
    if cleaned[0][0] != 0:
        cleaned[0] = (0, cleaned[0][1])
    if len(cleaned) <= 1:
        return [(0, total)]
    return cleaned


def _compute_tile_starts(total, n_tiles, overlap):
    regions = _compute_tile_regions(total, n_tiles, overlap)
    starts = [start for start, _end in regions]
    tile_size = max((end - start) for start, end in regions)
    return starts, tile_size


def _make_window_1d(length, ov_left, ov_right, dtype, device):
    w = torch.ones(length, dtype=dtype, device=device)
    if ov_left > 0:
        n = min(ov_left, length // 2 + 1)
        if n > 0:
            t = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[:-1]
            fade = 0.5 - 0.5 * torch.cos(t * _PI)
            w[:n] = torch.minimum(w[:n], fade)
    if ov_right > 0:
        n = min(ov_right, length // 2 + 1)
        if n > 0:
            t = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[:-1]
            fade = 0.5 - 0.5 * torch.cos((1 - t) * _PI)
            w[-n:] = torch.minimum(w[-n:], fade)
    return w


def _spatial_hw(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
        return None
    return int(tensor.shape[-2]), int(tensor.shape[-1])


def _crop_spatial(tensor, tile_axis, start, end):
    if tile_axis == "H":
        return tensor[..., start:end, :].contiguous()
    return tensor[..., start:end].contiguous()


def _pad_h3_patch(region):
    pad_h = (-region.shape[-2]) % 2
    pad_w = (-region.shape[-1]) % 2
    if pad_h or pad_w:
        return F.pad(region, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
    return region


def _cond_entry(cond):
    if isinstance(cond, dict):
        return cond
    if isinstance(cond, (list, tuple)) and len(cond) >= 2 and isinstance(cond[1], dict):
        return cond[1]
    return None


def _iter_cond_dicts(guider):
    original = getattr(guider, "original_conds", None)
    if not original:
        return
    for cond_list in original.values():
        if not cond_list:
            continue
        for cond in cond_list:
            item = _cond_entry(cond)
            if item is not None:
                yield item


def _latent_from_ref(ref):
    if isinstance(ref, torch.Tensor):
        return ref, "tensor"
    if isinstance(ref, dict) and isinstance(ref.get("latent"), torch.Tensor):
        return ref["latent"], "dict"
    return None, None


def _crop_ref_item(ref, tile_axis, start, end, full_hw):
    latent, kind = _latent_from_ref(ref)
    if latent is None:
        return ref
    if _spatial_hw(latent) != full_hw:
        return ref
    region = _pad_h3_patch(_crop_spatial(latent, tile_axis, start, end))
    if kind == "tensor":
        return region
    item = dict(ref)
    item["latent"] = region
    work = region.unsqueeze(0) if region.ndim == 4 else region
    item["latent_h"] = int(work.shape[-2])
    item["latent_w"] = int(work.shape[-1])
    if "latent_t" in item:
        item["latent_t"] = int(work.shape[-3])
    return item


def _crop_refs_list(refs, tile_axis, start, end, full_hw):
    if not refs:
        return refs
    return [_crop_ref_item(ref, tile_axis, start, end, full_hw) for ref in refs]


def _crop_refs_in_extra_args(extra_args, tile_axis, start, end, full_hw):
    if not extra_args:
        return extra_args
    cond = extra_args.get("cond")
    if not isinstance(cond, dict) or "minimax_refs" not in cond:
        return extra_args
    new_extra = dict(extra_args)
    new_cond = dict(cond)
    new_cond["minimax_refs"] = _crop_refs_list(
        cond.get("minimax_refs"), tile_axis, start, end, full_hw
    )
    new_extra["cond"] = new_cond
    return new_extra


def _iter_all_cond_dicts(guider):
    for store_name in ("original_conds", "conds"):
        store = getattr(guider, store_name, None)
        if not store:
            continue
        for cond_list in store.values():
            if not cond_list:
                continue
            for cond in cond_list:
                item = _cond_entry(cond)
                if item is not None:
                    yield item


def _crop_payload_inplace(payload, tile_axis, start, end, full_hw):
    restores = []
    if not isinstance(payload, dict):
        return restores
    if "layout" in payload:
        old_layout = payload.pop("layout", None)
        restores.append(lambda: payload.__setitem__("layout", old_layout))
    for key in ("keyframes", "refs"):
        items = payload.get(key)
        if not items:
            continue
        old = items
        payload[key] = _crop_refs_list(items, tile_axis, start, end, full_hw)
        restores.append(lambda old=old, key=key: payload.__setitem__(key, old))
    if "cond_video_latents" in payload:
        old_latents = payload["cond_video_latents"]
        new_latents = []
        for src in list(payload.get("keyframes") or []) + list(payload.get("refs") or []):
            if isinstance(src, dict) and isinstance(src.get("latent"), torch.Tensor):
                new_latents.append(src["latent"])
            elif isinstance(src, torch.Tensor):
                new_latents.append(src)
        if new_latents:
            payload["cond_video_latents"] = new_latents
        elif old_latents:
            payload["cond_video_latents"] = [
                _crop_ref_item(item, tile_axis, start, end, full_hw)
                for item in old_latents
            ]
        restores.append(lambda: payload.__setitem__("cond_video_latents", old_latents))
    return restores


def _crop_cond_dict(cond, tile_axis, start, end, full_hw):
    restores = []
    if not isinstance(cond, dict):
        return restores
    if cond.get("minimax_keyframes"):
        old = cond["minimax_keyframes"]
        cond["minimax_keyframes"] = _crop_refs_list(old, tile_axis, start, end, full_hw)
        restores.append(lambda: cond.__setitem__("minimax_keyframes", old))
    if cond.get("minimax_refs"):
        old = cond["minimax_refs"]
        cond["minimax_refs"] = _crop_refs_list(old, tile_axis, start, end, full_hw)
        restores.append(lambda: cond.__setitem__("minimax_refs", old))
    model_conds = cond.get("model_conds")
    if isinstance(model_conds, dict):
        payload_obj = model_conds.get("minimax_payload")
        payload = getattr(payload_obj, "cond", None)
        if isinstance(payload, dict):
            restores.extend(_crop_payload_inplace(payload, tile_axis, start, end, full_hw))
    return restores


class _TileCondCrop:
    def __init__(self, guider, full_hw):
        self.guider = guider
        self.full_hw = full_hw
        self._restores = []

    def apply(self, tile_axis, start, end):
        self.restore()
        patches = []
        for cond in _iter_all_cond_dicts(self.guider):
            patches.extend(_crop_cond_dict(cond, tile_axis, start, end, self.full_hw))
        self._restores = patches

    def restore(self):
        for fn in reversed(self._restores):
            fn()
        self._restores = []


def _slice_canvas_frame_grid(orig_frame_grid, full_h, full_w, tile_h, tile_w, tile_axis, start):
    full_h_e, full_w_e = _even(full_h), _even(full_w)
    tile_h_e, tile_w_e = _even(tile_h), _even(tile_w)
    frame, w_grid = orig_frame_grid(full_h_e, full_w_e)
    n_h, n_w = full_h_e // 2, full_w_e // 2
    grid = frame.reshape(n_h, n_w, 2)
    ph0 = (int(start) // 2) if tile_axis == "H" else 0
    pw0 = (int(start) // 2) if tile_axis == "W" else 0
    need_h, need_w = tile_h_e // 2, tile_w_e // 2
    ph1 = min(n_h, ph0 + need_h)
    pw1 = min(n_w, pw0 + need_w)
    sliced = grid[ph0:ph1, pw0:pw1, :]
    pad_h = need_h - sliced.shape[0]
    pad_w = need_w - sliced.shape[1]
    if pad_h or pad_w:
        sliced = F.pad(
            sliced.permute(2, 0, 1).unsqueeze(0),
            (0, pad_w, 0, pad_h),
            mode="replicate",
        )[0].permute(1, 2, 0)
    return sliced.reshape(-1, 2), w_grid


@contextmanager
def _tile_rope_context(full_h, full_w, tile_axis, start, end):
    """Rewrite H3 _frame_grid so a tile uses sliced full-canvas RoPE coords."""
    import comfy.ldm.minimax.model as mm

    tile_h = (end - start) if tile_axis == "H" else full_h
    tile_w = (end - start) if tile_axis == "W" else full_w
    tile_h_e, tile_w_e = _even(tile_h), _even(tile_w)
    orig = mm._frame_grid
    ctx = {
        "full_h": full_h,
        "full_w": full_w,
        "tile_axis": tile_axis,
        "start": start,
        "tile_h_e": tile_h_e,
        "tile_w_e": tile_w_e,
        "orig": orig,
    }
    prev = getattr(_tile_rope, "ctx", None)
    _tile_rope.ctx = ctx

    def patched(h, w):
        active = getattr(_tile_rope, "ctx", None)
        if active is None:
            return orig(h, w)
        if (int(h), int(w)) != (active["tile_h_e"], active["tile_w_e"]):
            return orig(h, w)
        return _slice_canvas_frame_grid(
            active["orig"],
            active["full_h"],
            active["full_w"],
            active["tile_h_e"],
            active["tile_w_e"],
            active["tile_axis"],
            active["start"],
        )

    mm._frame_grid = patched
    try:
        yield
    finally:
        mm._frame_grid = orig
        _tile_rope.ctx = prev


def _crop_packed(packed, shapes, tile_axis, start, end):
    if packed is None:
        return packed, shapes
    if isinstance(packed, torch.Tensor) and packed.dim() >= 4 and (not shapes or len(shapes) == 1):
        cropped = _crop_spatial(packed, tile_axis, start, end)
        return cropped, [tuple(cropped.shape)]
    if not shapes:
        return packed, shapes
    import comfy.utils

    streams = list(comfy.utils.unpack_latents(packed, shapes))
    if not streams or _spatial_hw(streams[0]) is None:
        return packed, shapes
    streams[0] = _crop_spatial(streams[0], tile_axis, start, end)
    return comfy.utils.pack_latents(streams)


def _override_latent_shapes(model_k, guider, tile_shapes):
    restores = []
    base = getattr(getattr(model_k, "inner_model", None), "inner_model", None)
    if base is not None and hasattr(base, "latent_shapes"):
        old = base.latent_shapes
        base.latent_shapes = tile_shapes
        restores.append(lambda old=old, base=base: setattr(base, "latent_shapes", old))
    for cond in _iter_all_cond_dicts(guider):
        model_conds = cond.get("model_conds") if isinstance(cond, dict) else None
        if not isinstance(model_conds, dict):
            continue
        shape_cond = model_conds.get("latent_shapes")
        if shape_cond is not None and hasattr(shape_cond, "cond"):
            old = shape_cond.cond
            shape_cond.cond = tile_shapes
            restores.append(
                lambda obj=shape_cond, old=old: setattr(obj, "cond", old)
            )
    return restores


def _prepare_minimax_conds(guider, full_h, full_w, debug=False):
    full_hw = (int(full_h), int(full_w))
    contexts = []
    for cond in _iter_cond_dicts(guider):
        original_kf = cond.get("minimax_keyframes")
        original_refs = cond.get("minimax_refs")
        prepared = []
        mismatched = []
        if original_kf:
            for kf in original_kf:
                if not isinstance(kf, dict):
                    continue
                item = dict(kf)
                latent = item.get("latent")
                if not isinstance(latent, torch.Tensor) or latent.dim() not in (4, 5):
                    continue
                old_hw = tuple(latent.shape[-2:])
                if old_hw != full_hw:
                    mismatched.append(old_hw)
                prepared.append(item)
        contexts.append(
            {
                "cond": cond,
                "original_kf": original_kf,
                "original_refs": original_refs,
                "prepared_kf": prepared,
                "disable_hard_injection": bool(mismatched),
                "full_hw": full_hw,
            }
        )
        if debug and mismatched:
            log.info(
                "tiled: skip low-res keyframe hard inject %s vs canvas %s",
                mismatched,
                full_hw,
            )
    return contexts


def _apply_minimax_region(contexts, tile_axis, start, end, debug=False):
    del debug
    for ctx in contexts:
        cond = ctx["cond"]
        if ctx.get("disable_hard_injection"):
            cond.pop("minimax_keyframes", None)
        else:
            tiled = []
            for kf in ctx["prepared_kf"]:
                item = dict(kf)
                latent = item["latent"]
                item["latent"] = _pad_h3_patch(_crop_spatial(latent, tile_axis, start, end))
                tiled.append(item)
            if tiled:
                cond["minimax_keyframes"] = tiled
        if ctx.get("original_refs"):
            cond["minimax_refs"] = _crop_refs_list(
                ctx["original_refs"], tile_axis, start, end, ctx["full_hw"]
            )


def _restore_minimax_conds(contexts):
    for ctx in contexts:
        cond = ctx["cond"]
        if ctx["original_kf"] is None:
            cond.pop("minimax_keyframes", None)
        else:
            cond["minimax_keyframes"] = ctx["original_kf"]
        if ctx["original_refs"] is None:
            cond.pop("minimax_refs", None)
        else:
            cond["minimax_refs"] = ctx["original_refs"]


def _clean_minimax_layout(guider):
    patcher = getattr(guider, "model_patcher", None)
    model = getattr(patcher, "model", None) if patcher is not None else None
    cached = getattr(model, "_cached_extra_conds", None) if model is not None else None
    if not isinstance(cached, dict):
        return
    for value in cached.values():
        payload = getattr(value, "cond", None)
        if isinstance(payload, dict):
            payload.pop("layout", None)
            payload.pop("cond_video_latents", None)


def _crop_noise_mask(mask, tile_axis, start, end, video_hw):
    if mask is None:
        return None
    try:
        video, audio, fmt = _h3_extract(mask, False)
    except Exception:
        if isinstance(mask, torch.Tensor) and _spatial_hw(mask) == video_hw:
            return _crop_spatial(mask, tile_axis, start, end)
        return mask
    if _spatial_hw(video) != video_hw:
        return mask
    return _h3_reconstruct(_crop_spatial(video, tile_axis, start, end), audio, fmt)


def _preview_x0(x0, shapes=None):
    """Video stream for Director TAE. Packed AV latents unpack with ``shapes``."""
    try:
        if hasattr(x0, "is_nested") and getattr(x0, "is_nested", False):
            parts = list(x0.unbind())
            for part in parts:
                if isinstance(part, torch.Tensor) and part.dim() == 5:
                    return part
            return parts[0] if parts else x0
        if isinstance(x0, torch.Tensor) and shapes and len(shapes) > 1:
            import comfy.utils

            streams = list(comfy.utils.unpack_latents(x0, shapes))
            if streams:
                return streams[0]
        video, _audio, _fmt = _h3_extract(x0, False)
        return video if video is not None else x0
    except Exception as exc:
        log.debug("tiled preview x0 unpack skipped: %s", exc)
        return x0


def _chain_step_preview(comfy_cb, on_step_preview, shapes=None, preview_every=1):
    if on_step_preview is None and comfy_cb is None:
        return None
    every = max(1, int(preview_every or 1))

    def callback(step, x0, x, total_steps):
        if on_step_preview is not None:
            try:
                last = max(0, int(total_steps) - 1)
                if int(step) % every == 0 or int(step) >= last:
                    on_step_preview(int(step), int(total_steps), _preview_x0(x0, shapes))
            except Exception as exc:
                log.debug("tiled step preview skipped: %s", exc)
        if comfy_cb is not None:
            try:
                comfy_cb(step, x0, x, total_steps)
            except Exception as exc:
                log.debug("tiled comfy preview skipped: %s", exc)

    return callback


def _preview_callback(guider, sigmas):
    try:
        import comfy.utils
        import latent_preview

        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output
        )
        disable_pbar = not getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True)
        return callback, disable_pbar
    except Exception:
        return None, True


def _tile_overlap_windows(regions, tile_axis, device):
    windows = []
    for tile_idx, (ax_start, ax_end) in enumerate(regions):
        actual_size = ax_end - ax_start
        prev_end = regions[tile_idx - 1][1] if tile_idx > 0 else ax_start
        next_start = regions[tile_idx + 1][0] if tile_idx < len(regions) - 1 else ax_end
        ov_left = min(actual_size, max(0, min(prev_end, ax_end) - ax_start))
        ov_right = min(actual_size, max(0, ax_end - max(ax_start, next_start)))
        window_1d = _make_window_1d(actual_size, ov_left, ov_right, torch.float32, device)
        window = window_1d.view(1, 1, 1, -1, 1) if tile_axis == "H" else window_1d.view(1, 1, 1, 1, -1)
        windows.append((ax_start, ax_end, window))
    return windows


def _seam_bands(regions):
    bands = []
    for i in range(len(regions) - 1):
        _s0, prev_end = regions[i]
        next_start, _e1 = regions[i + 1]
        if prev_end - next_start >= 2:
            bands.append((int(next_start), int(prev_end)))
    return bands


def _single_pass(
    noise,
    guider,
    sampler,
    sigmas,
    latent_dict,
    video_tensor,
    audio_tensor,
    fmt_info,
    contexts,
    debug=False,
    on_step_preview=None,
    preview_every=1,
):
    import comfy.model_management
    import comfy.sample

    latent_for_sample = _h3_reconstruct(video_tensor, audio_tensor, fmt_info)
    work = dict(latent_dict)
    try:
        latent_for_sample = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent_for_sample,
            work.get("downscale_ratio_spacial", None),
            work.get("downscale_ratio_temporal", None),
        )
    except Exception:
        pass
    work["samples"] = latent_for_sample
    comfy_cb, disable_pbar = _preview_callback(guider, sigmas)
    shapes = [tuple(video_tensor.shape)]
    if audio_tensor is not None:
        shapes.append(tuple(audio_tensor.shape))
    callback = _chain_step_preview(comfy_cb, on_step_preview, shapes, preview_every)
    noise_mask = work.get("noise_mask")
    _apply_minimax_region(contexts, "H", 0, int(video_tensor.shape[-2]), debug)
    _clean_minimax_layout(guider)
    try:
        samples = guider.sample(
            noise.generate_noise(work),
            latent_for_sample,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
    finally:
        _restore_minimax_conds(contexts)
    samples = samples.to(comfy.model_management.intermediate_device())
    out = dict(work)
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


class _TiledDenoise:
    """Wrap KSamplerX0Inpaint so each denoise evaluates overlapping tiles then blends."""

    def __init__(
        self,
        inner,
        *,
        guider,
        regions,
        tile_axis,
        full_shapes,
        full_hw,
        cropper,
    ):
        self._inner = inner
        self._guider = guider
        self._tile_axis = tile_axis
        self._full_shapes = full_shapes
        self._full_hw = full_hw
        self._cropper = cropper
        device = None
        for tensor in (getattr(inner, "latent_image", None), getattr(inner, "noise", None)):
            if isinstance(tensor, torch.Tensor):
                device = tensor.device
                break
        self._windows = _tile_overlap_windows(
            regions, tile_axis, device or torch.device("cpu")
        )
        axis_total = int(full_hw[0] if tile_axis == "H" else full_hw[1])
        weight_shape = (
            (1, 1, 1, axis_total, 1) if tile_axis == "H" else (1, 1, 1, 1, axis_total)
        )
        self._weights = torch.zeros(
            weight_shape, dtype=torch.float32, device=device or torch.device("cpu")
        )
        for ax_start, ax_end, window in self._windows:
            if tile_axis == "H":
                self._weights[:, :, :, ax_start:ax_end, :] += window
            else:
                self._weights[:, :, :, :, ax_start:ax_end] += window
        self._weights = self._weights.clamp(min=1e-8)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __call__(self, x, sigma, **kwargs):
        import comfy.utils

        packed = len(self._full_shapes) > 1
        if packed:
            streams = list(comfy.utils.unpack_latents(x, self._full_shapes))
            video_x = streams[0]
            audio_x = streams[1] if len(streams) > 1 else None
        else:
            video_x = x
            audio_x = None
        if self._weights.device != video_x.device:
            self._weights = self._weights.to(device=video_x.device)
            self._windows = [
                (s, e, w.to(device=video_x.device) if isinstance(w, torch.Tensor) else w)
                for s, e, w in self._windows
            ]
        video_acc = torch.zeros_like(video_x, dtype=torch.float32)
        audio_acc = (
            torch.zeros_like(audio_x, dtype=torch.float32) if audio_x is not None else None
        )
        mask = kwargs.get("denoise_mask")
        saved_noise = getattr(self._inner, "noise", None)
        saved_image = getattr(self._inner, "latent_image", None)
        try:
            for ax_start, ax_end, window in self._windows:
                tile_kw = dict(kwargs)
                if mask is not None:
                    cropped_mask, _ = _crop_packed(
                        mask, self._full_shapes, self._tile_axis, ax_start, ax_end
                    )
                    tile_kw["denoise_mask"] = cropped_mask
                if saved_noise is not None:
                    cropped_noise, _ = _crop_packed(
                        saved_noise, self._full_shapes, self._tile_axis, ax_start, ax_end
                    )
                    self._inner.noise = cropped_noise
                if saved_image is not None:
                    cropped_image, _ = _crop_packed(
                        saved_image, self._full_shapes, self._tile_axis, ax_start, ax_end
                    )
                    self._inner.latent_image = cropped_image
                video_tile = _crop_spatial(video_x, self._tile_axis, ax_start, ax_end)
                if packed:
                    tile_streams = [video_tile]
                    if audio_x is not None:
                        tile_streams.append(audio_x)
                    tile_x, tile_shapes = comfy.utils.pack_latents(tile_streams)
                else:
                    tile_x = video_tile
                    tile_shapes = [tuple(video_tile.shape)]
                shape_restores = _override_latent_shapes(
                    self._inner, self._guider, tile_shapes
                )
                self._cropper.apply(self._tile_axis, ax_start, ax_end)
                try:
                    with _tile_rope_context(
                        self._full_hw[0],
                        self._full_hw[1],
                        self._tile_axis,
                        ax_start,
                        ax_end,
                    ):
                        pred = self._inner(tile_x, sigma, **tile_kw)
                finally:
                    self._cropper.restore()
                    for restore in reversed(shape_restores):
                        restore()
                if packed:
                    pred_streams = list(comfy.utils.unpack_latents(pred, tile_shapes))
                    pred_video = pred_streams[0].float()
                    if audio_acc is not None and len(pred_streams) > 1:
                        audio_acc += pred_streams[1].float()
                else:
                    pred_video = pred.float()
                if self._tile_axis == "H":
                    video_acc[:, :, :, ax_start:ax_end, :] += pred_video * window
                else:
                    video_acc[:, :, :, :, ax_start:ax_end] += pred_video * window
        finally:
            if saved_noise is not None:
                self._inner.noise = saved_noise
            if saved_image is not None:
                self._inner.latent_image = saved_image
            self._cropper.restore()

        video_acc = video_acc / self._weights
        if not packed:
            return video_acc.to(dtype=video_x.dtype)
        merged = [video_acc.to(dtype=video_x.dtype)]
        if audio_acc is not None:
            audio_acc = audio_acc / float(len(self._windows))
            merged.append(audio_acc.to(dtype=audio_x.dtype))
        out, _ = comfy.utils.pack_latents(merged)
        return out


def _sample_step_synchronized(
    latent_dict,
    noise,
    guider,
    sampler,
    sigmas,
    video_tensor,
    audio_tensor,
    fmt_info,
    full_noise,
    regions,
    tile_axis,
    contexts,
    debug=False,
    on_step_preview=None,
    preview_every=1,
):
    import comfy.model_management
    import comfy.samplers

    del contexts, debug
    full_nested = _h3_make_nested(video_tensor, audio_tensor)
    full_shapes = [tuple(video_tensor.shape)]
    if audio_tensor is not None:
        full_shapes.append(tuple(audio_tensor.shape))
    full_hw = (int(video_tensor.shape[-2]), int(video_tensor.shape[-1]))
    comfy_cb, disable_pbar = _preview_callback(guider, sigmas)
    callback = _chain_step_preview(comfy_cb, on_step_preview, full_shapes, preview_every)
    _clean_minimax_layout(guider)
    cropper = _TileCondCrop(guider, full_hw)
    original_extra = dict(getattr(sampler, "extra_options", {}) or {})
    original_inpaint = dict(getattr(sampler, "inpaint_options", {}) or {})
    orig_fn = sampler.sampler_function

    def synced_sampler_fn(model, x, step_sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        tiled = _TiledDenoise(
            model,
            guider=guider,
            regions=regions,
            tile_axis=tile_axis,
            full_shapes=full_shapes,
            full_hw=full_hw,
            cropper=cropper,
        )
        try:
            return orig_fn(
                tiled,
                x,
                step_sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **kwargs,
            )
        finally:
            cropper.restore()

    sync_sampler = comfy.samplers.KSAMPLER(
        synced_sampler_fn,
        extra_options=original_extra,
        inpaint_options=original_inpaint,
    )
    try:
        samples = guider.sample(
            full_noise,
            full_nested,
            sync_sampler,
            sigmas,
            denoise_mask=latent_dict.get("noise_mask"),
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
    finally:
        cropper.restore()

    sampled_video, sampled_audio, _ = _h3_extract(samples, False)
    intermediate = comfy.model_management.intermediate_device()
    sampled_video = sampled_video.to(device=intermediate, dtype=video_tensor.dtype)
    if sampled_audio is not None:
        final_audio = sampled_audio.to(device=intermediate)
    elif audio_tensor is not None:
        final_audio = audio_tensor.to(intermediate)
    else:
        final_audio = None
    out = dict(latent_dict)
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = _h3_reconstruct(sampled_video, final_audio, fmt_info)
    if video_tensor.device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _refine_seams(
    output,
    full_video_noise,
    audio_tensor,
    full_audio_noise,
    regions,
    tile_axis,
    noise,
    guider,
    sampler,
    sigmas,
    refine_steps,
    device,
    contexts,
    noise_mask,
    debug=False,
):
    if refine_steps <= 0 or sigmas.shape[-1] <= 1:
        return output
    refine_sigmas = sigmas[-(int(refine_steps) + 1) :].clone()
    callback, disable_pbar = _preview_callback(guider, refine_sigmas)
    video_hw = (int(output.shape[-2]), int(output.shape[-1]))
    cropper = _TileCondCrop(guider, video_hw)

    for band_start, band_end in _seam_bands(regions):
        if band_end - band_start < 2:
            continue
        if tile_axis == "H":
            band_latent = output[:, :, :, band_start:band_end, :].contiguous()
            band_noise = full_video_noise[:, :, :, band_start:band_end, :].contiguous()
        else:
            band_latent = output[:, :, :, :, band_start:band_end].contiguous()
            band_noise = full_video_noise[:, :, :, :, band_start:band_end].contiguous()
        band_nested = _h3_make_nested(band_latent, audio_tensor)
        band_noise_nested = _h3_make_nested(band_noise, full_audio_noise)
        band_mask = _crop_noise_mask(noise_mask, tile_axis, band_start, band_end, video_hw)
        _apply_minimax_region(contexts, tile_axis, band_start, band_end, debug)
        cropper.apply(tile_axis, band_start, band_end)
        _clean_minimax_layout(guider)
        try:
            with _tile_rope_context(video_hw[0], video_hw[1], tile_axis, band_start, band_end):
                band_samples = guider.sample(
                    band_noise_nested,
                    band_nested,
                    sampler,
                    refine_sigmas,
                    denoise_mask=band_mask,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=noise.seed,
                )
        finally:
            cropper.restore()
            _restore_minimax_conds(contexts)
        if hasattr(band_samples, "is_nested") and band_samples.is_nested:
            band_video = band_samples.unbind()[0]
        else:
            band_video = band_samples
        band_video = band_video.to(device=device).float()
        length = band_end - band_start
        fade = _make_window_1d(length, length // 2, length - length // 2, torch.float32, device)
        if tile_axis == "H":
            window = fade.view(1, 1, 1, -1, 1)
            output[:, :, :, band_start:band_end, :] = (
                band_video * window + output[:, :, :, band_start:band_end, :] * (1.0 - window)
            )
        else:
            window = fade.view(1, 1, 1, 1, -1)
            output[:, :, :, :, band_start:band_end] = (
                band_video * window + output[:, :, :, :, band_start:band_end] * (1.0 - window)
            )
        del band_latent, band_noise, band_nested, band_noise_nested, band_samples, band_video
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def resolve_tile_plan(
    height: int,
    width: int,
    *,
    n_tiles: int = 2,
    tile_axis: str = "auto",
    tile_overlap: int = 8,
    max_size_for_no_tile: int = 64,
) -> dict[str, Any] | None:
    """Return tile geometry, or None when a single full-frame sample is enough."""
    try:
        n_tiles = int(n_tiles or 1)
    except (TypeError, ValueError):
        n_tiles = 1
    n_tiles = max(1, min(8, n_tiles))
    axis = str(tile_axis or "auto").strip()
    if axis not in TILE_AXES:
        axis = "auto"
    if axis == "auto":
        axis = "H" if int(height) >= int(width) else "W"
    axis_size = int(height) if axis == "H" else int(width)
    try:
        overlap = int(tile_overlap or 0)
    except (TypeError, ValueError):
        overlap = 8
    overlap = max(0, min(32, overlap))
    try:
        max_size = int(max_size_for_no_tile or 64)
    except (TypeError, ValueError):
        max_size = 64
    max_size = max(8, min(256, max_size))
    if n_tiles <= 1 or axis_size <= max_size:
        return None
    regions = _compute_tile_regions(axis_size, n_tiles, overlap)
    if len(regions) <= 1:
        return None
    starts = [start for start, _end in regions]
    tile_size = max(end - start for start, end in regions)
    return {
        "tile_axis": axis,
        "n_tiles": n_tiles,
        "tile_overlap": overlap,
        "max_size_for_no_tile": max_size,
        "axis_size": axis_size,
        "regions": regions,
        "starts": starts,
        "tile_size": tile_size,
    }


def sample_h3_tiled(
    *,
    noise,
    guider,
    sampler,
    sigmas,
    latent: dict,
    n_tiles: int = 2,
    tile_axis: str = "auto",
    tile_overlap: int = 8,
    max_size_for_no_tile: int = 64,
    refine_seams: bool = True,
    refine_steps: int = 8,
    debug: bool = False,
    on_step_preview=None,
    preview_every: int = 1,
) -> dict:
    """Sample an H3 AV latent, spatially tiling video when the canvas is large."""
    import comfy.model_management

    work = dict(latent)
    raw = work["samples"]
    video_tensor, audio_tensor, fmt_info = _h3_extract(raw, debug)
    if video_tensor.dim() != 5:
        raise ValueError(
            f"H3 tiled sample needs 5D video [B,C,T,H,W], got {video_tensor.dim()}D"
        )
    plan = resolve_tile_plan(
        int(video_tensor.shape[-2]),
        int(video_tensor.shape[-1]),
        n_tiles=n_tiles,
        tile_axis=tile_axis,
        tile_overlap=tile_overlap,
        max_size_for_no_tile=max_size_for_no_tile,
    )
    contexts = _prepare_minimax_conds(
        guider, int(video_tensor.shape[-2]), int(video_tensor.shape[-1]), debug
    )
    if plan is None:
        log.info(
            "Director tiled refine: full-frame (n_tiles=%s, %dx%d latent)",
            n_tiles,
            int(video_tensor.shape[-1]),
            int(video_tensor.shape[-2]),
        )
        return _single_pass(
            noise, guider, sampler, sigmas, work,
            video_tensor, audio_tensor, fmt_info, contexts, debug,
            on_step_preview=on_step_preview,
            preview_every=preview_every,
        )

    device = comfy.model_management.get_torch_device()
    video_tensor = video_tensor.to(device=device)
    if audio_tensor is not None:
        audio_tensor = audio_tensor.to(device=device)
    axis = plan["tile_axis"]
    regions = list(plan.get("regions") or [])
    if not regions:
        starts = plan["starts"]
        tile_size = int(plan["tile_size"])
        axis_total = int(video_tensor.shape[-2 if axis == "H" else -1])
        regions = [
            (int(start), int(min(start + tile_size, axis_total)))
            for start in starts
        ]
    H, W = int(video_tensor.shape[-2]), int(video_tensor.shape[-1])
    full_nested = _h3_make_nested(video_tensor, audio_tensor)
    full_noise = noise.generate_noise({"samples": full_nested})
    log.info(
        "Director tiled refine: axis=%s tiles=%d overlap=%d latent=%dx%d regions=%s",
        axis,
        len(regions),
        int(plan["tile_overlap"]),
        W,
        H,
        regions,
    )
    del refine_seams, refine_steps
    log.info("Director tiled refine: step-sync merge")
    return _sample_step_synchronized(
        work, noise, guider, sampler, sigmas,
        video_tensor, audio_tensor, fmt_info,
        full_noise, regions, axis, contexts, debug,
        on_step_preview=on_step_preview,
        preview_every=preview_every,
    )
