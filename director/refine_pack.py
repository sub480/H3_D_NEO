"""Built-in refine configuration for MiniMax H3 Director."""

from __future__ import annotations

import math
from typing import Any

from ..lib.image_prep import MINIMAX_CANVAS_STRIDE, ensure_minimax_canvas

MMX_DIR_REFINE = "MMX_DIR_REFINE"

REFINE_MODES = ("refine", "upscale", "latent_upscale")
SEED_MODES = ("inherit", "offset")
UPSCALE_METHODS = ("lanczos", "nvidia_rtx_vsr", "h3_latent")
TILE_AXES = ("auto", "H", "W")
MAX_REFINE_PASSES = 9999
DEFAULT_N_TILES = 4
DEFAULT_TILE_OVERLAP = 8
DEFAULT_MAX_SIZE_FOR_NO_TILE = 64
DEFAULT_SEAM_REFINE_STEPS = 8
# 海螺参考生视频二采：ManualSigmas 4 个数 = euler 3 步。
HAILUO_REFINE_SIGMAS = (0.85, 0.7250, 0.4219, 0.0)
DEFAULT_REFINE_SIGMA_SAMPLER = "euler"
DEFAULT_REFINE_SAMPLE_STEPS = 3
DEFAULT_REFINE_SCHEDULER = "simple"
DEFAULT_REFINE_DENOISE = 0.35
DEFAULT_REFINE_EXTRA_STEPS = 1
DEFAULT_REFINE_START_AT_SIGMA = 0.70
DEFAULT_REFINE_END_AT_SIGMA = 0.0
SIGMA_SPACINGS = ("cosine", "linear", "exponential")
DEFAULT_SIGMA_SPACING = "cosine"


def parse_refine_sigmas(raw: Any, *, fallback: bool = False) -> tuple[float, ...]:
    """Parse ManualSigmas text or a BasicScheduler SIGMAS tensor."""
    vals: list[float] = []
    from_tensor = is_refine_sigmas_tensor(raw)
    if from_tensor:
        try:
            vals = [float(x) for x in raw.detach().float().cpu().reshape(-1).tolist()]
        except Exception:
            vals = []
    elif isinstance(raw, (list, tuple)):
        for part in raw:
            try:
                vals.append(float(part))
            except (TypeError, ValueError):
                continue
    else:
        text = str(raw or "").replace(";", ",").replace("\n", ",")
        for part in text.split(","):
            token = str(part).strip()
            if not token:
                continue
            try:
                vals.append(float(token))
            except (TypeError, ValueError):
                continue
    if len(vals) < 2:
        if not fallback:
            raise ValueError(
                "Refine SIGMAS 至少需要 2 个数（步数 + 结尾 0）。"
                "请检查 BasicScheduler 的 steps / denoise。"
            )
        return HAILUO_REFINE_SIGMAS
    if abs(vals[-1]) > 1e-8:
        vals.append(0.0)
    return tuple(vals)


def is_refine_sigmas_tensor(raw: Any) -> bool:
    return raw is not None and not isinstance(raw, (str, bytes, list, tuple)) and hasattr(raw, "reshape")


def densify_refine_sigmas(
    sigmas,
    *,
    extra_steps: int = 0,
    start_at_sigma: float = DEFAULT_REFINE_START_AT_SIGMA,
    end_at_sigma: float = DEFAULT_REFINE_END_AT_SIGMA,
    spacing: str = DEFAULT_SIGMA_SPACING,
):
    """Insert extra low-sigma steps (same idea as H3 Sigma Refiner)."""
    extra = _clamp_int(extra_steps, 0, 0, 15)
    if extra <= 0 or sigmas is None:
        return sigmas
    try:
        import torch
    except Exception:
        return sigmas
    if torch.is_tensor(sigmas):
        cpu = sigmas.detach().float().cpu().reshape(-1)
        device, dtype = sigmas.device, sigmas.dtype
    else:
        try:
            cpu = torch.tensor([float(x) for x in sigmas], dtype=torch.float32)
        except Exception:
            return sigmas
        device, dtype = cpu.device, cpu.dtype
    if cpu.numel() < 2:
        return sigmas
    try:
        start = float(start_at_sigma)
    except (TypeError, ValueError):
        start = DEFAULT_REFINE_START_AT_SIGMA
    try:
        end = float(end_at_sigma)
    except (TypeError, ValueError):
        end = DEFAULT_REFINE_END_AT_SIGMA
    idx = -1
    for i, value in enumerate(cpu.tolist()):
        if value <= start:
            idx = i
            break
    if idx < 0 or idx >= int(cpu.numel()) - 1:
        return sigmas
    head = cpu[:idx]
    a = float(cpu[idx].item())
    b = max(end, float(cpu[-1].item()))
    new_len = int(cpu.numel()) - idx + extra
    t = torch.linspace(0.0, 1.0, steps=max(2, new_len))
    curve = str(spacing or DEFAULT_SIGMA_SPACING).strip().lower()
    if curve == "exponential":
        alpha = 3.0
        factor = (torch.exp(t * alpha) - 1.0) / (math.exp(alpha) - 1.0)
    elif curve == "linear":
        factor = t
    else:
        factor = (1.0 - torch.cos(t * math.pi)) / 2.0
    tail = a + (b - a) * factor
    if abs(float(cpu[-1].item())) <= 1e-8 and b > 0.0:
        tail = torch.cat([tail, torch.tensor([0.0])])
    out = torch.cat([head, tail]) if head.numel() else tail
    return out.to(device=device, dtype=dtype)


def resolve_latent_upscale_ref(raw: Any) -> tuple[Any, str]:
    """Return (loaded_module_or_None, filename)."""
    if raw is None or raw is False:
        return None, ""
    if isinstance(raw, str):
        name = raw.strip()
        return None, "" if name.startswith("(") else name
    if isinstance(raw, dict):
        name = str(raw.get("name") or raw.get("model_name") or "").strip()
        if name.startswith("("):
            name = ""
        return raw.get("model"), name
    name = str(getattr(raw, "name", None) or getattr(raw, "_h3_name", None) or "").strip()
    if hasattr(raw, "parameters") or hasattr(raw, "model"):
        return raw, name or type(raw).__name__
    return None, name


def refine_needs_canvas(pack: dict[str, Any] | None) -> bool:
    mode = str((pack or {}).get("mode") or "").strip().lower()
    return mode in {"upscale", "latent_upscale"}


def refine_uses_h3_latent(pack: dict[str, Any] | None) -> bool:
    pack = pack or {}
    mode = str(pack.get("mode") or "").strip().lower()
    if mode == "latent_upscale":
        return True
    method = str(pack.get("upscale_method") or "").strip().lower()
    return mode == "upscale" and method == "h3_latent"


def latent_upscale_model_name(pack: dict[str, Any] | None) -> str:
    pack = pack or {}
    _model, name = resolve_latent_upscale_ref(
        pack.get("latent_upscale_ref")
        if pack.get("latent_upscale_ref") is not None
        else pack.get("latent_upscale_model")
    )
    if name:
        return name
    return str(pack.get("h3_latent_model") or "").strip()


FOLLOW_DIRECTOR_ASPECT = "跟随导演台"
CUSTOM_ASPECT_RATIO = "自定义"
DEFAULT_UPSCALE_MEGAPIXELS = 2.0

# Same labels/ratios as Director output bar / official ResolutionSelector.
RESOLUTION_ASPECTS = (
    ("1:1 (方形)", 1, 1),
    ("2:3 (竖版照片)", 2, 3),
    ("3:2 (横版照片)", 3, 2),
    ("3:4 (竖版标准)", 3, 4),
    ("4:3 (标准)", 4, 3),
    ("9:16 (竖屏)", 9, 16),
    ("16:9 (宽屏)", 16, 9),
    ("21:9 (超宽)", 21, 9),
)

ASPECT_RATIO_CHOICES = (
    FOLLOW_DIRECTOR_ASPECT,
    *[row[0] for row in RESOLUTION_ASPECTS],
    CUSTOM_ASPECT_RATIO,
)


def infer_upscale_target(base_w: int, base_h: int) -> tuple[int, int]:
    """Default 720p-class canvas from a 480p-class source (snap to H3 ×32)."""
    w, h = int(base_w or 0), int(base_h or 0)
    if w <= 0 or h <= 0:
        return ensure_minimax_canvas(1280, 720)
    if w >= h:
        nh = 720
        nw = max(32, round(w * nh / h))
        return ensure_minimax_canvas(nw, nh)
    nw = 720
    nh = max(32, round(h * nw / w))
    return ensure_minimax_canvas(nw, nh)


def _closest_aspect_label(width: int, height: int) -> str | None:
    w, h = int(width or 0), int(height or 0)
    if w <= 0 or h <= 0:
        return None
    ratio = w / h
    best_label = None
    best_err = 1e9
    for label, wr, hr in RESOLUTION_ASPECTS:
        err = abs(ratio - (wr / hr))
        if err < best_err:
            best_label = label
            best_err = err
    if best_err > 0.08:
        return None
    return best_label


def canvas_from_director_aspect(
    base_w: int,
    base_h: int,
    megapixels: float,
) -> tuple[int, int]:
    """Keep Director output aspect; size from Refine megapixels."""
    w, h = int(base_w or 0), int(base_h or 0)
    if w <= 0 or h <= 0:
        return 0, 0
    label = _closest_aspect_label(w, h)
    if label:
        resolved = resolution_from_selector(label, megapixels)
        if resolved is not None:
            return resolved
    try:
        mp = float(megapixels)
    except (TypeError, ValueError):
        mp = DEFAULT_UPSCALE_MEGAPIXELS
    mp = min(16.0, max(0.1, mp))
    stride = MINIMAX_CANVAS_STRIDE
    scale = math.sqrt((mp * 1024 * 1024) / (w * h))
    nw = int(round((w * scale) / stride) * stride)
    nh = int(round((h * scale) / stride) * stride)
    return ensure_minimax_canvas(max(nw, stride), max(nh, stride))


def normalize_aspect_ratio(aspect_ratio: str | None) -> str:
    v = str(aspect_ratio).strip()
    if not v:
        return FOLLOW_DIRECTOR_ASPECT
    if v in ASPECT_RATIO_CHOICES:
        return v
    return FOLLOW_DIRECTOR_ASPECT


def is_follow_director_aspect(aspect_ratio: str | None) -> bool:
    return normalize_aspect_ratio(aspect_ratio) == FOLLOW_DIRECTOR_ASPECT


def is_custom_aspect_ratio(aspect_ratio: str | None) -> bool:
    return normalize_aspect_ratio(aspect_ratio) == CUSTOM_ASPECT_RATIO


def resolution_from_selector(
    aspect_ratio: str,
    megapixels: float,
    multiple: int = MINIMAX_CANVAS_STRIDE,
) -> tuple[int, int] | None:
    """Director ResolutionSelector math: aspect + MP → W×H snapped to ×32."""
    ar = normalize_aspect_ratio(aspect_ratio)
    if ar in {FOLLOW_DIRECTOR_ASPECT, CUSTOM_ASPECT_RATIO}:
        return None
    row = next((r for r in RESOLUTION_ASPECTS if r[0] == ar), None)
    if row is None:
        return None
    _, w_ratio, h_ratio = row
    try:
        mp = float(megapixels)
    except (TypeError, ValueError):
        mp = DEFAULT_UPSCALE_MEGAPIXELS
    mp = min(16.0, max(0.1, mp))
    mult = max(8, int(multiple or MINIMAX_CANVAS_STRIDE))
    scale = math.sqrt((mp * 1024 * 1024) / (w_ratio * h_ratio))
    width = int(round((w_ratio * scale) / mult) * mult)
    height = int(round((h_ratio * scale) / mult) * mult)
    return ensure_minimax_canvas(max(width, mult), max(height, mult))


def resolve_refine_target(
    *,
    aspect_ratio: str = FOLLOW_DIRECTOR_ASPECT,
    megapixels: float = DEFAULT_UPSCALE_MEGAPIXELS,
    width: int = 0,
    height: int = 0,
) -> tuple[int, int]:
    """Return (0, 0) to follow Director canvas; otherwise an explicit ×32 canvas."""
    ar = normalize_aspect_ratio(aspect_ratio)
    w, h = int(width or 0), int(height or 0)
    if is_follow_director_aspect(ar):
        return 0, 0
    if is_custom_aspect_ratio(ar):
        cw = w or 1280
        ch = h or 720
        return ensure_minimax_canvas(max(cw, 32), max(ch, 32))
    resolved = resolution_from_selector(ar, megapixels)
    if resolved is not None:
        return resolved
    return 0, 0


def _clamp_int(raw, default: int, lo: int, hi: int) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def _clamp_float(raw, default: float, lo: float, hi: float) -> float:
    try:
        n = float(raw)
    except (TypeError, ValueError):
        n = default
    if n != n:  # NaN
        n = default
    return max(lo, min(hi, n))


def _as_bool(raw, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off", ""}:
        return False
    return default


def _tile_fields(
    *,
    n_tiles=DEFAULT_N_TILES,
    tile_axis="auto",
    tile_overlap=DEFAULT_TILE_OVERLAP,
    max_size_for_no_tile=DEFAULT_MAX_SIZE_FOR_NO_TILE,
    refine_seams=True,
    refine_steps=DEFAULT_SEAM_REFINE_STEPS,
    tile_enabled=None,
    **_kwargs,
) -> dict[str, Any]:
    axis = str(tile_axis or "auto").strip()
    if axis not in TILE_AXES:
        axis = "auto"
    enabled = True if tile_enabled is None else _as_bool(tile_enabled, True)
    tiles = _clamp_int(n_tiles, DEFAULT_N_TILES, 1, 8)
    if not enabled:
        tiles = 1
    return {
        "tile_enabled": enabled,
        "n_tiles": tiles,
        "tile_axis": axis,
        "tile_overlap": _clamp_int(tile_overlap, DEFAULT_TILE_OVERLAP, 0, 32),
        "max_size_for_no_tile": _clamp_int(
            max_size_for_no_tile, DEFAULT_MAX_SIZE_FOR_NO_TILE, 8, 256
        ),
        "refine_seams": bool(refine_seams),
        "refine_steps": _clamp_int(refine_steps, DEFAULT_SEAM_REFINE_STEPS, 1, 25),
    }


def refine_tile_cfg(pack: dict[str, Any] | None) -> dict[str, Any] | None:
    """Tiling kwargs for second-sample. None = full-frame SamplerCustomAdvanced."""
    pack = pack or {}
    fields = _tile_fields(**pack)
    if not fields.get("tile_enabled", True):
        return None
    if int(fields["n_tiles"]) <= 1:
        return None
    return fields


def pack_refine(
    *,
    mode: str = "refine",
    passes: int = 1,
    seed_mode: str = "inherit",
    aspect_ratio: str = FOLLOW_DIRECTOR_ASPECT,
    megapixels: float = DEFAULT_UPSCALE_MEGAPIXELS,
    width: int = 0,
    height: int = 0,
    skip_fl2v: bool = False,
    upscale_method: str = "h3_latent",
    sample_model=None,
    sample_model_r2v=None,
    latent_upscale_model=None,
    upscale_model=None,
    sampler: str = "",
    confirm_first_pass: bool = False,
    n_tiles: int = DEFAULT_N_TILES,
    tile_axis: str = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    max_size_for_no_tile: int = DEFAULT_MAX_SIZE_FOR_NO_TILE,
    refine_seams: bool = True,
    refine_steps: int = DEFAULT_SEAM_REFINE_STEPS,
    tile_enabled=None,
    sample_steps: int = 0,
    scheduler: str = "",
    denoise: float = DEFAULT_REFINE_DENOISE,
    extra_steps: int = 0,
    start_at_sigma: float = DEFAULT_REFINE_START_AT_SIGMA,
    end_at_sigma: float = DEFAULT_REFINE_END_AT_SIGMA,
    spacing: str = DEFAULT_SIGMA_SPACING,
    builtin: bool = False,
) -> dict[str, Any]:
    mode = str(mode or "refine").strip().lower()
    if mode not in REFINE_MODES:
        mode = "refine"
    seed_mode = str(seed_mode or "inherit").strip().lower()
    if seed_mode not in SEED_MODES:
        seed_mode = "inherit"
    method = str(upscale_method or "h3_latent").strip().lower()
    if method not in UPSCALE_METHODS:
        method = "h3_latent"
    sampler = str(sampler or DEFAULT_REFINE_SIGMA_SAMPLER).strip() or DEFAULT_REFINE_SIGMA_SAMPLER
    ar = normalize_aspect_ratio(aspect_ratio)
    tw, th = resolve_refine_target(
        aspect_ratio=ar,
        megapixels=megapixels,
        width=width,
        height=height,
    )
    latent_mod, latent_name = resolve_latent_upscale_ref(latent_upscale_model)
    return {
        "enabled": True,
        "mode": mode,
        "passes": refine_passes_for({"passes": passes}),
        "seed_mode": seed_mode,
        "aspect_ratio": ar,
        "megapixels": float(megapixels or DEFAULT_UPSCALE_MEGAPIXELS),
        "target_width": tw,
        "target_height": th,
        "skip_fl2v": bool(skip_fl2v),
        "upscale_method": method,
        "upscale_model": upscale_model,
        "has_upscale_model": upscale_model is not None,
        "sample_model": sample_model,
        "has_sample_model": sample_model is not None,
        "sample_model_r2v": sample_model_r2v,
        "has_sample_model_r2v": sample_model_r2v is not None,
        "latent_upscale_ref": latent_upscale_model,
        "latent_upscale_module": latent_mod,
        "latent_upscale_model": latent_name,
        "h3_latent_model": latent_name,
        "has_latent_upscale_model": latent_upscale_model is not None,
        "sampler": sampler,
        "confirm_first_pass": bool(confirm_first_pass),
        "builtin": bool(builtin),
        "sample_steps": _clamp_int(sample_steps, 0, 0, 200),
        "scheduler": str(scheduler or "").strip(),
        "denoise": float(denoise if denoise is not None else DEFAULT_REFINE_DENOISE),
        "extra_steps": _clamp_int(extra_steps, 0, 0, 15),
        "start_at_sigma": float(start_at_sigma if start_at_sigma is not None else DEFAULT_REFINE_START_AT_SIGMA),
        "end_at_sigma": float(end_at_sigma if end_at_sigma is not None else DEFAULT_REFINE_END_AT_SIGMA),
        "spacing": (
            str(spacing or DEFAULT_SIGMA_SPACING).strip().lower()
            if str(spacing or "").strip().lower() in SIGMA_SPACINGS
            else DEFAULT_SIGMA_SPACING
        ),
        **_tile_fields(
            n_tiles=n_tiles,
            tile_axis=tile_axis,
            tile_overlap=tile_overlap,
            max_size_for_no_tile=max_size_for_no_tile,
            refine_seams=refine_seams,
            refine_steps=refine_steps,
            tile_enabled=tile_enabled,
        ),
    }


def pack_director_builtin_refine(
    *,
    enabled=False,
    refine_model=None,
    refine_model_r2v=None,
    upscale_model=None,
    **widgets,
) -> dict[str, Any] | None:
    """Director in-node 二采 widgets → refine pack. None when disabled."""
    if not _as_bool(enabled, False):
        return None
    tile_on = _as_bool(widgets.get("refine_tile"), True)
    n_tiles = widgets.get("refine_n_tiles", DEFAULT_N_TILES)
    if not tile_on:
        n_tiles = 1
    try:
        denoise = float(widgets.get("refine_denoise", DEFAULT_REFINE_DENOISE))
    except (TypeError, ValueError):
        denoise = DEFAULT_REFINE_DENOISE
    return pack_refine(
        mode=widgets.get("refine_mode") or "refine",
        passes=widgets.get("refine_passes") or 1,
        seed_mode=widgets.get("refine_seed_mode") or "inherit",
        aspect_ratio=widgets.get("refine_aspect_ratio") or FOLLOW_DIRECTOR_ASPECT,
        megapixels=widgets.get("refine_megapixels") or DEFAULT_UPSCALE_MEGAPIXELS,
        width=widgets.get("refine_width") or 1280,
        height=widgets.get("refine_height") or 720,
        skip_fl2v=_as_bool(widgets.get("refine_skip_fl2v"), False),
        n_tiles=n_tiles,
        tile_axis=widgets.get("refine_tile_axis") or "auto",
        tile_overlap=widgets.get("refine_tile_overlap", DEFAULT_TILE_OVERLAP),
        max_size_for_no_tile=widgets.get(
            "refine_max_size_for_no_tile", DEFAULT_MAX_SIZE_FOR_NO_TILE
        ),
        refine_seams=_as_bool(widgets.get("refine_seams"), True),
        refine_steps=widgets.get("refine_seam_steps", DEFAULT_SEAM_REFINE_STEPS),
        tile_enabled=tile_on,
        upscale_method=widgets.get("refine_upscale_method") or "h3_latent",
        sample_model=refine_model,
        sample_model_r2v=refine_model_r2v,
        latent_upscale_model=widgets.get("refine_latent_upscale_model"),
        upscale_model=upscale_model,
        sampler=widgets.get("refine_sampler") or DEFAULT_REFINE_SIGMA_SAMPLER,
        sample_steps=widgets.get("refine_sample_steps") or DEFAULT_REFINE_SAMPLE_STEPS,
        scheduler=widgets.get("refine_scheduler") or DEFAULT_REFINE_SCHEDULER,
        denoise=denoise,
        extra_steps=widgets.get("refine_extra_steps", DEFAULT_REFINE_EXTRA_STEPS),
        start_at_sigma=widgets.get("refine_start_at_sigma", DEFAULT_REFINE_START_AT_SIGMA),
        end_at_sigma=widgets.get("refine_end_at_sigma", DEFAULT_REFINE_END_AT_SIGMA),
        spacing=widgets.get("refine_spacing") or DEFAULT_SIGMA_SPACING,
        builtin=True,
    )


def normalize_refine_pack(
    raw,
    *,
    base_width: int = 0,
    base_height: int = 0,
) -> dict[str, Any] | None:
    """Director execute: None if unconnected / invalid."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("enabled") is False:
        return None
    mode = str(raw.get("mode") or "refine").strip().lower()
    if mode not in REFINE_MODES:
        mode = "refine"
    ar = normalize_aspect_ratio(raw.get("aspect_ratio"))
    tw, th = int(raw.get("target_width") or 0), int(raw.get("target_height") or 0)
    if mode in {"upscale", "latent_upscale"} and (tw <= 0 or th <= 0):
        if is_follow_director_aspect(ar):
            tw, th = canvas_from_director_aspect(
                base_width, base_height, raw.get("megapixels") or DEFAULT_UPSCALE_MEGAPIXELS
            )
        elif not is_custom_aspect_ratio(ar):
            resolved = resolution_from_selector(ar, raw.get("megapixels") or DEFAULT_UPSCALE_MEGAPIXELS)
            if resolved:
                tw, th = resolved
        if tw <= 0 or th <= 0:
            tw, th = infer_upscale_target(base_width, base_height)
    elif tw > 0 and th > 0:
        tw, th = ensure_minimax_canvas(tw, th)
    seed_mode = str(raw.get("seed_mode") or "inherit").strip().lower()
    if seed_mode not in SEED_MODES:
        seed_mode = "inherit"
    method = str(raw.get("upscale_method") or "h3_latent").strip().lower()
    if method not in UPSCALE_METHODS:
        method = "h3_latent"
    sampler = str(raw.get("sampler") or DEFAULT_REFINE_SIGMA_SAMPLER).strip() or DEFAULT_REFINE_SIGMA_SAMPLER
    sample_model = raw.get("sample_model")
    if sample_model is None:
        sample_model = raw.get("model")
    sample_model_r2v = raw.get("sample_model_r2v")
    latent_raw = raw.get("latent_upscale_ref")
    if latent_raw is None:
        latent_raw = raw.get("latent_upscale_model")
    latent_mod, latent_name = resolve_latent_upscale_ref(latent_raw)
    if not latent_name:
        latent_name = str(raw.get("h3_latent_model") or "").strip()
    upscale = raw.get("upscale_model")
    return {
        "enabled": True,
        "mode": mode,
        "passes": refine_passes_for(raw),
        "seed_mode": seed_mode,
        "aspect_ratio": ar,
        "megapixels": float(raw.get("megapixels") or DEFAULT_UPSCALE_MEGAPIXELS),
        "target_width": tw,
        "target_height": th,
        "skip_fl2v": bool(raw.get("skip_fl2v", False)),
        "upscale_method": method,
        "upscale_model": upscale,
        "has_upscale_model": upscale is not None,
        "sample_model": sample_model,
        "has_sample_model": sample_model is not None,
        "sample_model_r2v": sample_model_r2v,
        "has_sample_model_r2v": sample_model_r2v is not None,
        "latent_upscale_ref": latent_raw,
        "latent_upscale_module": latent_mod,
        "latent_upscale_model": latent_name,
        "h3_latent_model": latent_name,
        "has_latent_upscale_model": latent_raw is not None,
        "sampler": sampler,
        "confirm_first_pass": False,
        "builtin": bool(raw.get("builtin")),
        "sample_steps": _clamp_int(raw.get("sample_steps"), 0, 0, 200),
        "scheduler": str(raw.get("scheduler") or "").strip(),
        "denoise": _clamp_float(
            raw.get("denoise"), DEFAULT_REFINE_DENOISE, 0.0, 1.0
        ),
        "extra_steps": _clamp_int(raw.get("extra_steps"), 0, 0, 15),
        "start_at_sigma": _clamp_float(
            raw.get("start_at_sigma"), DEFAULT_REFINE_START_AT_SIGMA, 0.0, 20.0
        ),
        "end_at_sigma": _clamp_float(
            raw.get("end_at_sigma"), DEFAULT_REFINE_END_AT_SIGMA, 0.0, 5.0
        ),
        "spacing": (
            str(raw.get("spacing") or DEFAULT_SIGMA_SPACING).strip().lower()
            if str(raw.get("spacing") or "").strip().lower() in SIGMA_SPACINGS
            else DEFAULT_SIGMA_SPACING
        ),
        **_tile_fields(**raw),
    }


def refine_will_sample(plan, seg) -> bool:
    """True when this segment will run a second sample / upscale pass."""
    pack = getattr(plan, "refine", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return False
    if pack.get("skip_fl2v", False) and getattr(seg, "task_key", "") == "fl2v":
        return False
    return True


def refine_passes_for(pack: dict[str, Any] | None) -> int:
    try:
        n = int((pack or {}).get("passes") or 1)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(MAX_REFINE_PASSES, n))


def refine_model_for(pack: dict[str, Any] | None, fallback, task_key: str | None = None):
    """Resolve the second-pass UNET, including the optional r2v-family override."""
    if str(task_key or "").lower() in {"r2v", "v2v", "rv2v"}:
        custom_r2v = (pack or {}).get("sample_model_r2v")
        if custom_r2v is not None:
            return custom_r2v
    custom = (pack or {}).get("sample_model")
    if custom is None:
        custom = (pack or {}).get("model")
    return fallback if custom is None else custom


def refine_seed_for(pack: dict[str, Any], seed: int, pass_index: int = 0) -> int:
    if pack.get("seed_mode") == "offset":
        return (int(seed) + 1 + int(max(0, pass_index))) % (2**64)
    return int(seed)


def refine_fingerprint(plan) -> dict[str, Any]:
    pack = getattr(plan, "refine", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return {"refine": False}
    return {
        "refine": True,
        "refine_mode": pack.get("mode") or "refine",
        "refine_passes": refine_passes_for(pack),
        "refine_seed_mode": pack.get("seed_mode") or "inherit",
        "refine_target": f"{int(pack.get('target_width') or 0)}x{int(pack.get('target_height') or 0)}",
        "refine_aspect": pack.get("aspect_ratio") or FOLLOW_DIRECTOR_ASPECT,
        "refine_megapixels": round(float(pack.get("megapixels") or 0), 3),
        "refine_upscale_method": pack.get("upscale_method") or "h3_latent",
        "refine_upscale_model": bool(pack.get("has_upscale_model") or pack.get("upscale_model") is not None),
        "refine_latent_upscale_model": latent_upscale_model_name(pack),
        "refine_sampler": pack.get("sampler") or "",
        "refine_sample_model": bool(pack.get("has_sample_model") or pack.get("sample_model") is not None),
        "refine_sample_model_r2v": bool(
            pack.get("has_sample_model_r2v") or pack.get("sample_model_r2v") is not None
        ),
        "refine_skip_fl2v": bool(pack.get("skip_fl2v", False)),
        "refine_n_tiles": int(pack.get("n_tiles") or DEFAULT_N_TILES),
        "refine_tile_axis": pack.get("tile_axis") or "auto",
        "refine_tile_overlap": int(pack.get("tile_overlap") or DEFAULT_TILE_OVERLAP),
        "refine_max_size_for_no_tile": int(
            pack.get("max_size_for_no_tile") or DEFAULT_MAX_SIZE_FOR_NO_TILE
        ),
        "refine_seams": bool(pack.get("refine_seams", True)),
        "refine_seam_steps": int(pack.get("refine_steps") or DEFAULT_SEAM_REFINE_STEPS),
        "refine_tile": bool(pack.get("tile_enabled", True)),
        "refine_builtin": bool(pack.get("builtin")),
        "refine_sample_steps": int(pack.get("sample_steps") or 0),
        "refine_scheduler": pack.get("scheduler") or "",
        "refine_denoise": round(float(pack.get("denoise") or 0), 4),
        "refine_extra_steps": int(pack.get("extra_steps") or 0),
        "refine_start_at_sigma": round(float(pack.get("start_at_sigma") or 0), 4),
        "refine_end_at_sigma": round(float(pack.get("end_at_sigma") or 0), 4),
        "refine_spacing": pack.get("spacing") or DEFAULT_SIGMA_SPACING,
    }


def refine_report_line(plan) -> str | None:
    pack = getattr(plan, "refine", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return None
    mode = pack.get("mode") or "refine"
    extra = ""
    if refine_needs_canvas(pack):
        ar = pack.get("aspect_ratio") or FOLLOW_DIRECTOR_ASPECT
        if refine_uses_h3_latent(pack):
            how = latent_upscale_model_name(pack) or "h3_latent"
        else:
            method = pack.get("upscale_method") or "h3_latent"
            if method == "nvidia_rtx_vsr":
                how = "nvidia_rtx_vsr"
            elif pack.get("has_upscale_model") or pack.get("upscale_model") is not None:
                how = "upscale_model"
            else:
                how = "lanczos"
        extra = (
            f", {ar} → {int(pack.get('target_width') or 0)}×{int(pack.get('target_height') or 0)}"
            f", {how}"
        )
    n_passes = refine_passes_for(pack)
    pass_note = f", passes={n_passes}" if n_passes > 1 else ""
    model_note = (
        ", 二采模型" if (pack.get("has_sample_model") or pack.get("sample_model") is not None) else ""
    )
    if pack.get("has_sample_model_r2v") or pack.get("sample_model_r2v") is not None:
        model_note += ", R2V 二采模型"
    sampler = pack.get("sampler") or DEFAULT_REFINE_SIGMA_SAMPLER
    if mode == "latent_upscale":
        line = f"Refine: ON ({mode}{model_note}{extra})"
    else:
        builtin = bool(pack.get("builtin"))
        if builtin:
            steps = int(pack.get("sample_steps") or 0)
            sched = pack.get("scheduler") or DEFAULT_REFINE_SCHEDULER
            how = f"builtin {sampler} {sched} {steps}-step"
        else:
            how = sampler
        extra_steps = int(pack.get("extra_steps") or 0)
        densify_note = f", +{extra_steps} low-sigma" if extra_steps and builtin else ""
        line = (
            f"Refine: ON ({mode}, {how}{densify_note}"
            f"{pass_note}{model_note}{extra})"
        )
    tile = refine_tile_cfg(pack)
    if tile and mode != "latent_upscale":
        line += (
            f", tile {tile['tile_axis']}×{tile['n_tiles']}"
            f" overlap={tile['tile_overlap']}"
        )
    return line
