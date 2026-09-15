"""Release GPU memory between MiniMax H3 Director segment runs."""

from __future__ import annotations

import gc
import logging
import time

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")


def _evict_dead_loaded_models() -> int:
    """Remove dead ModelPatcher slots that ComfyUI cannot unload itself."""
    try:
        import comfy.model_management as mm
    except Exception:
        return 0
    models = getattr(mm, "current_loaded_models", None)
    if not models:
        return 0
    evicted = 0
    for index in range(len(models) - 1, -1, -1):
        current = models[index]
        try:
            if not current.is_dead():
                continue
            name = "?"
            try:
                real = current.real_model()
                name = type(real).__name__ if real is not None else "?"
            except Exception:
                pass
            models.pop(index)
            evicted += 1
            log.info("MiniMax H3 Director: evicted dead LoadedModel slot (%s)", name)
        except Exception:
            continue
    return evicted


def cleanup_segment_vram(
    *,
    enabled: bool = True,
    unload_models: bool = True,
    reason: str = "segment",
) -> None:
    """Release segment GPU memory: gc, optional unload of ComfyUI models, empty CUDA cache."""
    if not enabled:
        return
    started = time.perf_counter()
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.cleanup_models_gc()
        _evict_dead_loaded_models()
        if unload_models:
            mm.unload_all_models()
            mm.cleanup_models()
        _evict_dead_loaded_models()
        gc.collect()
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return
    elapsed = time.perf_counter() - started
    message = (
        "MiniMax H3 Director: VRAM cleanup %.2fs "
        "(reason=%s, models=%s)"
    )
    args = (elapsed, str(reason or "segment"), "unloaded" if unload_models else "kept")
    if elapsed >= 1.0:
        log.info(message, *args)
    else:
        log.debug(message, *args)
    if unload_models:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (models unloaded, cache cleared)")
    else:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (cache cleared, models kept loaded)")
