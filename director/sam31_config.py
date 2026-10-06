"""Versioned SAM drawer configuration; no ComfyUI or model imports."""

from __future__ import annotations

import hashlib
import json
import math

DEFAULT_CONFIG = {
    "version": 1,
    "enabled": False,
    "checkpoint": "",
    "threshold": 0.5,
    "max_objects": 4,
    "detect_interval": 5,
    "long_edge": 768,
    "target_group": "",
}


def _number(value, default, low, high, integer=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        result = float(default)
    if not math.isfinite(result):
        result = float(default)
    result = min(high, max(low, result))
    return int(result) if integer else result


def normalize_config(raw=None):
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("SAM3.1 configuration must be a JSON object.")
    if raw.get("version", 1) != 1:
        raise ValueError("Unsupported SAM3.1 configuration version.")
    enabled = raw.get("enabled", False)
    return {
        "version": 1,
        "enabled": enabled is True or str(enabled).lower() in {"1", "true", "yes", "on"},
        "checkpoint": str(raw.get("checkpoint") or "").strip()[:512],
        "threshold": _number(raw.get("threshold"), 0.5, 0, 1),
        "max_objects": _number(raw.get("max_objects"), 4, 1, 16, True),
        "detect_interval": _number(raw.get("detect_interval"), 5, 1, 256, True),
        "long_edge": _number(raw.get("long_edge"), 768, 256, 1024, True),
        "target_group": str(raw.get("target_group") or "")[:128],
    }


def normalize_group(raw=None):
    raw = raw if isinstance(raw, dict) else {}
    if raw.get("version", 1) != 1:
        raise ValueError("Unsupported SAM3.1 group version.")
    selected = raw.get("selected", [])
    if not isinstance(selected, list):
        selected = []
    return {
        "version": 1,
        "prompt": str(raw.get("prompt") or "").strip()[:256],
        "selected": sorted({
            value for value in selected
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 16
        }),
        "result": raw.get("result") if isinstance(raw.get("result"), dict) else None,
        "composite": normalize_composite(raw.get("composite")),
    }


def normalize_composite(raw=None):
    raw = raw if isinstance(raw, dict) else {}
    return {
        "enabled": raw.get("enabled") is True,
        "grow": _number(raw.get("grow"), 0, 0, 64, True),
        "feather": _number(raw.get("feather"), 4, 0, 64, True),
        "blend": _number(raw.get("blend"), 1, 0, 1),
    }


def identity_key(identity):
    payload = json.dumps(
        identity, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sam31_widget_inputs():
    return {
        "sam31_config": (
            "STRING",
            {
                "default": json.dumps(DEFAULT_CONFIG, separators=(",", ":")),
                "multiline": False,
                "tooltip": (
                    "Internal versioned SAM3.1 drawer settings. "
                    "Analysis is invoked from the drawer, not during generation."
                ),
            },
        ),
    }
