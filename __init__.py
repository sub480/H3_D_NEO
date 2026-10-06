"""H3_D_NEO — timeline plugin for MiniMax-H3 AV generation.

Based on ComfyUI official MiniMax H3 support (PR #15224 / #15228).
Licensed under the Apache License, Version 2.0. See LICENSE.
"""

from .nodes.director import H3_D_NEO
from .nodes.director_sam31 import DirectorSAM31Job

NODE_CLASS_MAPPINGS = {
    "H3_D_NEO": H3_D_NEO,
    "DirectorSAM31Job": DirectorSAM31Job,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3_D_NEO": "H3_D_NEO",
}

WEB_DIRECTORY = "./web/js"

import logging

_log = logging.getLogger("H3_D_NEO")

try:
    from .director.http_routes import register_routes as _register_director_routes

    if not _register_director_routes():
        message = (
            "H3_D_NEO HTTP routes deferred (PromptServer not ready). "
            "Restart ComfyUI if /minimax/director/* returns 404."
        )
        _log.warning(message)
except Exception as _director_routes_exc:
    _log.warning(
        "H3_D_NEO HTTP routes failed to load: %s",
        _director_routes_exc,
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
