"""First-pass SelfLift (progressive dual-resolution) for MiniMax H3 Director."""

from .pack import (
    selflift_enabled,
    selflift_fingerprint,
    selflift_report_line,
    selflift_will_run,
)
from .sample import sample_selflift_stage

__all__ = [
    "selflift_enabled",
    "selflift_fingerprint",
    "selflift_report_line",
    "selflift_will_run",
    "sample_selflift_stage",
]
