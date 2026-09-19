"""MiniMax H3 task-mode inference from connected inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MiniMaxH3Task(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    FL2V = "fl2v"
    R2V = "r2v"
    V2V = "v2v"
    RV2V = "rv2v"


TASK_DESCRIPTIONS = {
    MiniMaxH3Task.T2V: "Text-to AV (no keyframes or references)",
    MiniMaxH3Task.I2V: "Image-to AV — first keyframe conditioning",
    MiniMaxH3Task.FL2V: "First+last keyframe AV",
    MiniMaxH3Task.R2V: "Reference-to AV — subject images (+ optional tags) in prompt",
    MiniMaxH3Task.V2V: "Video edit — source timeline clip as <Video 1> reference",
    MiniMaxH3Task.RV2V: "Video edit with reference images — source <Video 1> + <Picture N>",
}


SUPPORTED_TASK_KEYS = frozenset(t.value for t in MiniMaxH3Task)


@dataclass(frozen=True)
class TaskSummary:
    mode: MiniMaxH3Task
    ref_image_count: int
    ref_video_count: int

    @property
    def label(self) -> str:
        return self.mode.value

    @property
    def description(self) -> str:
        return TASK_DESCRIPTIONS[self.mode]


def infer_task(
    ref_image_count: int,
    ref_video_count: int = 0,
    *,
    has_start_frame: bool = False,
    has_end_frame: bool = False,
) -> MiniMaxH3Task:
    """Infer the H3 mode from connected media.

    ``ref_image_count`` remains the generic reference-image input used by old
    callers.  Keyframes and source video are explicit keyword signals because
    they have different semantics from reference images.
    """
    if ref_video_count > 0:
        return MiniMaxH3Task.RV2V if ref_image_count > 0 else MiniMaxH3Task.V2V
    if has_start_frame and has_end_frame:
        return MiniMaxH3Task.FL2V
    if has_start_frame:
        return MiniMaxH3Task.I2V
    if has_end_frame:
        return MiniMaxH3Task.FL2V
    if ref_image_count > 0:
        return MiniMaxH3Task.R2V
    return MiniMaxH3Task.T2V
