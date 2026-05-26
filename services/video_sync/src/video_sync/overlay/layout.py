"""Horizontal top-bar layout for overlays."""

from __future__ import annotations

import cv2
import numpy as np

from video_sync.overlay.base import BAR_BG, BAR_BORDER, BAR_HEIGHT, BaseOverlay


class HorizontalTopBar:
    """Draws a solid bar across the top of the video and lays out overlays.

    The bar serves two purposes: a uniform backdrop for the overlay cells,
    and an opaque cover for the recording app's title / toolbar.
    """

    def __init__(
        self,
        overlays: list[BaseOverlay],
        video_width: int,
        bar_height: int = BAR_HEIGHT,
        padding: int = 10,
    ):
        self.overlays = overlays
        self.video_width = video_width
        self.bar_height = bar_height
        self.padding = padding
        self._assign_positions()

    def _assign_positions(self) -> None:
        n = len(self.overlays)
        if n == 0:
            return
        gap = self.padding
        total_min = sum(ov.width for ov in self.overlays) + gap * (n + 1)
        if total_min <= self.video_width:
            extra = (self.video_width - total_min) // n
        else:
            extra = 0
        x = gap
        for ov in self.overlays:
            w = ov.width + extra
            ov.set_cell(x, 0, w, self.bar_height)
            x += w + gap

    def render(self, frame: np.ndarray, ts_ms: int, trial: dict) -> np.ndarray:
        cv2.rectangle(
            frame, (0, 0), (frame.shape[1], self.bar_height), BAR_BG, -1
        )
        cv2.line(
            frame, (0, self.bar_height), (frame.shape[1], self.bar_height),
            BAR_BORDER, 1,
        )
        for ov in self.overlays:
            frame = ov.render(frame, ts_ms, trial)
        return frame
