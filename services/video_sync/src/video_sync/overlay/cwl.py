"""Overlay showing the current adaptation step."""

from __future__ import annotations

import cv2
import numpy as np

from video_sync.overlay.base import BaseOverlay, lookup_row


class AdaptationStepOverlay(BaseOverlay):
    width = 240

    def render(self, frame: np.ndarray, ts_ms: int, trial: dict) -> np.ndarray:
        x, y = self.position
        cv2.putText(
            frame, "Adaptation Step", (x + 8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )
        row = lookup_row(trial["commands"], ts_ms)
        if row is None:
            return frame
        step = int(row.get("cwl_current_step", 0))
        total = int(row.get("cwl_total_steps", 0))
        cv2.putText(
            frame, f"{step} / {max(total - 1, 0)}", (x + 8, y + 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (180, 255, 180), 2, cv2.LINE_AA,
        )
        return frame


# Back-compat alias
CWLStepOverlay = AdaptationStepOverlay
