"""Overlay showing elapsed trial time as an MM:SS clock."""

from __future__ import annotations

import cv2
import numpy as np

from video_sync.overlay.base import BaseOverlay


class ElapsedTimeOverlay(BaseOverlay):
    width = 200

    def render(self, frame: np.ndarray, ts_ms: int, trial: dict) -> np.ndarray:
        x, y = self.position
        cv2.putText(
            frame, "Race timer", (x + 8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )
        anchor = int(trial.get("anchor_ms", trial["t0_ms"]))
        elapsed_s = (ts_ms - anchor) / 1000.0
        sign = "-" if elapsed_s < 0 else "+"
        a = abs(elapsed_s)
        mm, ss = divmod(int(a), 60)
        cs = int((a - int(a)) * 100)
        cv2.putText(
            frame, f"T{sign}{mm:02d}:{ss:02d}.{cs:02d}", (x + 8, y + 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return frame
