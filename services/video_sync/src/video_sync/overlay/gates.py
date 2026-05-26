"""Overlay showing gate progress (gates passed / total)."""

from __future__ import annotations

import cv2
import numpy as np

from video_sync.overlay.base import BaseOverlay


class GateProgressOverlay(BaseOverlay):
    width = 280

    def render(self, frame: np.ndarray, ts_ms: int, trial: dict) -> np.ndarray:
        x, y = self.position
        w, _ = self.size
        cv2.putText(
            frame, "Gates", (x + 8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )

        gs = trial.get("gate_status")
        gates = trial.get("gates")
        if gs is None or gates is None or gs.empty:
            return frame

        gs_unique = gs.drop_duplicates(subset="id", keep="first")
        gates_unique = gates.drop_duplicates(subset="id", keep="first")
        total = len(gates_unique)
        passed_mask = (gs_unique["first_pass_timestamp"] > 0) & (
            gs_unique["first_pass_timestamp"] <= ts_ms
        )
        passed = int(passed_mask.sum())

        cv2.putText(
            frame, f"{passed} / {total}", (x + 8, y + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA,
        )

        bar_y = y + 78
        bar_h = 16
        bar_x = x + 8
        bar_w = w - 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        if total > 0:
            filled = int(bar_w * (passed / total))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (0, 200, 120), -1)
        return frame
