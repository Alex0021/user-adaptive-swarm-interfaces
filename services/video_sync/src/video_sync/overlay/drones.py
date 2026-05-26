"""Overlay showing live drone count and cumulative crashes."""

from __future__ import annotations

import cv2
import numpy as np

from video_sync.data import SWARM_SIZE
from video_sync.overlay.base import BaseOverlay


class DroneCountOverlay(BaseOverlay):
    width = 200

    def render(self, frame: np.ndarray, ts_ms: int, trial: dict) -> np.ndarray:
        drones = trial["drones"]
        x, y = self.position

        cv2.putText(
            frame, "Drones", (x + 8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )
        if drones is None or drones.empty:
            return frame

        past = drones[drones["timestamp"] <= ts_ms]
        if past.empty:
            return frame
        latest = past.sort_values("timestamp").groupby("id").tail(1)
        alive = int(latest["alive"].sum())

        crashes = 0
        for _, group in past.groupby("id"):
            a = group.sort_values("timestamp")["alive"].to_numpy()
            if len(a) >= 2:
                crashes += int(((a[:-1] == 1) & (a[1:] == 0)).sum())

        cv2.putText(
            frame, f"{alive}/{SWARM_SIZE} alive", (x + 8, y + 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"crashes: {crashes}", (x + 8, y + 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 2, cv2.LINE_AA,
        )
        return frame
