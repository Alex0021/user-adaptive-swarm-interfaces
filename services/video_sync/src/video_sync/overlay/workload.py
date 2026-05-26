"""Overlay showing the inferred cognitive workload state and probabilities."""

from __future__ import annotations

import cv2
import numpy as np

from video_sync.colors import WORKLOAD_COLORS_BGR, WORKLOAD_LABELS
from video_sync.overlay.base import BaseOverlay, lookup_row


class WorkloadStateOverlay(BaseOverlay):
    width = 320

    def render(self, frame: np.ndarray, ts_ms: int, trial: dict) -> np.ndarray:
        row = lookup_row(trial["inference"], ts_ms)
        x, y = self.position
        w, h = self.size

        cv2.putText(
            frame, "Workload", (x + 8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )
        if row is None:
            return frame

        state = int(row["filtered_state"])
        color = WORKLOAD_COLORS_BGR[state]
        label = WORKLOAD_LABELS[state]
        cv2.putText(
            frame, label, (x + 110, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )

        probs = [row["prob_low"], row["prob_medium"], row["prob_high"]]
        bar_y = y + 34
        bar_h = 14
        bar_x = x + 8
        bar_w = w - 16
        gap = 6
        for i, p in enumerate(probs):
            yi = bar_y + i * (bar_h + gap)
            cv2.rectangle(frame, (bar_x, yi), (bar_x + bar_w, yi + bar_h), (60, 60, 60), -1)
            filled = int(bar_w * float(p))
            c = WORKLOAD_COLORS_BGR[i]
            cv2.rectangle(frame, (bar_x, yi), (bar_x + filled, yi + bar_h), c, -1)
        return frame
