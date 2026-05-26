"""Base class for video overlays."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

BAR_HEIGHT = 110  # height of the top overlay bar (= workload widget height)
BAR_BG = (35, 35, 35)
BAR_BORDER = (90, 90, 90)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def lookup_row(df: pd.DataFrame, ts_ms: int, time_col: str = "timestamp"):
    """Return the row with the largest `time_col` value <= `ts_ms`, or None."""
    if df is None or df.empty:
        return None
    idx = df[time_col].searchsorted(ts_ms, side="right") - 1
    if idx < 0:
        return None
    return df.iloc[idx]


class BaseOverlay(ABC):
    """Abstract overlay rendered as a horizontal cell inside the top bar."""

    width: int = 260

    def __init__(self) -> None:
        self.position: tuple[int, int] = (0, 0)
        self.size: tuple[int, int] = (self.width, BAR_HEIGHT)

    def set_cell(self, x: int, y: int, width: int, height: int) -> None:
        self.position = (x, y)
        self.size = (width, height)

    @abstractmethod
    def render(
        self, frame: np.ndarray, ts_ms: int, trial: dict
    ) -> np.ndarray:  # pragma: no cover - abstract
        ...
