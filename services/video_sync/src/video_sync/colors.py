"""Colormap-based colors for the workload state overlay.

We sample the matplotlib ``magma`` colormap at three points so the
classifier states (Low / Medium / High) map onto a perceptually uniform
dark-to-bright sequence.
"""

from __future__ import annotations

from matplotlib import colormaps

_MAGMA = colormaps["magma"]

# Sample positions chosen to avoid the near-black tail and near-white head.
_SAMPLE_POINTS = {0: 0.30, 1: 0.55, 2: 0.80}


def _rgb_to_bgr(rgba: tuple[float, float, float, float]) -> tuple[int, int, int]:
    r, g, b, _ = rgba
    return (int(b * 255), int(g * 255), int(r * 255))


WORKLOAD_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    state: _rgb_to_bgr(_MAGMA(pos)) for state, pos in _SAMPLE_POINTS.items()
}

WORKLOAD_LABELS: dict[int, str] = {0: "Low", 1: "Medium", 2: "High"}
