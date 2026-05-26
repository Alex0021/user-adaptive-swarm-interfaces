"""Time alignment between a video timeline and experiment data timestamps.

Supports one or two sync points:

- **Single point** — one (video time, data timestamp) pair. The mapping is a
  pure shift using `fps` to convert video frames to seconds.
- **Two points** — two pairs (start, end). The mapping linearly interpolates
  between them, which absorbs constant fps drift / wall-clock skew over the
  course of the recording.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VideoSync:
    """Map between video frame indices and Unix-ms data timestamps.

    If `anchor_end_ms` and `sync_offset_end_s` are both provided, two-point
    linear interpolation is used. Otherwise a single-point shift is applied.
    """

    anchor_ms: int
    sync_offset_s: float
    fps: float
    anchor_end_ms: int | None = None
    sync_offset_end_s: float | None = None

    def __post_init__(self) -> None:
        if (self.anchor_end_ms is None) ^ (self.sync_offset_end_s is None):
            raise ValueError(
                "Both anchor_end_ms and sync_offset_end_s must be set together."
            )
        if self.anchor_end_ms is not None:
            if self.anchor_end_ms == self.anchor_ms:
                raise ValueError("End anchor equals start anchor; cannot interpolate.")

    @property
    def two_point(self) -> bool:
        return self.anchor_end_ms is not None

    def _ms_per_video_second(self) -> float:
        """How many data-ms elapse per second of video time."""
        if not self.two_point:
            return 1000.0
        d_data = self.anchor_end_ms - self.anchor_ms
        d_video = (self.sync_offset_end_s or 0.0) - self.sync_offset_s
        return d_data / d_video

    def frame_to_timestamp(self, frame_idx: int) -> int:
        video_time_s = frame_idx / self.fps
        ms_per_s = self._ms_per_video_second()
        return int(self.anchor_ms + (video_time_s - self.sync_offset_s) * ms_per_s)

    def timestamp_to_frame(self, ts_ms: int) -> int:
        ms_per_s = self._ms_per_video_second()
        video_time_s = self.sync_offset_s + (ts_ms - self.anchor_ms) / ms_per_s
        return max(0, int(round(video_time_s * self.fps)))
