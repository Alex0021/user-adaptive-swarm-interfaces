"""Build a crash montage video for a subject."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from video_sync.crashes import (
    CrashEvent,
    detect_all_crashes,
    merge_close_crashes,
)
from video_sync.overlay import BaseOverlay, HorizontalTopBar
from video_sync.render import make_writer, open_video
from video_sync.sync import VideoSync

logger = logging.getLogger(__name__)


def _title_card(
    width: int, height: int, lines: list[str], fps: float, duration_s: float = 1.5
) -> list[np.ndarray]:
    """Generate a list of black frames containing centered text."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, 1.0, 2)[0] for t in lines]
    total_h = sum(s[1] for s in sizes) + 20 * (len(lines) - 1)
    y = (height - total_h) // 2 + sizes[0][1]
    for text, (tw, th) in zip(lines, sizes, strict=True):
        x = (width - tw) // 2
        cv2.putText(frame, text, (x, y), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        y += th + 20
    n = max(1, int(round(duration_s * fps)))
    return [frame.copy() for _ in range(n)]


def build_crash_montage(
    video_path: Path,
    trials: list[dict],
    sync: VideoSync,
    overlays: list[BaseOverlay],
    output_path: Path,
    padding_s: float = 3.0,
    merge_window_s: float = 0.5,
    title_card_s: float = 1.5,
) -> int:
    """Render a concatenated montage of all crash clips. Returns clip count."""
    events = detect_all_crashes(trials)
    if not events:
        logger.warning("No crashes detected; nothing to render.")
        return 0

    groups: list[list[CrashEvent]] = merge_close_crashes(events, merge_window_s)
    logger.info(
        "Detected %d crashes across %d trials → %d clips after merging",
        len(events), len({e.trial_idx for e in events}), len(groups),
    )

    cap, fps, width, height, n_frames_total = open_video(video_path)
    writer = make_writer(output_path, fps, width, height)
    padding_frames = int(round(padding_s * fps))
    bar = HorizontalTopBar(overlays, video_width=width)

    try:
        for clip_i, group in enumerate(groups, start=1):
            first = group[0]
            trial = trials[first.trial_idx]
            center_frame = sync.timestamp_to_frame(first.timestamp_ms)
            start_frame = max(0, center_frame - padding_frames)
            end_frame = min(n_frames_total, center_frame + padding_frames + 1)

            elapsed_s = (first.timestamp_ms - int(trial["t0_ms"])) / 1000.0
            ids = ",".join(str(e.drone_id) for e in group)
            title_lines = [
                f"Clip {clip_i}/{len(groups)} - {trial.get('name', '?')}",
                f"Crash at T+{elapsed_s:0.1f}s  (drone id: {ids})",
            ]
            for tf in _title_card(width, height, title_lines, fps, title_card_s):
                writer.write(tf)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            while frame_idx < end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                ts_ms = sync.frame_to_timestamp(frame_idx)
                frame = bar.render(frame, ts_ms, trial)
                writer.write(frame)
                frame_idx += 1
    finally:
        cap.release()
        writer.release()

    logger.info("Wrote montage with %d clips to %s", len(groups), output_path)
    return len(groups)
