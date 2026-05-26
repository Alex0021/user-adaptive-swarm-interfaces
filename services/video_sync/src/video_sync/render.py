"""Main video rendering pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from video_sync.overlay import BaseOverlay, HorizontalTopBar
from video_sync.sync import VideoSync

logger = logging.getLogger(__name__)


def open_video(video_path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, n_frames


def make_writer(
    output_path: Path, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for {output_path}")
    return writer


def render_trial(
    video_path: Path,
    trial: dict,
    sync: VideoSync,
    overlays: list[BaseOverlay],
    output_path: Path,
    trim: bool = True,
) -> None:
    """Render a single trial to `output_path` with all overlays applied.

    If `trim` is True, clip the output to the trial's [t0, last-timestamp] window.
    """
    cap, fps, width, height, _ = open_video(video_path)
    writer = make_writer(output_path, fps, width, height)

    t0_ms = int(trial["t0_ms"])
    end_ts_candidates = [
        trial[key]["timestamp"].max()
        for key in ("inference", "drones", "commands")
        if trial.get(key) is not None and not trial[key].empty
    ]
    end_ms = int(max(end_ts_candidates)) if end_ts_candidates else None

    if trim and end_ms is not None:
        start_frame = max(0, sync.timestamp_to_frame(t0_ms))
        end_frame = sync.timestamp_to_frame(end_ms) + 1
    else:
        start_frame = 0
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Rendering trial %s: frames %d -> %d", trial.get("name"), start_frame, end_frame
    )
    bar = HorizontalTopBar(overlays, video_width=width)
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

    cap.release()
    writer.release()
    logger.info("Wrote %s", output_path)
