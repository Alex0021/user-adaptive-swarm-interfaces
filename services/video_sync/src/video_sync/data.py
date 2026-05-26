"""Thin wrapper around workload_inference's racing-data loaders."""

from __future__ import annotations

from pathlib import Path

from workload_inference.plot_results._common import (
    STATE_COLORS,
    STATE_LABELS,
    SWARM_SIZE,
    _load_racing_trials,
)

__all__ = ["load_subject", "STATE_COLORS", "STATE_LABELS", "SWARM_SIZE"]


def load_subject(subject_dir: Path) -> list[dict]:
    """Load all racing trials for a subject directory.

    Each trial dict has keys: name, gates, gate_status, inference, commands,
    drones, centroid, t0_ms.
    """
    subject_dir = Path(subject_dir)
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
    trials = _load_racing_trials(subject_dir)
    if not trials:
        raise RuntimeError(f"No racing trials found under {subject_dir}")
    return trials


def find_subject_video(subject_dir: Path) -> Path:
    """Locate the screen recording inside a subject folder.

    Handles both English ('Recording_<CODE>.mp4') and French naming variants.
    """
    subject_dir = Path(subject_dir)
    candidates = list(subject_dir.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"No .mp4 file in {subject_dir}")
    for c in candidates:
        if c.name.lower().startswith("recording_"):
            return c
    return candidates[0]
