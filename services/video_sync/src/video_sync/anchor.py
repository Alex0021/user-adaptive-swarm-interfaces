"""Compute per-trial and per-experiment sync anchor timestamps.

Anchor modes for the START sync point (trial 1):

- ``trigger-plane`` (default): estimate when the swarm centroid crosses an
  invisible trigger plane just before gate 1. This is the moment the
  on-screen race timer typically starts.
- ``first-gate``: timestamp of the first gate's `first_pass_timestamp`.
- ``t0``: trial's `t0_ms` (earliest data sample).

The END sync point is always the last gate pass of the LAST trial.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Distance (meters) the trigger plane sits before gate 1.
DEFAULT_TRIGGER_DISTANCE_M = 2.0


def _dedup_gate_status(gs: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate rows that share the same gate id (keep first occurrence)."""
    if gs is None or gs.empty:
        return gs
    return gs.drop_duplicates(subset="id", keep="first")


def _first_gate_pass_ms(trial: dict) -> int | None:
    gs = _dedup_gate_status(trial.get("gate_status"))
    if gs is None or gs.empty:
        return None
    valid = gs.loc[gs["first_pass_timestamp"] > 0, "first_pass_timestamp"]
    if valid.empty:
        return None
    return int(valid.min())


def _last_gate_pass_ms(trial: dict) -> int | None:
    gs = _dedup_gate_status(trial.get("gate_status"))
    if gs is None or gs.empty:
        return None
    valid = gs.loc[gs["first_pass_timestamp"] > 0, "first_pass_timestamp"]
    if valid.empty:
        return None
    return int(valid.max())


def _trigger_plane_ms(
    trial: dict, trigger_distance_m: float = DEFAULT_TRIGGER_DISTANCE_M
) -> int | None:
    """Estimate when the swarm centroid crosses a plane in front of gate 1.

    Method: project centroid positions onto the unit vector from the
    centroid's starting position toward gate 1's center. Return the
    timestamp where that projection first equals
    (distance_start_to_gate1 - trigger_distance_m).
    """
    centroid: pd.DataFrame | None = trial.get("centroid")
    gates: pd.DataFrame | None = trial.get("gates")
    if centroid is None or centroid.empty or gates is None or gates.empty:
        return None

    g1 = gates.sort_values("id").iloc[0]
    g = np.array([g1["center_x"], g1["center_y"], g1["center_z"]], dtype=float)

    start = centroid.iloc[0]
    c0 = np.array([start["x"], start["y"], start["z"]], dtype=float)
    delta = g - c0
    dist = float(np.linalg.norm(delta))
    if dist < 1e-3:
        return None
    direction = delta / dist
    target_projection = dist - trigger_distance_m

    pos = centroid[["x", "y", "z"]].to_numpy(dtype=float)
    proj = (pos - c0) @ direction
    mask = proj >= target_projection
    if not mask.any():
        return None
    idx = int(np.argmax(mask))
    return int(centroid["timestamp"].iloc[idx])


def trial_anchor_ms(
    trial: dict,
    mode: str = "trigger-plane",
    trigger_distance_m: float = DEFAULT_TRIGGER_DISTANCE_M,
) -> int:
    """Return the START anchor timestamp (Unix ms) for a single trial."""
    if mode == "t0":
        return int(trial["t0_ms"])
    if mode == "first-gate":
        v = _first_gate_pass_ms(trial)
        return v if v is not None else int(trial["t0_ms"])
    if mode == "trigger-plane":
        v = _trigger_plane_ms(trial, trigger_distance_m)
        if v is not None:
            return v
        logger.warning(
            "Trigger-plane estimate unavailable for trial %s; falling back to first-gate.",
            trial.get("name"),
        )
        v = _first_gate_pass_ms(trial)
        return v if v is not None else int(trial["t0_ms"])
    raise ValueError(f"Unknown anchor mode: {mode}")


def experiment_end_anchor_ms(trials: list[dict]) -> int | None:
    """Last gate pass time of the LAST trial (used as 2nd sync point)."""
    for trial in reversed(trials):
        v = _last_gate_pass_ms(trial)
        if v is not None:
            return v
    return None


def annotate_trials(
    trials: list[dict],
    mode: str = "trigger-plane",
    trigger_distance_m: float = DEFAULT_TRIGGER_DISTANCE_M,
) -> None:
    """Attach `anchor_ms` and `anchor_mode` to each trial dict in-place."""
    for t in trials:
        t["anchor_ms"] = trial_anchor_ms(t, mode, trigger_distance_m)
        t["anchor_mode"] = mode
