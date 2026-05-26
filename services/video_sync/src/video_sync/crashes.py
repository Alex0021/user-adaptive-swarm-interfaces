"""Detect drone crash events from drone_data.csv (alive 1 → 0 transitions)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CrashEvent:
    trial_idx: int     # 0-based index into the subject's trial list
    trial_name: str
    timestamp_ms: int  # Unix ms of the crash (first 1→0 sample)
    drone_id: int


def detect_crashes_in_trial(drone_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Return list of (timestamp_ms, drone_id) for each 1→0 alive transition."""
    if drone_df is None or drone_df.empty:
        return []
    events: list[tuple[int, int]] = []
    for drone_id, group in drone_df.sort_values("timestamp").groupby("id"):
        a = group["alive"].to_numpy()
        ts = group["timestamp"].to_numpy()
        if len(a) < 2:
            continue
        transitions = (a[:-1] == 1) & (a[1:] == 0)
        for idx in transitions.nonzero()[0]:
            events.append((int(ts[idx + 1]), int(drone_id)))
    events.sort(key=lambda x: x[0])
    return events


def detect_all_crashes(trials: list[dict]) -> list[CrashEvent]:
    """Detect crashes across all trials in a subject."""
    out: list[CrashEvent] = []
    for idx, trial in enumerate(trials):
        for ts, did in detect_crashes_in_trial(trial["drones"]):
            out.append(
                CrashEvent(
                    trial_idx=idx,
                    trial_name=trial.get("name", f"trial_{idx + 1}"),
                    timestamp_ms=ts,
                    drone_id=did,
                )
            )
    return out


def merge_close_crashes(
    events: list[CrashEvent], window_s: float
) -> list[list[CrashEvent]]:
    """Group crashes that occur within `window_s` of each other (per-trial)."""
    if not events:
        return []
    groups: list[list[CrashEvent]] = []
    window_ms = int(window_s * 1000)
    by_trial: dict[int, list[CrashEvent]] = {}
    for ev in events:
        by_trial.setdefault(ev.trial_idx, []).append(ev)

    for trial_idx in sorted(by_trial.keys()):
        trial_events = sorted(by_trial[trial_idx], key=lambda e: e.timestamp_ms)
        current = [trial_events[0]]
        for ev in trial_events[1:]:
            if ev.timestamp_ms - current[-1].timestamp_ms <= window_ms:
                current.append(ev)
            else:
                groups.append(current)
                current = [ev]
        groups.append(current)
    return groups
