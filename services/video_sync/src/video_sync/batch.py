"""Batch-process multiple subjects from a YAML config."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from video_sync.anchor import (
    DEFAULT_TRIGGER_DISTANCE_M,
    annotate_trials,
    experiment_end_anchor_ms,
)
from video_sync.data import find_subject_video, load_subject
from video_sync.montage import build_crash_montage
from video_sync.overlay import OVERLAY_REGISTRY, BaseOverlay
from video_sync.render import open_video, render_trial
from video_sync.sync import VideoSync

logger = logging.getLogger(__name__)


@dataclass
class BatchDefaults:
    anchor: str = "trigger-plane"
    trigger_distance_m: float = DEFAULT_TRIGGER_DISTANCE_M
    overlays: list[str] | None = None  # None = all
    padding_s: float = 3.0
    merge_window_s: float = 0.5
    do_trials: bool = True
    do_montage: bool = True
    trim: bool = True


@dataclass
class SubjectConfig:
    code: str
    sync_offset: float
    sync_offset_end: float | None = None
    skip: bool = False


def _build_overlays(names: list[str]) -> list[BaseOverlay]:
    return [OVERLAY_REGISTRY[n]() for n in names]


def _coerce_subject(entry: dict | str) -> SubjectConfig:
    if isinstance(entry, str):
        return SubjectConfig(code=entry, sync_offset=0.0, skip=True)
    raw_offset = entry.get("sync_offset")
    raw_offset_end = entry.get("sync_offset_end")
    skip = bool(entry.get("skip", False))
    if raw_offset is None:
        skip = True
        sync_offset = 0.0
    else:
        sync_offset = float(raw_offset)
    return SubjectConfig(
        code=entry["code"],
        sync_offset=sync_offset,
        sync_offset_end=float(raw_offset_end) if raw_offset_end is not None else None,
        skip=skip,
    )


def load_batch_config(path: Path) -> tuple[Path, Path, BatchDefaults, list[SubjectConfig]]:
    with path.open() as fh:
        raw = yaml.safe_load(fh)

    data_dir = Path(raw["data_dir"]).expanduser().resolve()
    output_dir = Path(raw["output_dir"]).expanduser().resolve()
    d = raw.get("defaults", {}) or {}
    defaults = BatchDefaults(
        anchor=d.get("anchor", "trigger-plane"),
        trigger_distance_m=float(d.get("trigger_distance_m", DEFAULT_TRIGGER_DISTANCE_M)),
        overlays=d.get("overlays") or list(OVERLAY_REGISTRY),
        padding_s=float(d.get("padding_s", 3.0)),
        merge_window_s=float(d.get("merge_window_s", 0.5)),
        do_trials=bool(d.get("do_trials", True)),
        do_montage=bool(d.get("do_montage", True)),
        trim=bool(d.get("trim", True)),
    )
    subjects = [_coerce_subject(e) for e in raw["subjects"]]
    return data_dir, output_dir, defaults, subjects


def process_subject(
    subject_cfg: SubjectConfig,
    data_dir: Path,
    output_dir: Path,
    defaults: BatchDefaults,
) -> None:
    code = subject_cfg.code
    if subject_cfg.skip:
        logger.info("Skipping %s (skip=true or sync_offset unset)", code)
        return

    subject_dir = data_dir / code
    out_dir = output_dir / code
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        trials = load_subject(subject_dir)
        video_path = find_subject_video(subject_dir)
    except (FileNotFoundError, RuntimeError) as err:
        logger.error("Skipping %s: %s", code, err)
        return

    annotate_trials(
        trials,
        mode=defaults.anchor,
        trigger_distance_m=defaults.trigger_distance_m,
    )

    _, fps, _, _, _ = open_video(video_path)
    end_anchor = (
        experiment_end_anchor_ms(trials)
        if subject_cfg.sync_offset_end is not None
        else None
    )
    sync = VideoSync(
        anchor_ms=int(trials[0]["anchor_ms"]),
        sync_offset_s=subject_cfg.sync_offset,
        fps=fps,
        anchor_end_ms=end_anchor,
        sync_offset_end_s=subject_cfg.sync_offset_end,
    )

    overlay_names = defaults.overlays or list(OVERLAY_REGISTRY)
    logger.info(
        "[%s] anchors=(%s, end=%s) overlays=%s",
        code, sync.anchor_ms, end_anchor, overlay_names,
    )

    if defaults.do_trials:
        for i, trial in enumerate(trials, start=1):
            out = out_dir / f"{code}_{trial.get('name', f'trial_{i}')}.mp4"
            logger.info("[%s] rendering trial %d -> %s", code, i, out.name)
            try:
                render_trial(
                    video_path=video_path,
                    trial=trial,
                    sync=sync,
                    overlays=_build_overlays(overlay_names),
                    output_path=out,
                    trim=defaults.trim,
                )
            except Exception as err:  # noqa: BLE001
                logger.exception("[%s] trial %d failed: %s", code, i, err)

    if defaults.do_montage:
        out = out_dir / f"{code}_crash_montage.mp4"
        logger.info("[%s] building crash montage -> %s", code, out.name)
        try:
            build_crash_montage(
                video_path=video_path,
                trials=trials,
                sync=sync,
                overlays=_build_overlays(overlay_names),
                output_path=out,
                padding_s=defaults.padding_s,
                merge_window_s=defaults.merge_window_s,
            )
        except Exception as err:  # noqa: BLE001
            logger.exception("[%s] montage failed: %s", code, err)


def run_batch(config_path: Path) -> None:
    data_dir, output_dir, defaults, subjects = load_batch_config(config_path)
    logger.info(
        "Batch: %d subject(s) from %s -> %s",
        len(subjects), data_dir, output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        process_subject(s, data_dir, output_dir, defaults)
