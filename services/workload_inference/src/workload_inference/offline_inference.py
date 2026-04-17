"""Offline batch inference from recorded gaze data.

Given an experiment base folder, recursively finds all folders containing
``gaze_data.csv`` and runs workload inference using the same engine and settings
as the live system.  Ground truth is read from ``nback_data.csv`` when present
(e.g. N-Back trials) or defaults to 0 (e.g. FlyingPractice).

Produces ``inference_data.csv`` in each processed folder, in the exact format
written by the live :class:`~workload_inference.experiment.ExperimentManager`.

Usage::

    offline_inference --data path/to/experiments [--model path/to/model.zip]
                      [--config path/to/settings.yaml] [--overwrite] [--dry-run]
"""

# Import torch before PyQt6 to avoid DLL conflicts on Windows
import contextlib
import os

with contextlib.suppress(ImportError):
    import torch  # noqa: F401

os.environ["QT_API"] = "PyQt6"  # Ensure PyQt6 is used for matplotlib backend

import argparse
import bisect
import csv
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from eye_metrics.config import EyeMetricsConfig

from workload_inference.constants import DATA_DIR
from workload_inference.experiments.base import INFERENCE_CSV_HEADER
from workload_inference.experiments.data_structures import (
    GazeData,
    InferenceRecord,
    NBackData,
)
from workload_inference.inference.engine import (
    DEFAULT_EYE_METRICS_CONFIG_FILENAME,
    WorkloadInferenceEngine,
)
from workload_inference.inference.settings import InferenceSettings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FolderResult:
    folder: Path
    n_inferences: int = 0
    skipped: bool = False
    has_ground_truth: bool = False
    raw_accuracy: float | None = None
    filtered_accuracy: float | None = None
    error: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

_GAZE_OPENNESS_COLS = {
    "left_openness_validity",
    "right_openness_validity",
    "left_openness",
    "right_openness",
}
_OPENNESS_DEFAULTS: dict[str, object] = {
    "left_openness_validity": np.int8(0),
    "right_openness_validity": np.int8(0),
    "left_openness": np.float32(0.0),
    "right_openness": np.float32(0.0),
}


def load_gaze_csv(path: Path) -> list[GazeData]:
    """Load ``gaze_data.csv`` into a list of :class:`GazeData` objects.

    Handles both the 15-column legacy format (no openness columns) and the
    current 19-column format.  Rows are returned sorted by timestamp.
    """
    df = pd.read_csv(path)
    if df.empty:
        return []

    # Fill missing openness columns with defaults (legacy 15-col files)
    for col, default in _OPENNESS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    records: list[GazeData] = []
    for row in df.itertuples(index=False):
        records.append(
            GazeData(
                timestamp=np.int64(row.timestamp),
                left_gaze_point_x=np.float32(row.left_gaze_point_x),
                left_gaze_point_y=np.float32(row.left_gaze_point_y),
                left_gaze_point_z=np.float32(row.left_gaze_point_z),
                right_gaze_point_x=np.float32(row.right_gaze_point_x),
                right_gaze_point_y=np.float32(row.right_gaze_point_y),
                right_gaze_point_z=np.float32(row.right_gaze_point_z),
                left_point_screen_x=np.float32(row.left_point_screen_x),
                left_point_screen_y=np.float32(row.left_point_screen_y),
                right_point_screen_x=np.float32(row.right_point_screen_x),
                right_point_screen_y=np.float32(row.right_point_screen_y),
                left_validity=np.int8(row.left_validity),
                right_validity=np.int8(row.right_validity),
                left_pupil_diameter=np.float32(row.left_pupil_diameter),
                right_pupil_diameter=np.float32(row.right_pupil_diameter),
                left_openness_validity=np.int8(row.left_openness_validity),
                right_openness_validity=np.int8(row.right_openness_validity),
                left_openness=np.float32(row.left_openness),
                right_openness=np.float32(row.right_openness),
            )
        )

    records.sort(key=lambda g: g.timestamp)
    return records


def load_nback_csv(path: Path) -> list[NBackData] | None:
    """Load ``nback_data.csv``.  Returns *None* if the file is absent or unreadable."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        events: list[NBackData] = []
        for row in df.itertuples(index=False):
            events.append(
                NBackData(
                    timestamp=np.int64(row.timestamp),
                    response_timestamp=np.int64(row.response_timestamp),
                    nback_level=np.int8(row.nback_level),
                    stimulus=np.int8(row.stimulus),
                    participant_response=np.int8(row.participant_response),
                    is_correct=np.int8(row.is_correct),
                )
            )
        events.sort(key=lambda e: e.timestamp)
        return events
    except Exception:
        logger.warning("Could not read nback_data.csv at %s — treating as absent", path)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth resolution
# ──────────────────────────────────────────────────────────────────────────────


def resolve_ground_truth(
    inference_ts: int,
    nback_events: list[NBackData] | None,
    nback_timestamps: list[int] | None = None,
) -> int:
    """Return the N-back level active at *inference_ts*.

    Uses a bisect lookup into the sorted *nback_timestamps* list to find the
    most recent stimulus event that started at or before the inference
    timestamp.  Returns ``0`` when no ``nback_data.csv`` is available (e.g.
    FlyingPractice) or when the inference precedes any recorded stimulus.
    """
    if not nback_events:
        return 0
    if nback_timestamps is None:
        nback_timestamps = [int(e.timestamp) for e in nback_events]
    idx = bisect.bisect_right(nback_timestamps, inference_ts) - 1
    if idx < 0:
        return 0
    return int(nback_events[idx].nback_level)


# ──────────────────────────────────────────────────────────────────────────────
# CSV writer
# ──────────────────────────────────────────────────────────────────────────────


def write_inference_csv(records: list[InferenceRecord], output_path: Path) -> None:
    """Write *records* to *output_path* using the same format as the live system."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = list(INFERENCE_CSV_HEADER)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in records:
            writer.writerow([getattr(r, col) for col in header])


# ──────────────────────────────────────────────────────────────────────────────
# Core per-folder processing
# ──────────────────────────────────────────────────────────────────────────────


def run_folder(
    folder: Path,
    data_root: Path,
    settings: InferenceSettings,
    model_path: Path | None,
    output_root: Path | None,
    overwrite: bool,
    engine: WorkloadInferenceEngine | None = None,
    eye_metrics_config: EyeMetricsConfig | None = None,
) -> FolderResult:
    """Run offline inference for a single experiment folder.

    Args:
        folder: Directory containing ``gaze_data.csv``.
        data_root: The top-level data directory (used for relative-path display).
        settings: Inference settings to use.
        model_path: Path to the trained model file, or *None* to run without a model.
        output_root: If given, write ``inference_data.csv`` under this root
            mirroring the source folder structure.  If *None*, write alongside
            the source ``gaze_data.csv``.
        overwrite: If *False*, skip folders that already contain
            ``inference_data.csv``.
        engine: Optional pre-built engine to reuse.  When provided the engine's
            running state (normalizer, extractor buffers, Schmitt filter) is
            preserved across folders — exactly as it is in the live experiment
            where practice phases warm up the engine before the first trial.
            When *None* a fresh engine is created for this folder.
        eye_metrics_config: Preprocessing configuration.  Forwarded to a newly
            created engine when *engine* is *None*; ignored when reusing an
            existing engine (config is already baked in at construction time).

    Returns:
        A :class:`FolderResult` summarising the outcome.
    """
    # ── Determine output path ────────────────────────────────────────────────
    if output_root is not None:
        try:
            rel = folder.relative_to(data_root)
        except ValueError:
            rel = folder
        out_path = output_root / rel / "inference_data.csv"
    else:
        out_path = folder / "inference_data.csv"

    if out_path.exists() and not overwrite:
        return FolderResult(folder=folder, skipped=True)

    # ── Load gaze data ───────────────────────────────────────────────────────
    try:
        gaze_records = load_gaze_csv(folder / "gaze_data.csv")
    except Exception as exc:
        return FolderResult(folder=folder, error=f"gaze_data.csv load error: {exc}")

    interval = settings.inference_interval_samples
    if len(gaze_records) < interval:
        return FolderResult(
            folder=folder,
            error=f"too few samples: {len(gaze_records)} < {interval}",
        )

    # ── Load nback data (optional) ───────────────────────────────────────────
    nback_events = load_nback_csv(folder / "nback_data.csv")
    nback_timestamps: list[int] = (
        [int(e.timestamp) for e in nback_events] if nback_events else []
    )

    # ── Replay through engine ────────────────────────────────────────────────
    if engine is None:
        engine = WorkloadInferenceEngine.create(
            model_path=model_path,
            settings=settings,
            eye_metrics_config=eye_metrics_config,
        )
    else:
        # Reset per-task state: flush pupil buffers, raw gaze buffer, and Schmitt
        # filter so the previous task's data does not bleed into this one.
        # The WelfordNormalizer and OnlinePupilStats remain warm (subject-level).
        engine.reset_pupil_buffers()

    # Clear any listener registered by the previous folder.
    engine._listeners.clear()

    collected: list[tuple[int, int, np.ndarray, int]] = []
    current_window_ts: list[int] = [0]

    def listener(raw: int, filtered: int, proba: np.ndarray) -> None:
        collected.append((raw, filtered, proba.copy(), current_window_ts[0]))

    engine.register_listener(listener)

    chunks = [
        gaze_records[i : i + interval] for i in range(0, len(gaze_records), interval)
    ]

    for i, chunk in enumerate(chunks):
        end_idx = min((i + 1) * interval, len(gaze_records)) - 1
        current_window_ts[0] = int(gaze_records[end_idx].timestamp)
        engine.gaze_datas_callback(chunk, batch_update=(i == 0))
        # Join before updating current_window_ts for the next chunk to avoid races
        if engine._inference_thread is not None and engine._inference_thread.is_alive():
            engine._inference_thread.join(timeout=30.0)

    # ── Build InferenceRecord list ───────────────────────────────────────────
    records: list[InferenceRecord] = []
    for raw, filtered, proba, ts in collected:
        gt = resolve_ground_truth(ts, nback_events, nback_timestamps)
        records.append(
            InferenceRecord(
                timestamp=ts,
                prob_low=float(proba[0]),
                prob_medium=float(proba[1]),
                prob_high=float(proba[2]),
                raw_state=raw,
                filtered_state=filtered,
                nback_level=gt,
            )
        )

    # ── Write output ─────────────────────────────────────────────────────────
    try:
        write_inference_csv(records, out_path)
    except Exception as exc:
        return FolderResult(folder=folder, error=f"write error: {exc}")

    # ── Compute accuracy (if ground truth available) ─────────────────────────
    has_gt = nback_events is not None
    raw_acc: float | None = None
    filtered_acc: float | None = None
    if has_gt and records:
        gt_arr = np.array([r.nback_level for r in records])
        raw_arr = np.array([r.raw_state for r in records])
        filt_arr = np.array([r.filtered_state for r in records])
        raw_acc = float((gt_arr == raw_arr).mean())
        filtered_acc = float((gt_arr == filt_arr).mean())

    return FolderResult(
        folder=folder,
        n_inferences=len(records),
        has_ground_truth=has_gt,
        raw_accuracy=raw_acc,
        filtered_accuracy=filtered_acc,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Folder discovery
# ──────────────────────────────────────────────────────────────────────────────


def find_gaze_folders(data_dir: Path) -> list[Path]:
    """Return all directories under *data_dir* that contain ``gaze_data.csv``."""
    return sorted(p.parent for p in data_dir.rglob("gaze_data.csv"))


# ──────────────────────────────────────────────────────────────────────────────
# Model path resolution
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_model_path(
    cli_model: Path | None,
    data_root: Path,
) -> Path | None:
    """Resolve the model file to use.

    Priority:
    1. Explicit ``--model`` CLI argument.
    2. First ``*.zip`` found under ``<data_root>/../../data/models/`` (TabNet).
    3. First ``*.pkl`` or ``*.joblib`` found there (sklearn).
    4. *None* — engine runs in no-model mode (uniform probabilities).
    """
    if cli_model is not None:
        if not cli_model.exists():
            raise FileNotFoundError(f"Model file not found: {cli_model}")
        return cli_model

    # Try default model directory relative to the script's package root
    candidates = [
        Path(__file__).parents[2] / "data" / "models",
        data_root.parent / "models",
        data_root / "models",
    ]
    for models_dir in candidates:
        if not models_dir.is_dir():
            continue
        for pattern in ("*.zip", "*.pkl", "*.joblib"):
            found = sorted(models_dir.glob(pattern))
            if found:
                logger.info("Auto-detected model: %s", found[0])
                return found[0]

    logger.warning(
        "No model file found; engine will run in no-model mode (uniform probabilities)"
    )
    return None


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="offline_inference",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parents[2] / "data" / "experiments",
        metavar="DIR",
        help="Base experiment folder to scan (default: data/experiments/)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="YAML",
        help="Path to InferenceSettings YAML.  Uses defaults when omitted.",
    )
    parser.add_argument(
        "--eye-metrics",
        type=Path,
        default=None,
        metavar="YAML",
        help="Path to eye_metrics.yml preprocessing config.  "
        "Auto-detected from data/eye_metrics.yml when omitted.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        metavar="MODEL",
        help="Path to a trained model file (.zip for TabNet, .pkl/.joblib for sklearn). "
        "Auto-detected from data/models/ when omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write inference_data.csv files here, mirroring the source folder "
        "structure.  Defaults to writing alongside each gaze_data.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-process folders that already have inference_data.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List folders that would be processed without running inference.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    return parser


def _fmt_rel(folder: Path, data_root: Path) -> str:
    try:
        return str(folder.relative_to(data_root))
    except ValueError:
        return str(folder)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    data_root: Path = args.data.resolve()
    if not data_root.exists():
        parser.error(f"Data directory does not exist: {data_root}")

    # ── Load settings ────────────────────────────────────────────────────────
    if args.config is not None:
        if not args.config.exists():
            parser.error(f"Settings file not found: {args.config}")
        settings = InferenceSettings.from_yaml(args.config)
    else:
        settings = InferenceSettings()

    # ── Load eye metrics preprocessing config ────────────────────────────────
    eye_metrics_path: Path | None = args.eye_metrics
    if eye_metrics_path is None:
        default_em = DATA_DIR / DEFAULT_EYE_METRICS_CONFIG_FILENAME
        if default_em.exists():
            eye_metrics_path = default_em
    if eye_metrics_path is not None:
        if not eye_metrics_path.exists():
            parser.error(f"Eye metrics config not found: {eye_metrics_path}")
        eye_metrics_config = EyeMetricsConfig.from_yaml(eye_metrics_path)
        logger.info("Loaded eye metrics config from %s", eye_metrics_path)
    else:
        eye_metrics_config = EyeMetricsConfig()

    # ── Resolve model ────────────────────────────────────────────────────────
    try:
        model_path = _resolve_model_path(args.model, data_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    # ── Discover folders ─────────────────────────────────────────────────────
    folders = find_gaze_folders(data_root)
    if not folders:
        print(f"No gaze_data.csv files found under {data_root}")
        return

    total = len(folders)

    if args.dry_run:
        print(f"Dry run — {total} folder(s) found:")
        for folder in folders:
            print(f"  {_fmt_rel(folder, data_root)}")
        return

    # ── Group folders by subject and process in session order ────────────────
    # Mirrors the live experiment: one engine per subject whose state persists
    # across FlyingPractice → NBackPractice → trials, so the normalizer and
    # feature buffers are warm before the first scored trial.
    # Subject identity is the first 4-character alphanumeric folder found in
    # the path (e.g. "BEN0", "SIM0"), keyed together with its path prefix so
    # that identical codes in different experiments don't collide.
    _SUBJECT_RE = re.compile(r"^[A-Z0-9]{4}$")

    def _subject_key(f: Path) -> str:
        try:
            parts = f.relative_to(data_root).parts
            for i, part in enumerate(parts):
                if _SUBJECT_RE.match(part):
                    return "/".join(parts[: i + 1])
            return parts[0] if parts else ""
        except (ValueError, IndexError):
            return ""

    subject_groups: dict[str, list[Path]] = {}
    for folder in folders:
        key = _subject_key(folder)
        subject_groups.setdefault(key, []).append(folder)

    # ── Process ──────────────────────────────────────────────────────────────
    results: list[FolderResult] = []
    idx = 0
    for subject in sorted(subject_groups):
        group = sorted(subject_groups[subject])  # chronological by name
        subject_engine = WorkloadInferenceEngine.create(
            model_path=model_path,
            settings=settings,
            eye_metrics_config=eye_metrics_config,
        )
        for folder in group:
            idx += 1
            rel = _fmt_rel(folder, data_root)
            prefix = f"[{idx:>{len(str(total))}}/{total}] {rel}"

            result = run_folder(
                folder=folder,
                data_root=data_root,
                settings=settings,
                model_path=model_path,
                output_root=args.output,
                overwrite=args.overwrite,
                engine=subject_engine,
                eye_metrics_config=eye_metrics_config,
            )
            results.append(result)

            if result.skipped:
                print(f"{prefix} ... SKIPPED (already processed)")
            elif result.error:
                print(f"{prefix} ... ERROR: {result.error}")
            elif result.has_ground_truth and result.raw_accuracy is not None:
                print(
                    f"{prefix} ... {result.n_inferences} inferences, "
                    f"raw={result.raw_accuracy:.1%}, filtered={result.filtered_accuracy:.1%}"
                )
            else:
                print(f"{prefix} ... {result.n_inferences} inferences, no ground truth")

    # ── Summary ──────────────────────────────────────────────────────────────
    processed = [r for r in results if not r.skipped and r.error is None]
    skipped = [r for r in results if r.skipped]
    errors = [r for r in results if r.error is not None]
    total_inferences = sum(r.n_inferences for r in processed)

    gt_results = [
        r
        for r in processed
        if r.has_ground_truth and r.raw_accuracy is not None and r.n_inferences > 0
    ]

    print()
    print("=== Offline Inference Summary ===")
    print(f"Processed:  {len(processed):>4} folder(s)")
    print(f"Skipped:    {len(skipped):>4} folder(s) (already had inference_data.csv)")
    print(f"Errors:     {len(errors):>4} folder(s)")
    print(f"Total inferences: {total_inferences}")

    if gt_results:
        mean_raw = float(np.mean([r.raw_accuracy for r in gt_results]))  # type: ignore[arg-type]
        mean_filt = float(np.mean([r.filtered_accuracy for r in gt_results]))  # type: ignore[arg-type]
        print()
        print(f"Accuracy (ground-truth folders only, n={len(gt_results)}):")
        print(f"  Raw:      {mean_raw:.1%}")
        print(f"  Filtered: {mean_filt:.1%}")

    if errors:
        print()
        print("Errors detail:")
        for r in errors:
            print(f"  {_fmt_rel(r.folder, data_root)}: {r.error}")


if __name__ == "__main__":
    main()
