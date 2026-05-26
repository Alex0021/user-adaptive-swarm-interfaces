"""Command-line entry point for video_sync."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from video_sync.anchor import (
    DEFAULT_TRIGGER_DISTANCE_M,
    annotate_trials,
    experiment_end_anchor_ms,
)
from video_sync.batch import run_batch
from video_sync.crashes import detect_all_crashes
from video_sync.data import find_subject_video, load_subject
from video_sync.montage import build_crash_montage
from video_sync.overlay import OVERLAY_REGISTRY, BaseOverlay
from video_sync.render import open_video, render_trial
from video_sync.sync import VideoSync

DEFAULT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[3]
    / "workload_inference"
    / "data"
    / "experiments"
    / "experiment_racing_gates"
)


def _build_overlays(names: list[str]) -> list[BaseOverlay]:
    overlays: list[BaseOverlay] = []
    for n in names:
        cls = OVERLAY_REGISTRY.get(n)
        if cls is None:
            raise SystemExit(
                f"Unknown overlay '{n}'. Available: {sorted(OVERLAY_REGISTRY)}"
            )
        overlays.append(cls())
    return overlays


def _print_sync_info(trials: list[dict], video_path: Path, mode: str) -> None:
    _, fps, w, h, n_frames = open_video(video_path)
    print(f"Video: {video_path.name}  {w}x{h}  {fps:.2f} fps  {n_frames} frames")
    print(f"Anchor mode: {mode}")
    anchor = int(trials[0]["anchor_ms"])
    end_anchor = experiment_end_anchor_ms(trials)
    print(f"START anchor (trial 1, {mode}):           {anchor} ms (Unix)")
    if end_anchor is not None:
        dur_s = (end_anchor - anchor) / 1000.0
        print(f"END   anchor (last-gate of last trial):  {end_anchor} ms")
        print(f"  -> span between anchors: {dur_s:.2f} s")
    print()
    print("Per-trial offsets relative to trial-1 anchor (assuming --sync-offset 0):")
    print(f"  {'trial':<22} {'t0':>10} {'1st gate':>12} {'last gate':>12}")
    for i, t in enumerate(trials, start=1):
        rel_t0 = (int(t["t0_ms"]) - anchor) / 1000.0
        gs = t.get("gate_status")
        if gs is not None and not gs.empty:
            valid = gs["first_pass_timestamp"][gs["first_pass_timestamp"] > 0]
            fg = (int(valid.min()) - anchor) / 1000.0 if not valid.empty else float("nan")
            lg = (int(valid.max()) - anchor) / 1000.0 if not valid.empty else float("nan")
        else:
            fg = lg = float("nan")
        print(
            f"  trial {i:>2} {t.get('name', '?'):<14} {rel_t0:>+9.2f}s "
            f"{fg:>+11.2f}s {lg:>+11.2f}s"
        )
    crashes = detect_all_crashes(trials)
    print(f"\nDetected {len(crashes)} crash events across all trials")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="video_sync", description=__doc__)
    parser.add_argument(
        "--subject", help="Subject code (e.g. CXOE). Required unless --batch is used."
    )
    parser.add_argument(
        "--batch", type=Path, default=None,
        help="Path to a YAML batch config. Runs trials + crash montage for "
             "every subject defined in the file (ignores other flags).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help=f"Root experiment dir (default: {DEFAULT_EXPERIMENT_DIR})",
    )
    parser.add_argument(
        "--sync-offset", type=float, default=0.0,
        help="Seconds into the video where trial 1's START anchor happens.",
    )
    parser.add_argument(
        "--sync-offset-end", type=float, default=None,
        help="Seconds into the video where the END anchor happens "
             "(last gate pass of the LAST trial). Enables two-point drift correction.",
    )
    parser.add_argument(
        "--anchor",
        choices=["trigger-plane", "first-gate", "t0"],
        default="trigger-plane",
        help="START anchor: 'trigger-plane' (default, estimates race-timer 00:00), "
             "'first-gate' (first gate pass), or 't0' (drones spawn).",
    )
    parser.add_argument(
        "--trigger-distance", type=float, default=DEFAULT_TRIGGER_DISTANCE_M,
        help="Meters before gate 1 to place the virtual trigger plane "
             "(only used by --anchor trigger-plane).",
    )
    parser.add_argument(
        "--overlays", nargs="+", default=list(OVERLAY_REGISTRY),
        choices=list(OVERLAY_REGISTRY),
        help="Overlays to render (default: all).",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output video file path."
    )
    parser.add_argument(
        "--trial", type=int, default=None,
        help="1-based trial index for single-trial render mode.",
    )
    parser.add_argument(
        "--montage", action="store_true",
        help="Build a crash montage across all trials.",
    )
    parser.add_argument(
        "--padding", type=float, default=3.0,
        help="Seconds before/after each crash for montage clips.",
    )
    parser.add_argument(
        "--merge-window", type=float, default=0.5,
        help="Crashes within this many seconds are grouped into one clip.",
    )
    parser.add_argument(
        "--trim", action="store_true",
        help="Trim single-trial output to trial data window.",
    )
    parser.add_argument(
        "--print-sync-info", action="store_true",
        help="Print video / trial timing info and exit.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    if args.batch is not None:
        run_batch(args.batch)
        return 0

    if args.subject is None:
        parser.error("--subject is required unless --batch is used.")

    subject_dir = args.data_dir / args.subject
    trials = load_subject(subject_dir)
    video_path = find_subject_video(subject_dir)
    annotate_trials(trials, mode=args.anchor, trigger_distance_m=args.trigger_distance)

    if args.print_sync_info:
        _print_sync_info(trials, video_path, args.anchor)
        return 0

    _, fps, _, _, _ = open_video(video_path)
    end_anchor = (
        experiment_end_anchor_ms(trials)
        if args.sync_offset_end is not None
        else None
    )
    if args.sync_offset_end is not None and end_anchor is None:
        parser.error("--sync-offset-end given but no last-gate timestamp found.")
    sync = VideoSync(
        anchor_ms=int(trials[0]["anchor_ms"]),
        sync_offset_s=args.sync_offset,
        fps=fps,
        anchor_end_ms=end_anchor,
        sync_offset_end_s=args.sync_offset_end,
    )
    if sync.two_point:
        drift_ms = (sync._ms_per_video_second() - 1000.0)
        logging.info(
            "Two-point sync enabled. Effective drift: %.2f ms per video second.",
            drift_ms,
        )
    overlays = _build_overlays(args.overlays)

    if args.montage:
        out = args.output or subject_dir / f"{args.subject}_crash_montage.mp4"
        build_crash_montage(
            video_path=video_path,
            trials=trials,
            sync=sync,
            overlays=overlays,
            output_path=out,
            padding_s=args.padding,
            merge_window_s=args.merge_window,
        )
        return 0

    if args.trial is None:
        parser.error("Either --trial N or --montage must be provided.")
    if not 1 <= args.trial <= len(trials):
        parser.error(f"--trial must be in 1..{len(trials)}")
    trial = trials[args.trial - 1]
    out = args.output or subject_dir / f"{args.subject}_{trial.get('name', 'trial')}.mp4"
    render_trial(
        video_path=video_path,
        trial=trial,
        sync=sync,
        overlays=overlays,
        output_path=out,
        trim=args.trim,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
