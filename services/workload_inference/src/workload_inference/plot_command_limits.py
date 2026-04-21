"""Analyse the distribution of adaptive flight-control limits across an experiment.

Reads all command_data.csv files found under a given experiment folder, extracts the
limit values at each CWL step-transition (run-length encoding on cwl_current_step),
and produces two figures:

  1. command_limits_histograms.png — 2×3 histogram grid, one subplot per limit
     parameter, with per-subject colour coding and a recommended-value annotation.

  2. command_limits_summary.png — a formatted table of per-subject medians and the
     recommended (cross-subject median-of-medians) flight profile.

Usage
-----
    plot_command_limits [--data DIR] [--output DIR] [--show]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from workload_inference.plot_results import ADAPTATION_PARAMS

# ── Paths ─────────────────────────────────────────────────────────────────────
_SERVICE_ROOT = Path(__file__).parents[2]
_DEFAULT_DATA = _SERVICE_ROOT / "data" / "experiments"
_DEFAULT_OUTPUT = _SERVICE_ROOT / "data" / "results"

COMMAND_FILE_NAME = "command_data.csv"

_SUBJECT_RE = re.compile(r"^[A-Z0-9]{4}$")
_TASK_RE = re.compile(r"^task_\d+$")
_TRIAL_RE = re.compile(r"^trial_\d+$")

# Six limit parameters to analyse (in 2×3 display order)
_PARAM_ORDER = [
    "max_pitch",
    "max_roll",
    "max_yaw_rate",
    "max_speed",
    "max_altitude_rate",
    "max_alpha",
]

# Pull label + (min, max) from the shared ADAPTATION_PARAMS dict
LIMIT_PARAMS: dict[str, tuple[str, float, float]] = {
    col: ADAPTATION_PARAMS[col]
    for col in _PARAM_ORDER
    if col in ADAPTATION_PARAMS
}

_REQUIRED_COLS = frozenset(
    {"cwl_current_step"} | set(_PARAM_ORDER)
)

# ── Mode detection ─────────────────────────────────────────────────────────────


def _detect_mode(data_dir: Path) -> str:
    """Return 'trial', 'subject', or 'experiment'.

    - trial      : data_dir contains command_data.csv directly.
    - subject    : folder name matches the 4-char subject code regex.
    - experiment : folder contains one or more 4-char subject sub-folders.
    """
    if (data_dir / COMMAND_FILE_NAME).exists():
        return "trial"
    if _SUBJECT_RE.match(data_dir.name):
        return "subject"
    return "experiment"


# ── File discovery ─────────────────────────────────────────────────────────────


def _find_command_files(data_dir: Path) -> list[Path]:
    """Return all command_data.csv files under data_dir, sorted by path."""
    return sorted(data_dir.rglob(COMMAND_FILE_NAME))


# ── Metadata parsing ───────────────────────────────────────────────────────────


def _parse_metadata(csv_path: Path, data_dir: Path) -> dict[str, str]:
    """Extract subject_id, task, and trial from a csv path relative to data_dir.

    Scans path parts for segments matching the expected regex patterns.
    Missing segments receive empty-string defaults.
    """
    try:
        rel_parts = csv_path.relative_to(data_dir).parts
    except ValueError:
        rel_parts = csv_path.parts

    meta: dict[str, str] = {"subject_id": "", "task": "", "trial": ""}
    for part in rel_parts:
        if not meta["subject_id"] and _SUBJECT_RE.match(part):
            meta["subject_id"] = part
        elif not meta["task"] and _TASK_RE.match(part):
            meta["task"] = part
        elif not meta["trial"] and _TRIAL_RE.match(part):
            meta["trial"] = part
    return meta


# ── Step-event extraction ──────────────────────────────────────────────────────


def _extract_step_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply run-length encoding to cwl_current_step and return one row per step-event.

    A step-event is a maximal contiguous run of identical cwl_current_step values.
    For each run the limit values are summarised as the median of that run (handles
    boundary rows where a numeric value straddles two consecutive step indices).

    Returns an empty DataFrame if none of the required columns are present or if the
    DataFrame itself is empty.
    """
    if df.empty:
        return pd.DataFrame()

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        return pd.DataFrame()

    step_col = df["cwl_current_step"]
    change_mask = step_col != step_col.shift(1)
    group_id = change_mask.cumsum()

    agg_spec: dict[str, tuple[str, str]] = {
        "cwl_current_step": ("cwl_current_step", "first"),
    }
    for col in _PARAM_ORDER:
        if col in df.columns:
            agg_spec[col] = (col, "median")

    result = df.groupby(group_id).agg(**agg_spec)
    return result.reset_index(drop=True)


# ── Combined DataFrame builder ─────────────────────────────────────────────────


def load_limit_events(data_dir: Path) -> pd.DataFrame:
    """Discover all command_data.csv files, extract step-events, and return a
    combined DataFrame with metadata columns.

    Output columns: subject_id, task, trial, cwl_current_step, max_pitch,
    max_roll, max_yaw_rate, max_speed, max_altitude_rate, max_alpha.

    Files that lack the limit columns are silently skipped (with a printed warning).
    Raises FileNotFoundError if no valid files are found.
    """
    frames: list[pd.DataFrame] = []

    for csv_path in _find_command_files(data_dir):
        try:
            df = pd.read_csv(
                csv_path,
                usecols=lambda c: c in (
                    {"cwl_current_step", "cwl_total_steps"} | set(_PARAM_ORDER)
                ),
            )
        except Exception:
            continue

        if not _REQUIRED_COLS.issubset(df.columns):
            rel = csv_path.relative_to(data_dir) if data_dir in csv_path.parents else csv_path
            print(f"  Skipping (missing limit columns): {rel}")
            continue

        step_events = _extract_step_values(df)
        if step_events.empty:
            continue

        meta = _parse_metadata(csv_path, data_dir)
        for key, val in meta.items():
            step_events[key] = val

        frames.append(step_events)

    if not frames:
        raise FileNotFoundError(
            f"No command_data.csv with limit columns found under {data_dir}"
        )

    return pd.concat(frames, ignore_index=True)


# ── Statistics ─────────────────────────────────────────────────────────────────


def _compute_subject_medians(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per subject_id with the median limit value per parameter.

    Groups events by subject_id and takes the median across all step-events for that
    subject, so each step-event contributes equally (not time-weighted).
    """
    available = [c for c in _PARAM_ORDER if c in events.columns]
    return events.groupby("subject_id")[available].median()


def _recommended_profile(subject_medians: pd.DataFrame) -> pd.Series:
    """Return the recommended value per parameter as the median of per-subject medians.

    This median-of-medians treats each subject as one equal vote and is robust to
    subjects with differing numbers of trials or step-events.
    """
    available = [c for c in _PARAM_ORDER if c in subject_medians.columns]
    return subject_medians[available].median(axis=0)


# ── Subject colour assignment ──────────────────────────────────────────────────


def _subject_color_map(subject_ids: list[str]) -> dict[str, tuple]:
    """Assign a distinct tab10 colour to each subject_id (sorted for reproducibility)."""
    cmap = plt.get_cmap("tab10")
    sorted_ids = sorted(set(subject_ids))
    return {sid: cmap(i % 10) for i, sid in enumerate(sorted_ids)}


# ── Figure 1: histogram grid ───────────────────────────────────────────────────


def plot_limit_histograms(
    events: pd.DataFrame,
    subject_medians: pd.DataFrame,
    recommended: pd.Series,
    color_map: dict[str, tuple],
) -> plt.Figure:
    """Plot a 2×3 grid of histograms — one per limit parameter.

    Per subplot:
    - Semi-transparent filled histogram per subject (alpha=0.35), tab10 colours.
    - Dashed vertical line at each subject's median.
    - Solid black vertical line at the cross-subject recommended value.
    - Annotation box with the recommended value.
    - X-axis clipped to parameter range from LIMIT_PARAMS.
    - Legend on the first subplot only.
    """
    available_params = [p for p in _PARAM_ORDER if p in events.columns]
    n_params = len(available_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8), constrained_layout=True)
    axes_flat = axes.flatten() if n_params > 1 else [axes]

    subject_ids = sorted(events["subject_id"].unique())

    for idx, param in enumerate(available_params):
        ax = axes_flat[idx]
        label, p_min, p_max = LIMIT_PARAMS.get(param, (param, None, None))

        # Per-subject histograms
        for sid in subject_ids:
            mask = events["subject_id"] == sid
            vals = events.loc[mask, param].dropna()
            if vals.empty:
                continue
            color = color_map[sid]
            n_unique = vals.nunique()
            n_bins = max(10, min(40, n_unique))
            ax.hist(
                vals,
                bins=n_bins,
                alpha=0.35,
                color=color,
                label=sid,
                density=False,
            )
            # Per-subject median vline
            if sid in subject_medians.index and param in subject_medians.columns:
                s_med = subject_medians.loc[sid, param]
                ax.axvline(
                    s_med,
                    color=color,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.85,
                )

        # Recommended value vline + annotation
        if param in recommended.index:
            rec_val = recommended[param]
            ax.axvline(rec_val, color="black", linewidth=2.0, linestyle="-")
            ax.annotate(
                f"Rec: {rec_val:.3f}",
                xy=(rec_val, 1),
                xycoords=("data", "axes fraction"),
                xytext=(6, -4),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                va="top",
            )

        # Axis formatting
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Step-event count", fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if p_min is not None and p_max is not None:
            ax.set_xlim(p_min * 0.85, p_max * 1.1)

        # Legend on first subplot only
        if idx == 0:
            ax.legend(fontsize=7, ncol=2, title="Subject", title_fontsize=7)

    # Hide unused subplots
    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    n_subjects = len(subject_ids)
    n_trials = events[["subject_id", "task", "trial"]].drop_duplicates().shape[0]
    n_events = len(events)
    fig.suptitle(
        f"Command Limit Distribution — {events['subject_id'].iloc[0] if n_subjects == 1 else 'All Subjects'}\n"
        f"{n_subjects} subject(s), {n_trials} trial(s), {n_events} step-events",
        fontsize=12,
        fontweight="bold",
    )
    return fig


# ── Figure 2: summary table ────────────────────────────────────────────────────


def plot_summary_table(
    subject_medians: pd.DataFrame,
    recommended: pd.Series,
) -> plt.Figure:
    """Render a figure with a formatted table of per-subject medians and the
    recommended flight profile.

    Columns: Subject | one column per limit parameter.
    Final row (Recommended) is bold with a light-yellow background.
    """
    available = [c for c in _PARAM_ORDER if c in subject_medians.columns]
    col_labels = ["Subject"] + [LIMIT_PARAMS.get(c, (c,))[0] for c in available]

    rows: list[list[str]] = []
    for sid in sorted(subject_medians.index):
        row = [sid] + [f"{subject_medians.loc[sid, c]:.4f}" for c in available]
        rows.append(row)
    # Recommended row
    rec_row = ["Recommended\n(median)"] + [
        f"{recommended[c]:.4f}" if c in recommended.index else "—" for c in available
    ]
    rows.append(rec_row)

    n_subjects = len(subject_medians)
    fig_height = max(3.0, 2.5 + 0.45 * n_subjects)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    # Style the recommended (last data) row
    rec_row_idx = len(rows)  # ax.table uses 0=header, 1..N=data rows
    for col_idx in range(len(col_labels)):
        cell = tbl[rec_row_idx, col_idx]
        cell.set_facecolor("#FFFDE7")
        cell.set_text_props(fontweight="bold")

    fig.suptitle("Recommended Flight Profile", fontsize=12, fontweight="bold")
    return fig


# ── Save / show helper ─────────────────────────────────────────────────────────


def _save_or_show(figs: list[tuple[plt.Figure, Path]], show: bool) -> None:
    for fig, path in figs:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show:
        plt.show()


# ── Orchestration ──────────────────────────────────────────────────────────────


def run_command_limits(show: bool, output_dir: Path, data_dir: Path) -> None:
    """Discover data, compute step-events, build both figures, and save outputs."""
    mode = _detect_mode(data_dir)
    print(f"Loading command data from: {data_dir}  [{mode} mode]")

    events = load_limit_events(data_dir)

    n_subjects = events["subject_id"].nunique()
    n_trials = events[["subject_id", "task", "trial"]].drop_duplicates().shape[0]
    n_events = len(events)
    print(f"  Found {n_subjects} subject(s), {n_trials} trial(s), {n_events} step-events")

    subject_medians = _compute_subject_medians(events)
    recommended = _recommended_profile(subject_medians)

    print("\nRecommended flight profile:")
    for param in _PARAM_ORDER:
        if param in recommended.index:
            label = LIMIT_PARAMS.get(param, (param,))[0]
            print(f"  {label:25s}: {recommended[param]:.4f}")

    color_map = _subject_color_map(events["subject_id"].unique().tolist())

    fig1 = plot_limit_histograms(events, subject_medians, recommended, color_map)
    fig2 = plot_summary_table(subject_medians, recommended)

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_or_show(
        [
            (fig1, output_dir / "command_limits_histograms.png"),
            (fig2, output_dir / "command_limits_summary.png"),
        ],
        show,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display figures interactively after saving.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_DATA,
        metavar="DIR",
        help="Experiment data folder (trial / subject / experiment mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="DIR",
        help="Output directory for saved figures.",
    )
    args = parser.parse_args()
    run_command_limits(show=args.show, output_dir=args.output, data_dir=args.data)


if __name__ == "__main__":
    main()
