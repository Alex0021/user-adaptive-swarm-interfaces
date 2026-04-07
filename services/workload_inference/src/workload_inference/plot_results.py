"""
Plot and visualize workload inference experiment results.

Three modes are selected automatically based on --data:

* **Trial folder**   (contains inference_data.csv directly)
  → inference_time_series.png only

* **Subject folder** (4-char code, e.g. BEN0 — contains task_N/trial_M sub-folders)
  → inference_time_series.png  (task trials only, concatenated)
  → inference_accuracy_summary.png  (per-task bars + per-CWL-level bars, trial dots)

* **Experiment / root folder** (contains multiple subjects)
  → inference_time_series.png  (all sessions concatenated)
  → inference_accuracy_summary.png  (overall + per-class bars)

When ``--cwl`` is given, trajectory-based CWL plots are produced instead.
The task folder is resolved automatically per subject so randomized task order
across subjects is handled transparently:

* **Subject + cwl**     → trajectory colored by filtered CWL + per-trial accuracy
* **Experiment + cwl**  → aggregate trajectory (mean ± std) + per-subject accuracy

Usage:
    plot_results inference [--show] [--data DIR] [--output DIR] [--cwl {0,1,2}]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import CheckButtons
from scipy.spatial import cKDTree

_SERVICE_ROOT = Path(__file__).parents[2]
_DEFAULT_DATA = _SERVICE_ROOT / "data" / "experiments"
_DEFAULT_OUTPUT = _SERVICE_ROOT / "data" / "results"
_SPLINE_FILE = _SERVICE_ROOT / "data" / "spline_trajectory.csv"

INFERENCE_FILE_NAME = "inference_data.csv"
DRONE_FILE_NAME = "drone_data.csv"
INFERENCE_SAMPLE_RATE = 60  # samples per second
STATE_LABELS = {0: "Low", 1: "Medium", 2: "High"}
STATE_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}

_TASK_RE = re.compile(r"^task_\d+/")
_SUBJECT_RE = re.compile(r"^[A-Z0-9]{4}$")


# ─────────────────────────────────────────────────────────────────────────────
# Mode detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_mode(data_dir: Path) -> str:
    """Return 'trial', 'subject', or 'experiment'."""
    if (data_dir / INFERENCE_FILE_NAME).exists():
        return "trial"
    if _SUBJECT_RE.match(data_dir.name):
        return "subject"
    return "experiment"


def _find_task_for_cwl(subject_dir: Path, cwl_level: int) -> str | None:
    """Return the task folder name whose dominant nback_level equals *cwl_level*.

    Iterates task_N sub-folders, reads the first available inference_data.csv
    from any trial, and returns the task name whose majority nback_level matches
    the requested CWL level.  Returns None if no matching task is found.
    """
    task_dirs = sorted(
        d for d in subject_dir.iterdir()
        if d.is_dir() and re.match(r"^task_\d+$", d.name)
    )
    if not task_dirs:
        print(f"  {subject_dir.name}: no task folders found.")
        return None

    all_csvs = sorted(subject_dir.rglob(INFERENCE_FILE_NAME))
    if not all_csvs:
        print(
            f"  {subject_dir.name}: no {INFERENCE_FILE_NAME} files found — "
            "run offline_inference first."
        )
        return None

    for task_dir in task_dirs:
        trial_csvs = sorted(task_dir.rglob(INFERENCE_FILE_NAME))
        if not trial_csvs:
            continue
        try:
            df = pd.read_csv(trial_csvs[0], usecols=["nback_level"])
        except Exception:
            continue
        if df.empty:
            continue
        dominant = int(df["nback_level"].mode().iloc[0])
        if dominant == cwl_level:
            return task_dir.name

    cwl_label = STATE_LABELS.get(cwl_level, str(cwl_level))
    print(
        f"  {subject_dir.name}: no task matched CWL level {cwl_label} "
        f"({cwl_level}) — check nback_level values in inference data."
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_inference_data(experiments_dir: Path) -> pd.DataFrame:
    csv_files = sorted(experiments_dir.rglob(INFERENCE_FILE_NAME))
    if not csv_files:
        raise FileNotFoundError(
            f"No {INFERENCE_FILE_NAME} files found under {experiments_dir}"
        )

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        rel = f.relative_to(experiments_dir)
        df["_source"] = "/".join(rel.parts[:-1])
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["elapsed_s"] = data.groupby("_source")["timestamp"].transform(
        lambda t: (t - t.iloc[0]) / 1000.0
    )
    return data


def _task_trials_only(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only task_N/trial_M sources (drop FlyingPractice, NBackPractice)."""
    return data[data["_source"].str.match(_TASK_RE)].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _state_step_plot(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, color: str, **kwargs
):
    ax.step(x, y, where="post", label=label, color=color, **kwargs)


def _accuracy_over_time(
    y_true: np.ndarray, y_pred: np.ndarray, window: int = INFERENCE_SAMPLE_RATE
) -> np.ndarray:
    correct = (y_true == y_pred).astype(float)
    return pd.Series(correct).rolling(window, min_periods=1).mean().to_numpy()


def _contiguous_regions(mask: np.ndarray):
    """Yield (start, end) index pairs for contiguous True regions in mask."""
    in_region = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_region:
            start = i
            in_region = True
        elif not val and in_region:
            yield start, i
            in_region = False
    if in_region:
        yield start, len(mask)


def _bar_label(ax: plt.Axes, bars, fmt: str = "{:.0%}", offset: float = 0.02):
    for bar in bars:
        val = bar.get_height()
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                fmt.format(val),
                ha="center", va="bottom", fontsize=8,
            )


def _hbar_label(ax: plt.Axes, bars, fmt: str = "{:.0%}", offset: float = 0.01):
    for bar in bars:
        val = bar.get_width()
        if val > 0:
            ax.text(
                val + offset, bar.get_y() + bar.get_height() / 2,
                fmt.format(val),
                ha="left", va="center", fontsize=8,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Time series plot  (shared by all modes)
# ─────────────────────────────────────────────────────────────────────────────

def plot_inference_time_series(
    data: pd.DataFrame, ax_states: plt.Axes, ax_rolling: plt.Axes
):
    sources = list(data["_source"].unique())
    offset = 0.0
    x_ticks, x_tick_labels = [], []

    for src in sources:
        seg = data[data["_source"] == src].copy().sort_values("elapsed_s")
        t = seg["elapsed_s"].to_numpy() + offset
        gt = seg["nback_level"].to_numpy()
        raw = seg["raw_state"].to_numpy()
        filt = seg["filtered_state"].to_numpy()

        for level, color in STATE_COLORS.items():
            mask = gt == level
            if not mask.any():
                continue
            for start_i, end_i in _contiguous_regions(mask):
                ax_states.axvspan(
                    t[start_i],
                    t[min(end_i, len(t) - 1)],
                    alpha=0.12,
                    color=color,
                    linewidth=0,
                )

        first = src == sources[0]
        _state_step_plot(
            ax_states, t, gt,
            label="Ground truth" if first else "_",
            color="#333333", linewidth=1.5, linestyle="--",
        )
        _state_step_plot(
            ax_states, t, raw,
            label="Raw inference" if first else "_",
            color="#1976D2", linewidth=1.2, alpha=0.85,
        )
        _state_step_plot(
            ax_states, t, filt,
            label="Filtered inference" if first else "_",
            color="#E91E63", linewidth=1.5,
        )

        ax_rolling.plot(
            t, _accuracy_over_time(gt, raw),
            color="#1976D2", linewidth=1.2, alpha=0.85,
            label="Raw accuracy" if first else "_",
        )
        ax_rolling.plot(
            t, _accuracy_over_time(gt, filt),
            color="#E91E63", linewidth=1.5,
            label="Filtered accuracy" if first else "_",
        )

        if len(sources) > 1:
            sep_x = t[-1] + 2.0
            ax_states.axvline(sep_x - 1.0, color="gray", linewidth=0.5, linestyle=":")
            ax_rolling.axvline(sep_x - 1.0, color="gray", linewidth=0.5, linestyle=":")
            x_ticks.append((offset + t[-1]) / 2)
            x_tick_labels.append(src.split("/")[-1])
            offset = sep_x
        else:
            offset = t[-1] + 2.0

    ax_states.set_yticks([0, 1, 2])
    ax_states.set_yticklabels([STATE_LABELS[i] for i in range(3)])
    ax_states.set_ylabel("Workload Level")
    ax_states.set_ylim(-0.3, 2.5)
    ax_states.legend(loc="upper right", fontsize=8)
    ax_states.set_title("Inference Time Series")
    ax_states.grid(axis="y", linestyle=":", alpha=0.4)
    ax_states.set_xticklabels([])

    ax_rolling.set_ylabel(f"Rolling Accuracy\n(window={INFERENCE_SAMPLE_RATE} s)")
    ax_rolling.set_xlabel("Time (s)")
    ax_rolling.set_ylim(-0.05, 1.05)
    ax_rolling.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_rolling.legend(loc="lower right", fontsize=8)
    ax_rolling.grid(axis="y", linestyle=":", alpha=0.4)
    ax_rolling.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    if x_ticks:
        for ax in (ax_states, ax_rolling):
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels, rotation=20, ha="right", fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Subject-level accuracy summary
# ─────────────────────────────────────────────────────────────────────────────

def _build_trial_summary(data: pd.DataFrame) -> pd.DataFrame:
    """One row per source with task, trial, CWL level, raw/filtered accuracy."""
    records = []
    for src, grp in data.groupby("_source"):
        parts = src.split("/")
        task = parts[0] if len(parts) >= 1 else src
        trial = parts[1] if len(parts) >= 2 else "trial_1"
        nback_level = int(grp["nback_level"].mode().iloc[0])
        raw_acc = float((grp["nback_level"] == grp["raw_state"]).mean())
        filt_acc = float((grp["nback_level"] == grp["filtered_state"]).mean())
        records.append({
            "source": src,
            "task": task,
            "trial": trial,
            "nback_level": nback_level,
            "raw_acc": raw_acc,
            "filt_acc": filt_acc,
        })
    return pd.DataFrame(records)


def plot_subject_accuracy_summary(
    data: pd.DataFrame, ax_task: plt.Axes, ax_level: plt.Axes
):
    """Two panels: accuracy per task (with trial dots) + accuracy per CWL level."""
    summary = _build_trial_summary(data)
    width = 0.35
    chance = 1 / 3

    # ── Panel A: per task ────────────────────────────────────────────────────
    tasks = sorted(summary["task"].unique())
    x = np.arange(len(tasks))

    task_raw = [summary[summary["task"] == t]["raw_acc"].mean() for t in tasks]
    task_filt = [summary[summary["task"] == t]["filt_acc"].mean() for t in tasks]

    bars_r = ax_task.bar(
        x - width / 2, task_raw, width, label="Raw", color="#1976D2", alpha=0.85,
        edgecolor="white",
    )
    bars_f = ax_task.bar(
        x + width / 2, task_filt, width, label="Filtered", color="#E91E63", alpha=0.85,
        edgecolor="white",
    )
    _bar_label(ax_task, bars_r)
    _bar_label(ax_task, bars_f)

    # Trial dots (jittered)
    for xi, task in zip(x, tasks):
        t_rows = summary[summary["task"] == task]
        n = len(t_rows)
        jitter = np.linspace(-0.07, 0.07, n) if n > 1 else [0.0]
        for j, (_, row) in zip(jitter, t_rows.iterrows()):
            ax_task.scatter(xi - width / 2 + j, row["raw_acc"],
                            color="#1976D2", s=35, zorder=5, alpha=0.7,
                            edgecolors="white", linewidths=0.5)
            ax_task.scatter(xi + width / 2 + j, row["filt_acc"],
                            color="#E91E63", s=35, zorder=5, alpha=0.7,
                            edgecolors="white", linewidths=0.5)

    # x-axis labels include CWL level
    task_xlabels = []
    for task in tasks:
        lvl = int(summary[summary["task"] == task]["nback_level"].mode().iloc[0])
        n_trials = len(summary[summary["task"] == task])
        task_xlabels.append(f"{task}\n({STATE_LABELS[lvl]}, n={n_trials})")

    ax_task.set_xticks(x)
    ax_task.set_xticklabels(task_xlabels, fontsize=9)
    ax_task.set_ylim(0, 1.18)
    ax_task.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_task.set_ylabel("Accuracy")
    ax_task.set_title("Accuracy per Task")
    ax_task.legend(fontsize=8)
    ax_task.grid(axis="y", linestyle=":", alpha=0.4)
    ax_task.axhline(chance, color="gray", linewidth=0.8, linestyle="--", alpha=0.6,
                    label="Chance")

    # ── Panel B: per CWL level ───────────────────────────────────────────────
    levels = sorted(summary["nback_level"].unique())
    x2 = np.arange(len(levels))

    level_raw = [summary[summary["nback_level"] == l]["raw_acc"].mean() for l in levels]
    level_filt = [summary[summary["nback_level"] == l]["filt_acc"].mean() for l in levels]
    level_counts = [int((summary["nback_level"] == l).sum()) for l in levels]

    bars_r2 = ax_level.bar(
        x2 - width / 2, level_raw, width, label="Raw", color="#1976D2", alpha=0.85,
        edgecolor="white",
    )
    bars_f2 = ax_level.bar(
        x2 + width / 2, level_filt, width, label="Filtered", color="#E91E63", alpha=0.85,
        edgecolor="white",
    )
    _bar_label(ax_level, bars_r2)
    _bar_label(ax_level, bars_f2)

    # Trial dots per level
    for xi, level in zip(x2, levels):
        l_rows = summary[summary["nback_level"] == level]
        n = len(l_rows)
        jitter = np.linspace(-0.07, 0.07, n) if n > 1 else [0.0]
        for j, (_, row) in zip(jitter, l_rows.iterrows()):
            ax_level.scatter(xi - width / 2 + j, row["raw_acc"],
                             color="#1976D2", s=35, zorder=5, alpha=0.7,
                             edgecolors="white", linewidths=0.5)
            ax_level.scatter(xi + width / 2 + j, row["filt_acc"],
                             color="#E91E63", s=35, zorder=5, alpha=0.7,
                             edgecolors="white", linewidths=0.5)

    ax_level.set_xticks(x2)
    ax_level.set_xticklabels(
        [f"{STATE_LABELS[l]}\n(n={level_counts[i]})" for i, l in enumerate(levels)],
        fontsize=9,
    )
    ax_level.set_ylim(0, 1.18)
    ax_level.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_level.set_ylabel("Accuracy")
    ax_level.set_title("Accuracy per CWL Level")
    ax_level.legend(fontsize=8)
    ax_level.grid(axis="y", linestyle=":", alpha=0.4)
    ax_level.axhline(chance, color="gray", linewidth=0.8, linestyle="--", alpha=0.6,
                     label="Chance")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment-level accuracy summary (legacy, multiple subjects)
# ─────────────────────────────────────────────────────────────────────────────

def plot_inference_accuracy_summary(
    data: pd.DataFrame, ax_overall: plt.Axes, ax_per_class: plt.Axes
):
    gt = data["nback_level"].to_numpy()
    raw = data["raw_state"].to_numpy()
    filt = data["filtered_state"].to_numpy()

    acc_raw = (gt == raw).mean()
    acc_filt = (gt == filt).mean()

    bars = ax_overall.bar(
        ["Raw", "Filtered"], [acc_raw, acc_filt],
        color=["#1976D2", "#E91E63"], width=0.5, edgecolor="white",
    )
    ax_overall.set_ylim(0, 1.1)
    ax_overall.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_overall.set_ylabel("Accuracy")
    ax_overall.set_title("Overall Accuracy")
    ax_overall.grid(axis="y", linestyle=":", alpha=0.4)
    for bar, val in zip(bars, [acc_raw, acc_filt]):
        ax_overall.text(
            bar.get_x() + bar.get_width() / 2, val + 0.02,
            f"{val:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    levels = [0, 1, 2]
    x = np.arange(len(levels))
    width = 0.35

    raw_per_class, filt_per_class, counts = [], [], []
    for lvl in levels:
        mask = gt == lvl
        n = mask.sum()
        counts.append(n)
        raw_per_class.append((raw[mask] == lvl).mean() if n > 0 else 0.0)
        filt_per_class.append((filt[mask] == lvl).mean() if n > 0 else 0.0)

    bars_raw = ax_per_class.bar(
        x - width / 2, raw_per_class, width, label="Raw",
        color="#1976D2", edgecolor="white",
    )
    bars_filt = ax_per_class.bar(
        x + width / 2, filt_per_class, width, label="Filtered",
        color="#E91E63", edgecolor="white",
    )
    ax_per_class.set_xticks(x)
    ax_per_class.set_xticklabels(
        [f"{STATE_LABELS[l]}\n(n={counts[i]})" for i, l in enumerate(levels)]
    )
    ax_per_class.set_ylim(0, 1.15)
    ax_per_class.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_per_class.set_ylabel("Accuracy")
    ax_per_class.set_title("Per-Class Accuracy")
    ax_per_class.legend(fontsize=8)
    ax_per_class.grid(axis="y", linestyle=":", alpha=0.4)
    for bars_group in (bars_raw, bars_filt):
        _bar_label(ax_per_class, bars_group)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory CWL helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_spline() -> pd.DataFrame | None:
    """Load spline_trajectory.csv (x, z columns used for the track outline)."""
    if not _SPLINE_FILE.exists():
        print(f"  Warning: spline trajectory not found at {_SPLINE_FILE}")
        return None
    return pd.read_csv(_SPLINE_FILE)


def _load_trial_drone(trial_dir: Path) -> pd.DataFrame | None:
    """Load drone_data.csv from a trial folder, return mean position per timestamp."""
    drone_path = trial_dir / DRONE_FILE_NAME
    if not drone_path.exists():
        return None
    df = pd.read_csv(drone_path)
    if df.empty:
        return None
    # Mean position across all drone IDs per timestamp
    return (
        df.groupby("timestamp")[["position_x", "position_z"]]
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )


def _load_trial_inference(trial_dir: Path) -> pd.DataFrame | None:
    """Load inference_data.csv from a trial folder."""
    inf_path = trial_dir / INFERENCE_FILE_NAME
    if not inf_path.exists():
        return None
    df = pd.read_csv(inf_path)
    return df.sort_values("timestamp") if not df.empty else None


def _join_cwl_to_drone(
    drone_df: pd.DataFrame, inf_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach filtered_state and nback_level to each drone row via merge_asof."""
    drone_sorted = drone_df.sort_values("timestamp")
    inf_sorted = inf_df[["timestamp", "filtered_state", "nback_level"]].sort_values(
        "timestamp"
    )
    return pd.merge_asof(
        drone_sorted, inf_sorted, on="timestamp", direction="backward"
    ).dropna(subset=["filtered_state"])


def _draw_spline_background(ax: plt.Axes, spline_df: pd.DataFrame):
    """Draw the track as a gray dashed background and set axis limits."""
    sx, sz = spline_df["x"].values, spline_df["z"].values
    ax.plot(sz, sx, color="lightgray", linewidth=2, linestyle="--", zorder=0)
    pad_x = (sx.max() - sx.min()) * 0.1
    pad_z = (sz.max() - sz.min()) * 0.1
    ax.set_ylim(sx.min() - pad_x, sx.max() + pad_x)
    # Invert horizontal axis for clockwise motion (matching visualize.py)
    ax.set_xlim(sz.max() + pad_z, sz.min() - pad_z)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Z (m)")
    ax.set_ylabel("X (m)")


def _compute_arc_param(spline_df: pd.DataFrame) -> np.ndarray:
    """Cumulative arc-length along the spline, normalised to [0, 1]."""
    sx, sz = spline_df["x"].values, spline_df["z"].values
    ds = np.sqrt(np.diff(sx) ** 2 + np.diff(sz) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(ds)])
    return arc / arc[-1]


def _project_to_arc(
    pos_x: np.ndarray,
    pos_z: np.ndarray,
    spline_df: pd.DataFrame,
    arc_param: np.ndarray,
) -> np.ndarray:
    """Project (x, z) positions onto the nearest spline point's arc parameter."""
    sx, sz = spline_df["x"].values, spline_df["z"].values
    tree = cKDTree(np.column_stack([sz, sx]))  # query in (z, x) order
    _, idx = tree.query(np.column_stack([pos_z, pos_x]))
    return arc_param[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Per-CWL-level checkbox widget
# ─────────────────────────────────────────────────────────────────────────────

def _add_cwl_checkboxes(
    ax_traj: plt.Axes,
    per_level_artists: dict[int, list],
    mean_artists: list,
) -> None:
    """Add Low / Medium / High CheckButtons to the trajectory figure.

    All three are checked by default (matches the initial full view).
    When only a subset is checked, individual traces for those levels are
    shown and *mean_artists* (aggregate mean line) are hidden — the mean
    only makes sense when all levels are visible.
    The widget reference is pinned to the figure to prevent GC.
    """
    fig = ax_traj.figure
    labels = [STATE_LABELS[lvl] for lvl in (0, 1, 2)]
    # Taller axes to fit three rows
    cb_ax = fig.add_axes([0.01, 0.01, 0.09, 0.14])
    cb_ax.set_facecolor("#f8f8f8")
    cb = CheckButtons(cb_ax, labels, [True, True, True])
    for i, lbl in enumerate(cb.labels):
        lbl.set_fontsize(9)
        lbl.set_color(STATE_COLORS[i])

    def _on_toggle(_label: str) -> None:
        statuses = cb.get_status()
        all_checked = all(statuses)
        for i, level in enumerate((0, 1, 2)):
            vis = statuses[i]
            for artist in per_level_artists.get(level, []):
                artist.set_visible(vis)
        for artist in mean_artists:
            artist.set_visible(all_checked)
        fig.canvas.draw_idle()

    cb.on_clicked(_on_toggle)
    fig._cwl_checkboxes = cb  # keep alive


# ─────────────────────────────────────────────────────────────────────────────
# Subject + task trajectory plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_subject_task_trajectory(
    data_dir: Path, cwl_level: int, spline_df: pd.DataFrame,
    ax_traj: plt.Axes, ax_acc: plt.Axes,
):
    """Left: trajectory colored by CWL per trial.  Right: per-trial accuracy bars."""
    task = _find_task_for_cwl(data_dir, cwl_level)
    if task is None:
        cwl_label = STATE_LABELS.get(cwl_level, str(cwl_level))
        print(f"  No task found for CWL level {cwl_label} under {data_dir}")
        return
    task_dir = data_dir / task
    if not task_dir.exists():
        print(f"  Task folder not found: {task_dir}")
        return

    trial_dirs = sorted(
        d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")
    )
    if not trial_dirs:
        print(f"  No trial folders found under {task_dir}")
        return

    _draw_spline_background(ax_traj, spline_df)

    # Collect accuracy data for the right panel
    trial_names, raw_accs, filt_accs, gt_levels = [], [], [], []
    drawn_levels: set[int] = set()
    per_level_artists: dict[int, list] = {0: [], 1: [], 2: []}

    for trial_dir in trial_dirs:
        drone_df = _load_trial_drone(trial_dir)
        inf_df = _load_trial_inference(trial_dir)
        if drone_df is None or inf_df is None:
            continue

        merged = _join_cwl_to_drone(drone_df, inf_df)
        if merged.empty:
            continue

        # Scatter drone positions colored by filtered CWL
        for level, color in STATE_COLORS.items():
            mask = merged["filtered_state"] == level
            if not mask.any():
                continue
            sub = merged[mask]
            label = STATE_LABELS[level] if level not in drawn_levels else "_"
            drawn_levels.add(level)
            sc = ax_traj.scatter(
                sub["position_z"], sub["position_x"],
                c=color, s=6, alpha=0.6, label=label, zorder=2,
            )
            per_level_artists[level].append(sc)

        # Mark trial start
        first = merged.iloc[0]
        ax_traj.annotate(
            trial_dir.name, (first["position_z"], first["position_x"]),
            fontsize=6, color="#555", ha="center", va="bottom",
            textcoords="offset points", xytext=(0, 4),
        )

        # Accuracy for right panel
        gt_arr = inf_df["nback_level"].to_numpy()
        raw_arr = inf_df["raw_state"].to_numpy()
        filt_arr = inf_df["filtered_state"].to_numpy()
        trial_names.append(trial_dir.name)
        raw_accs.append(float((gt_arr == raw_arr).mean()))
        filt_accs.append(float((gt_arr == filt_arr).mean()))
        gt_levels.append(int(pd.Series(gt_arr).mode().iloc[0]))

    _add_cwl_checkboxes(ax_traj, per_level_artists, [])

    gt_level = gt_levels[0] if gt_levels else -1
    gt_label = STATE_LABELS.get(gt_level, "?")
    cwl_label = STATE_LABELS.get(cwl_level, str(cwl_level))
    ax_traj.set_title(f"Trajectory — CWL: {cwl_label} ({task}) (GT: {gt_label})")
    ax_traj.legend(loc="upper right", fontsize=7, markerscale=2)

    # ── Right panel: per-trial accuracy bars ─────────────────────────────────
    if not trial_names:
        ax_acc.text(0.5, 0.5, "No data", transform=ax_acc.transAxes, ha="center")
        return

    y = np.arange(len(trial_names))
    height = 0.35
    bars_r = ax_acc.barh(
        y - height / 2, raw_accs, height, label="Raw",
        color="#1976D2", alpha=0.85, edgecolor="white",
    )
    bars_f = ax_acc.barh(
        y + height / 2, filt_accs, height, label="Filtered",
        color="#E91E63", alpha=0.85, edgecolor="white",
    )
    _hbar_label(ax_acc, bars_r)
    _hbar_label(ax_acc, bars_f)

    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels(trial_names, fontsize=9)
    ax_acc.set_xlim(0, 1.15)
    ax_acc.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_acc.set_xlabel("Accuracy")
    ax_acc.set_title(f"Per-Trial Accuracy — CWL: {cwl_label} ({task})")
    ax_acc.legend(fontsize=8, loc="lower right")
    ax_acc.grid(axis="x", linestyle=":", alpha=0.4)
    ax_acc.axvline(1 / 3, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_acc.invert_yaxis()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment + task aggregate trajectory plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_aggregate_task_trajectory(
    data_dir: Path, cwl_level: int, spline_df: pd.DataFrame,
    ax_traj: plt.Axes, ax_acc: plt.Axes,
):
    """Left: aggregate trajectory (mean ± std).  Right: per-subject accuracy."""
    cwl_label = STATE_LABELS.get(cwl_level, str(cwl_level))

    # Discover all subjects (task order may differ per subject)
    subject_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and _SUBJECT_RE.match(d.name)
    )
    if not subject_dirs:
        print(f"  No subject folders found under {data_dir}")
        return

    _draw_spline_background(ax_traj, spline_df)
    arc_param = _compute_arc_param(spline_df)

    # Per-subject: collect individual traces + accuracy
    subject_names, subject_filt_accs, subject_raw_accs = [], [], []
    all_arc_cwl: list[pd.DataFrame] = []  # for binning
    drawn_levels: set[int] = set()
    per_level_artists: dict[int, list] = {0: [], 1: [], 2: []}
    mean_artists: list = []

    for subj_dir in subject_dirs:
        # Resolve which task holds this CWL level for this subject
        task = _find_task_for_cwl(subj_dir, cwl_level)
        if task is None:
            print(
                f"  {subj_dir.name}: no task found for CWL={cwl_label}, skipping."
            )
            continue
        task_dir = subj_dir / task
        trial_dirs = sorted(
            d for d in task_dir.iterdir()
            if d.is_dir() and d.name.startswith("trial_")
        )
        if not trial_dirs:
            continue

        subj_merged_frames = []
        subj_gt, subj_raw, subj_filt = [], [], []

        for trial_dir in trial_dirs:
            drone_df = _load_trial_drone(trial_dir)
            inf_df = _load_trial_inference(trial_dir)
            if drone_df is None or inf_df is None:
                continue
            merged = _join_cwl_to_drone(drone_df, inf_df)
            if merged.empty:
                continue
            subj_merged_frames.append(merged)
            subj_gt.extend(inf_df["nback_level"].tolist())
            subj_raw.extend(inf_df["raw_state"].tolist())
            subj_filt.extend(inf_df["filtered_state"].tolist())

        if not subj_merged_frames:
            continue

        subj_all = pd.concat(subj_merged_frames, ignore_index=True)

        # Individual trace (faint)
        for level, color in STATE_COLORS.items():
            mask = subj_all["filtered_state"] == level
            if not mask.any():
                continue
            sub = subj_all[mask]
            label = STATE_LABELS[level] if level not in drawn_levels else "_"
            drawn_levels.add(level)
            sc = ax_traj.scatter(
                sub["position_z"], sub["position_x"],
                c=color, s=4, alpha=0.15, label=label, zorder=1,
            )
            per_level_artists[level].append(sc)

        # Project onto arc for binned aggregation
        arcs = _project_to_arc(
            subj_all["position_x"].values,
            subj_all["position_z"].values,
            spline_df, arc_param,
        )
        subj_all = subj_all.copy()
        subj_all["arc"] = arcs
        arc_cols = ["arc", "position_x", "position_z", "filtered_state"]
        all_arc_cwl.append(subj_all[arc_cols])

        # Accuracy
        subj_gt_arr = np.array(subj_gt)
        subj_raw_arr = np.array(subj_raw)
        subj_filt_arr = np.array(subj_filt)
        subject_names.append(f"{subj_dir.name} ({task})")
        subject_raw_accs.append(float((subj_gt_arr == subj_raw_arr).mean()))
        subject_filt_accs.append(float((subj_gt_arr == subj_filt_arr).mean()))

    # ── Mean trajectory (thick colored line) ─────────────────────────────────
    if all_arc_cwl:
        combined = pd.concat(all_arc_cwl, ignore_index=True)
        n_bins = 200
        combined["arc_bin"] = pd.cut(combined["arc"], bins=n_bins, labels=False)
        binned = combined.groupby("arc_bin").agg(
            x_mean=("position_x", "mean"),
            z_mean=("position_z", "mean"),
            x_std=("position_x", "std"),
            z_std=("position_z", "std"),
            cwl_mode=(
                "filtered_state",
                lambda s: int(s.mode().iloc[0]) if len(s) > 0 else 0,
            ),
        ).dropna()

        # Draw mean line, colored per-bin
        for _, row in binned.iterrows():
            color = STATE_COLORS.get(int(row["cwl_mode"]), "#999")
            sc = ax_traj.scatter(
                row["z_mean"], row["x_mean"],
                c=color, s=30, zorder=3, edgecolors="white", linewidths=0.3,
            )
            mean_artists.append(sc)

    _add_cwl_checkboxes(ax_traj, per_level_artists, mean_artists)

    n_subjects = len(subject_names)
    ax_traj.set_title(
        f"Aggregate Trajectory — CWL: {cwl_label} (n={n_subjects} subjects)"
    )
    ax_traj.legend(loc="upper right", fontsize=7, markerscale=2)

    # ── Right panel: per-subject accuracy ────────────────────────────────────
    if not subject_names:
        ax_acc.text(0.5, 0.5, "No data", transform=ax_acc.transAxes, ha="center")
        return

    y = np.arange(len(subject_names))
    height = 0.35
    bars_r = ax_acc.barh(
        y - height / 2, subject_raw_accs, height, label="Raw",
        color="#1976D2", alpha=0.85, edgecolor="white",
    )
    bars_f = ax_acc.barh(
        y + height / 2, subject_filt_accs, height, label="Filtered",
        color="#E91E63", alpha=0.85, edgecolor="white",
    )
    _hbar_label(ax_acc, bars_r)
    _hbar_label(ax_acc, bars_f)

    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels(subject_names, fontsize=9)
    ax_acc.set_xlim(0, 1.15)
    ax_acc.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_acc.set_xlabel("Accuracy")
    ax_acc.set_title(f"Per-Subject Accuracy — CWL: {cwl_label}")
    ax_acc.legend(fontsize=8, loc="lower right")
    ax_acc.grid(axis="x", linestyle=":", alpha=0.4)
    ax_acc.axvline(1 / 3, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_acc.invert_yaxis()

    # Summary stats text
    mean_filt = np.mean(subject_filt_accs)
    std_filt = np.std(subject_filt_accs)
    mean_raw = np.mean(subject_raw_accs)
    std_raw = np.std(subject_raw_accs)
    ax_acc.text(
        0.98, 0.02,
        f"Filtered: {mean_filt:.1%} ± {std_filt:.1%}\n"
        f"Raw: {mean_raw:.1%} ± {std_raw:.1%}",
        transform=ax_acc.transAxes, ha="right", va="bottom",
        fontsize=8, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Spline accuracy ribbon plot
# ─────────────────────────────────────────────────────────────────────────────

def _collect_merged_frames(data_dir: Path, cwl_level: int) -> list[pd.DataFrame]:
    """Return all drone+inference merged DataFrames for *cwl_level*.

    Works for both subject dirs (4-char code) and experiment dirs (containing
    subject sub-folders).  Task assignment is resolved per subject so that
    randomised task order is handled transparently.
    """
    mode = _detect_mode(data_dir)
    subject_dirs = (
        [data_dir] if mode == "subject"
        else sorted(
            d for d in data_dir.iterdir()
            if d.is_dir() and _SUBJECT_RE.match(d.name)
        )
    )
    frames: list[pd.DataFrame] = []
    for subj_dir in subject_dirs:
        task = _find_task_for_cwl(subj_dir, cwl_level)
        if task is None:
            continue
        task_dir = subj_dir / task
        trial_dirs = sorted(
            d for d in task_dir.iterdir()
            if d.is_dir() and d.name.startswith("trial_")
        )
        for trial_dir in trial_dirs:
            drone_df = _load_trial_drone(trial_dir)
            inf_df = _load_trial_inference(trial_dir)
            if drone_df is None or inf_df is None:
                continue
            merged = _join_cwl_to_drone(drone_df, inf_df)
            if not merged.empty:
                frames.append(merged)
    return frames


def _plot_spline_accuracy_ribbon(
    ax: plt.Axes,
    spline_df: pd.DataFrame,
    all_merged_frames: list[pd.DataFrame],
    cwl_level: int,
    n_bins: int = 120,
) -> None:
    """Draw a per-class prediction ribbon on the spline.

    For each arc bin, three colored bands are drawn perpendicular to the
    track and stacked outward from the spline center (Low → Medium → High).
    The width of each band is proportional to the prediction count for that
    class.  Total stacked width at the busiest bin fills *max_hw*, so the
    dominant class at each segment is immediately visible.

    A thin gray centerline replaces the previous class-colored line.
    """
    if not all_merged_frames:
        return

    arc_param = _compute_arc_param(spline_df)
    sx = spline_df["x"].values
    sz = spline_df["z"].values

    # Spline tangent + normal in plot space (z on x-axis, x on y-axis)
    tgz = np.gradient(sz)
    tgx = np.gradient(sx)
    mag = np.sqrt(tgz ** 2 + tgx ** 2) + 1e-9
    tgz /= mag
    tgx /= mag
    nz = tgx   # 90° CW normal → points outside the track for a CW loop
    nx = -tgz

    # Project inference rows onto arc and bin
    combined = pd.concat(all_merged_frames, ignore_index=True)
    arcs = _project_to_arc(
        combined["position_x"].values,
        combined["position_z"].values,
        spline_df, arc_param,
    )
    combined = combined.copy()
    combined["arc"] = arcs
    combined["arc_bin"] = pd.cut(combined["arc"], bins=n_bins, labels=False)

    # Per-bin count of each predicted class
    counts_df = (
        combined.groupby(["arc_bin", "filtered_state"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1, 2], fill_value=0)
    )
    if counts_df.empty:
        return

    # Normalize: max total across all bins → max_hw
    totals = counts_df.sum(axis=1)
    max_total = totals.max() or 1
    track_extent = min(sz.max() - sz.min(), sx.max() - sx.min())
    max_hw = track_extent * 0.06  # total stacked width at busiest bin

    # Map each spline index to its arc bin
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    spline_bin = np.clip(
        np.searchsorted(bin_edges, arc_param, side="right") - 1, 0, n_bins - 1
    )

    def _stacked_quad(i: int, j: int, inner: float, outer: float):
        """Return the four corners of a quad offset by [inner, outer] along the normal."""
        return [
            (sz[i] + nz[i] * inner, sx[i] + nx[i] * inner),
            (sz[j] + nz[j] * inner, sx[j] + nx[j] * inner),
            (sz[j] + nz[j] * outer, sx[j] + nx[j] * outer),
            (sz[i] + nz[i] * outer, sx[i] + nx[i] * outer),
        ]

    # ── Per-class stacked ribbons ─────────────────────────────────────────────
    # Draw each spline segment; for each segment look up bin counts, compute
    # cumulative offsets [0 → w0 → w0+w1 → w0+w1+w2] on the normal direction.
    for i in range(len(sz) - 1):
        b = int(spline_bin[i])
        if b not in counts_df.index:
            continue
        row = counts_df.loc[b]
        offset = 0.0
        for level in (0, 1, 2):
            w = max_hw * float(row[level]) / max_total
            if w < 1e-6:
                offset += w
                continue
            verts = _stacked_quad(i, i + 1, offset, offset + w)
            ax.add_patch(MplPolygon(
                verts, closed=True,
                facecolor=STATE_COLORS[level], alpha=0.55,
                linewidth=0, zorder=2,
            ))
            offset += w

    # ── Neutral gray centerline ───────────────────────────────────────────────
    segments = [
        [(sz[i], sx[i]), (sz[i + 1], sx[i + 1])]
        for i in range(len(sz) - 1)
    ]
    lc = LineCollection(segments, colors="#555555", linewidths=1.5, zorder=3)
    ax.add_collection(lc)

    # ── Legend ────────────────────────────────────────────────────────────────
    cwl_label = STATE_LABELS.get(cwl_level, str(cwl_level))
    for level, color in STATE_COLORS.items():
        ax.scatter([], [], c=color, s=80, marker="s", alpha=0.55,
                   label=f"{STATE_LABELS[level]} predicted")

    ax.set_title(
        f"Prediction Ribbon — CWL: {cwl_label}\n"
        "Stacked bands per arc segment · width ∝ prediction count"
    )
    ax.legend(loc="upper right", fontsize=8, markerscale=1.2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry points per mode
# ─────────────────────────────────────────────────────────────────────────────

def _save_or_show(figs: list[tuple[plt.Figure, Path]], show: bool):
    for fig, path in figs:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close("all")


def _make_time_series_figure(data: pd.DataFrame, title: str) -> plt.Figure:
    fig, (ax_states, ax_rolling) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plot_inference_time_series(data, ax_states, ax_rolling)
    fig.tight_layout()
    return fig


def run_inference(
    show: bool, output_dir: Path, data_dir: Path, cwl: int | None = None,
):
    mode = _detect_mode(data_dir)
    print(f"Loading inference data from: {data_dir}  [{mode} mode]")

    output_dir.mkdir(parents=True, exist_ok=True)
    subject = data_dir.name
    figs: list[tuple[plt.Figure, Path]] = []

    # ── CWL trajectory mode ──────────────────────────────────────────────────
    if cwl is not None:
        cwl_label = STATE_LABELS.get(cwl, str(cwl)).lower()
        spline_df = _load_spline()
        if spline_df is None:
            print("  Cannot produce trajectory plots without spline_trajectory.csv")
            return

        if mode == "subject":
            fig, (ax_traj, ax_acc) = plt.subplots(
                1, 2, figsize=(16, 7),
                gridspec_kw={"width_ratios": [2, 1], "wspace": 0.25},
            )
            fig.suptitle(
                f"CWL Trajectory — {subject} — {STATE_LABELS.get(cwl, cwl)}",
                fontsize=13, fontweight="bold",
            )
            _plot_subject_task_trajectory(data_dir, cwl, spline_df, ax_traj, ax_acc)
            fig.tight_layout()
            figs.append((fig, output_dir / f"trajectory_cwl_{cwl_label}.png"))

        elif mode == "experiment":
            fig, (ax_traj, ax_acc) = plt.subplots(
                1, 2, figsize=(16, 7),
                gridspec_kw={"width_ratios": [2, 1], "wspace": 0.25},
            )
            fig.suptitle(
                f"Aggregate CWL Trajectory — {STATE_LABELS.get(cwl, cwl)}",
                fontsize=13, fontweight="bold",
            )
            _plot_aggregate_task_trajectory(
                data_dir, cwl, spline_df, ax_traj, ax_acc,
            )
            fig.tight_layout()
            out_name = f"trajectory_cwl_aggregate_{cwl_label}.png"
            figs.append((fig, output_dir / out_name))

        # ── Accuracy ribbon plot (both modes) ────────────────────────────────
        merged_frames = _collect_merged_frames(data_dir, cwl)
        if merged_frames:
            fig_r, ax_r = plt.subplots(figsize=(10, 9))
            fig_r.suptitle(
                f"Accuracy Ribbon — {STATE_LABELS.get(cwl, cwl)}"
                + (f" — {subject}" if mode == "subject" else ""),
                fontsize=13, fontweight="bold",
            )
            _draw_spline_background(ax_r, spline_df)
            _plot_spline_accuracy_ribbon(ax_r, spline_df, merged_frames, cwl)
            fig_r.tight_layout()
            ribbon_name = f"trajectory_cwl_ribbon_{cwl_label}.png"
            figs.append((fig_r, output_dir / ribbon_name))

        else:
            print(f"  --cwl is not supported in {mode} mode.")
            return

        _save_or_show(figs, show)
        return

    # ── Standard mode (no --task) ────────────────────────────────────────────
    data = load_inference_data(data_dir)
    n_sources = data["_source"].nunique()
    print(f"  Loaded {len(data)} rows from {n_sources} session(s).")

    if mode == "trial":
        fig = _make_time_series_figure(
            data, f"Workload Inference — {subject}"
        )
        figs.append((fig, output_dir / "inference_time_series.png"))

    elif mode == "subject":
        task_data = _task_trials_only(data)
        if task_data.empty:
            print("  No task trial data found — skipping time series.")
        else:
            fig1 = _make_time_series_figure(
                task_data, f"Workload Inference — {subject} — Task Trials"
            )
            figs.append((fig1, output_dir / "inference_time_series.png"))

            fig2, (ax_task, ax_level) = plt.subplots(1, 2, figsize=(12, 5))
            fig2.suptitle(
                f"Workload Inference — {subject} — Accuracy Summary",
                fontsize=13, fontweight="bold",
            )
            plot_subject_accuracy_summary(task_data, ax_task, ax_level)
            fig2.tight_layout()
            figs.append((fig2, output_dir / "inference_accuracy_summary.png"))

    else:
        fig1 = _make_time_series_figure(
            data, "Real-Time Workload Inference — Time Series"
        )
        figs.append((fig1, output_dir / "inference_time_series.png"))

        fig2, (ax_overall, ax_per_class) = plt.subplots(1, 2, figsize=(10, 5))
        fig2.suptitle(
            "Real-Time Workload Inference — Accuracy Summary",
            fontsize=13, fontweight="bold",
        )
        plot_inference_accuracy_summary(data, ax_overall, ax_per_class)
        fig2.tight_layout()
        figs.append((fig2, output_dir / "inference_accuracy_summary.png"))

    _save_or_show(figs, show)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

RESULT_TYPES = {
    "inference": run_inference,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported result types: " + ", ".join(RESULT_TYPES),
    )
    parser.add_argument("result_type", choices=list(RESULT_TYPES))
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA, metavar="DIR")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, metavar="DIR")
    parser.add_argument(
        "--cwl", type=int, default=None, choices=[0, 1, 2], metavar="CWL",
        help="CWL level to visualize as a trajectory plot: 0=Low, 1=Medium, "
        "2=High.  The corresponding task is resolved automatically per subject.",
    )

    args = parser.parse_args()
    RESULT_TYPES[args.result_type](
        show=args.show, output_dir=args.output, data_dir=args.data, cwl=args.cwl
    )


if __name__ == "__main__":
    main()
