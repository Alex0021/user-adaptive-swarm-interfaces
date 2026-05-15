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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import CheckButtons
from scipy.spatial import cKDTree

_SERVICE_ROOT = Path(__file__).parents[3]
_DEFAULT_DATA = _SERVICE_ROOT / "data" / "experiments"
_DEFAULT_OUTPUT = _SERVICE_ROOT / "data" / "results"
_SPLINE_FILE = _SERVICE_ROOT / "data" / "spline_trajectory.csv"

INFERENCE_FILE_NAME = "inference_data.csv"
DRONE_FILE_NAME = "drone_data.csv"
INFERENCE_SAMPLE_RATE = 60  # samples per second
STATE_LABELS = {0: "Low", 1: "Medium", 2: "High"}
# Okabe-Ito colorblind-safe palette: green=Low, sky-blue=Medium, orange=High
STATE_COLORS = {0: "#009E73", 1: "#56B4E9", 2: "#E69F00"}
# Continuous 3-stop colormap built from the same Okabe-Ito colors so the
# ribbon can be rendered consistently with the other sequential cmaps.
OKABE_ITO_CMAP = LinearSegmentedColormap.from_list(
    "okabe_ito", [STATE_COLORS[0], STATE_COLORS[1], STATE_COLORS[2]]
)

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
        d
        for d in subject_dir.iterdir()
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
                ha="center",
                va="bottom",
                fontsize=8,
            )


def _hbar_label(ax: plt.Axes, bars, fmt: str = "{:.0%}", offset: float = 0.01):
    for bar in bars:
        val = bar.get_width()
        if val > 0:
            ax.text(
                val + offset,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(val),
                ha="left",
                va="center",
                fontsize=8,
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
            ax_states,
            t,
            gt,
            label="Ground truth" if first else "_",
            color="#333333",
            linewidth=1.5,
            linestyle="--",
        )
        _state_step_plot(
            ax_states,
            t,
            raw,
            label="Raw inference" if first else "_",
            color="#1976D2",
            linewidth=1.2,
            alpha=0.85,
        )
        _state_step_plot(
            ax_states,
            t,
            filt,
            label="Filtered inference" if first else "_",
            color="#E91E63",
            linewidth=1.5,
        )

        ax_rolling.plot(
            t,
            _accuracy_over_time(gt, raw),
            color="#1976D2",
            linewidth=1.2,
            alpha=0.85,
            label="Raw accuracy" if first else "_",
        )
        ax_rolling.plot(
            t,
            _accuracy_over_time(gt, filt),
            color="#E91E63",
            linewidth=1.5,
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


def _join_cwl_to_drone(drone_df: pd.DataFrame, inf_df: pd.DataFrame) -> pd.DataFrame:
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
    # Append first point to close the loop visually.
    sz_closed = np.append(sz, sz[0])
    sx_closed = np.append(sx, sx[0])
    ax.plot(sz_closed, sx_closed, color="lightgray", linewidth=2, linestyle="--", zorder=0)
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



GATE_LAYOUT_FILE = "gate_layout.csv"
GATE_STATUS_FILE = "gate_status.csv"
COMMAND_DATA_FILE = "command_data.csv"
SWARM_SIZE = 9

# Penalty weights for the composite trial-performance score (s-equivalent).
# Tune these to weight crash/miss severity vs lap time.
RACING_DEAD_PENALTY_S = 30.0
RACING_MISS_PENALTY_S = 5.0

# Adaptation-group visualisation
GROUP_COLORS = {
    "adaptive": "#FB8C00",
    "non_adaptive": "#1976D2",
    "unknown": "#9E9E9E",
}
GROUP_LABELS = {
    "adaptive": "Adaptive",
    "non_adaptive": "Non-Adaptive (Control)",
    "unknown": "Unknown",
}

DIFFICULTY_COL = "is_hard"


def _load_csv(trial_dir: Path, filename: str) -> pd.DataFrame | None:
    path = trial_dir / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _compute_centroid(drones: pd.DataFrame) -> pd.DataFrame | None:
    valid = drones[drones["timestamp"] > 0]
    if "alive" in valid.columns:
        alive = valid[valid["alive"] > 0]
        if not alive.empty:
            valid = alive
    if valid.empty:
        return None
    centroid = (
        valid.groupby("timestamp")
        .agg(
            x=("position_x", "mean"),
            y=("position_y", "mean"),
            z=("position_z", "mean"),
            n_alive=("id", "nunique"),
        )
        .reset_index()
        .sort_values("timestamp")
    )
    t0 = centroid["timestamp"].iloc[0]
    centroid["elapsed_s"] = (centroid["timestamp"] - t0) / 1000.0
    return centroid


def _gate_passage_times(gate_status: pd.DataFrame) -> pd.DataFrame:
    passed = gate_status[gate_status["first_pass_timestamp"] > 0].copy()
    # Deduplicate: keep first occurrence of each gate ID
    passed = passed.drop_duplicates(subset="id", keep="first")
    return passed.sort_values("first_pass_timestamp")


def _find_t0(inference, centroid, commands, gate_status) -> int:
    candidates = []
    for df in [inference, centroid, commands]:
        if df is not None and "timestamp" in df.columns:
            valid_ts = df[df["timestamp"] > 0]["timestamp"]
            if not valid_ts.empty:
                candidates.append(int(valid_ts.min()))
    if gate_status is not None:
        valid_ts = gate_status[gate_status["first_pass_timestamp"] > 0][
            "first_pass_timestamp"
        ]
        if not valid_ts.empty:
            candidates.append(int(valid_ts.min()))
    return min(candidates) if candidates else 0


def _shade_difficulty_z(ax, gates, alpha: float = 0.06):
    if DIFFICULTY_COL not in gates.columns:
        return
    sorted_gates = gates.sort_values("center_z")
    for i in range(len(sorted_gates) - 1):
        g_next = sorted_gates.iloc[i + 1]
        hard = bool(g_next[DIFFICULTY_COL])
        color = "#F44336" if hard else "#4CAF50"
        ax.axvspan(
            sorted_gates.iloc[i]["center_z"],
            g_next["center_z"],
            alpha=alpha,
            color=color,
            linewidth=0,
        )


def _draw_gate_lines_z(ax, gates):
    for _, g in gates.sort_values("center_z").iterrows():
        ax.axvline(g["center_z"], color="#aaa", linewidth=0.5, linestyle=":", alpha=0.5)


def _alive_at_timestamp(drones: pd.DataFrame, ts: int) -> int:
    """Return count of unique alive drone IDs at or just before ts."""
    if drones is None or drones.empty:
        return SWARM_SIZE
    before = drones[drones["timestamp"] <= ts]
    if before.empty:
        return SWARM_SIZE
    latest_ts = int(before["timestamp"].max())
    frame = before[before["timestamp"] == latest_ts]
    return int(frame[frame["alive"] > 0]["id"].nunique())


def _compute_gate_breakdown(gates, gate_status, drones) -> dict:
    """Returns mapping gate_id -> {inside, outside, dead, reached, first_pass_ts}."""
    breakdown = {}
    if gate_status is None:
        for _, g in gates.iterrows():
            breakdown[int(g["id"])] = dict(
                inside=0, outside=0, dead=SWARM_SIZE, reached=False, first_pass_ts=0
            )
        return breakdown

    # Deduplicate gate_status (keep first occurrence of each gate ID)
    gate_status_unique = gate_status.drop_duplicates(subset="id", keep="first")

    for _, row in gate_status_unique.iterrows():
        gid = int(row["id"])
        ts = int(row.get("first_pass_timestamp", 0))
        pc = int(row.get("pass_count", 0))
        reached = ts > 0
        alive_ids = _alive_at_timestamp(drones, ts) if (reached and drones is not None) else SWARM_SIZE
        dead = max(0, SWARM_SIZE - alive_ids)
        outside = max(0, alive_ids - pc)
        breakdown[gid] = dict(
            inside=pc, outside=outside, dead=dead, reached=reached, first_pass_ts=ts
        )
    return breakdown


def _no_data_placeholder(ax, title: str):
    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=10)
    ax.set_title(title)



def _save_or_show(figs: list[tuple[plt.Figure, Path]], show: bool):
    for fig, path in figs:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close("all")



def _detect_racing_mode(data_dir: Path) -> str:
    """Return 'trial', 'subject', or 'experiment' for racing data."""
    if (data_dir / GATE_LAYOUT_FILE).exists():
        return "trial"
    if _SUBJECT_RE.match(data_dir.name):
        return "subject"
    return "experiment"


# ── Per-subject racing analysis ──────────────────────────────────────────────


def _load_racing_trials(subject_dir: Path) -> list[dict]:
    """Load all racing trial folders under a subject directory.

    Returns one dict per trial with: name, gates, gate_status, inference,
    commands, drones, centroid, t0_ms.  Trials missing gate_layout or
    gate_status are skipped.

    Searches both directly in subject_dir and in task_N/ subdirectories.
    """
    trial_dirs = []

    # Look for trials directly in subject_dir (old structure)
    trial_dirs.extend(
        d for d in subject_dir.iterdir()
        if d.is_dir() and d.name.startswith("trial_")
    )

    # Also look in task_N folders (new structure)
    for task_dir in subject_dir.iterdir():
        if not (task_dir.is_dir() and task_dir.name.startswith("task_")):
            continue
        trial_dirs.extend(
            d for d in task_dir.iterdir()
            if d.is_dir() and d.name.startswith("trial_")
        )

    trial_dirs = sorted(trial_dirs)
    trials: list[dict] = []
    for trial_dir in trial_dirs:
        gates = _load_csv(trial_dir, GATE_LAYOUT_FILE)
        if gates is not None:
            gates = gates.drop_duplicates(subset=["id"], keep="first")
        gate_status = _load_csv(trial_dir, GATE_STATUS_FILE)
        if gates is None or gate_status is None:
            continue
        inference = _load_csv(trial_dir, INFERENCE_FILE_NAME)
        commands = _load_csv(trial_dir, COMMAND_DATA_FILE)
        drones = _load_csv(trial_dir, DRONE_FILE_NAME)
        centroid = _compute_centroid(drones) if drones is not None else None
        t0_ms = _find_t0(inference, centroid, commands, gate_status)
        trials.append({
            "name": trial_dir.name,
            "gates": gates,
            "gate_status": gate_status,
            "inference": inference,
            "commands": commands,
            "drones": drones,
            "centroid": centroid,
            "t0_ms": t0_ms,
        })
    return trials


def _segment_metadata(
    gates: pd.DataFrame,
) -> tuple[list[int], list[int], list[str], list[bool]]:
    """Return 8-gate block segments with alternating easy/hard difficulty.

    Gates are ordered by center_z. Each segment spans 8 consecutive gates.
    Difficulty alternates between segments (easy, hard, easy, hard, ...).
    """
    sorted_gates = gates.sort_values("center_z").reset_index(drop=True)

    a_ids: list[int] = []
    b_ids: list[int] = []
    names: list[str] = []
    diffs: list[bool] = []

    gates_per_block = 8
    seg_idx = 0
    for i in range(0, len(sorted_gates), gates_per_block):
        end_idx = min(i + gates_per_block, len(sorted_gates))
        if end_idx - i < gates_per_block:
            break

        start_gate_id = int(sorted_gates.iloc[i]["id"])
        end_gate_id = int(sorted_gates.iloc[end_idx - 1]["id"])

        is_hard = seg_idx % 2 == 1

        a_ids.append(start_gate_id)
        b_ids.append(end_gate_id)
        diffs.append(is_hard)
        names.append(f"{'Hard' if is_hard else 'Easy'} {seg_idx // 2 + 1}")
        seg_idx += 1

    return a_ids, b_ids, names, diffs



def _trial_metrics(tr: dict) -> dict | None:
    """Compute summary metrics for one trial.

    completion_s   — first → last gate in seconds.
    min_alive      — minimum centroid n_alive over the trial.
    final_alive    — n_alive at the last centroid sample.
    dead_drones    — SWARM_SIZE − final_alive.
    missed_drones  — Σ over reached gates of (alive − inside) at gate time.
                     Counts every "alive drone outside the gate" event, so
                     a chronic straggler is penalised at every gate it skips.
    penalty_s      — completion + dead·DEAD_PENALTY + miss·MISS_PENALTY.
    """
    gates = tr["gates"]
    gs = tr["gate_status"]
    drones = tr["drones"]
    centroid = tr["centroid"]
    if gates is None or gs is None:
        return None

    passed = _gate_passage_times(gs)
    if len(passed) < 2:
        return None

    completion_s = (
        passed["first_pass_timestamp"].max()
        - passed["first_pass_timestamp"].min()
    ) / 1000.0

    breakdown = _compute_gate_breakdown(gates, gs, drones)
    missed_drones = sum(
        b.get("outside", 0) for b in breakdown.values() if b.get("reached")
    )

    if centroid is not None and "n_alive" in centroid.columns:
        min_alive = int(centroid["n_alive"].min())
        final_alive = int(centroid["n_alive"].iloc[-1])
    else:
        min_alive = SWARM_SIZE
        final_alive = SWARM_SIZE
    dead_drones = SWARM_SIZE - final_alive

    gates_reached = (
        int((gs["pass_count"] > 0).sum())
        if "pass_count" in gs.columns
        else len(passed)
    )

    penalty_s = (
        completion_s
        + dead_drones * RACING_DEAD_PENALTY_S
        + missed_drones * RACING_MISS_PENALTY_S
    )

    return {
        "completion_s": completion_s,
        "min_alive": min_alive,
        "final_alive": final_alive,
        "dead_drones": dead_drones,
        "missed_drones": missed_drones,
        "gates_reached": gates_reached,
        "gates_total": len(gates),
        "penalty_s": penalty_s,
    }


def _load_experiment_racing(experiment_dir: Path) -> dict[str, list[dict]]:
    """Load racing trials for every 4-char subject folder under experiment_dir."""
    by_subject: dict[str, list[dict]] = {}
    for subj_dir in sorted(experiment_dir.iterdir()):
        if not subj_dir.is_dir() or not _SUBJECT_RE.match(subj_dir.name):
            continue
        trials = _load_racing_trials(subj_dir)
        if trials:
            by_subject[subj_dir.name] = trials
    return by_subject



def _count_crashes(drones: pd.DataFrame) -> int:
    """Count drone alive 1→0 transitions."""
    if drones is None or drones.empty or "alive" not in drones.columns:
        return 0
    crashes = 0
    for _, grp in drones.groupby("id"):
        grp = grp.sort_values("timestamp")
        alive = grp["alive"].values
        if len(alive) > 0:
            transitions = np.diff(alive.astype(int))
            crashes += int((transitions < 0).sum())
    return crashes


def _dead_count_per_gate(gate_status: pd.DataFrame, drones: pd.DataFrame) -> dict:
    """Return dict mapping gate_id -> dead_count (at time of first_pass_timestamp)."""
    result = {}
    if gate_status is None or drones is None or drones.empty:
        return result
    # Deduplicate gate_status (keep first occurrence of each gate ID)
    gate_status_unique = gate_status.drop_duplicates(subset="id", keep="first")
    for _, row in gate_status_unique.iterrows():
        gid = int(row["id"])
        ts = int(row.get("first_pass_timestamp", 0))
        if ts <= 0:
            continue
        result[gid] = SWARM_SIZE - _alive_at_timestamp(drones, ts)
    return result


def _trial_metrics_enhanced(
    tr: dict,
    p_crash: float = 1.0,
    w_dead: float = 1.0,
    p_outside: float = 1.0,
    p_skip: float = 2.0,
    t_max: float | None = None,
) -> dict | None:
    """Compute enhanced penalty metrics for one trial.

    Penalized_Time = t_completion
        + p_crash   × n_crashes
        + w_dead    × Σ_gates(dead_count_at_gate_i)
        + p_outside × Σ_gates(outside_count_at_gate_i)
        + p_skip    × n_completely_missed_gates

    p_crash, w_dead, p_outside, p_skip are penalty weights (in seconds).
    t_max is the max completion time to assign to incomplete trials.
    """
    gates = tr["gates"]
    gs = tr["gate_status"]
    drones = tr["drones"]
    if gates is None or gs is None:
        return None

    passed = _gate_passage_times(gs)
    if len(passed) < 2:
        return None

    completion_s = (
        passed["first_pass_timestamp"].max()
        - passed["first_pass_timestamp"].min()
    ) / 1000.0

    # Check if trial is incomplete and assign t_max if so
    max_gates_reached = int((gs["pass_count"] > 0).sum())
    if max_gates_reached < len(gates):
        if t_max is not None:
            completion_s = t_max

    breakdown = _compute_gate_breakdown(gates, gs, drones)

    # Component 1: crash count
    n_crashes = _count_crashes(drones)

    # Component 2: dead drones at each gate (positional penalty)
    dead_per_gate = _dead_count_per_gate(gs, drones)
    sum_dead_at_gates = sum(dead_per_gate.values())

    # Component 3: alive but outside (per gate)
    sum_outside_at_gates = sum(
        b.get("outside", 0) for b in breakdown.values() if b.get("reached")
    )

    # Component 4: completely missed gates
    n_completely_missed = sum(1 for b in breakdown.values() if b.get("reached") is False)

    penalized_s = (
        completion_s
        + p_crash * n_crashes
        + w_dead * sum_dead_at_gates
        + p_outside * sum_outside_at_gates
        + p_skip * n_completely_missed
    )

    return {
        "completion_s": completion_s,
        "n_crashes": n_crashes,
        "sum_dead_at_gates": sum_dead_at_gates,
        "sum_outside_at_gates": sum_outside_at_gates,
        "n_completely_missed": n_completely_missed,
        "penalized_s": penalized_s,
        "gates_reached": max_gates_reached,
        "gates_total": len(gates),
    }

