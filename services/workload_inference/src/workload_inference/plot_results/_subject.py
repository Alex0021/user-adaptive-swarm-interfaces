from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ._common import (
    _detect_mode, _find_task_for_cwl, load_inference_data,
    _task_trials_only, plot_inference_time_series, _load_spline,
    _load_trial_drone, _load_trial_inference, _join_cwl_to_drone,
    _draw_spline_background, _compute_arc_param, _project_to_arc,
    _add_cwl_checkboxes, _save_or_show, _bar_label,
    _load_racing_trials, _find_t0, _gate_passage_times,
    _shade_difficulty_z, _draw_gate_lines_z, _alive_at_timestamp,
    _compute_gate_breakdown, _no_data_placeholder,
    _segment_metadata, INFERENCE_FILE_NAME, DRONE_FILE_NAME,
    GATE_LAYOUT_FILE, GATE_STATUS_FILE, COMMAND_DATA_FILE,
    STATE_COLORS, STATE_LABELS, SWARM_SIZE, _DEFAULT_DATA, _DEFAULT_OUTPUT,
)


# ── Inference accuracy helpers (subject-level) ────────────────────────────────

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

    bars_r = ax_task.bar(x - width / 2, task_raw, width, label="Raw",
                         color="#1976D2", alpha=0.85, edgecolor="white")
    bars_f = ax_task.bar(x + width / 2, task_filt, width, label="Filtered",
                         color="#E91E63", alpha=0.85, edgecolor="white")
    _bar_label(ax_task, bars_r)
    _bar_label(ax_task, bars_f)

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
    ax_task.axhline(chance, color="gray", linewidth=0.8, linestyle="--",
                    alpha=0.6, label="Chance")

    # ── Panel B: per CWL level ───────────────────────────────────────────────
    levels = sorted(summary["nback_level"].unique())
    x2 = np.arange(len(levels))

    level_raw = [summary[summary["nback_level"] == l]["raw_acc"].mean() for l in levels]
    level_filt = [summary[summary["nback_level"] == l]["filt_acc"].mean() for l in levels]
    level_counts = [int((summary["nback_level"] == l).sum()) for l in levels]

    bars_r2 = ax_level.bar(x2 - width / 2, level_raw, width, label="Raw",
                           color="#1976D2", alpha=0.85, edgecolor="white")
    bars_f2 = ax_level.bar(x2 + width / 2, level_filt, width, label="Filtered",
                           color="#E91E63", alpha=0.85, edgecolor="white")
    _bar_label(ax_level, bars_r2)
    _bar_label(ax_level, bars_f2)

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
    ax_level.grid(axis="y", linestyle=":", alpha=0.4)
    ax_level.axhline(chance, color="gray", linewidth=0.8, linestyle="--",
                     alpha=0.6, label="Chance")
    ax_level.legend(fontsize=8, loc=1)


def _plot_subject_task_trajectory(
    data_dir: Path,
    cwl_level: int,
    spline_df: pd.DataFrame,
    ax_traj: plt.Axes,
    ax_acc: plt.Axes,
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
                sub["position_z"],
                sub["position_x"],
                c=color,
                s=6,
                alpha=0.6,
                label=label,
                zorder=2,
            )
            per_level_artists[level].append(sc)

        # Mark trial start
        first = merged.iloc[0]
        ax_traj.annotate(
            trial_dir.name,
            (first["position_z"], first["position_x"]),
            fontsize=6,
            color="#555",
            ha="center",
            va="bottom",
            textcoords="offset points",
            xytext=(0, 4),
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
        y - height / 2,
        raw_accs,
        height,
        label="Raw",
        color="#1976D2",
        alpha=0.85,
        edgecolor="white",
    )
    bars_f = ax_acc.barh(
        y + height / 2,
        filt_accs,
        height,
        label="Filtered",
        color="#E91E63",
        alpha=0.85,
        edgecolor="white",
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




def _plot_subject_completion_times(ax, trials):
    """Horizontal stacked bars per trial, segments split at gates.

    Easy/hard segments use distinct hues (green/red) and alternate between two
    shades of that hue so adjacent segment boundaries are clearly visible.
    Per-segment split times are annotated inside each segment when wide enough,
    and total completion time (mm:ss) is annotated at the right of each bar.
    """
    if not trials:
        _no_data_placeholder(ax, "Trial Completion Times")
        return

    easy_shades = ["#43A047", "#A5D6A7"]
    hard_shades = ["#E53935", "#FFCDD2"]

    y_positions = np.arange(len(trials))
    bar_height = 0.62
    max_total = 0.0
    drew_easy = False
    drew_hard = False

    for yi, tr in zip(y_positions, trials):
        gates = tr["gates"]
        passed = _gate_passage_times(tr["gate_status"])
        if len(passed) < 2 or gates is None:
            continue

        gates_idx = gates.set_index("id")
        ts = passed["first_pass_timestamp"].values / 1000.0
        ids = passed["id"].values
        ts0 = ts[0]

        easy_alt = 0
        hard_alt = 0
        for i in range(1, len(ts)):
            seg_start = ts[i - 1] - ts0
            seg_dur = ts[i] - ts[i - 1]
            if seg_dur <= 0:
                continue

            dest_id = int(ids[i])
            hard = (
                bool(gates_idx.loc[dest_id, DIFFICULTY_COL])
                if (
                    DIFFICULTY_COL in gates_idx.columns
                    and dest_id in gates_idx.index
                )
                else False
            )
            if hard:
                color = hard_shades[hard_alt % 2]
                hard_alt += 1
                drew_hard = True
            else:
                color = easy_shades[easy_alt % 2]
                easy_alt += 1
                drew_easy = True

            ax.barh(
                yi, seg_dur, left=seg_start, height=bar_height,
                color=color, edgecolor="white", linewidth=1.2, zorder=2,
            )

        total = ts[-1] - ts0
        max_total = max(max_total, total)
        mm = int(total // 60)
        ss = total % 60
        ax.text(
            total + 0.5, yi,
            f"{mm}m {ss:.1f}s",
            ha="left", va="center",
            fontsize=9, fontweight="bold", color="#212121",
        )

    if max_total <= 0:
        _no_data_placeholder(ax, "Trial Completion Times")
        return

    ax.set_yticks(y_positions)
    ax.set_yticklabels([tr["name"] for tr in trials], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, max_total * 1.18)
    ax.set_title(
        "Trial Completion Times — gate splits stacked, hue = difficulty"
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    from matplotlib.patches import Patch
    handles = []
    if drew_easy:
        handles.append(Patch(facecolor=easy_shades[0], edgecolor="white", label="Easy segment"))
    if drew_hard:
        handles.append(Patch(facecolor=hard_shades[0], edgecolor="white", label="Hard segment"))
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8)


def _plot_subject_segment_cwl_summary(ax, trials):
    """Per-segment CWL distribution: horizontal stacked bars, one per segment per trial.

    Rows = segments. Columns = trials. Each bar stacked: LOW | MED | HIGH.
    Gaps between segments and between trials. Uses full plot area dynamically.
    Summary stats (LOW% / MED% / HIGH%) shown inline for each trial-segment combo.
    """
    if not trials:
        _no_data_placeholder(ax, "Segment CWL Summary")
        return

    gates = trials[0]["gates"]
    if gates is None or len(gates) < 2:
        _no_data_placeholder(ax, "Segment CWL Summary")
        return

    a_ids, b_ids, seg_names, seg_diffs = _segment_metadata(gates)
    if not a_ids:
        _no_data_placeholder(ax, "Segment CWL Summary")
        return

    n_trials = len(trials)
    n_segs = len(a_ids)

    # Dynamic spacing to use full available space
    total_height = max(8, n_segs * 1.8)
    seg_height = total_height / n_segs
    trial_width = 0.85 / n_trials
    seg_gap = 0.25
    trial_gap = 0.01
    bar_height = seg_height - seg_gap
    bar_width = trial_width - trial_gap

    y_offset = total_height - seg_height / 2

    # Precompute trial-wide stats for labels
    trial_stats = []
    for tr in trials:
        inf = tr["inference"]
        if inf is None or len(inf) == 0:
            trial_stats.append((0, 0, 0))
        else:
            low = (inf["filtered_state"] == 0).sum() / len(inf) * 100
            med = (inf["filtered_state"] == 1).sum() / len(inf) * 100
            high = (inf["filtered_state"] == 2).sum() / len(inf) * 100
            trial_stats.append((low, med, high))

    for seg_idx in range(n_segs):
        ga, gb = a_ids[seg_idx], b_ids[seg_idx]
        y_base = y_offset - seg_idx * seg_height

        for trial_idx, tr in enumerate(trials):
            inf = tr["inference"]
            gs = tr["gate_status"]
            if inf is None or gs is None or len(inf) == 0:
                continue

            gs_idx = gs.drop_duplicates(subset=["id"]).set_index("id")

            try:
                t_start = float(gs_idx.loc[ga, "first_pass_timestamp"])
                t_end = float(gs_idx.loc[gb, "first_pass_timestamp"])
            except (KeyError, TypeError):
                continue
            if t_start <= 0 or t_end <= 0 or t_end <= t_start:
                continue

            mask = (inf["timestamp"] >= t_start) & (inf["timestamp"] <= t_end)
            inf_seg = inf.loc[mask]
            if len(inf_seg) < 2:
                continue

            low_count = (inf_seg["filtered_state"] == 0).sum()
            med_count = (inf_seg["filtered_state"] == 1).sum()
            high_count = (inf_seg["filtered_state"] == 2).sum()
            total = len(inf_seg)

            low_pct = low_count / total
            med_pct = med_count / total
            high_pct = high_count / total

            x_base = trial_idx * trial_width + trial_gap / 2

            # Horizontal stacked bar: LOW | MED | HIGH
            left = x_base
            if low_pct > 0:
                ax.barh(y_base, bar_width * low_pct, left=left, height=bar_height,
                       color="#43A047", edgecolor="white", linewidth=0.5, zorder=2)
                left += bar_width * low_pct
            if med_pct > 0:
                ax.barh(y_base, bar_width * med_pct, left=left, height=bar_height,
                       color="#FFC107", edgecolor="white", linewidth=0.5, zorder=2)
                left += bar_width * med_pct
            if high_pct > 0:
                ax.barh(y_base, bar_width * high_pct, left=left, height=bar_height,
                       color="#E53935", edgecolor="white", linewidth=0.5, zorder=2)

    # Y-axis: segment labels
    seg_y_positions = [y_offset - seg_idx * seg_height for seg_idx in range(n_segs)]
    ax.set_yticks(seg_y_positions)
    ax.set_yticklabels(seg_names, fontsize=9)

    # X-axis: trial labels at trial centers
    trial_x_positions = [trial_idx * trial_width + trial_width / 2 for trial_idx in range(n_trials)]
    ax.set_xticks(trial_x_positions)
    ax.set_xticklabels([tr["name"].replace("trial_", "T") for tr in trials], fontsize=9)

    # Add per-trial summary stats above the plot
    summary_y = total_height + 0.8
    for trial_idx, (low, med, high) in enumerate(trial_stats):
        x_pos = trial_idx * trial_width + trial_width / 2
        stats_text = f"{low:.0f}% / {med:.0f}% / {high:.0f}%"
        ax.text(x_pos, summary_y, stats_text, ha="center", va="bottom", fontsize=7,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ddd", alpha=0.9))

    ax.set_xlim(-0.05, n_trials * trial_width + 0.15)
    ax.set_ylim(-0.5, total_height + 1.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Segment")
    ax.set_title("Per-Segment CWL Distribution — each bar shows LOW/MED/HIGH % per segment per trial")
    ax.grid(axis="both", linestyle=":", alpha=0.15)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=STATE_COLORS[0], edgecolor="white", label="LOW"),
        Patch(facecolor=STATE_COLORS[1], edgecolor="white", label="MED"),
        Patch(facecolor=STATE_COLORS[2], edgecolor="white", label="HIGH"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=3)


def _plot_subject_segment_cwl(ax, trials):
    """Per-segment CWL stripes — y groups course segments, sub-rows per trial.

    Each course segment (between consecutive gates ordered by center_z) becomes
    a horizontal band on the y-axis.  Within each band, each trial gets its own
    horizontal stripe colored by `filtered_state` (Low / Medium / High).  The
    CWL signal is time-normalized within the segment so all trials are
    horizontally aligned regardless of completion speed.

    Difficulty is shown as a soft background band (green = easy, red = hard).
    """
    if not trials:
        _no_data_placeholder(ax, "Segment CWL Projection")
        return

    gates = trials[0]["gates"]
    if gates is None or len(gates) < 2:
        _no_data_placeholder(ax, "Segment CWL Projection")
        return

    a_ids, b_ids, seg_names, seg_diffs = _segment_metadata(gates)
    n_segments = len(a_ids)
    if n_segments == 0:
        _no_data_placeholder(ax, "Segment CWL Projection")
        return

    n_trials = len(trials)
    seg_block = 1.0
    seg_gap = 0.3
    seg_pitch = seg_block + seg_gap
    line_pitch = seg_block / (n_trials + 1)

    from matplotlib.collections import LineCollection as _LC
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    y_centers: list[float] = []
    drew_any = False
    for s_idx in range(n_segments):
        seg_top = s_idx * seg_pitch
        y_centers.append(seg_top + seg_block / 2)

        bg_color = "#FFEBEE" if seg_diffs[s_idx] else "#E8F5E9"
        ax.axhspan(seg_top, seg_top + seg_block, color=bg_color, alpha=0.55, zorder=0)

        ga, gb = a_ids[s_idx], b_ids[s_idx]
        for t_idx, tr in enumerate(trials):
            inf = tr["inference"]
            gs = tr["gate_status"]
            if inf is None or gs is None:
                continue

            try:
                gs_idx = gs.drop_duplicates(subset=["id"]).set_index("id")
                t_start = float(gs_idx.loc[ga, "first_pass_timestamp"])
                t_end = float(gs_idx.loc[gb, "first_pass_timestamp"])
            except KeyError:
                continue
            if t_start <= 0 or t_end <= 0 or t_end <= t_start:
                continue

            mask = (inf["timestamp"] >= t_start) & (inf["timestamp"] <= t_end)
            inf_seg = inf.loc[mask].sort_values("timestamp")
            if len(inf_seg) < 2:
                continue

            t_seg_min = inf_seg["timestamp"].min()
            t_seg_max = inf_seg["timestamp"].max()
            x_norm = (inf_seg["timestamp"].values - t_seg_min) / (t_seg_max - t_seg_min)
            cwl = inf_seg["filtered_state"].fillna(0).values.astype(int)

            y_line = seg_top + (t_idx + 1) * line_pitch
            points = np.column_stack([x_norm, np.full_like(x_norm, y_line)])
            segs = np.stack([points[:-1], points[1:]], axis=1)
            colors = [STATE_COLORS.get(int(c), "#999") for c in cwl[:-1]]
            ax.add_collection(
                _LC(segs, colors=colors, linewidths=8, capstyle="butt", zorder=3)
            )
            drew_any = True

            if s_idx == 0:
                ax.text(
                    -0.015, y_line, tr["name"].replace("trial_", "T"),
                    ha="right", va="center", fontsize=8, color="#444",
                )

    if not drew_any:
        _no_data_placeholder(ax, "Segment CWL Projection")
        return

    ax.set_yticks(y_centers)
    ax.set_yticklabels(seg_names, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.06, 1.02)
    ax.set_ylim(n_segments * seg_pitch - seg_gap, -0.05)
    ax.set_xlabel("Normalized segment progress  (0 = entry → 1 = exit)")
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    handles = [
        Line2D([0], [0], color=STATE_COLORS[i], linewidth=6, label=f"CWL {STATE_LABELS[i]}")
        for i in range(3)
    ]
    handles += [
        Patch(facecolor="#E8F5E9", edgecolor="none", label="Easy bg"),
        Patch(facecolor="#FFEBEE", edgecolor="none", label="Hard bg"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=5)


def _plot_subject_segment_adaptation(ax, trials):
    """Per-segment flight profile (adaptation step) time-normalized visualization.

    Similar to CWL visualization but shows flight profile step (0-23) with Viridis colormap
    (colorblind-safe): 0 (dark purple, slow) → 23 (bright yellow, fast).
    """
    if not trials:
        _no_data_placeholder(ax, "Segment Adaptation Profile")
        return

    gates = trials[0]["gates"]
    if gates is None or len(gates) < 2:
        _no_data_placeholder(ax, "Segment Adaptation Profile")
        return

    a_ids, b_ids, seg_names, seg_diffs = _segment_metadata(gates)
    n_segments = len(a_ids)
    if n_segments == 0:
        _no_data_placeholder(ax, "Segment Adaptation Profile")
        return

    n_trials = len(trials)
    seg_block = 1.0
    seg_gap = 0.3
    seg_pitch = seg_block + seg_gap
    line_pitch = seg_block / (n_trials + 1)

    from matplotlib.collections import LineCollection as _LC
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.cm as cm

    # Use Viridis: colorblind-safe, perceptually uniform
    # 0 (dark purple) → 23 (bright yellow)
    cmap = cm.get_cmap("viridis")

    y_centers: list[float] = []
    drew_any = False
    for s_idx in range(n_segments):
        seg_top = s_idx * seg_pitch
        y_centers.append(seg_top + seg_block / 2)

        bg_color = "#FFEBEE" if seg_diffs[s_idx] else "#E8F5E9"
        ax.axhspan(seg_top, seg_top + seg_block, color=bg_color, alpha=0.55, zorder=0)

        ga, gb = a_ids[s_idx], b_ids[s_idx]
        for t_idx, tr in enumerate(trials):
            commands = tr["commands"]
            gs = tr["gate_status"]
            if commands is None or gs is None:
                continue
            if "cwl_current_step" not in commands.columns:
                continue

            gs_idx = gs.drop_duplicates(subset=["id"]).set_index("id")

            try:
                t_start = float(gs_idx.loc[ga, "first_pass_timestamp"])
                t_end = float(gs_idx.loc[gb, "first_pass_timestamp"])
            except (KeyError, TypeError):
                continue
            if t_start <= 0 or t_end <= 0 or t_end <= t_start:
                continue

            mask = (commands["timestamp"] >= t_start) & (commands["timestamp"] <= t_end)
            cmd_seg = commands.loc[mask].sort_values("timestamp")
            if len(cmd_seg) < 2:
                continue

            t_seg_min = cmd_seg["timestamp"].min()
            t_seg_max = cmd_seg["timestamp"].max()
            x_norm = (cmd_seg["timestamp"].values - t_seg_min) / (t_seg_max - t_seg_min)
            steps = cmd_seg["cwl_current_step"].fillna(0).values.astype(float)

            y_line = seg_top + (t_idx + 1) * line_pitch
            points = np.column_stack([x_norm, np.full_like(x_norm, y_line)])
            segs = np.stack([points[:-1], points[1:]], axis=1)

            # Normalize steps to [0, 1] for colormap (0-23 → 0-1)
            step_colors = [cmap(int(s) / 23.0) for s in steps[:-1]]
            ax.add_collection(
                _LC(segs, colors=step_colors, linewidths=8, capstyle="butt", zorder=3)
            )
            drew_any = True

            if s_idx == 0:
                ax.text(
                    -0.015, y_line, tr["name"].replace("trial_", "T"),
                    ha="right", va="center", fontsize=8, color="#444",
                )

    if not drew_any:
        _no_data_placeholder(ax, "Segment Adaptation Profile")
        return

    ax.set_yticks(y_centers)
    ax.set_yticklabels(seg_names, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.06, 1.02)
    ax.set_ylim(n_segments * seg_pitch - seg_gap, -0.05)
    ax.set_xlabel("Normalized segment progress  (0 = entry → 1 = exit)")
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    # Add colorbar for step values
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import ColorbarBase
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=23))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, aspect=20, fraction=0.03)
    cbar.set_label("Flight Step (0=slow → 23=fast)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _plot_subject_segment_adaptation_summary(ax, trials):
    """Per-segment adaptation profile: rows = segments, cols = trials.

    Each bar is stacked with segments proportional to occurrence of each flight step.
    Colors use Viridis (colorblind-safe): dark purple (slow) → bright yellow (fast).
    Summary stats show average step per trial.
    """
    if not trials:
        _no_data_placeholder(ax, "Segment Adaptation Summary")
        return

    gates = trials[0]["gates"]
    if gates is None or len(gates) < 2:
        _no_data_placeholder(ax, "Segment Adaptation Summary")
        return

    a_ids, b_ids, seg_names, seg_diffs = _segment_metadata(gates)
    if not a_ids:
        _no_data_placeholder(ax, "Segment Adaptation Summary")
        return

    n_trials = len(trials)
    n_segs = len(a_ids)

    import matplotlib.cm as cm

    # Use Viridis: colorblind-safe, perceptually uniform (0 = dark purple → 23 = bright yellow)
    cmap = cm.get_cmap("viridis")

    # Dynamic spacing to use full available space
    total_height = max(8, n_segs * 1.8)
    seg_height = total_height / n_segs
    trial_width = 0.85 / n_trials
    seg_gap = 0.25
    trial_gap = 0.01
    bar_height = seg_height - seg_gap
    bar_width = trial_width - trial_gap

    y_offset = total_height - seg_height / 2

    # Precompute trial-wide stats for labels
    trial_stats = []
    for tr in trials:
        commands = tr["commands"]
        if commands is None or "cwl_current_step" not in commands.columns or len(commands) == 0:
            trial_stats.append(0)
        else:
            avg_step = commands["cwl_current_step"].dropna().mean()
            trial_stats.append(avg_step)

    for seg_idx in range(n_segs):
        ga, gb = a_ids[seg_idx], b_ids[seg_idx]
        y_base = y_offset - seg_idx * seg_height

        for trial_idx, tr in enumerate(trials):
            commands = tr["commands"]
            gs = tr["gate_status"]
            if commands is None or gs is None or "cwl_current_step" not in commands.columns:
                continue

            gs_idx = gs.drop_duplicates(subset=["id"]).set_index("id")

            try:
                t_start = float(gs_idx.loc[ga, "first_pass_timestamp"])
                t_end = float(gs_idx.loc[gb, "first_pass_timestamp"])
            except (KeyError, TypeError):
                continue
            if t_start <= 0 or t_end <= 0 or t_end <= t_start:
                continue

            mask = (commands["timestamp"] >= t_start) & (commands["timestamp"] <= t_end)
            cmd_seg = commands.loc[mask]
            if len(cmd_seg) < 2:
                continue

            steps = cmd_seg["cwl_current_step"].dropna()
            if len(steps) == 0:
                continue

            # Get step value distribution: bin steps into groups and show as stacked bar
            step_counts = steps.value_counts().sort_index()
            total_samples = len(steps)

            x_base = trial_idx * trial_width + trial_gap / 2
            x_pos = x_base

            # Draw stacked bar with width proportional to occurrence of each step
            for step_val, count in step_counts.items():
                width_frac = count / total_samples
                seg_width = bar_width * width_frac
                color = cmap(min(step_val / 23.0, 1.0))  # Normalize to [0, 1]

                ax.barh(y_base, seg_width, left=x_pos, height=bar_height,
                       color=color, edgecolor="white", linewidth=0.3, zorder=2)
                x_pos += seg_width

    # Y-axis: segment labels
    seg_y_positions = [y_offset - seg_idx * seg_height for seg_idx in range(n_segs)]
    ax.set_yticks(seg_y_positions)
    ax.set_yticklabels(seg_names, fontsize=9)

    # X-axis: trial labels at trial centers
    trial_x_positions = [trial_idx * trial_width + trial_width / 2 for trial_idx in range(n_trials)]
    ax.set_xticks(trial_x_positions)
    ax.set_xticklabels([tr["name"].replace("trial_", "T") for tr in trials], fontsize=9)

    # Add per-trial summary stats above the plot
    summary_y = total_height + 0.8
    for trial_idx, avg_step in enumerate(trial_stats):
        x_pos = trial_idx * trial_width + trial_width / 2
        stats_text = f"Avg: {avg_step:.1f}"
        ax.text(x_pos, summary_y, stats_text, ha="center", va="bottom", fontsize=7,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ddd", alpha=0.9))

    ax.set_xlim(-0.05, n_trials * trial_width + 0.15)
    ax.set_ylim(-0.5, total_height + 1.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Segment")
    ax.set_title("Per-Segment Adaptation Profile — bar width shows step distribution")
    ax.grid(axis="both", linestyle=":", alpha=0.15)

    # Add colorbar for step values
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=23))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, aspect=20, fraction=0.03)
    cbar.set_label("Flight Step (0=slow → 23=fast)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _run_racing_subject(show: bool, output_dir: Path, data_dir: Path, traj_type: str = "inference"):
    trials = _load_racing_trials(data_dir)
    if not trials:
        print("  No trial folders with gate_layout + gate_status found.")
        return
    print(f"  Loaded {len(trials)} trial(s)")

    figs: list[tuple[plt.Figure, Path]] = []
    subject = data_dir.name

    fig1_h = max(3.5, 0.9 + 0.7 * len(trials))
    fig1, ax1 = plt.subplots(figsize=(15, fig1_h))
    fig1.suptitle(
        f"Racing — {subject} --- Trial Completion Times",
        fontsize=13, fontweight="bold",
    )
    _plot_subject_completion_times(ax1, trials)
    fig1.tight_layout()
    figs.append((fig1, output_dir / f"racing_subject_{subject}_completion_times.png"))

    n_segs_est = max(1, len(trials[0]["gates"]) - 1) if trials[0]["gates"] is not None else 5
    fig2_h = max(6.5, 0.7 * n_segs_est + 0.25 * n_segs_est * len(trials))
    fig2, ax2 = plt.subplots(figsize=(15, min(fig2_h, 14)))
    fig2.suptitle(
        f"Racing — {subject} --- Segment CWL Profile",
        fontsize=13, fontweight="bold",
    )
    _plot_subject_segment_cwl(ax2, trials)
    fig2.tight_layout()
    figs.append((fig2, output_dir / f"racing_subject_{subject}_segment_cwl.png"))

    _save_or_show(figs, show)
