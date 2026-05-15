from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as sp_stats

from ._common import (
    _DEFAULT_DATA,
    _DEFAULT_OUTPUT,
    _SUBJECT_RE,
    COMMAND_DATA_FILE,
    DRONE_FILE_NAME,
    GATE_STATUS_FILE,
    GROUP_COLORS,
    GROUP_LABELS,
    INFERENCE_FILE_NAME,
    STATE_COLORS,
    STATE_LABELS,
    SWARM_SIZE,
    LineCollection,
    MplPolygon,
    _add_cwl_checkboxes,
    _alive_at_timestamp,
    _bar_label,
    _compute_arc_param,
    _compute_gate_breakdown,
    _count_crashes,
    _dead_count_per_gate,
    _detect_mode,
    _draw_spline_background,
    _find_task_for_cwl,
    _gate_passage_times,
    _hbar_label,
    _join_cwl_to_drone,
    _load_experiment_racing,
    _load_spline,
    _load_trial_drone,
    _load_trial_inference,
    _no_data_placeholder,
    _project_to_arc,
    _save_or_show,
    _segment_metadata,
    _trial_metrics,
    _trial_metrics_enhanced,
    load_inference_data,
)

# ── Colormap registry ─────────────────────────────────────────────────────────
# Curated colorblind-safe colormaps. All custom diverging maps are built via
# _make_diverging_cmap(), which supports a tunable neutral plateau so that
# near-zero deltas stay white while larger changes snap to strong color.
#
# ── Tuning knob ──────────────────────────────────────────────────────────────
# DIVERGING_WHITE_WIDTH: fraction of the full colormap range that maps to the
# neutral mid-color (white by default).
#   0.00 → linear transition, no plateau (classic 3-stop diverging)
#   0.20 → ±10 % of the data range is white — sharpens visible transitions
#   0.40 → ±20 % is white — only strong deviations show color
# After changing this value, reload the module (importlib.reload) to rebuild.
DIVERGING_WHITE_WIDTH: float = 0.20


def _make_diverging_cmap(
    name: str,
    low: str,
    high: str,
    mid: str = "#FFFFFF",
    white_width: float | None = None,
    N: int = 256,
) -> LinearSegmentedColormap:
    """Build a diverging colormap with an optional neutral plateau at the centre.

    Parameters
    ----------
    low / high : hex colours for the negative / positive extremes.
    mid        : neutral centre colour (default white).
    white_width: width of the flat mid-colour plateau as a fraction of the
                 full [0, 1] normalised range. ``None`` uses the module-level
                 ``DIVERGING_WHITE_WIDTH``. Pass ``0.0`` for a plain 3-stop
                 linear map with no plateau.
    """
    w = DIVERGING_WHITE_WIDTH if white_width is None else white_width
    w = max(0.0, min(w, 0.98))
    half = w / 2.0
    anchors: list[tuple[float, str]] = [
        (0.0, low),
        (0.5 - half, mid),
        (0.5 + half, mid),
        (1.0, high),
    ]
    return LinearSegmentedColormap.from_list(name, anchors, N=N)


_DIVERGING_PRESETS: dict[str, tuple[str, str]] = {
    # name           : (low_hex,    high_hex)
    "ok_div": ("#0072B2", "#D55E00"),  # Okabe-Ito blue ↔ vermillion
    "vik_like": ("#001959", "#7B0700"),  # deep navy ↔ deep red
    "broc_like": ("#1A4314", "#4A148C"),  # dark green ↔ dark purple
    "cork_like": ("#003545", "#59083A"),  # teal ↔ magenta
    "ok_div_strong": ("#01355A", "#5B0000"),  # muted Okabe, darker ends
}

_SEQUENTIAL_CMAPS = {
    "ok_red": LinearSegmentedColormap.from_list(
        "ok_red", ["#FFFFFF", "#FFE0B5", "#D55E00", "#5B0000"], N=256
    ),
    "ok_blue": LinearSegmentedColormap.from_list(
        "ok_blue", ["#FFFFFF", "#CDE5F2", "#0072B2", "#00263F"], N=256
    ),
    "ok_green": LinearSegmentedColormap.from_list(
        "ok_green", ["#FFFFFF", "#CFE8DC", "#009E73", "#003D2C"], N=256
    ),
}

WINDOW_CMAPS: dict[str, object] = {
    **{
        k: _make_diverging_cmap(k, low, high)
        for k, (low, high) in _DIVERGING_PRESETS.items()
    },
    **_SEQUENTIAL_CMAPS,
    # built-in matplotlib names — all colorblind-safe / perceptual
    "RdBu_r": "RdBu_r",
    "PuOr_r": "PuOr_r",
    "BrBG_r": "BrBG_r",
    "coolwarm": "coolwarm",
    "viridis": "viridis",
    "cividis": "cividis",
    "plasma": "plasma",
    "magma": "magma",
    "inferno": "inferno",
    "YlOrRd": "YlOrRd",
    "YlGnBu": "YlGnBu",
}


def _resolve_cmap(name):
    """Return a Colormap (or matplotlib-known string) from the registry; passthrough otherwise."""
    if name is None:
        return None
    if not isinstance(name, str):
        return name  # already a Colormap object
    return WINDOW_CMAPS.get(name, name)


# ── Crash-figure colormap defaults (tweak to explore) ─────────────────────────
# All names must exist in WINDOW_CMAPS or be valid matplotlib cmap names.
#
# Crash density / outside-gate heatmaps: sequential, single-hue.
#   Suggestions: "ok_red", "ok_blue", "ok_green", "YlOrRd", "YlGnBu",
#                "viridis", "cividis", "plasma", "magma", "inferno"
CRASH_HEATMAP_CMAP: str | None = None  # None → metric default ("YlOrRd")
OUTSIDE_HEATMAP_CMAP: str | None = None  # None → metric default ("YlGnBu")

# Crash-window CWL/step heatmaps: diverging when WINDOW_USE_DELTA=True.
#   Suggestions: "ok_div", "vik_like", "broc_like", "cork_like",
#                "ok_div_strong", "RdBu_r", "PuOr_r", "BrBG_r", "coolwarm"
WINDOW_CWL_CMAP: str = "vik_like"
WINDOW_STEP_CMAP: str = "vik_like"
WINDOW_USE_DELTA: bool = True  # plot variation vs. start-of-window
WINDOW_AVERAGED: bool = False  # per-subject (False) vs. per-group (True)


# ── Inference accuracy summary (experiment-level) ─────────────────────────────


def plot_inference_accuracy_summary(
    data: pd.DataFrame, ax_overall: plt.Axes, ax_per_class: plt.Axes
):
    gt = data["nback_level"].to_numpy()
    raw = data["raw_state"].to_numpy()
    filt = data["filtered_state"].to_numpy()

    acc_raw = (gt == raw).mean()
    acc_filt = (gt == filt).mean()

    bars = ax_overall.bar(
        ["Raw", "Filtered"],
        [acc_raw, acc_filt],
        color=["#1976D2", "#E91E63"],
        width=0.5,
        edgecolor="white",
    )
    ax_overall.set_ylim(0, 1.1)
    ax_overall.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_overall.set_ylabel("Accuracy")
    ax_overall.set_title("Overall Accuracy")
    ax_overall.grid(axis="y", linestyle=":", alpha=0.4)
    for bar, val in zip(bars, [acc_raw, acc_filt]):
        ax_overall.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
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
        x - width / 2,
        raw_per_class,
        width,
        label="Raw",
        color="#1976D2",
        edgecolor="white",
    )
    bars_filt = ax_per_class.bar(
        x + width / 2,
        filt_per_class,
        width,
        label="Filtered",
        color="#E91E63",
        edgecolor="white",
    )
    ax_per_class.set_xticks(x)
    ax_per_class.set_xticklabels(
        [f"{STATE_LABELS[l]}\n(n={counts[i]})" for i, l in enumerate(levels)]
    )
    ax_per_class.set_ylim(0, 1.15)
    ax_per_class.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_per_class.set_ylabel("Accuracy")
    ax_per_class.set_title("Per-Class Accuracy")
    ax_per_class.legend(fontsize=8, loc=1)
    ax_per_class.grid(axis="y", linestyle=":", alpha=0.4)
    for bars_group in (bars_raw, bars_filt):
        _bar_label(ax_per_class, bars_group)


def _read_subject_adaptive_flag(subject_dir: Path) -> str:
    """Read adaptive flag from subject's extra_info.yaml/.yml.

    Returns 'adaptive' / 'non_adaptive' / 'unknown'.
    """
    for name in ("extra_info.yaml", "extra_info.yml"):
        info_file = subject_dir / name
        if not info_file.exists():
            continue
        try:
            with info_file.open() as f:
                info = yaml.safe_load(f)
            val = info.get("adaptive")
            if val is True:
                return "adaptive"
            if val is False:
                return "non_adaptive"
        except Exception:
            pass
    return "unknown"


def _plot_aggregate_task_trajectory(
    data_dir: Path,
    cwl_level: int,
    spline_df: pd.DataFrame,
    ax_traj: plt.Axes,
    ax_acc: plt.Axes,
):
    """Left: aggregate trajectory (mean ± std).  Right: per-subject accuracy."""
    cwl_label = STATE_LABELS.get(cwl_level, str(cwl_level))

    # Discover all subjects (task order may differ per subject)
    subject_dirs = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and _SUBJECT_RE.match(d.name)
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
            print(f"  {subj_dir.name}: no task found for CWL={cwl_label}, skipping.")
            continue
        task_dir = subj_dir / task
        trial_dirs = sorted(
            d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")
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
                sub["position_z"],
                sub["position_x"],
                c=color,
                s=4,
                alpha=0.15,
                label=label,
                zorder=1,
            )
            per_level_artists[level].append(sc)

        # Project onto arc for binned aggregation
        arcs = _project_to_arc(
            subj_all["position_x"].values,
            subj_all["position_z"].values,
            spline_df,
            arc_param,
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
        binned = (
            combined.groupby("arc_bin")
            .agg(
                x_mean=("position_x", "mean"),
                z_mean=("position_z", "mean"),
                x_std=("position_x", "std"),
                z_std=("position_z", "std"),
                cwl_mode=(
                    "filtered_state",
                    lambda s: int(s.mode().iloc[0]) if len(s) > 0 else 0,
                ),
            )
            .dropna()
        )

        # Draw mean line, colored per-bin
        for _, row in binned.iterrows():
            color = STATE_COLORS.get(int(row["cwl_mode"]), "#999")
            sc = ax_traj.scatter(
                row["z_mean"],
                row["x_mean"],
                c=color,
                s=30,
                zorder=3,
                edgecolors="white",
                linewidths=0.3,
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
        y - height / 2,
        subject_raw_accs,
        height,
        label="Raw",
        color="#1976D2",
        alpha=0.85,
        edgecolor="white",
    )
    bars_f = ax_acc.barh(
        y + height / 2,
        subject_filt_accs,
        height,
        label="Filtered",
        color="#E91E63",
        alpha=0.85,
        edgecolor="white",
    )
    _hbar_label(ax_acc, bars_r)
    _hbar_label(ax_acc, bars_f)

    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels(subject_names, fontsize=9)
    ax_acc.set_xlim(0, 1.15)
    ax_acc.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_acc.set_xlabel("Accuracy")
    ax_acc.set_title(f"Per-Subject Accuracy — CWL: {cwl_label}")
    ax_acc.legend(fontsize=8, loc="upper right")
    ax_acc.grid(axis="x", linestyle=":", alpha=0.4)
    ax_acc.axvline(1 / 3, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_acc.invert_yaxis()

    # Summary stats text
    mean_filt = np.mean(subject_filt_accs)
    std_filt = np.std(subject_filt_accs)
    mean_raw = np.mean(subject_raw_accs)
    std_raw = np.std(subject_raw_accs)
    ax_acc.text(
        0.98,
        0.02,
        f"Filtered: {mean_filt:.1%} ± {std_filt:.1%}\n"
        f"Raw: {mean_raw:.1%} ± {std_raw:.1%}",
        transform=ax_acc.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def _collect_merged_frames(data_dir: Path) -> pd.DataFrame:
    """Load drone+inference data for all CWL levels in a single pass.

    Returns a concatenated DataFrame with an added *cwl_level* column so the
    caller can efficiently slice per level without re-reading any files.
    Works for both subject dirs (4-char code) and experiment dirs (containing
    subject sub-folders).  Task assignment is resolved per subject so that
    randomised task order is handled transparently.
    """
    mode = _detect_mode(data_dir)
    subject_dirs = (
        [data_dir]
        if mode == "subject"
        else sorted(
            d for d in data_dir.iterdir() if d.is_dir() and _SUBJECT_RE.match(d.name)
        )
    )
    frames: list[pd.DataFrame] = []
    for cwl_level in (0, 1, 2):
        for subj_dir in subject_dirs:
            task = _find_task_for_cwl(subj_dir, cwl_level)
            if task is None:
                continue
            task_dir = subj_dir / task
            trial_dirs = sorted(
                d
                for d in task_dir.iterdir()
                if d.is_dir() and d.name.startswith("trial_")
            )
            for trial_dir in trial_dirs:
                drone_df = _load_trial_drone(trial_dir)
                inf_df = _load_trial_inference(trial_dir)
                if drone_df is None or inf_df is None:
                    continue
                merged = _join_cwl_to_drone(drone_df, inf_df)
                if not merged.empty:
                    merged = merged.copy()
                    merged["cwl_level"] = cwl_level
                    frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot_spline_accuracy_ribbon(
    ax: plt.Axes,
    spline_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    cwl_level: int,
    n_bins: int = 120,
    ax_summary: plt.Axes | None = None,
    cmap=None,
    show_legend: bool = True,
) -> None:
    """Draw a per-class prediction ribbon on the spline.

    *merged_df* is the combined DataFrame returned by _collect_merged_frames
    (all CWL levels, with a *cwl_level* column).  This function filters it to
    *cwl_level* internally so the caller can reuse the same DataFrame for all
    three levels without re-loading data.

    For each arc bin, three colored bands are drawn perpendicular to the
    track and stacked outward from the spline center (Low → Medium → High).
    The width of each band is proportional to the prediction count for that
    class.  Total stacked width at the busiest bin fills *max_hw*.

    *cmap* controls the three band colors.  Pass a colormap name (str) or any
    Matplotlib Colormap object.  The three class colors are sampled at
    positions 0.15 / 0.5 / 0.85 so the full dynamic range of the colormap
    is used without hitting the extreme dark/light ends.
    Pass ``None`` (default) to fall back to the Okabe-Ito STATE_COLORS palette.

    If *ax_summary* is provided, a horizontal stacked bar is drawn showing the
    global prediction distribution across the entire track.
    """
    combined = merged_df[merged_df["cwl_level"] == cwl_level]
    if combined.empty:
        return

    # ── Resolve per-class colors ──────────────────────────────────────────────
    if cmap is None:
        level_colors = STATE_COLORS
    else:
        import matplotlib.cm as _mcm

        _cm = _mcm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        _positions = (0.15, 0.5, 0.85)
        level_colors = {lvl: _cm(_positions[i]) for i, lvl in enumerate((0, 1, 2))}

    arc_param = _compute_arc_param(spline_df)
    sx = spline_df["x"].values
    sz = spline_df["z"].values

    # Spline tangent + normal in plot space (z on x-axis, x on y-axis)
    tgz = np.gradient(sz)
    tgx = np.gradient(sx)
    mag = np.sqrt(tgz**2 + tgx**2) + 1e-9
    tgz /= mag
    tgx /= mag
    nz = tgx  # 90° CW normal → points outside the track for a CW loop
    nx = -tgz

    # Use consistent bin edges for both spline segments and data points so that
    # the last few segments near the loop seam are never skipped due to a
    # mismatch between pd.cut auto-edges and the [0, 1] linspace used below.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Project inference rows onto arc and bin
    combined = combined.reset_index(drop=True)
    arcs = _project_to_arc(
        combined["position_x"].values,
        combined["position_z"].values,
        spline_df,
        arc_param,
    )
    combined = combined.copy()
    combined["arc"] = arcs
    # Use the same searchsorted binning as the spline so edges always align.
    combined["arc_bin"] = np.clip(
        np.searchsorted(bin_edges, arcs, side="right") - 1, 0, n_bins - 1
    ).astype(int)

    # ── Closed-loop seam correction ───────────────────────────────────────────
    # For a closed track the KD-tree assigns drone positions near the seam to
    # arc≈0 (start, lower index wins) rather than arc≈1 (end).  Fix this by
    # comparing each point's distance to the first vs last N spline points.
    # Points that are actually closer to the end are remapped to their correct
    # arc bin using the nearest end-spline index (not just forced to bin N-1).
    from scipy.spatial import cKDTree as _KDT

    dist_seam = float(np.sqrt((sz[0] - sz[-1]) ** 2 + (sx[0] - sx[-1]) ** 2))
    track_scale = float(max(sz.max() - sz.min(), sx.max() - sx.min()))
    if dist_seam < track_scale * 0.12:  # nearly-closed loop
        n_end = max(8, len(sz) // 15)
        positions = np.column_stack(
            [combined["position_z"].values, combined["position_x"].values]
        )
        d_start, _ = _KDT(np.column_stack([sz[:n_end], sx[:n_end]])).query(positions)
        d_end, end_idx = _KDT(np.column_stack([sz[-n_end:], sx[-n_end:]])).query(
            positions
        )
        seam_thresh = max(3, n_bins // 20)
        should_be_end = (combined["arc_bin"] <= seam_thresh) & (d_end < d_start)
        if should_be_end.any():
            # Map each reassigned point to its correct arc bin via the nearest
            # end-spline index rather than blindly forcing to n_bins-1.
            global_idx = (len(sz) - n_end) + end_idx[should_be_end]
            correct_bins = np.clip(
                np.searchsorted(bin_edges, arc_param[global_idx], side="right") - 1,
                0,
                n_bins - 1,
            )
            correct_bins = correct_bins.astype(int)
            combined.loc[should_be_end, "arc_bin"] = correct_bins

    # Per-bin count of each predicted class
    counts_df = (
        combined.groupby(["arc_bin", "filtered_state"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1, 2], fill_value=0)
    )
    if counts_df.empty:
        return

    # Ensure every bin 0..n_bins-1 exists; fill any remaining gaps via ffill/bfill.
    counts_df = counts_df.reindex(range(n_bins), fill_value=0).astype(float)
    empty_mask = counts_df.sum(axis=1) == 0
    if empty_mask.any():
        counts_df.loc[empty_mask, :] = np.nan
        counts_df = counts_df.ffill().bfill().fillna(0)

    # ── Spatial CWL variance ──────────────────────────────────────────────────
    # For each arc bin compute the expected CWL value E[CWL_b] = Σ level*P(level|b)
    # then take the std over all bins and normalise to [0,1] (÷2 because CWL∈{0,1,2}).
    # Low σ_spatial ≈ steady CWL across the track (n-back drives perception).
    # High σ_spatial = track sections add/remove cognitive load on top of the task.
    _bin_totals = counts_df.sum(axis=1).replace(0, np.nan)
    _probs = counts_df.div(_bin_totals, axis=0).fillna(0)
    _expected_cwl = _probs[0] * 0 + _probs[1] * 1 + _probs[2] * 2
    sigma_spatial = float(_expected_cwl.std()) / 2.0  # normalised to [0, 1]

    # Normalize: max total across all bins → max_hw
    totals = counts_df.sum(axis=1)
    max_total = totals.max() or 1
    track_extent = min(sz.max() - sz.min(), sx.max() - sx.min())
    max_hw = track_extent * 0.06  # total stacked width at busiest bin

    # Map each spline index to its arc bin
    spline_bin = np.clip(
        np.searchsorted(bin_edges, arc_param, side="right") - 1, 0, n_bins - 1
    )

    def _stacked_quad(i: int, j: int, inner: float, outer: float):
        return [
            (sz[i] + nz[i] * inner, sx[i] + nx[i] * inner),
            (sz[j] + nz[j] * inner, sx[j] + nx[j] * inner),
            (sz[j] + nz[j] * outer, sx[j] + nx[j] * outer),
            (sz[i] + nz[i] * outer, sx[i] + nx[i] * outer),
        ]

    # ── Per-class stacked ribbons ─────────────────────────────────────────────
    # Use modular indexing so the loop-closing segment (last→first spline point)
    # is always rendered, eliminating the visible gap at the track seam.
    n_pts = len(sz)
    for i in range(n_pts):
        j = (i + 1) % n_pts
        b = int(spline_bin[i])
        row = counts_df.loc[b]
        offset = 0.0
        for level in (0, 1, 2):
            w = max_hw * float(row[level]) / max_total
            if w < 1e-6:
                offset += w
                continue
            verts = _stacked_quad(i, j, offset, offset + w)
            ax.add_patch(
                MplPolygon(
                    verts,
                    closed=True,
                    facecolor=level_colors[level],
                    linewidth=0,
                    zorder=2,
                )
            )
            offset += w

    # ── Neutral gray centerline (closed loop) ────────────────────────────────
    segments = [
        [(sz[i], sx[i]), (sz[(i + 1) % n_pts], sx[(i + 1) % n_pts])]
        for i in range(n_pts)
    ]
    lc = LineCollection(segments, colors="#555555", linewidths=1.5, zorder=3)
    ax.add_collection(lc)

    # ── σ_spatial — centred on the ribbon plot ────────────────────────────────
    ax.text(
        0.5,
        0.5,
        f"σ = {sigma_spatial:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="black",
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.6
        ),
    )

    # ── Legend (first panel only) ─────────────────────────────────────────────
    if show_legend:
        for level, color in level_colors.items():
            ax.scatter(
                [],
                [],
                c=color,
                s=80,
                marker="s",
                label=f"{STATE_LABELS[level]} predicted",
            )
        ax.legend(loc=(0.35, 0.2), fontsize=8, markerscale=1.2, framealpha=0.85)

    # ── Global summary bar ────────────────────────────────────────────────────
    if ax_summary is not None:
        global_counts = combined["filtered_state"].value_counts().sort_index()
        grand_total = global_counts.sum() or 1
        left = 0.0
        for level in (0, 1, 2):
            pct = float(global_counts.get(level, 0)) / grand_total
            if pct < 1e-6:
                left += pct
                continue
            ax_summary.barh(
                0,
                pct,
                left=left,
                height=0.8,
                color=level_colors[level],
                alpha=0.9,
                edgecolor="white",
                linewidth=1.5,
            )
            if pct > 0.04:
                ax_summary.text(
                    left + pct / 2,
                    0,
                    f"{pct:.1%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
            left += pct

        ax_summary.set_xlim(0, 1)
        ax_summary.set_ylim(-0.5, 0.5)
        ax_summary.axis("off")


def _build_experiment_metrics(
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
) -> pd.DataFrame:
    """Long DataFrame: one row per (subject, trial) with metrics + group."""
    rows = []
    for sid, trials in by_subject.items():
        group = groups.get(sid, "unknown")
        for tr in trials:
            m = _trial_metrics(tr)
            if m is None:
                continue
            inf = tr["inference"]
            mean_cwl = (
                float(inf["filtered_state"].dropna().mean())
                if inf is not None and "filtered_state" in inf.columns
                else np.nan
            )
            cmd = tr["commands"]
            mean_step = (
                float(cmd["cwl_current_step"].dropna().mean())
                if cmd is not None and "cwl_current_step" in cmd.columns
                else np.nan
            )
            rows.append(
                {
                    "subject_id": sid,
                    "trial": tr["name"],
                    "group": group,
                    "mean_cwl": mean_cwl,
                    "mean_step": mean_step,
                    **m,
                }
            )
    return pd.DataFrame(rows)


def _iqr_filter(vals: np.ndarray) -> np.ndarray:
    """Remove values outside 1.5×IQR fence — matches matplotlib boxplot whiskers."""
    if len(vals) < 2:
        return vals
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    return vals[(vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)]


def _add_mwu_brackets(
    ax,
    df: pd.DataFrame,
    value_col: str,
    trials: list,
    width: float = 0.35,
):
    """
    For each trial, run Mann-Whitney U on IQR-filtered adaptive vs non-adaptive values.
    Draw a significance bracket (line + stars) above the pair of boxes when p < 0.05.
    """
    from scipy.stats import mannwhitneyu

    offsets = [-width / 2, width / 2]
    brackets = []  # (x1, x2, y_whisker_top, stars)

    for t_idx, trial in enumerate(trials):
        a_raw = (
            df[(df["group"] == "adaptive") & (df["trial"] == trial)][value_col]
            .dropna()
            .values
        )
        b_raw = (
            df[(df["group"] == "non_adaptive") & (df["trial"] == trial)][value_col]
            .dropna()
            .values
        )

        if len(a_raw) < 2 or len(b_raw) < 2:
            continue

        a_filt = _iqr_filter(a_raw)
        b_filt = _iqr_filter(b_raw)

        if len(a_filt) < 2 or len(b_filt) < 2:
            continue

        _, p = mannwhitneyu(a_filt, b_filt, alternative="two-sided")
        if p >= 0.05:
            continue

        stars = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
        x1 = t_idx + 1 + offsets[0]
        x2 = t_idx + 1 + offsets[1]
        # bracket sits above the highest IQR-filtered whisker top
        y_top = float(np.nanmax(np.concatenate([a_filt, b_filt])))
        brackets.append((x1, x2, y_top, stars))

    drawn: set[str] = set()

    if not brackets:
        return drawn

    all_vals = df[value_col].dropna().values
    data_range = (
        float(np.nanmax(all_vals) - np.nanmin(all_vals)) if len(all_vals) >= 2 else 1.0
    )
    if data_range == 0:
        data_range = 1.0
    tick_h = data_range * 0.03
    margin = data_range * 0.06

    y_lo, y_hi = ax.get_ylim()
    y_ceiling = y_hi

    for x1, x2, y_top, stars in brackets:
        y_line = y_top + margin
        # stack brackets that share the same trial x-range to avoid overlap
        y_line = max(y_line, y_ceiling - (y_hi - y_lo) * 0.0)
        ax.plot(
            [x1, x1, x2, x2],
            [y_line - tick_h, y_line, y_line, y_line - tick_h],
            color="black",
            linewidth=1.0,
            clip_on=False,
        )
        ax.text(
            (x1 + x2) / 2,
            y_line + tick_h * 0.2,
            stars,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        drawn.add(stars)
        y_ceiling = max(y_ceiling, y_line + tick_h * 2.5)

    # Extend y-axis to show all brackets
    if y_ceiling > y_hi:
        ax.set_ylim(y_lo, y_ceiling)

    return drawn


def _grouped_boxplot(
    ax,
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    better_low: bool,
):
    """Per-subject boxes coloured by adaptation group, with per-trial dots.

    Subjects are ordered: adaptive → non_adaptive → unknown, alphabetical
    within group. A dashed horizontal line marks each group's mean across
    its trials and is annotated with μ. A vertical dotted line separates
    groups for readability.
    """
    if df.empty or value_col not in df.columns:
        _no_data_placeholder(ax, title)
        return

    groups_order = ["adaptive", "non_adaptive", "unknown"]
    box_data: list[np.ndarray] = []
    box_colors: list[str] = []
    box_labels: list[str] = []
    box_groups: list[str] = []

    for g in groups_order:
        sids = sorted(df[df["group"] == g]["subject_id"].unique())
        for sid in sids:
            vals = df[df["subject_id"] == sid][value_col].dropna().values
            if len(vals) == 0:
                continue
            box_data.append(vals)
            box_colors.append(GROUP_COLORS[g])
            box_labels.append(sid)
            box_groups.append(g)

    if not box_data:
        _no_data_placeholder(ax, title)
        return

    positions = np.arange(1, len(box_data) + 1)
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.6},
        flierprops={"marker": ".", "markersize": 4, "alpha": 0.5},
    )
    import matplotlib.colors as _mcolors

    for patch, c in zip(bp["boxes"], box_colors, strict=True):
        rgba = (*_mcolors.to_rgb(c), 0.55)
        patch.set_facecolor(rgba)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)

    # Group means + group separators
    from matplotlib.lines import Line2D

    last_g = None
    legend_handles = []
    for i, (pos, g) in enumerate(zip(positions, box_groups)):
        if last_g is not None and g != last_g:
            ax.axvline(pos - 0.5, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
        last_g = g

    for g in groups_order:
        gdata = df[df["group"] == g][value_col].dropna().values
        if len(gdata) == 0:
            continue
        gxs = [pos for pos, gg in zip(positions, box_groups) if gg == g]
        if not gxs:
            continue
        gmean = float(np.mean(gdata))
        x0, x1 = min(gxs) - 0.45, max(gxs) + 0.45
        ax.hlines(
            gmean,
            x0,
            x1,
            colors=GROUP_COLORS[g],
            linestyles="--",
            linewidth=1.6,
            alpha=0.95,
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=GROUP_COLORS[g],
                linewidth=1.6,
                linestyle="--",
                label=f"{GROUP_LABELS[g].split()[0]} μ={gmean:.2f}",
            )
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    indicator = "↓ better" if better_low else "↑ better"
    if legend_handles:
        legend_handles.append(
            Line2D([0], [0], color="#444", linewidth=0, marker="", label=indicator)
        )
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
            edgecolor="#ccc",
        )


def _plot_per_trial_group_boxplot(
    ax,
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    better_low: bool = True,
    debug: bool = False,
    show_datapoints: bool = False,
    show_global: bool = True,
    show_legend: bool = True,
):
    """Grouped boxplot: x = trial, two boxes per trial (adaptive vs non-adaptive).

    Shows mean (not median) as the central line. Per-subject dots overlaid.
    """
    import matplotlib.colors as _mcolors

    groups_shown = ["non_adaptive", "adaptive"]
    trials = sorted(df["trial"].unique())
    n_trials = len(trials)
    n_groups = len(groups_shown)
    width = 0.35
    offsets = [-width / 2, width / 2]

    legend_handles = []

    box_alpha = 0.28
    rng = np.random.RandomState(42)
    has_step = "mean_step" in df.columns
    _MAX_STEP = 23.0

    for g_idx, g in enumerate(groups_shown):
        color = GROUP_COLORS[g]
        rgba = (*_mcolors.to_rgb(color), box_alpha)
        gdf = df[df["group"] == g]
        box_data = []
        positions = []

        for t_idx, trial in enumerate(trials):
            vals = gdf[gdf["trial"] == trial][value_col].dropna().values
            box_data.append(vals if len(vals) > 0 else np.array([np.nan]))
            positions.append(t_idx + 1 + offsets[g_idx])

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            showfliers=not show_datapoints,  # hide fliers when dots replace them
            medianprops={"linewidth": 0, "color": (0, 0, 0, 0)},
            meanprops={"color": color, "linewidth": 2.0, "linestyle": "-"},
            whiskerprops={"color": color, "linewidth": 1.2},
            capprops={"color": color, "linewidth": 1.2},
            boxprops={"linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 4,
                "alpha": 0.7,
                "markerfacecolor": color,
                "markeredgecolor": color,
            },
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(rgba)
            patch.set_edgecolor(color)

        if show_datapoints:
            from matplotlib.colors import LinearSegmentedColormap

            # adaptive only: light→dark blue gradient mapped to step value
            _adaptive_cmap = LinearSegmentedColormap.from_list(
                "adaptive_steps", ["#C6DBEF", "#08519C"], N=256
            )
            is_non_adaptive = g == "non_adaptive"
            for t_idx, trial in enumerate(trials):
                pos = positions[t_idx]
                extra_cols = ["mean_step"] if has_step else []
                trial_rows = gdf[gdf["trial"] == trial][
                    [value_col] + extra_cols
                ].dropna(subset=[value_col])
                for _, row in trial_rows.iterrows():
                    if is_non_adaptive:
                        dot_color = GROUP_COLORS["non_adaptive"]
                        dot_s = 10 + 70 * 0.5  # fixed at mid-step (~12)
                    else:
                        raw_step = row.get("mean_step", np.nan) if has_step else np.nan
                        step = (
                            float(np.clip(raw_step, 0, _MAX_STEP))
                            if not np.isnan(raw_step)
                            else _MAX_STEP / 2
                        )
                        dot_color = _adaptive_cmap(step / _MAX_STEP)
                        dot_s = 10 + 70 * (step / _MAX_STEP)
                    jitter = (rng.rand() - 0.5) * width * 0.55
                    ax.scatter(
                        pos + jitter,
                        row[value_col],
                        s=dot_s,
                        color=dot_color,
                        alpha=0.92,
                        edgecolors=color,
                        linewidths=1.0,
                        zorder=5,
                    )

        mean_vals = [np.nanmean(d) for d in box_data]
        ax.plot(
            positions,
            mean_vals,
            color=color,
            linewidth=1.2,
            linestyle=":",
            alpha=0.8,
            zorder=2,
        )

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_handles.append(
            Patch(facecolor=rgba, edgecolor=color, label=GROUP_LABELS[g].split()[0])
        )

    ax.set_xticks(np.arange(1, n_trials + 1))
    ax.set_xticklabels([t.replace("trial_", "T") for t in trials], fontsize=13)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # debug: show raw p-value for every trial below the x-axis tick labels
    if debug:
        from scipy.stats import mannwhitneyu

        for t_idx, trial in enumerate(trials):
            a_raw = (
                df[(df["group"] == "adaptive") & (df["trial"] == trial)][value_col]
                .dropna()
                .values
            )
            b_raw = (
                df[(df["group"] == "non_adaptive") & (df["trial"] == trial)][value_col]
                .dropna()
                .values
            )
            if len(a_raw) < 2 or len(b_raw) < 2:
                label = "p=n/a"
            else:
                a_filt = _iqr_filter(a_raw)
                b_filt = _iqr_filter(b_raw)
                if len(a_filt) < 2 or len(b_filt) < 2:
                    label = "p=n/a"
                else:
                    _, p = mannwhitneyu(a_filt, b_filt, alternative="two-sided")
                    label = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
                    print(
                        f"  [debug] {trial}: {label}  (n_a={len(a_filt)}, n_b={len(b_filt)})"
                    )
            # get_xaxis_transform: x = data coords, y = axes fraction (0=bottom, 1=top)
            # use a small negative y fraction to place text just below the tick labels
            ax.text(
                t_idx + 1,
                -0.18,
                label,
                ha="center",
                va="top",
                fontsize=6.5,
                color="#aaaaaa",
                style="italic",
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

    indicator = "↓ better" if better_low else "↑ better"
    legend_handles.append(
        Line2D([0], [0], color="#444", linewidth=0, marker="", label=indicator)
    )
    if show_legend:
        ax.legend(
            handles=legend_handles,
            fontsize=8,
            framealpha=0.9,
            loc="upper center",
        )

    # significance brackets (must come after ax limits are set by boxplot)
    drawn_stars = _add_mwu_brackets(ax, df, value_col, trials, width=width)

    # ── Global "All" column ───────────────────────────────────────────────────
    if show_global:
        from scipy.stats import mannwhitneyu

        all_pos = n_trials + 1

        # Grey background behind the "All" column
        ax.axvspan(
            all_pos - 0.5,
            all_pos + 0.5,
            color="#888888",
            alpha=0.18,
            zorder=0,
        )

        for g_idx, g in enumerate(groups_shown):
            color = GROUP_COLORS[g]
            rgba = (*_mcolors.to_rgb(color), box_alpha)
            gdf = df[df["group"] == g]
            all_vals = gdf[value_col].dropna().values
            if len(all_vals) == 0:
                continue
            pos = all_pos + offsets[g_idx]
            bp_all = ax.boxplot(
                [all_vals],
                positions=[pos],
                widths=width * 0.85,
                patch_artist=True,
                showmeans=True,
                meanline=True,
                showfliers=True,
                medianprops={"linewidth": 0, "color": (0, 0, 0, 0)},
                meanprops={"color": color, "linewidth": 2.0, "linestyle": "-"},
                whiskerprops={"color": color, "linewidth": 1.2},
                capprops={"color": color, "linewidth": 1.2},
                boxprops={"linewidth": 1.2},
                flierprops={
                    "marker": "o",
                    "markersize": 4,
                    "alpha": 0.7,
                    "markerfacecolor": color,
                    "markeredgecolor": color,
                },
            )
            for patch in bp_all["boxes"]:
                patch.set_facecolor(rgba)
                patch.set_edgecolor(color)

        # MWU bracket for the "All" pair
        a_all = _iqr_filter(df[df["group"] == "adaptive"][value_col].dropna().values)
        b_all = _iqr_filter(
            df[df["group"] == "non_adaptive"][value_col].dropna().values
        )
        if len(a_all) >= 2 and len(b_all) >= 2:
            _, p_all = mannwhitneyu(a_all, b_all, alternative="two-sided")
            if p_all < 0.05:
                stars = "***" if p_all < 0.001 else ("**" if p_all < 0.01 else "*")
                x1 = all_pos + offsets[0]
                x2 = all_pos + offsets[1]
                all_data = df[value_col].dropna().values
                dr = (
                    float(np.nanmax(all_data) - np.nanmin(all_data))
                    if len(all_data) >= 2
                    else 1.0
                )
                if dr == 0:
                    dr = 1.0
                tick_h = dr * 0.03
                margin = dr * 0.06
                y_top = float(np.nanmax(np.concatenate([a_all, b_all])))
                y_line = y_top + margin
                y_lo, y_hi = ax.get_ylim()
                ax.plot(
                    [x1, x1, x2, x2],
                    [y_line - tick_h, y_line, y_line, y_line - tick_h],
                    color="black",
                    linewidth=1.0,
                    clip_on=False,
                )
                ax.text(
                    (x1 + x2) / 2,
                    y_line + tick_h * 0.2,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
                if y_line + tick_h * 2.5 > y_hi:
                    ax.set_ylim(y_lo, y_line + tick_h * 2.5)
                drawn_stars.add(stars)

        # Overwrite ticks to include "All"
        ax.set_xticks(list(np.arange(1, n_trials + 1)) + [all_pos])
        ax.set_xticklabels(
            [t.replace("trial_", "T") for t in trials] + ["All"],
            fontsize=9,
        )
        ax.set_xlim(0.5, all_pos + 0.5)

    return drawn_stars


def _plot_experiment_summary_boxplots(
    axes, df: pd.DataFrame, apply_penalty: bool = True
):
    """4-panel grid: completion, min alive, missed, penalty (or raw completion)."""
    last_panel = (
        (
            "penalty_s",
            "Penalty score (s-equivalent)",
            f"Composite Penalty\n(time + {RACING_DEAD_PENALTY_S:.0f}s/dead "
            f"+ {RACING_MISS_PENALTY_S:.0f}s/miss)",
            True,
        )
        if apply_penalty
        else (
            "completion_s",
            "Completion time (s)",
            "Trial Completion Time (raw)",
            True,
        )
    )
    panels = [
        ("completion_s", "Completion time (s)", "Trial Completion Time", True),
        (
            "min_alive",
            f"Min drones alive (/{SWARM_SIZE})",
            "Min Drones Alive per Trial",
            False,
        ),
        (
            "missed_drones",
            "Σ drones outside gates",
            "Drone-Gate Misses per Trial",
            True,
        ),
        last_panel,
    ]
    flat = np.asarray(axes).ravel()
    for ax, (col, ylabel, title, low_better) in zip(flat, panels):
        _grouped_boxplot(ax, df, col, ylabel, title, better_low=low_better)


def _plot_experiment_cwl_distribution(
    ax,
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
):
    """Stacked horizontal bars: % time in Low/Med/High CWL per subject."""
    rows = []
    for sid, trials in by_subject.items():
        all_states: list[int] = []
        for tr in trials:
            inf = tr["inference"]
            if inf is None or "filtered_state" not in inf.columns:
                continue
            all_states.extend(inf["filtered_state"].dropna().astype(int).tolist())
        if not all_states:
            continue
        s = np.array(all_states)
        total = len(s)
        rows.append(
            {
                "subject_id": sid,
                "group": groups.get(sid, "unknown"),
                "low_pct": float((s == 0).sum()) / total,
                "med_pct": float((s == 1).sum()) / total,
                "high_pct": float((s == 2).sum()) / total,
            }
        )

    if not rows:
        _no_data_placeholder(ax, "CWL Distribution")
        return

    df = pd.DataFrame(rows)
    order = {"adaptive": 0, "non_adaptive": 1, "unknown": 2}
    df["g_ord"] = df["group"].map(order).fillna(3)
    df = df.sort_values(["g_ord", "subject_id"]).reset_index(drop=True)

    y = np.arange(len(df))
    ax.barh(
        y,
        df["low_pct"],
        color=STATE_COLORS[0],
        label=STATE_LABELS[0],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.barh(
        y,
        df["med_pct"],
        left=df["low_pct"],
        color=STATE_COLORS[1],
        label=STATE_LABELS[1],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.barh(
        y,
        df["high_pct"],
        left=df["low_pct"] + df["med_pct"],
        color=STATE_COLORS[2],
        label=STATE_LABELS[2],
        edgecolor="white",
        linewidth=0.6,
    )

    labels = [
        f"{r.subject_id} [{GROUP_LABELS[r.group].split()[0]}]" for r in df.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Time fraction in CWL state")
    ax.set_title("CWL Distribution per Subject", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=3)


def _plot_experiment_adaptation_steps(
    ax,
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
):
    """Box plot of cwl_current_step per adaptive subject + grand mean line.

    Skipped for non_adaptive subjects since their step is constant by
    construction. The grand mean across all adaptive trials is a useful
    reference for picking constant-step values for new control subjects.
    """
    rows = []
    for sid, trials in by_subject.items():
        if groups.get(sid) != "adaptive":
            continue
        steps: list[float] = []
        for tr in trials:
            commands = tr["commands"]
            if commands is None or "cwl_current_step" not in commands.columns:
                continue
            steps.extend(commands["cwl_current_step"].dropna().tolist())
        if steps:
            rows.append({"subject_id": sid, "steps": np.array(steps)})

    if not rows:
        _no_data_placeholder(ax, "Adaptation Step Distribution")
        return

    rows.sort(key=lambda r: r["subject_id"])
    box_data = [r["steps"] for r in rows]
    positions = np.arange(1, len(rows) + 1)

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.6},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
    )
    import matplotlib.colors as _mcolors

    color = GROUP_COLORS["adaptive"]
    rgba = (*_mcolors.to_rgb(color), 0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor(rgba)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    grand_mean = float(np.concatenate(box_data).mean())
    ax.axhline(
        grand_mean,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label=f"Adaptive grand μ = {grand_mean:.1f}",
    )

    # Indicate the experiment's max step from the first adaptive trial we find
    max_step = None
    for sid, trials in by_subject.items():
        if groups.get(sid) != "adaptive":
            continue
        for tr in trials:
            commands = tr["commands"]
            if commands is None or "cwl_total_steps" not in commands.columns:
                continue
            tot = commands["cwl_total_steps"].dropna()
            if not tot.empty:
                max_step = int(tot.iloc[0]) - 1
                break
        if max_step is not None:
            break
    if max_step is not None:
        ax.axhline(
            max_step,
            color="#888",
            linewidth=0.8,
            linestyle=":",
            label=f"Max step ({max_step})",
        )
        ax.axhline(
            0,
            color="#888",
            linewidth=0.8,
            linestyle=":",
            label="Min step (0)",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [r["subject_id"] for r in rows],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("CWL current step")
    ax.set_title(
        "Adaptation Step Distribution per Subject (adaptive group)",
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)


def _print_experiment_group_summary(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("GROUP SUMMARY — mean ± std across all trials")
    print("=" * 70)
    cols = [
        ("completion_s", "Completion time (s)", "{:>7.1f}"),
        ("min_alive", f"Min alive (/{SWARM_SIZE})", "{:>7.2f}"),
        ("missed_drones", "Drone-gate misses", "{:>7.2f}"),
        ("dead_drones", f"Dead (/{SWARM_SIZE})", "{:>7.2f}"),
        ("penalty_s", "Penalty (s)", "{:>7.1f}"),
    ]
    for group in ["adaptive", "non_adaptive", "unknown"]:
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        n_subj = sub["subject_id"].nunique()
        print(f"\n  {GROUP_LABELS[group]}  (n={len(sub)} trials, {n_subj} subject(s))")
        for col, label, fmt in cols:
            mu = sub[col].mean()
            sd = sub[col].std()
            print(f"    {label:<22}  {fmt.format(mu)}  ± {fmt.format(sd).strip()}")


def _build_experiment_metrics_enhanced(
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
    p_crash: float = 1.0,
    w_dead: float = 1.0,
    p_outside: float = 1.0,
    p_skip: float = 2.0,
    t_max: float | None = None,
) -> pd.DataFrame:
    """Long DataFrame: one row per (subject, trial) with enhanced metrics + group."""
    if t_max is None:
        all_times = []
        for trials in by_subject.values():
            for tr in trials:
                m = _trial_metrics_enhanced(tr)
                if m is not None:
                    all_times.append(m["completion_s"])
        t_max = 2 * max(all_times) if all_times else 600.0

    rows = []
    for sid, trials in by_subject.items():
        group = groups.get(sid, "unknown")
        for i, tr in enumerate(trials):
            m = _trial_metrics_enhanced(tr, p_crash, w_dead, p_outside, p_skip, t_max)
            if m is None:
                continue
            rows.append(
                {
                    "subject_id": sid,
                    "trial_num": i + 1,
                    "trial": tr["name"],
                    "group": group,
                    **m,
                }
            )
    return pd.DataFrame(rows)


def _plot_raw_distributions(axes, df: pd.DataFrame):
    """Plot KDE + violin distributions for all metric terms, grouped by control/adaptive."""
    from scipy import stats as sp_stats

    groups = df["group"].unique()
    terms = [
        ("completion_s", "Completion Time (s)"),
        ("n_crashes", "# Crashes"),
        ("sum_dead_at_gates", "Σ Dead Drones @ Gates"),
        ("sum_outside_at_gates", "Σ Drones Outside Gates"),
        ("n_completely_missed", "# Completely Missed Gates"),
    ]

    for ax, (col, title) in zip(axes, terms):
        for group in sorted(groups):
            sub = df[df["group"] == group][col].dropna()
            if sub.empty:
                continue
            # KDE
            try:
                density = sp_stats.gaussian_kde(sub)
                x = np.linspace(sub.min(), sub.max(), 200)
                ax.fill_between(
                    x,
                    density(x),
                    alpha=0.35,
                    color=GROUP_COLORS.get(group, "#999999"),
                    label=GROUP_LABELS.get(group, group),
                )
            except:
                pass
            # Rug plot
            ax.scatter(
                sub,
                np.zeros(len(sub)),
                s=20,
                alpha=0.5,
                color=GROUP_COLORS.get(group, "#999999"),
            )
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)


def _plot_correlation_scatter(axes, df: pd.DataFrame):
    """Plot scatter grids for metric correlations, colored by group."""
    terms = [
        "completion_s",
        "n_crashes",
        "sum_dead_at_gates",
        "sum_outside_at_gates",
    ]
    n_terms = len(terms)

    # Flatten axes grid
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else axes

    ax_idx = 0
    for i, term1 in enumerate(terms):
        for j, term2 in enumerate(terms):
            if i >= j:
                continue
            if ax_idx >= len(axes_flat):
                break

            ax = axes_flat[ax_idx]
            for group in sorted(df["group"].unique()):
                sub = df[df["group"] == group]
                ax.scatter(
                    sub[term1],
                    sub[term2],
                    s=50,
                    alpha=0.6,
                    color=GROUP_COLORS.get(group, "#999999"),
                    label=GROUP_LABELS.get(group, group),
                )
            ax.set_xlabel(term1.replace("_", "\n"))
            ax.set_ylabel(term2.replace("_", "\n"))
            ax.grid(alpha=0.3)
            if ax_idx == 0:
                ax.legend(fontsize=8)
            ax_idx += 1

    for ax in axes_flat[ax_idx:]:
        ax.axis("off")


def _plot_penalized_time_distribution(ax, df: pd.DataFrame):
    """KDE plot of penalized time, grouped by control/adaptive."""
    from scipy import stats as sp_stats

    for group in sorted(df["group"].unique()):
        sub = df[df["group"] == group]["penalized_s"].dropna()
        if sub.empty:
            continue
        try:
            density = sp_stats.gaussian_kde(sub)
            x = np.linspace(df["penalized_s"].min(), df["penalized_s"].max(), 200)
            ax.fill_between(
                x,
                density(x),
                alpha=0.35,
                color=GROUP_COLORS.get(group, "#999999"),
                label=GROUP_LABELS.get(group, group),
            )
        except:
            pass

    ax.set_xlabel("Penalized Time (s)")
    ax.set_ylabel("Density")
    ax.set_title("Penalized Time Distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def _plot_learning_curve(ax, df: pd.DataFrame):
    """Mean penalized time per trial number, with CI bands."""
    from scipy import stats as sp_stats

    trial_nums = sorted(df["trial_num"].unique())
    for group in sorted(df["group"].unique()):
        sub = df[df["group"] == group]
        means = []
        cis_lo = []
        cis_hi = []

        for tn in trial_nums:
            vals = sub[sub["trial_num"] == tn]["penalized_s"].dropna()
            if vals.empty:
                continue
            m = vals.mean()
            se = vals.sem()
            ci = se * sp_stats.t.ppf(0.975, len(vals) - 1) if len(vals) > 1 else 0
            means.append(m)
            cis_lo.append(m - ci)
            cis_hi.append(m + ci)

        trial_nums_present = [
            tn for tn in trial_nums if len(sub[sub["trial_num"] == tn]) > 0
        ]
        ax.plot(
            trial_nums_present[: len(means)],
            means,
            marker="o",
            linewidth=2,
            color=GROUP_COLORS.get(group, "#999999"),
            label=GROUP_LABELS.get(group, group),
        )
        ax.fill_between(
            trial_nums_present[: len(means)],
            cis_lo,
            cis_hi,
            alpha=0.15,
            color=GROUP_COLORS.get(group, "#999999"),
        )

    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Mean Penalized Time (s)")
    ax.set_title("Learning Curve")
    ax.set_xticks(trial_nums)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def _plot_gate_level_heatmap(ax, by_subject: dict, groups: dict):
    """Heatmap of pass_count / alive_at_gate per gate, grouped by control/adaptive."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    gate_data = {}
    for group in ["adaptive", "non_adaptive"]:
        gate_stats = []
        for sid, trials in by_subject.items():
            if groups.get(sid) != group:
                continue
            for tr in trials:
                gs = tr["gate_status"]
                drones = tr["drones"]
                if gs is None:
                    continue
                for _, row in gs.iterrows():
                    gid = int(row["id"])
                    ts = int(row.get("first_pass_timestamp", 0))
                    pc = int(row.get("pass_count", 0))
                    if ts > 0:
                        alive = _alive_at_timestamp(drones, ts)
                        if alive > 0:
                            if gid not in gate_data:
                                gate_data[gid] = {"adaptive": [], "non_adaptive": []}
                            gate_data[gid][group].append(pc / alive)

    gate_ids = sorted(gate_data.keys())
    data_matrix = []
    for gid in gate_ids:
        row = [
            np.mean(gate_data[gid]["adaptive"]) if gate_data[gid]["adaptive"] else 0,
            np.mean(gate_data[gid]["non_adaptive"])
            if gate_data[gid]["non_adaptive"]
            else 0,
        ]
        data_matrix.append(row)

    if not data_matrix:
        ax.text(0.5, 0.5, "No gate data", ha="center", va="center")
        return

    data_array = np.array(data_matrix)
    im = ax.imshow(
        data_array.T,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        origin="upper",
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Adaptive", "Non-Adaptive"], fontsize=9)
    ax.set_xlabel("Gate ID")
    ax.set_title("Pass Rate per Gate (pass_count / alive)")
    plt.colorbar(im, ax=ax, label="Pass Rate")


def _plot_experiment_segment_cwl_limits(
    axes, by_subject: dict[str, list[dict]], subjects: list[str]
):
    """N-column grid (one per trial) for a given list of subjects.

    Per subject: two horizontal image bars — CWL estimate (top, Okabe-Ito) and
    flight-step limit (bottom, Viridis) — with a visible gap between them.
    Uses ax.imshow with exact data-coordinate extents so bar height is always
    correct regardless of figure size.
    """
    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    viridis = get_cmap("viridis")
    n_subjects = len(subjects)

    bar_h = 0.8  # height of each bar in data units
    bar_gap = 0.35  # gap between step bar (bottom) and CWL bar (top)
    subj_margin = 0.3  # blank space above/below each subject block
    subj_pitch = 2 * bar_h + bar_gap + 2 * subj_margin

    _N_RESAMP = 256  # horizontal pixel resolution per segment

    # CWL bar sits above, step bar below, with bar_gap between them
    def cwl_yrange(si):  # (bottom, top)
        return (
            si * subj_pitch + subj_margin + bar_gap + bar_h,
            si * subj_pitch + subj_margin + bar_gap + 2 * bar_h,
        )

    def step_yrange(si):  # (bottom, top)
        return (si * subj_pitch + subj_margin, si * subj_pitch + subj_margin + bar_h)

    ref_gates = next(
        (
            tr["gates"]
            for sid in subjects
            for tr in by_subject.get(sid, [])
            if tr["gates"] is not None
        ),
        None,
    )
    if ref_gates is None:
        for ax in axes:
            _no_data_placeholder(ax, "No gate data")
        return

    a_ids_ref, b_ids_ref, seg_names, seg_diffs = _segment_metadata(ref_gates)
    n_segs = len(a_ids_ref)
    if n_segs == 0:
        for ax in axes:
            _no_data_placeholder(ax, "No segments")
        return

    xgrid = np.linspace(0, 1, _N_RESAMP)

    for col_idx, ax in enumerate(axes):
        for s_idx, sid in enumerate(subjects):
            trials = by_subject.get(sid, [])
            if col_idx >= len(trials):
                continue
            tr = trials[col_idx]

            inf = tr["inference"]
            commands = tr["commands"]
            gs = tr["gate_status"]
            gates = tr["gates"]
            if gs is None or gates is None:
                continue

            gs_idx = gs.drop_duplicates(subset=["id"]).set_index("id")
            a_ids_tr, b_ids_tr, _, _ = _segment_metadata(gates)

            for seg_idx, (ga, gb) in enumerate(zip(a_ids_tr, b_ids_tr)):
                if seg_idx >= n_segs:
                    break
                try:
                    t_start = float(gs_idx.loc[ga, "first_pass_timestamp"])
                    t_end = float(gs_idx.loc[gb, "first_pass_timestamp"])
                except (KeyError, TypeError):
                    continue
                if t_start <= 0 or t_end <= 0 or t_end <= t_start:
                    continue

                x0, x1 = float(seg_idx), float(seg_idx + 1)

                # ── CWL estimate bar ─────────────────────────────────────────
                if inf is not None and "filtered_state" in inf.columns:
                    mask = (inf["timestamp"] >= t_start) & (inf["timestamp"] <= t_end)
                    seg_inf = inf.loc[mask].sort_values("timestamp")
                    if len(seg_inf) >= 2:
                        t_lo = seg_inf["timestamp"].min()
                        t_hi = seg_inf["timestamp"].max()
                        xn = (seg_inf["timestamp"].values - t_lo) / (t_hi - t_lo)
                        cwl = seg_inf["filtered_state"].fillna(0).values.astype(int)
                        idxs = np.clip(np.searchsorted(xn, xgrid), 0, len(cwl) - 1)
                        rgba = np.array(
                            [
                                to_rgba(STATE_COLORS.get(int(cwl[i]), "#aaa"))
                                for i in idxs
                            ],
                            dtype=float,
                        )[np.newaxis, :, :]
                        yb, yt = cwl_yrange(s_idx)
                        ax.imshow(
                            rgba,
                            extent=[x0, x1, yb, yt],
                            aspect="auto",
                            interpolation="nearest",
                            zorder=3,
                        )

                # ── Flight-step limit bar ────────────────────────────────────
                if commands is not None and "cwl_current_step" in commands.columns:
                    mask = (commands["timestamp"] >= t_start) & (
                        commands["timestamp"] <= t_end
                    )
                    seg_cmd = commands.loc[mask].sort_values("timestamp")
                    if len(seg_cmd) >= 2:
                        t_lo = seg_cmd["timestamp"].min()
                        t_hi = seg_cmd["timestamp"].max()
                        xn = (seg_cmd["timestamp"].values - t_lo) / (t_hi - t_lo)
                        steps = (
                            seg_cmd["cwl_current_step"].fillna(0).values.astype(float)
                        )
                        idxs = np.clip(np.searchsorted(xn, xgrid), 0, len(steps) - 1)
                        rgba = np.array(
                            [viridis(min(steps[i] / 23.0, 1.0)) for i in idxs],
                            dtype=float,
                        )[np.newaxis, :, :]
                        yb, yt = step_yrange(s_idx)
                        ax.imshow(
                            rgba,
                            extent=[x0, x1, yb, yt],
                            aspect="auto",
                            interpolation="nearest",
                            zorder=3,
                        )

        # ── Axis decoration ──────────────────────────────────────────────────
        total_h = n_subjects * subj_pitch
        ax.set_xlim(-0.08, n_segs + 0.08)
        ax.set_ylim(0, total_h)

        # Segment backgrounds (easy/hard)
        for i, is_hard in enumerate(seg_diffs):
            ax.axvspan(
                i,
                i + 1,
                color="#FFEBEE" if is_hard else "#E8F5E9",
                alpha=0.35,
                zorder=0,
            )

        # Dotted vertical segment separators
        for i in range(n_segs + 1):
            ax.axvline(i, color="#666", linewidth=0.9, linestyle=":", zorder=4)

        ax.set_xticks([i + 0.5 for i in range(n_segs)])
        ax.set_xticklabels(seg_names, fontsize=9, rotation=30, ha="right")

        # Thin dashed line in the gap between the two bars of each subject
        for s_idx in range(n_subjects):
            gap_mid = (step_yrange(s_idx)[1] + cwl_yrange(s_idx)[0]) / 2
            ax.hlines(
                gap_mid,
                -0.05,
                n_segs + 0.05,
                colors="#aaa",
                linewidths=0.6,
                linestyles="--",
                zorder=5,
            )

        # Subject separator bands (between subjects)
        for s_idx in range(1, n_subjects):
            sep_y = s_idx * subj_pitch
            ax.axhspan(
                sep_y - subj_margin,
                sep_y + subj_margin,
                color="#f0f0f0",
                alpha=0.8,
                zorder=1,
            )
            ax.axhline(sep_y, color="#bbb", linewidth=1.0, linestyle="-", zorder=2)

        # Trial title
        first_trials = by_subject.get(subjects[0], [])
        trial_name = (
            first_trials[col_idx]["name"]
            if col_idx < len(first_trials)
            else f"Trial {col_idx + 1}"
        )
        ax.set_title(
            trial_name.replace("trial_", "Trial "), fontsize=11, fontweight="bold"
        )

        # Y-tick labels: centred on the CWL bar, first column only
        y_ticks = [
            (cwl_yrange(s_idx)[0] + cwl_yrange(s_idx)[1]) / 2
            for s_idx in range(n_subjects)
        ]
        if col_idx == 0:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(subjects, fontsize=10)
        else:
            ax.set_yticks([])

    # ── Shared legend and colorbar on last axis ──────────────────────────────
    handles = [
        Line2D([0], [0], color=STATE_COLORS[0], linewidth=5, label="CWL Low"),
        Line2D([0], [0], color=STATE_COLORS[1], linewidth=5, label="CWL Med"),
        Line2D([0], [0], color=STATE_COLORS[2], linewidth=5, label="CWL High"),
    ]
    sm = ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=0, vmax=23))
    sm.set_array([])
    cbar = plt.colorbar(
        sm,
        ax=axes[-1],
        orientation="vertical",
        pad=0.04,
        aspect=25,
        fraction=0.06,
        shrink=0.6,
    )
    cbar.set_label("Flight step (limit)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    axes[0].legend(
        handles=handles,
        loc="upper left",
        fontsize=9,
        ncol=1,
        framealpha=0.85,
        title="CWL estimate",
        title_fontsize=9,
    )


_CONTROL_COLS = ["pitch_rate", "roll_rate", "yaw_rate", "altitude_rate"]
_CONTROL_LABELS = {
    "pitch_rate": "Cmd. velocity Z (forward)",
    "roll_rate": "Cmd. velocity X (lateral)",
    "yaw_rate": "Cmd. yaw rate",
    "altitude_rate": "Cmd. altitude rate",
}
_N_PER_SEG = 120  # resampling resolution per segment
_N_RESAMP_PROFILE = 40  # per-segment resampling for adaptive profile summary
# Colorblind-safe sequential palettes to compare for the adaptive profile figure.
# All are perceptually uniform or single-hue; none rely on red-green contrast.
_PROFILE_STEP_CMAPS = ["viridis", "plasma", "cividis", "magma", "Blues", "YlOrBr"]


def _plot_experiment_control_inputs(
    axes,
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
):
    """One subplot per control axis: average normalised input per segment.

    For each (subject, trial, segment) the signal is time-normalised within
    the segment and resampled to _N_PER_SEG points.  Mean ± 1 std are plotted
    separately for adaptive and non-adaptive groups.  Segment boundaries are
    dotted vertical lines; easy/hard bands shade the background.
    """
    # Collect reference segment metadata from first available trial
    ref_gates = next(
        (
            tr["gates"]
            for trials in by_subject.values()
            for tr in trials
            if tr["gates"] is not None
        ),
        None,
    )
    if ref_gates is None:
        for ax in axes:
            _no_data_placeholder(ax, "No gate data")
        return

    a_ids_ref, b_ids_ref, seg_names, seg_diffs = _segment_metadata(ref_gates)
    n_segs = len(a_ids_ref)
    if n_segs == 0:
        for ax in axes:
            _no_data_placeholder(ax, "No segments")
        return

    n_pts = n_segs * _N_PER_SEG
    x_full = np.linspace(0, n_segs, n_pts)
    x_grid = np.linspace(0, 1, _N_PER_SEG)  # local grid within one segment

    groups_shown = ["adaptive", "non_adaptive"]

    # Build traces dict: {group: {col: list of 1-D arrays of length n_pts}}
    traces: dict[str, dict[str, list]] = {
        g: {c: [] for c in _CONTROL_COLS} for g in groups_shown
    }

    for sid, sid_trials in by_subject.items():
        g = groups.get(sid, "unknown")
        if g not in groups_shown:
            continue
        for tr in sid_trials:
            commands = tr["commands"]
            gs = tr["gate_status"]
            gates = tr["gates"]
            if commands is None or gs is None or gates is None:
                continue

            gs_idx = gs.drop_duplicates(subset=["id"]).set_index("id")
            a_ids_tr, b_ids_tr, _, _ = _segment_metadata(gates)

            for col in _CONTROL_COLS:
                if col not in commands.columns:
                    continue
                trace = np.full(n_pts, np.nan)

                for seg_idx, (ga, gb) in enumerate(zip(a_ids_tr, b_ids_tr)):
                    if seg_idx >= n_segs:
                        break
                    try:
                        t_start = float(gs_idx.loc[ga, "first_pass_timestamp"])
                        t_end = float(gs_idx.loc[gb, "first_pass_timestamp"])
                    except (KeyError, TypeError):
                        continue
                    if t_start <= 0 or t_end <= 0 or t_end <= t_start:
                        continue

                    mask = (commands["timestamp"] >= t_start) & (
                        commands["timestamp"] <= t_end
                    )
                    seg_cmd = commands.loc[mask].sort_values("timestamp")
                    if len(seg_cmd) < 2:
                        continue

                    t_lo, t_hi = seg_cmd["timestamp"].min(), seg_cmd["timestamp"].max()
                    x_raw = (seg_cmd["timestamp"].values - t_lo) / (t_hi - t_lo)
                    vals = seg_cmd[col].values.astype(float)

                    resampled = np.interp(x_grid, x_raw, vals)
                    start = seg_idx * _N_PER_SEG
                    trace[start : start + _N_PER_SEG] = resampled

                traces[g][col].append(trace)

    # Plot
    for ax, col in zip(axes, _CONTROL_COLS):
        for g in groups_shown:
            arr = np.array(traces[g][col])  # shape (n_traces, n_pts)
            if arr.size == 0:
                continue
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            color = GROUP_COLORS[g]
            label = GROUP_LABELS[g].split()[0]

            ax.plot(x_full, mean, color=color, linewidth=1.6, label=label)
            ax.fill_between(x_full, mean - std, mean + std, color=color, alpha=0.15)

        # Segment backgrounds and dotted boundaries
        for i, is_hard in enumerate(seg_diffs):
            ax.axvspan(
                i, i + 1, color="#FFEBEE" if is_hard else "#E8F5E9", alpha=0.3, zorder=0
            )
        for i in range(n_segs + 1):
            ax.axvline(i, color="#666", linewidth=0.8, linestyle=":", zorder=2)

        ax.axhline(0, color="#aaa", linewidth=0.6, linestyle="--", zorder=1)
        ax.set_xticks([i + 0.5 for i in range(n_segs)])
        ax.set_xticklabels(seg_names, fontsize=10)
        ax.set_xlim(0, n_segs)
        ax.set_ylabel(_CONTROL_LABELS[col], fontsize=11)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.legend(fontsize=10, loc="upper right")

    axes[0].set_title("Mean ± 1 std across subjects & trials", fontsize=11)


_CRASH_WINDOW_PRE_S = 30.0  # seconds before crash gate to include in event window
_CRASH_WINDOW_POST_S = 5.0  # seconds after  crash gate
_CRASH_PROFILE_BINS = 70  # time bins in event-triggered profiles
_MOCK_GATE_AMP = 0.38  # lateral S-amplitude for hard segments (schematic)


def _get_sorted_gate_ids(ref_gates: pd.DataFrame) -> list[int]:
    """All gate IDs in course order (ascending center_z)."""
    return list(ref_gates.sort_values("center_z")["id"].astype(int))


def _crash_density_per_gate(tr: dict, sorted_gate_ids: list[int]) -> np.ndarray:
    """Crash count approaching each gate, indexed by position in sorted_gate_ids."""
    gs = tr["gate_status"]
    drones = tr["drones"]
    if gs is None or drones is None or drones.empty:
        return np.zeros(len(sorted_gate_ids))
    gs_idx = gs.drop_duplicates(subset=["id"]).set_index("id")
    counts = np.zeros(len(sorted_gate_ids), dtype=float)
    prev_alive = int(SWARM_SIZE)
    for i, gid in enumerate(sorted_gate_ids):
        if gid not in gs_idx.index:
            continue
        t_gate = float(gs_idx.loc[gid, "first_pass_timestamp"])
        if t_gate <= 0:
            continue
        cur_alive = _alive_at_timestamp(drones, t_gate)
        deaths = max(0, prev_alive - cur_alive)
        counts[i] = deaths
        prev_alive = cur_alive
    return counts


def _outside_density_per_gate(tr: dict, sorted_gate_ids: list[int]) -> np.ndarray:
    """Alive-but-outside count at each gate, indexed by position in sorted_gate_ids."""
    gs = tr["gate_status"]
    gates = tr["gates"]
    drones = tr["drones"]
    if gs is None or gates is None:
        return np.zeros(len(sorted_gate_ids))
    breakdown = _compute_gate_breakdown(gates, gs, drones)
    counts = np.zeros(len(sorted_gate_ids), dtype=float)
    for i, gid in enumerate(sorted_gate_ids):
        b = breakdown.get(gid, {})
        if b.get("reached"):
            counts[i] = float(b.get("outside", 0))
    return counts


def _collect_crash_gate_events(tr: dict, sorted_gate_ids: list[int]) -> list[dict]:
    """Return [{t_gate, n_deaths}] for each gate where at least one drone died."""
    gs = tr["gate_status"]
    drones = tr["drones"]
    if gs is None or drones is None or drones.empty:
        return []
    gs_idx = gs.drop_duplicates(subset=["id"]).set_index("id")
    events = []
    prev_alive = int(SWARM_SIZE)
    for gid in sorted_gate_ids:
        if gid not in gs_idx.index:
            continue
        t_gate = float(gs_idx.loc[gid, "first_pass_timestamp"])
        if t_gate <= 0:
            continue
        cur_alive = _alive_at_timestamp(drones, t_gate)
        deaths = max(0, prev_alive - cur_alive)
        if deaths > 0:
            events.append({"t_gate": t_gate, "n_deaths": deaths})
        prev_alive = cur_alive
    return events


def _build_mock_course_ax(
    ax,
    seg_names: list[str],
    seg_diffs: list[bool],
    n_gates: int = 8,
):
    """Schematic course: straight easy segments, symmetric S-shape hard segments.

    Start (★) and End (■) markers are placed in the unshaded margins outside
    the first and last segments.
    """
    n_segs = len(seg_names)
    margin = 0.28  # horizontal space reserved outside segments for markers
    all_gx: list[float] = []
    all_gy: list[float] = []
    all_labels: list[str] = []

    for seg_idx, is_hard in enumerate(seg_diffs):
        x0, x1 = float(seg_idx), float(seg_idx + 1)
        xs = np.linspace(x0 + 0.04, x1 - 0.04, n_gates)
        if is_hard:
            t = np.linspace(0, 2 * np.pi, n_gates, endpoint=False)
            ys = _MOCK_GATE_AMP * np.sin(t)
        else:
            ys = np.zeros(n_gates)
        for gi in range(n_gates):
            gnum = seg_idx * n_gates + gi + 1
            all_gx.append(float(xs[gi]))
            all_gy.append(float(ys[gi]))
            all_labels.append(f"G{gnum}")

    # Start / end marker positions — in the unshaded margins
    x_start = -margin * 0.55
    x_end = n_segs + margin * 0.55

    # Connecting path (including approach to start/end markers)
    path_x = [x_start] + all_gx + [x_end]
    path_y = [0.0] + all_gy + [0.0]
    ax.plot(path_x, path_y, "-", color="#999", linewidth=1.0, alpha=0.5, zorder=1)

    # Segment backgrounds — only between x=0 and x=n_segs (no shading in margins)
    for i, is_hard in enumerate(seg_diffs):
        ax.axvspan(
            i, i + 1, color="#FFCDD2" if is_hard else "#C8E6C9", alpha=0.6, zorder=0
        )
    for i in range(n_segs + 1):
        ax.axvline(i, color="#666", linewidth=0.8, linestyle=":", zorder=2)

    # Gate circles and labels
    for gx, gy, lbl in zip(all_gx, all_gy, all_labels):
        ax.plot(gx, gy, "o", color="#444", markersize=5, zorder=3)
        ax.text(gx, gy + 0.07, lbl, ha="center", va="bottom", fontsize=7, color="#222")

    # Start marker (★) in left margin
    ax.plot(
        x_start,
        0.0,
        "*",
        color="green",
        markersize=14,
        zorder=5,
        markeredgewidth=0.5,
        markeredgecolor="darkgreen",
    )
    ax.text(
        x_start,
        0.12,
        "START",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="darkgreen",
        fontweight="bold",
    )

    # End marker (■) in right margin
    ax.plot(
        x_end,
        0.0,
        "s",
        color="#c00000",
        markersize=9,
        zorder=5,
        markeredgewidth=0.5,
        markeredgecolor="#800000",
    )
    ax.text(
        x_end,
        0.12,
        "END",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#800000",
        fontweight="bold",
    )

    ax.set_xlim(-margin, n_segs + margin)
    ax.set_ylim(-0.65, 0.75)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Course Layout Schematic", fontsize=12, fontweight="bold")


# ── Okabe-Ito-compatible palette reused by the controller diagram ─────────────
_CTRL_C_LOW = "#009E73"
_CTRL_C_MED = "#E69F00"
_CTRL_C_HIGH = "#D55E00"
_CTRL_MAX_JUMP = 6
_CTRL_MAX_STEP = 23
_CTRL_INSET_W = 0.17
_CTRL_INSET_H = 0.13
_CTRL_SCENARIOS = [
    {
        "probs": [0.70, 0.20, 0.10],
        "net": -0.60,
        "delta": -4,
        "inset": [0.02, 0.50, _CTRL_INSET_W, _CTRL_INSET_H],
        "label": "P(Low) dom.",
    },
    {
        "probs": [0.10, 0.80, 0.10],
        "net": 0.00,
        "delta": 0,
        "inset": [0.415, 0.63, _CTRL_INSET_W, _CTRL_INSET_H],
        "label": "P(Med) dom.",
    },
    {
        "probs": [0.10, 0.20, 0.70],
        "net": +0.60,
        "delta": +4,
        "inset": [0.81, 0.50, _CTRL_INSET_W, _CTRL_INSET_H],
        "label": "P(High) dom.",
    },
]


def _plot_step_controller_ax(ax: plt.Axes) -> None:
    """Draw the adaptive step controller decision-function diagram.

    Carves *ax*'s subplot spec into a 3-row × 2-column nested grid.
    Left column: model confidence stacked bar.  Right column: step gradient
    with transition shading and arrow.  Row order: LOW → HIGH → MED.
    """
    from matplotlib.colors import LinearSegmentedColormap

    _BG = "#d8d8d8"
    _C_LOW = "#f4a000"
    _C_MED = "#be4b86"
    _C_HIGH = "#2d0d5e"
    _COLORS = [_C_HIGH, _C_MED, _C_LOW]
    _TEXT_INSIDE = ["#ffffff", "#ffffff", "#333333"]
    _LABEL_COLS = [_C_HIGH, _C_MED, _C_LOW]

    _STEP_CMAP = LinearSegmentedColormap.from_list(
        "div_cwl_ctrl", [_C_LOW, _C_MED, _C_HIGH], N=256
    )
    _N_STEPS = 24
    # Scenario color = dominant-class color (LOW dominant → dark purple, HIGH → orange)
    _SCENARIOS = [
        {
            "probs": [0.90, 0.05, 0.05],
            "from_step": 12,
            "to_step": 17,
            "color": _C_HIGH,
            "delta": "Δ+5",
            "title": "Low (highly) dominant",
        },
        {
            "probs": [0.25, 0.35, 0.50],
            "from_step": 12,
            "to_step": 10,
            "color": _C_LOW,
            "delta": "Δ−2",
            "title": "High (slightly) dominant",
        },
        {
            "probs": [0.15, 0.70, 0.15],
            "from_step": 12,
            "to_step": 12,
            "color": _C_MED,
            "delta": None,
            "title": "Medium dominant",
        },
    ]

    _BAR_H = 1.0
    _BAR_BOT = 0.0
    _BAR_TOP = 1.0
    _BAR_CEN = 0.5
    _STEP_YLIM = (0.0, 1.5)

    fig = ax.get_figure()

    # Parent ax: uniform gray panel filling the whole section including title.
    from matplotlib.patches import Rectangle as _Rect

    ax.set_facecolor(_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Full-width gray band from figure bottom to the top of this ax section.
    # Bounds are set via draw_event so they reflect the final layout (including
    # any subplots_adjust calls made after this function returns).
    # bbox_inches="tight" then finds no out-of-bounds artists and doesn't overflow.
    _bg_rect = _Rect(
        (0, 0),
        1.0,
        0.5,
        transform=fig.transFigure,
        facecolor=_BG,
        zorder=-2,
        clip_on=True,
    )
    fig.add_artist(_bg_rect)

    def _update_bg_rect(event, _ref=ax, _rect=_bg_rect):
        bbox = _ref.get_position()
        _rect.set_bounds(0, 0, 1.0, bbox.y1)

    fig.canvas.mpl_connect("draw_event", _update_bg_rect)
    ax.text(
        0.5,
        0.97,
        "Adaptive Controller — Step adjustment logic",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#222",
    )

    # Row 0 is an empty title-spacer row; data rows are 1-3.
    inner_gs = ax.get_subplotspec().subgridspec(
        4,
        2,
        width_ratios=[1, 5],
        height_ratios=[0.18, 1, 1, 1],
        hspace=0.2,
        wspace=0.03,
    )

    ax_step_shared = None
    for row, sc in enumerate(_SCENARIOS):
        c = sc["color"]
        fs = sc["from_step"]
        ts = sc["to_step"]

        # ── Left: model confidence bar ────────────────────────────────────────
        ax_prob = fig.add_subplot(inner_gs[row + 1, 0])
        ax_prob.set_facecolor(_BG)

        cum = 0.0
        for p, cls_c, txt_c, _ in zip(
            sc["probs"], _COLORS, _TEXT_INSIDE, _LABEL_COLS, strict=False
        ):
            ecol = "#cccccc" if cls_c == _C_MED else "none"
            ax_prob.barh(
                [_BAR_CEN],
                [p],
                left=cum,
                color=cls_c,
                height=0.5,
                edgecolor=ecol,
                linewidth=0.7,
                alpha=0.95,
            )
            if p >= 0.15:
                ax_prob.text(
                    cum + p / 2,
                    _BAR_CEN,
                    f"{p:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=txt_c,
                    fontweight="bold",
                )
            cum += p

        cum = 0.0
        for lbl, lbl_c, p in zip(
            ["LOW", "MED", "HIGH"], _LABEL_COLS, sc["probs"], strict=False
        ):
            if p >= 0.08:
                ax_prob.text(
                    cum + p / 2,
                    _BAR_CEN - 0.31,
                    lbl,
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=lbl_c,
                    fontweight="bold",
                )
            cum += p

        ax_prob.text(
            0.5,
            _BAR_TOP + 0.05,
            sc["title"],
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=c,
            transform=ax_prob.transData,
        )
        ax_prob.set_xlim(0, 1)
        ax_prob.set_ylim(*_STEP_YLIM)
        ax_prob.set_xticks([])
        ax_prob.set_yticks([])
        for spine in ax_prob.spines.values():
            spine.set_visible(False)

        # ── Right: step gradient bar ──────────────────────────────────────────
        ax_step = fig.add_subplot(inner_gs[row + 1, 1], sharex=ax_step_shared)
        if ax_step_shared is None:
            ax_step_shared = ax_step

        ax_step.set_facecolor(_BG)
        for s in range(_N_STEPS):
            ax_step.barh(
                [_BAR_CEN],
                [1],
                left=s - 0.5,
                color=_STEP_CMAP(s / (_N_STEPS - 1)),
                height=_BAR_H,
                edgecolor="none",
            )

        ax_step.set_xlim(-0.5, _N_STEPS - 0.5)
        ax_step.set_ylim(*_STEP_YLIM)
        ax_step.set_yticks([])
        ax_step.set_xticks(range(_N_STEPS))
        ax_step.tick_params(axis="x", colors="#444", labelsize=9)
        for sp in ["top", "left", "right"]:
            ax_step.spines[sp].set_visible(False)

        if row < 2:
            ax_step.tick_params(labelbottom=False, bottom=False)
            ax_step.spines["bottom"].set_visible(False)
        else:
            ax_step.spines["bottom"].set_color("#aaa")
            ax_step.set_xticklabels([str(s) for s in range(_N_STEPS)], fontsize=9)
            ax_step.set_xlabel("Adaptation Step", fontsize=11, color="#444")
            ax_step.xaxis.label.set_color("#444")

        # Transition: gray shading
        if fs != ts:
            x0, x1 = sorted([fs, ts])
            ax_step.fill_between(
                [x0, x1],
                _BAR_BOT,
                _BAR_TOP,
                color="#868686",
                alpha=0.55,
                zorder=3,
            )

        ax_step.plot([fs, fs], [_BAR_BOT, _BAR_TOP], color=c, lw=1.8, ls="--", zorder=6)
        if ts != fs:
            ax_step.plot([ts, ts], [_BAR_BOT, _BAR_TOP], color=c, lw=2.8, zorder=7)

        if sc["delta"] is not None:
            ax_step.annotate(
                "",
                xy=(ts, _BAR_CEN + 0.2),
                xytext=(fs, _BAR_CEN + 0.2),
                arrowprops=dict(
                    arrowstyle="->",
                    color="white",
                    lw=2.0,
                    mutation_scale=16,
                ),
            )
            ax_step.text(
                (fs + ts) / 2,
                _BAR_CEN,
                sc["delta"],
                ha="center",
                va="top",
                fontsize=9.5,
                color="white",
                fontweight="bold",
                zorder=9,
            )
        else:
            ax_step.text(
                fs,
                _BAR_CEN,
                "No Change",
                ha="center",
                va="center",
                fontsize=9.5,
                color="#ffffff",
                fontweight="bold",
                zorder=12,
                bbox=dict(facecolor="#868686", edgecolor="none", pad=3.5, alpha=1.0),
            )


def _plot_crash_heatmap(
    ax,
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
    sorted_gate_ids: list[int],
    n_segs: int,
    seg_diffs: list[bool],
    seg_names: list[str],
    n_per_seg: int = 8,
    metric: str = "crashes",
    averaged: bool = False,
    cmap_name: str | None = None,
    show_xlabels: bool = True,
    show_legend: bool = False,
):
    """2D heatmap: subjects (or groups) × gate position.

    Parameters
    ----------
    metric:    "crashes"  — drone deaths approaching each gate
               "outside"  — alive-but-outside count at each gate
    averaged:  False — one row per subject (default)
               True  — one row per group, values averaged across subjects
    cmap_name: name from WINDOW_CMAPS or any matplotlib cmap. Defaults to a
               sensible sequential map per metric.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_cbar

    density_fn = (
        _crash_density_per_gate if metric == "crashes" else _outside_density_per_gate
    )
    default_cmap = "YlOrRd" if metric == "crashes" else "YlGnBu"
    cmap = _resolve_cmap(cmap_name) if cmap_name else default_cmap
    cbar_label = (
        "Average\ncrashes\ncount" if metric == "crashes" else "Average\noutside\ncount"
    )
    no_data_msg = "No crash data" if metric == "crashes" else "No outside-gate data"

    groups_order = ["adaptive", "non_adaptive"]

    if averaged:
        # One row per group; value = mean across subjects of (sum-over-trials per gate)
        row_labels: list[str] = []
        row_colors: list[str] = []
        row_data: list[np.ndarray] = []
        for g in groups_order:
            sids_g = sorted(
                sid for sid, grp in groups.items() if grp == g and sid in by_subject
            )
            if not sids_g:
                continue
            per_subject = []
            for sid in sids_g:
                v = np.zeros(len(sorted_gate_ids), dtype=float)
                for tr in by_subject.get(sid, []):
                    if tr["gates"] is not None:
                        v += density_fn(tr, sorted_gate_ids)
                per_subject.append(v)
            row_data.append(np.mean(per_subject, axis=0))
            label = "Adaptive" if g == "adaptive" else "Non-adaptive"
            row_labels.append(label)
            row_colors.append(GROUP_COLORS.get(g, "#333"))

        n_rows = len(row_data)
        if n_rows == 0:
            _no_data_placeholder(ax, no_data_msg)
            return

        matrix = np.array(row_data)
        x_edges = np.arange(len(sorted_gate_ids) + 1) / n_per_seg
        y_edges = np.arange(n_rows + 1, dtype=float)
        mesh = ax.pcolormesh(x_edges, y_edges, matrix, cmap=cmap, zorder=3, vmin=0)

        for i, is_hard in enumerate(seg_diffs):
            ax.axvspan(
                i,
                i + 1,
                color="#FFEBEE" if is_hard else "#E8F5E9",
                alpha=0.18,
                zorder=0,
            )
        for i in range(n_segs + 1):
            ax.axvline(i, color="#666", linewidth=0.8, linestyle=":", zorder=4)

        ax.set_yticks([i + 0.5 for i in range(n_rows)])
        ax.set_yticklabels(row_labels, fontsize=9)
        for label, color in zip(ax.get_yticklabels(), row_colors, strict=False):
            label.set_color(color)
            label.set_fontweight("bold")

        ax.set_ylim(0, n_rows)
        ax.invert_yaxis()

    else:
        # One row per subject
        ordered_sids: list[str] = []
        boundary_ys: list[float] = []
        for g in groups_order:
            sids_g = sorted(
                sid for sid, grp in groups.items() if grp == g and sid in by_subject
            )
            if sids_g:
                if ordered_sids:
                    boundary_ys.append(float(len(ordered_sids)))
                ordered_sids.extend(sids_g)

        n_subjects = len(ordered_sids)
        n_gates = len(sorted_gate_ids)
        if n_subjects == 0 or n_gates == 0:
            _no_data_placeholder(ax, no_data_msg)
            return

        matrix = np.zeros((n_subjects, n_gates), dtype=float)
        for s_idx, sid in enumerate(ordered_sids):
            for tr in by_subject.get(sid, []):
                if tr["gates"] is not None:
                    matrix[s_idx] += density_fn(tr, sorted_gate_ids)

        x_edges = np.arange(n_gates + 1) / n_per_seg
        y_edges = np.arange(n_subjects + 1, dtype=float)
        mesh = ax.pcolormesh(x_edges, y_edges, matrix, cmap=cmap, zorder=3, vmin=0)

        for i, is_hard in enumerate(seg_diffs):
            ax.axvspan(
                i,
                i + 1,
                color="#FFEBEE" if is_hard else "#E8F5E9",
                alpha=0.18,
                zorder=0,
            )
        for i in range(n_segs + 1):
            ax.axvline(i, color="#666", linewidth=0.8, linestyle=":", zorder=4)

        for yb in boundary_ys:
            ax.axhline(yb, color="#333", linewidth=2.0, zorder=5)

        ax.set_yticks([i + 0.5 for i in range(n_subjects)])
        ax.set_yticklabels(ordered_sids, fontsize=10)
        for label, sid in zip(ax.get_yticklabels(), ordered_sids, strict=False):
            g = groups.get(sid, "unknown")
            label.set_color(GROUP_COLORS.get(g, "#333"))
            label.set_fontweight("bold")

        ax.set_ylim(0, n_subjects)
        ax.invert_yaxis()

    ax.set_xticks([i + 0.5 for i in range(n_segs)])
    if show_xlabels:
        ax.set_xticklabels(seg_names, fontsize=10)
    else:
        ax.tick_params(labelbottom=False)
    suffix = "avg. across subjects" if averaged else "per subject, all trials"
    title_metric = "Crash Density" if metric == "crashes" else "Outside-Gate Passes"
    ax.set_title(
        f"{title_metric} Heatmap  ({suffix})",
        fontsize=11,
        fontweight="bold",
    )

    if show_legend and not averaged:
        from matplotlib.lines import Line2D as _Line2D

        ax.legend(
            handles=[
                _Line2D(
                    [0],
                    [0],
                    color=GROUP_COLORS["adaptive"],
                    linewidth=4,
                    label="Adaptive",
                ),
                _Line2D(
                    [0],
                    [0],
                    color=GROUP_COLORS["non_adaptive"],
                    linewidth=4,
                    label="Non-Adaptive",
                ),
            ],
            fontsize=9,
            loc="upper right",
            framealpha=0.85,
            title="Group",
            title_fontsize=9,
        )

    cbar_ax = _inset_cbar(
        ax,
        width="2%",
        height="90%",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.05, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = ax.get_figure().colorbar(mesh, cax=cbar_ax)
    cb.set_label(cbar_label, fontsize=10)
    cb.ax.tick_params(labelsize=9)


def _find_drone_death_timestamps(drones: pd.DataFrame) -> list[float]:
    """Return timestamps of alive→dead transitions for each drone."""
    if drones is None or drones.empty or "alive" not in drones.columns:
        return []
    death_times: list[float] = []
    for _, grp in drones.groupby("id"):
        grp = grp.sort_values("timestamp")
        alive = grp["alive"].values.astype(int)
        times = grp["timestamp"].values
        transitions = np.diff(alive)
        for i, delta in enumerate(transitions):
            if delta < 0:
                death_times.append(float(times[i + 1]))
    return sorted(death_times)


def _build_crash_window_matrices(
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
    pre_s: float = 10.0,
    post_s: float = 10.0,
    n_bins: int = 80,
    delta: bool = False,
    averaged: bool = False,
):
    """
    Returns (row_labels, row_groups, t_axis, cwl_matrix, step_matrix, counts).

    - delta=False  → traces are absolute CWL state / adaptation step.
      delta=True   → each per-event trace is subtracted from its first valid
                     value (initial sample at the start of the window).
    - averaged=False → rows are subjects (existing behaviour). row_labels=sids,
                       row_groups[i]=group of subject i, counts[i]=#crashes.
      averaged=True  → 2 rows max: one per group ("Adaptive", "Non-adaptive"),
                       row trace = mean of subject-mean traces (each subject
                       contributes equally), counts[i]=total events in group.
    """
    ordered_sids = [
        sid
        for g in ["adaptive", "non_adaptive"]
        for sid in sorted(
            s for s, grp in groups.items() if grp == g and s in by_subject
        )
        if not sid.startswith("_")
    ]
    t_axis = np.linspace(-pre_s, post_s, n_bins)

    # Per-subject collection of per-event traces
    cwl_by_sid: dict[str, list[np.ndarray]] = {sid: [] for sid in ordered_sids}
    step_by_sid: dict[str, list[np.ndarray]] = {sid: [] for sid in ordered_sids}

    def _maybe_delta(trace: np.ndarray) -> np.ndarray:
        if not delta:
            return trace
        valid = trace[~np.isnan(trace)]
        if len(valid) == 0:
            return trace
        return trace - valid[0]

    for sid in ordered_sids:
        for tr in by_subject[sid]:
            drones = tr["drones"]
            inf = tr["inference"]
            commands = tr["commands"]
            for t_death in _find_drone_death_timestamps(drones):
                # timestamps are in ms; pre_s/post_s are seconds → convert
                t0_ms = t_death - pre_s * 1000
                t1_ms = t_death + post_s * 1000
                if inf is not None and "filtered_state" in inf.columns:
                    seg = inf[(inf["timestamp"] >= t0_ms) & (inf["timestamp"] <= t1_ms)]
                    if len(seg) >= 2:
                        t_rel = (seg["timestamp"].values - t_death) / 1000.0
                        states = seg["filtered_state"].fillna(0).values.astype(float)
                        idxs = np.clip(
                            np.searchsorted(t_rel, t_axis), 0, len(states) - 1
                        )
                        in_rng = (t_axis >= t_rel[0]) & (t_axis <= t_rel[-1])
                        trace = np.where(in_rng, states[idxs], np.nan)
                        cwl_by_sid[sid].append(_maybe_delta(trace))
                if commands is not None and "cwl_current_step" in commands.columns:
                    seg = commands[
                        (commands["timestamp"] >= t0_ms)
                        & (commands["timestamp"] <= t1_ms)
                    ]
                    if len(seg) >= 2:
                        t_rel = (seg["timestamp"].values - t_death) / 1000.0
                        steps = (
                            seg["cwl_current_step"].fillna(np.nan).values.astype(float)
                        )
                        idxs = np.clip(
                            np.searchsorted(t_rel, t_axis), 0, len(steps) - 1
                        )
                        in_rng = (t_axis >= t_rel[0]) & (t_axis <= t_rel[-1])
                        trace = np.where(in_rng, steps[idxs], np.nan)
                        step_by_sid[sid].append(_maybe_delta(trace))

    if averaged:
        row_labels: list[str] = []
        row_groups: list[str] = []
        cwl_rows: list[np.ndarray] = []
        step_rows: list[np.ndarray] = []
        counts: list[int] = []
        for g in ["adaptive", "non_adaptive"]:
            sids_g = [s for s in ordered_sids if groups.get(s) == g]
            if not sids_g:
                continue
            cwl_subj_means = []
            step_subj_means = []
            n_events = 0
            for sid in sids_g:
                if cwl_by_sid[sid]:
                    cwl_subj_means.append(np.nanmean(np.array(cwl_by_sid[sid]), axis=0))
                if step_by_sid[sid]:
                    step_subj_means.append(
                        np.nanmean(np.array(step_by_sid[sid]), axis=0)
                    )
                n_events += len(cwl_by_sid[sid])
            cwl_rows.append(
                np.nanmean(np.array(cwl_subj_means), axis=0)
                if cwl_subj_means
                else np.full(n_bins, np.nan)
            )
            step_rows.append(
                np.nanmean(np.array(step_subj_means), axis=0)
                if step_subj_means
                else np.full(n_bins, np.nan)
            )
            row_labels.append("Adaptive" if g == "adaptive" else "Non-adaptive")
            row_groups.append(g)
            counts.append(n_events)
        cwl_matrix = np.array(cwl_rows) if cwl_rows else np.empty((0, n_bins))
        step_matrix = np.array(step_rows) if step_rows else np.empty((0, n_bins))
        return (
            row_labels,
            row_groups,
            t_axis,
            cwl_matrix,
            step_matrix,
            np.array(counts),
        )

    n_sids = len(ordered_sids)
    cwl_matrix = np.full((n_sids, n_bins), np.nan)
    step_matrix = np.full((n_sids, n_bins), np.nan)
    crash_counts = np.zeros(n_sids, dtype=int)
    for i, sid in enumerate(ordered_sids):
        crash_counts[i] = len(cwl_by_sid[sid])
        if cwl_by_sid[sid]:
            cwl_matrix[i] = np.nanmean(np.array(cwl_by_sid[sid]), axis=0)
        if step_by_sid[sid]:
            step_matrix[i] = np.nanmean(np.array(step_by_sid[sid]), axis=0)
    row_groups = [groups.get(s, "unknown") for s in ordered_sids]
    return ordered_sids, row_groups, t_axis, cwl_matrix, step_matrix, crash_counts


def _plot_crash_window_heatmap(
    ax_cwl,
    ax_step,
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
    pre_s: float = 10.0,
    post_s: float = 10.0,
    delta: bool = True,
    averaged: bool = False,
    cwl_cmap_name: str = "vik_like",
    step_cmap_name: str = "vik_like",
):
    """
    Two side-by-side heatmaps (rows × time) showing CWL state and adaptation
    step around drone crashes (±pre/post_s).

    Parameters
    ----------
    delta:    if True, plot variation w.r.t. each event's initial value
              (highlights the post-crash transient; recommended for spotting
              the inference window's ~5 s reaction lag).
              if False, plot absolute mean values.
    averaged: if True, two rows total (Adaptive / Non-adaptive), each row =
              mean of subject-mean traces within group.
              if False, one row per subject.
    cwl_cmap_name / step_cmap_name: keys into WINDOW_CMAPS (or any matplotlib
              colormap name). For delta plots, use a diverging map
              ("ok_div", "vik_like", "broc_like", "cork_like", "ok_div_strong",
              "RdBu_r", "PuOr_r", "BrBG_r", "coolwarm").
              For absolute plots, sequential maps work better.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_cbar

    row_labels, row_groups, t_axis, cwl_matrix, step_matrix, counts = (
        _build_crash_window_matrices(
            by_subject,
            groups,
            pre_s,
            post_s,
            delta=delta,
            averaged=averaged,
        )
    )

    n_rows = len(row_labels)
    if n_rows == 0:
        _no_data_placeholder(ax_cwl, "No subjects")
        _no_data_placeholder(ax_step, "No subjects")
        return

    dt = t_axis[1] - t_axis[0]
    t_edges = np.append(t_axis - dt / 2, t_axis[-1] + dt / 2)
    y_edges = np.arange(n_rows + 1) - 0.5

    # Step panel: adaptive subjects only (non-adaptive have no cwl_current_step).
    step_idxs = [i for i, g in enumerate(row_groups) if g == "adaptive"]
    step_row_labels = [row_labels[i] for i in step_idxs]
    step_row_groups = [row_groups[i] for i in step_idxs]
    step_counts_f = counts[step_idxs] if len(step_idxs) else counts[:0]
    step_matrix_f = (
        step_matrix[step_idxs]
        if len(step_idxs)
        else np.empty((0, step_matrix.shape[1]))
    )
    n_step_rows = len(step_row_labels)
    y_edges_step = np.arange(n_step_rows + 1) - 0.5

    def _sym_lim(matrix: np.ndarray, fallback: float) -> float:
        if not np.isfinite(matrix).any():
            return fallback
        return max(float(np.nanmax(np.abs(matrix))), 1e-3)

    # ── CWL panel ────────────────────────────────────────────────────────────
    if delta:
        cwl_lim = _sym_lim(cwl_matrix, 1.0)
        cwl_kwargs = dict(
            cmap=_resolve_cmap(cwl_cmap_name), vmin=-cwl_lim, vmax=cwl_lim
        )
        cwl_cb_label = "Δ CWL\nstate"
        cwl_title = (
            f"Δ CWL state ±{int(post_s)} s around crash "
            f"({'group avg' if averaged else 'per subject'})"
        )
    else:
        cwl_kwargs = dict(
            cmap=_resolve_cmap(cwl_cmap_name)
            if cwl_cmap_name not in (None, "vik_like")
            else LinearSegmentedColormap.from_list(
                "cwl3",
                [STATE_COLORS[0], STATE_COLORS[1], STATE_COLORS[2]],
                N=256,
            ),
            vmin=0,
            vmax=2,
        )
        cwl_cb_label = "CWL state"
        cwl_title = (
            f"Mean CWL state ±{int(post_s)} s around crash "
            f"({'group avg' if averaged else 'per subject'})"
        )

    mesh_cwl = ax_cwl.pcolormesh(t_edges, y_edges, cwl_matrix, zorder=3, **cwl_kwargs)
    ax_cwl.axvspan(-pre_s, 0, color="k", alpha=0.07, zorder=1)
    ax_cwl.axvline(0, color="gray", linewidth=1.8, linestyle="--", zorder=5)
    ax_cwl.set_xlim(-pre_s, post_s)
    ax_cwl.set_ylim(-0.5, n_rows - 0.5)
    ax_cwl.invert_yaxis()
    ax_cwl.set_yticks(range(n_rows))
    ax_cwl.set_yticklabels(
        [f"{lab}" for lab in row_labels],
        fontsize=9 if not averaged else 11,
    )
    for label, grp in zip(ax_cwl.get_yticklabels(), row_groups, strict=False):
        label.set_color(GROUP_COLORS.get(grp, "#333"))
        label.set_fontweight("bold")
    ax_cwl.set_title(cwl_title, fontsize=11, fontweight="bold")
    ax_cwl.tick_params(axis="x", labelbottom=False)

    # Group separator between adaptive and non-adaptive rows
    if not averaged:
        n_adaptive_cwl = sum(1 for g in row_groups if g == "adaptive")
        if 0 < n_adaptive_cwl < n_rows:
            ax_cwl.axhline(n_adaptive_cwl - 0.5, color="white", linewidth=1.5, zorder=6)

    cbar_ax = _inset_cbar(
        ax_cwl,
        width="2%",
        height="90%",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.05, 1, 1),
        bbox_transform=ax_cwl.transAxes,
        borderpad=0,
    )
    cb = ax_cwl.get_figure().colorbar(mesh_cwl, cax=cbar_ax)
    if delta:
        cb.set_label(cwl_cb_label, fontsize=9)
    else:
        cb.set_ticks([0, 1, 2])
        cb.set_ticklabels(["Low", "Med", "High"])
    cb.ax.tick_params(labelsize=9)

    # ── Step panel (adaptive subjects only) ──────────────────────────────────
    if n_step_rows == 0:
        _no_data_placeholder(ax_step, "No adaptive subjects")
        return

    if delta:
        step_lim = _sym_lim(step_matrix_f, 1.0)
        step_kwargs = dict(
            cmap=_resolve_cmap(step_cmap_name), vmin=-step_lim, vmax=step_lim
        )
        step_cb_label = "Δ adapt.\nstep"
        step_title = (
            f"Δ adaptation step ±{int(post_s)} s around crash "
            f"({'group avg' if averaged else 'adaptive subjects only'})"
        )
    else:
        step_kwargs = dict(cmap=_resolve_cmap(step_cmap_name) or "Blues", vmin=0)
        step_cb_label = "step"
        step_title = f"Mean adaptation step ±{int(post_s)} s around crash "

    mesh_step = ax_step.pcolormesh(
        t_edges, y_edges_step, step_matrix_f, zorder=3, **step_kwargs
    )
    ax_step.axvspan(-pre_s, 0, color="k", alpha=0.07, zorder=1)
    ax_step.axvline(0, color="gray", linewidth=1.8, linestyle="--", zorder=5)
    ax_step.set_xlim(-pre_s, post_s)
    ax_step.set_ylim(-0.5, n_step_rows - 0.5)
    ax_step.invert_yaxis()
    ax_step.set_yticks(range(n_step_rows))
    ax_step.set_yticklabels(
        [f"{lab}" for lab in step_row_labels],
        fontsize=9 if not averaged else 11,
    )
    for label, grp in zip(ax_step.get_yticklabels(), step_row_groups, strict=False):
        label.set_color(GROUP_COLORS.get(grp, "#333"))
        label.set_fontweight("bold")
    ax_step.set_xlabel("Time relative to crash (s)", fontsize=10)
    ax_step.set_title(step_title, fontsize=11, fontweight="bold")
    ax_step.tick_params(axis="x", labelsize=9)

    cbar_ax2 = _inset_cbar(
        ax_step,
        width="2%",
        height="90%",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.05, 1, 1),
        bbox_transform=ax_step.transAxes,
        borderpad=0,
    )
    cb2 = ax_step.get_figure().colorbar(mesh_step, cax=cbar_ax2)
    cb2.set_label(step_cb_label, fontsize=9)
    cb2.ax.tick_params(labelsize=9)


def _plot_subject_crash_cwl(
    ax_cwl,
    ax_step,
    sid: str,
    group: str,
    trials: list[dict],
    pre_s: float = 10.0,
    post_s: float = 10.0,
):
    """Per-subject event-triggered CWL + step around actual drone deaths (±pre/post seconds)."""
    n_bins = 60
    t_axis = np.linspace(-pre_s, post_s, n_bins)
    cwl_events: list[np.ndarray] = []
    step_events: list[np.ndarray] = []

    for tr in trials:
        drones = tr["drones"]
        inf = tr["inference"]
        commands = tr["commands"]

        death_times = _find_drone_death_timestamps(drones)

        for t_death in death_times:
            # timestamps are in ms; pre_s/post_s are seconds → convert
            t0_ms = t_death - pre_s * 1000
            t1_ms = t_death + post_s * 1000

            if inf is not None and "filtered_state" in inf.columns:
                seg = inf[
                    (inf["timestamp"] >= t0_ms) & (inf["timestamp"] <= t1_ms)
                ].copy()
                if len(seg) >= 2:
                    # t_rel in seconds, matching t_axis
                    t_rel = (seg["timestamp"].values - t_death) / 1000.0
                    states = seg["filtered_state"].fillna(0).values.astype(int)
                    idxs = np.clip(np.searchsorted(t_rel, t_axis), 0, len(states) - 1)
                    in_rng = (t_axis >= t_rel[0]) & (t_axis <= t_rel[-1])
                    cwl_events.append(
                        np.where(in_rng, states[idxs].astype(float), np.nan)
                    )

            if commands is not None and "cwl_current_step" in commands.columns:
                seg = commands[
                    (commands["timestamp"] >= t0_ms) & (commands["timestamp"] <= t1_ms)
                ].copy()
                if len(seg) >= 2:
                    t_rel = (seg["timestamp"].values - t_death) / 1000.0
                    steps = seg["cwl_current_step"].fillna(np.nan).values.astype(float)
                    idxs = np.clip(np.searchsorted(t_rel, t_axis), 0, len(steps) - 1)
                    in_rng = (t_axis >= t_rel[0]) & (t_axis <= t_rel[-1])
                    step_events.append(np.where(in_rng, steps[idxs], np.nan))

    color = GROUP_COLORS.get(group, "#888")
    n_ev = len(cwl_events)

    # ── CWL probability ───────────────────────────────────────────────────────
    if cwl_events:
        arr = np.array(cwl_events, dtype=float)
        for state in range(3):
            probs = np.nanmean(arr == state, axis=0)
            ax_cwl.plot(
                t_axis,
                probs,
                color=STATE_COLORS[state],
                linewidth=1.6,
                label=STATE_LABELS[state],
            )
            ax_cwl.fill_between(t_axis, probs, color=STATE_COLORS[state], alpha=0.08)
    else:
        _no_data_placeholder(ax_cwl, "No crash events")

    ax_cwl.axvspan(-pre_s, 0, color="#FFF3E0", alpha=0.25, zorder=0)
    ax_cwl.axvline(0, color="#555", linewidth=1.2, linestyle="--", zorder=5)
    ax_cwl.set_xlim(-pre_s, post_s)
    ax_cwl.set_ylim(0, 1.05)
    ax_cwl.set_ylabel(f"P(CWL)", fontsize=7)
    ax_cwl.set_title(
        f"{sid}  [{GROUP_LABELS[group].split()[0]}]  — n={n_ev} events",
        fontsize=8,
        fontweight="bold",
        color=color,
    )
    ax_cwl.grid(linestyle=":", alpha=0.3)
    if cwl_events:
        ax_cwl.legend(fontsize=6.5, ncol=3, loc="upper left", framealpha=0.8)

    # ── Adaptation step ───────────────────────────────────────────────────────
    if step_events:
        arr = np.array(step_events, dtype=float)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        for trace in arr:
            ax_step.plot(t_axis, trace, color=color, linewidth=0.5, alpha=0.2, zorder=1)
        ax_step.plot(t_axis, mean, color=color, linewidth=1.8, zorder=3, label="mean")
        ax_step.fill_between(
            t_axis, mean - std, mean + std, color=color, alpha=0.18, zorder=2
        )
    else:
        _no_data_placeholder(ax_step, "No step data")

    ax_step.axvspan(-pre_s, 0, color="#FFF3E0", alpha=0.25, zorder=0)
    ax_step.axvline(0, color="#555", linewidth=1.2, linestyle="--", zorder=5)
    ax_step.set_xlim(-pre_s, post_s)
    ax_step.set_ylabel("Step", fontsize=7)
    ax_step.set_title("Adaptation step", fontsize=8)
    ax_step.grid(linestyle=":", alpha=0.3)


def _plot_crash_triggered_cwl(
    ax_cwl,
    ax_step,
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
    pre_s: float = 10.0,
    post_s: float = 10.0,
):
    """Event-triggered average of CWL state and adaptation step ±10 s around crash gates.

    Reference t=0 is the gate-passage timestamp where a death was detected.
    Crashes occurred somewhere in [-10, 0], so CWL elevation before t=0
    reveals high workload preceding crashes.
    """
    n_bins = 60
    t_axis = np.linspace(-pre_s, post_s, n_bins)

    groups_shown = ["adaptive", "non_adaptive"]
    cwl_traces: dict[str, list[np.ndarray]] = {g: [] for g in groups_shown}
    step_traces: dict[str, list[np.ndarray]] = {g: [] for g in groups_shown}
    n_events: dict[str, int] = {g: 0 for g in groups_shown}

    for sid, trials in by_subject.items():
        g = groups.get(sid, "unknown")
        if g not in groups_shown:
            continue
        ref_gates = next(
            (tr["gates"] for tr in trials if tr["gates"] is not None), None
        )
        if ref_gates is None:
            continue
        sorted_gids = _get_sorted_gate_ids(ref_gates)

        for tr in trials:
            inf = tr["inference"]
            commands = tr["commands"]
            events = _collect_crash_gate_events(tr, sorted_gids)
            n_events[g] += len(events)

            for ev in events:
                t_ref = ev["t_gate"]
                t0, t1 = t_ref - pre_s, t_ref + post_s

                # CWL state trace
                if inf is not None and "filtered_state" in inf.columns:
                    seg = inf[
                        (inf["timestamp"] >= t0) & (inf["timestamp"] <= t1)
                    ].copy()
                    if len(seg) >= 2:
                        t_rel = seg["timestamp"].values - t_ref
                        states = seg["filtered_state"].fillna(0).values.astype(int)
                        idxs = np.clip(
                            np.searchsorted(t_rel, t_axis), 0, len(states) - 1
                        )
                        in_rng = (t_axis >= t_rel[0]) & (t_axis <= t_rel[-1])
                        cwl_traces[g].append(
                            np.where(in_rng, states[idxs].astype(float), np.nan)
                        )

                # Step trace
                if commands is not None and "cwl_current_step" in commands.columns:
                    seg = commands[
                        (commands["timestamp"] >= t0) & (commands["timestamp"] <= t1)
                    ].copy()
                    if len(seg) >= 2:
                        t_rel = seg["timestamp"].values - t_ref
                        steps = (
                            seg["cwl_current_step"].fillna(np.nan).values.astype(float)
                        )
                        idxs = np.clip(
                            np.searchsorted(t_rel, t_axis), 0, len(steps) - 1
                        )
                        in_rng = (t_axis >= t_rel[0]) & (t_axis <= t_rel[-1])
                        step_traces[g].append(np.where(in_rng, steps[idxs], np.nan))

    # ── CWL probability plot ──────────────────────────────────────────────────
    any_cwl = False
    for g in groups_shown:
        traces = cwl_traces[g]
        if not traces:
            continue
        any_cwl = True
        arr = np.array(traces, dtype=float)  # (n_events, n_bins)
        ls = "-" if g == "adaptive" else "--"
        prefix = f"{GROUP_LABELS[g].split()[0]} (n={n_events[g]})"
        for state in range(3):
            probs = np.nanmean(arr == state, axis=0)
            color = STATE_COLORS[state]
            ax_cwl.plot(
                t_axis,
                probs,
                color=color,
                linestyle=ls,
                linewidth=1.8,
                label=f"{prefix} — {STATE_LABELS[state]}",
            )
            ax_cwl.fill_between(t_axis, probs, color=color, alpha=0.07)

    if not any_cwl:
        _no_data_placeholder(ax_cwl, "No crash events detected")
    else:
        ax_cwl.axvspan(-pre_s, 0, color="#FFF3E0", alpha=0.30, zorder=0)
        ax_cwl.axvline(
            0,
            color="#333",
            linewidth=1.4,
            linestyle="--",
            zorder=5,
            label="Crash gate (t = 0)",
        )
        ax_cwl.set_xlabel("Time relative to crash gate (s)", fontsize=8)
        ax_cwl.set_ylabel("P(CWL state)", fontsize=8)
        ax_cwl.set_ylim(0, 1.05)
        ax_cwl.set_xlim(-pre_s, post_s)
        ax_cwl.set_title(
            f"CWL estimate ±{int(pre_s)}s around crash gates\n(event-triggered average)",
            fontsize=9,
            fontweight="bold",
        )
        ax_cwl.legend(fontsize=6.5, ncol=2, loc="upper left", framealpha=0.9)
        ax_cwl.grid(linestyle=":", alpha=0.3)

    # ── Adaptation step plot ──────────────────────────────────────────────────
    any_step = False
    for g in groups_shown:
        traces = step_traces[g]
        if not traces:
            continue
        any_step = True
        arr = np.array(traces, dtype=float)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        color = GROUP_COLORS[g]
        ls = "-" if g == "adaptive" else "--"
        label = f"{GROUP_LABELS[g].split()[0]} (n={n_events[g]})"
        ax_step.plot(
            t_axis, mean, color=color, linestyle=ls, linewidth=1.8, label=label
        )
        ax_step.fill_between(t_axis, mean - std, mean + std, color=color, alpha=0.15)

    if not any_step:
        _no_data_placeholder(ax_step, "No step data around crashes")
    else:
        ax_step.axvspan(-pre_s, 0, color="#FFF3E0", alpha=0.30, zorder=0)
        ax_step.axvline(0, color="#333", linewidth=1.4, linestyle="--", zorder=5)
        ax_step.set_xlabel("Time relative to crash gate (s)", fontsize=8)
        ax_step.set_ylabel("Adaptation step", fontsize=8)
        ax_step.set_xlim(-pre_s, post_s)
        ax_step.set_title(
            f"Adaptation step ±{int(pre_s)}s around crash gates\n(event-triggered average)",
            fontsize=9,
            fontweight="bold",
        )
        ax_step.legend(fontsize=8, loc="upper left", framealpha=0.9)
        ax_step.grid(linestyle=":", alpha=0.3)


def _collect_trial_segment_traces(
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
    n_segs: int,
    a_ids_ref: list[int],
    b_ids_ref: list[int],
    signal: str,
    group_filter: set[str] | None,
    n_resamp: int,
) -> dict[int, list[np.ndarray]]:
    """Collect per-trial, per-subject resampled signal traces aligned to segments.

    For each (subject, trial_index), the signal ('step' = cwl_current_step,
    'cwl' = filtered_state) is sliced per segment using gate passage timestamps,
    time-normalised within the segment, and resampled to *n_resamp* points.

    Returns {trial_idx: [trace_array(n_segs * n_resamp), ...]} with NaN where
    data is missing for a segment.
    """
    result: dict[int, list[np.ndarray]] = {}

    for sid, trials in by_subject.items():
        if group_filter is not None and groups.get(sid) not in group_filter:
            continue
        for t_idx, tr in enumerate(trials):
            gs = tr["gate_status"]
            gates = tr["gates"]
            if gs is None or gates is None:
                continue
            gs_idx = gs.drop_duplicates(subset=["id"]).set_index("id")
            a_ids_tr, b_ids_tr, _, _ = _segment_metadata(gates)

            if signal == "step":
                df_sig = tr["commands"]
                val_col = "cwl_current_step"
            else:
                df_sig = tr["inference"]
                val_col = "filtered_state"

            if df_sig is None or val_col not in df_sig.columns:
                continue

            trace = np.full(n_segs * n_resamp, np.nan)
            for seg_idx in range(n_segs):
                if seg_idx >= len(a_ids_tr):
                    break
                ga, gb = a_ids_tr[seg_idx], b_ids_tr[seg_idx]
                try:
                    t_start = float(gs_idx.loc[ga, "first_pass_timestamp"])
                    t_end = float(gs_idx.loc[gb, "first_pass_timestamp"])
                except (KeyError, TypeError):
                    continue
                if t_start <= 0 or t_end <= 0 or t_end <= t_start:
                    continue

                mask = (df_sig["timestamp"] >= t_start) & (df_sig["timestamp"] <= t_end)
                seg_df = df_sig.loc[mask].sort_values("timestamp")
                if len(seg_df) < 2:
                    continue

                t_lo = seg_df["timestamp"].min()
                t_hi = seg_df["timestamp"].max()
                xn = (seg_df["timestamp"].values - t_lo) / (t_hi - t_lo)
                vals = seg_df[val_col].fillna(0).values.astype(float)
                x_grid = np.linspace(0, 1, n_resamp)
                resampled = np.interp(x_grid, xn, vals)
                trace[seg_idx * n_resamp : (seg_idx + 1) * n_resamp] = resampled

            if not np.all(np.isnan(trace)):
                result.setdefault(t_idx, []).append(trace)

    return result


def _plot_adaptive_profile_summary(
    ax_mock: plt.Axes,
    step_axes: list | None,
    cwl_axes: list | None,
    cbar_step_ax: plt.Axes | None,
    cbar_cwl_ax: plt.Axes | None,
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
    step_cmap_name: str = "viridis",
    cwl_cmap_name: str | None = None,
    n_resamp: int = _N_RESAMP_PROFILE,
) -> None:
    """Adaptive flight profile summary: mock course + per-trial segment-aligned bars.

    Layout (shared X = segment space):
      ax_mock        — schematic course (_build_mock_course_ax)
      step_axes[:-1] — one row per trial: mean adaptation step (adaptive subjects)
      step_axes[-1]  — global mean adaptation step (all trials × adaptive subjects)
      cwl_axes[:-1]  — one row per trial: mean CWL estimate (adaptive subjects)
      cwl_axes[-1]   — global mean CWL
      cbar_step_ax   — dedicated colorbar axes for the step section (no size theft)
      cbar_cwl_ax    — dedicated colorbar axes for the CWL section

    Colorbars are drawn into pre-allocated axes so data rows and mock course stay
    the same width (no `ax=` shrinkage).
    """
    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    ref_gates = next(
        (
            tr["gates"]
            for trials in by_subject.values()
            for tr in trials
            if tr["gates"] is not None
        ),
        None,
    )
    if ref_gates is None:
        all_data_axes = [ax_mock]
        if step_axes is not None:
            all_data_axes.extend(step_axes)
        if cwl_axes is not None:
            all_data_axes.extend(cwl_axes)
        for ax in all_data_axes:
            _no_data_placeholder(ax, "No gate data")
        return

    a_ids_ref, b_ids_ref, seg_names, seg_diffs = _segment_metadata(ref_gates)
    n_segs = len(a_ids_ref)

    _build_mock_course_ax(ax_mock, seg_names, seg_diffs)

    # Determine max step from experiment data
    max_step = 23.0
    for sid, trials in by_subject.items():
        if groups.get(sid) != "adaptive":
            continue
        for tr in trials:
            cmd = tr["commands"]
            if cmd is not None and "cwl_total_steps" in cmd.columns:
                tot = cmd["cwl_total_steps"].dropna()
                if not tot.empty:
                    max_step = float(tot.iloc[0]) - 1
                    break
        else:
            continue
        break

    step_cmap = get_cmap(step_cmap_name)
    cwl_cmap = (
        get_cmap(cwl_cmap_name)
        if cwl_cmap_name is not None
        else LinearSegmentedColormap.from_list(
            "cwl3", [STATE_COLORS[0], STATE_COLORS[1], STATE_COLORS[2]], N=256
        )
    )

    # Adaptive subjects only for both step and CWL
    step_traces = (
        _collect_trial_segment_traces(
            by_subject,
            groups,
            n_segs,
            a_ids_ref,
            b_ids_ref,
            "step",
            group_filter={"adaptive"},
            n_resamp=n_resamp,
        )
        if step_axes is not None
        else {}
    )
    cwl_traces = (
        _collect_trial_segment_traces(
            by_subject,
            groups,
            n_segs,
            a_ids_ref,
            b_ids_ref,
            "cwl",
            group_filter={"adaptive"},
            n_resamp=n_resamp,
        )
        if cwl_axes is not None
        else {}
    )

    n_pts = n_segs * n_resamp

    _ROW_MARGIN = 0.04  # fraction of row height left as whitespace above/below image
    _ALL_ROW_BG = "#888888"  # background tint for the aggregate "All" row
    _ALL_ROW_BG_ALPHA = 0.30
    _ALL_V_PAD = 0.05  # extra vertical padding for the "All" row image

    def _draw_row(
        ax, mean_trace, cmap, vmin, vmax, ylabel, show_xticks=False, v_pad=0.0
    ):
        for i, is_hard in enumerate(seg_diffs):
            ax.axvspan(
                i,
                i + 1,
                color="#FFEBEE" if is_hard else "#E8F5E9",
                alpha=0.35,
                zorder=0,
            )

        bot = 0.02 + v_pad
        top = 1 - _ROW_MARGIN - v_pad
        if not np.all(np.isnan(mean_trace)):
            norm_vals = np.clip((mean_trace - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
            rgba = cmap(norm_vals).astype(float)
            rgba[np.isnan(mean_trace), 3] = 0.0
            ax.imshow(
                rgba[np.newaxis, :, :],
                extent=[0, n_segs, bot, top],
                aspect="auto",
                interpolation="nearest",
                zorder=3,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#aaa",
            )

        # Segment dividers drawn after the image so they sit on top
        for i in range(n_segs + 1):
            ax.axvline(i, color="white", linewidth=0.8, alpha=0.7, zorder=4)

        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([ylabel], fontsize=9)
        ax.tick_params(axis="y", which="both", length=0, pad=2)

        if show_xticks:
            ax.set_xticks([i + 0.5 for i in range(n_segs)])
            ax.set_xticklabels(seg_names, fontsize=9, rotation=0, ha="center")
        else:
            ax.tick_params(labelbottom=False)

    def _mean_of(traces_dict, t_idx):
        traces = traces_dict.get(t_idx, [])
        if traces:
            return np.nanmean(np.array(traces, dtype=float), axis=0), len(traces)
        return np.full(n_pts, np.nan), 0

    def _add_full_width_bg(ax, color, alpha):
        """Gray rect spanning the full figure width at *ax*'s row height."""
        import matplotlib.transforms as mtransforms
        from matplotlib.patches import Rectangle

        blend = mtransforms.blended_transform_factory(
            ax.get_figure().transFigure, ax.transAxes
        )
        ax.add_patch(
            Rectangle(
                (0, 0),
                1.0,
                1.0,
                transform=blend,
                facecolor=color,
                alpha=alpha,
                zorder=-1,
                clip_on=False,
            )
        )

    # ── Step rows ─────────────────────────────────────────────────────────────
    # Last visible section gets show_xticks=True; if only step exists, its last
    # row is the bottom; otherwise CWL section owns the bottom x-axis labels.
    step_is_last = step_axes is not None and cwl_axes is None

    if step_axes is not None:
        step_axes[0].set_title(
            "Avg. Adaptation",
            fontsize=10,
            fontweight="bold",
            loc="left",
            pad=2,
        )
        for t_idx, ax in enumerate(step_axes[:-1]):
            mean_tr, _ = _mean_of(step_traces, t_idx)
            _draw_row(ax, mean_tr, step_cmap, 0.0, max_step, f"Trial {t_idx + 1}")

        all_step_flat = [t for traces in step_traces.values() for t in traces]
        global_step = (
            np.nanmean(np.array(all_step_flat, dtype=float), axis=0)
            if all_step_flat
            else np.full(n_pts, np.nan)
        )
        _add_full_width_bg(step_axes[-1], _ALL_ROW_BG, _ALL_ROW_BG_ALPHA)
        _draw_row(
            step_axes[-1],
            global_step,
            step_cmap,
            0.0,
            max_step,
            "All",
            show_xticks=step_is_last,
            v_pad=_ALL_V_PAD,
        )
        for spine in step_axes[-1].spines.values():
            spine.set_linewidth(1.4)

        if cbar_step_ax is not None:
            sm_step = ScalarMappable(
                cmap=step_cmap, norm=Normalize(vmin=0, vmax=max_step)
            )
            sm_step.set_array([])
            cb_step = ax_mock.get_figure().colorbar(sm_step, cax=cbar_step_ax)
            cb_step.set_label("Adaptation Steps", fontsize=9)
            cb_step.ax.tick_params(labelsize=8)

    # ── CWL rows ──────────────────────────────────────────────────────────────
    if cwl_axes is not None:
        cwl_axes[0].set_title(
            "Avg. CWL estimate",
            fontsize=10,
            fontweight="bold",
            loc="left",
            pad=2,
        )
        for t_idx, ax in enumerate(cwl_axes[:-1]):
            mean_tr, _ = _mean_of(cwl_traces, t_idx)
            _draw_row(ax, mean_tr, cwl_cmap, 0.0, 2.0, f"Trial {t_idx + 1}")

        all_cwl_flat = [t for traces in cwl_traces.values() for t in traces]
        global_cwl = (
            np.nanmean(np.array(all_cwl_flat, dtype=float), axis=0)
            if all_cwl_flat
            else np.full(n_pts, np.nan)
        )
        _add_full_width_bg(cwl_axes[-1], _ALL_ROW_BG, _ALL_ROW_BG_ALPHA)
        _draw_row(
            cwl_axes[-1],
            global_cwl,
            cwl_cmap,
            0.0,
            2.0,
            "All",
            show_xticks=True,
            v_pad=_ALL_V_PAD,
        )
        for spine in cwl_axes[-1].spines.values():
            spine.set_linewidth(1.4)

        if cbar_cwl_ax is not None:
            sm_cwl = ScalarMappable(cmap=cwl_cmap, norm=Normalize(vmin=0, vmax=2))
            sm_cwl.set_array([])
            cb_cwl = ax_mock.get_figure().colorbar(sm_cwl, cax=cbar_cwl_ax)
            cb_cwl.set_ticks([0, 1, 2])
            cb_cwl.set_label("Cwl Level", fontsize=9)
            cb_cwl.set_ticklabels(["Low", "Med", "High"])
            cb_cwl.ax.tick_params(labelsize=8)

    # Suppress tick labels on all rows that don't have show_xticks=True.
    # set_xticks() on the last row propagates tick *positions* to all sharex
    # axes — explicitly hide them everywhere except the intentional bottom row.
    ax_mock.tick_params(bottom=False, labelbottom=False)
    if step_axes is not None:
        rows_to_hide = step_axes[:-1] if step_is_last else step_axes
        for ax in rows_to_hide:
            ax.tick_params(labelbottom=False)
    if cwl_axes is not None:
        for ax in cwl_axes[:-1]:
            ax.tick_params(labelbottom=False)


def _run_racing_experiment(
    show: bool,
    output_dir: Path,
    data_dir: Path,
    apply_penalty: bool = False,
    debug: bool = False,
    figures: list[str] | None = None,
):
    _figs = set(figures or ["all"])

    def _want(key: str) -> bool:
        return "all" in _figs or key in _figs

    by_subject = _load_experiment_racing(data_dir)
    if not by_subject:
        print("  No subject folders with racing trials found.")
        return

    groups = {sid: _read_subject_adaptive_flag(data_dir / sid) for sid in by_subject}
    n_total_trials = sum(len(t) for t in by_subject.values())
    print(f"  Loaded {len(by_subject)} subject(s), {n_total_trials} trial(s)")
    print("\n  Group classification (from extra_info.yml):")
    for sid in sorted(by_subject):
        print(f"    {sid}  ->  {GROUP_LABELS[groups[sid]]}")

    df = _build_experiment_metrics(by_subject, groups)
    if df.empty:
        print("  No trial metrics could be computed.")
        return

    _print_experiment_group_summary(df)

    figs: list[tuple[plt.Figure, Path]] = []

    n_subj = df["subject_id"].nunique()
    suptitle = f"Racing Experiment  ({n_subj} subjects, {len(df)} trials)"

    if _want("completion") or _want("completion-points"):
        both = _want("completion") and _want("completion-points")
        nrows = 2 if both else 1
        fig1, axes1 = plt.subplots(nrows, 1, figsize=(12, 5 * nrows), sharex=True)
        if nrows == 1:
            axes1 = [axes1]
        fig1.suptitle(
            f"{suptitle} — Completion Time",
            fontsize=13,
            fontweight="bold",
        )
        ax_idx = 0
        if _want("completion"):
            _plot_per_trial_group_boxplot(
                axes1[ax_idx],
                df,
                "completion_s",
                "Completion time (s)",
                "Distribution",
                better_low=True,
                debug=debug,
            )
            ax_idx += 1
        if _want("completion-points"):
            ax_pts = axes1[ax_idx]
            _plot_per_trial_group_boxplot(
                ax_pts,
                df,
                "completion_s",
                "Completion time (s)",
                "Subjects  (dot size ∝ adapt. step)",
                better_low=True,
                debug=debug,
                show_datapoints=True,
            )
            from matplotlib.colors import LinearSegmentedColormap

            _sc = LinearSegmentedColormap.from_list(
                "adaptive_steps", ["#C6DBEF", "#08519C"], N=256
            )
            _na_color = GROUP_COLORS["non_adaptive"]
            _a_color = GROUP_COLORS["adaptive"]
            _step_legend = [
                plt.scatter(
                    [],
                    [],
                    s=10,
                    color=_sc(0.0),
                    label="Adaptive — step 0",
                    alpha=0.92,
                    edgecolors=_a_color,
                    linewidths=1.0,
                ),
                plt.scatter(
                    [],
                    [],
                    s=45,
                    color=_sc(0.5),
                    label="Adaptive — step ~12",
                    alpha=0.92,
                    edgecolors=_a_color,
                    linewidths=1.0,
                ),
                plt.scatter(
                    [],
                    [],
                    s=80,
                    color=_sc(1.0),
                    label="Adaptive — step 23",
                    alpha=0.92,
                    edgecolors=_a_color,
                    linewidths=1.0,
                ),
                plt.scatter(
                    [],
                    [],
                    s=45,
                    color=_na_color,
                    label="Non-Adaptive",
                    alpha=0.92,
                    edgecolors=_na_color,
                    linewidths=1.0,
                ),
            ]
            step_leg = ax_pts.legend(
                handles=_step_legend,
                title="Adapt. step",
                fontsize=8,
                loc="upper right",
                framealpha=0.9,
                title_fontsize=8,
            )
            ax_pts.add_artist(step_leg)
        fig1.tight_layout(rect=[0, 0, 1, 0.95])
        figs.append((fig1, output_dir / "racing_experiment_completion_time.png"))

    if _want("drones"):
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 5))
        fig2.suptitle(
            f"{suptitle} — Drone Loss & Gate Misses",
            fontsize=13,
            fontweight="bold",
        )
        _plot_per_trial_group_boxplot(
            ax2a,
            df,
            "dead_drones",
            "Dead drones",
            "Dead Drones per Trial",
            better_low=True,
        )
        _plot_per_trial_group_boxplot(
            ax2b,
            df,
            "missed_drones",
            "Σ drones outside gates",
            "Outside Gate Count per Trial",
            better_low=True,
        )
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        figs.append((fig2, output_dir / "racing_experiment_drones.png"))

    if _want("controls"):
        n_ctrl = len(_CONTROL_COLS)
        fig3, axes3 = plt.subplots(
            n_ctrl,
            1,
            figsize=(16, 2.8 * n_ctrl),
            sharex=True,
            gridspec_kw={"hspace": 0.15},
        )
        fig3.suptitle(
            f"{suptitle} — Normalised Control Inputs (mean ± 1 std, per segment)",
            fontsize=13,
            fontweight="bold",
        )
        _plot_experiment_control_inputs(axes3, by_subject, groups)
        axes3[-1].set_xlabel("Segment (normalised progress)", fontsize=9)
        fig3.tight_layout(rect=[0, 0, 1, 0.96])
        figs.append((fig3, output_dir / "racing_experiment_control_inputs.png"))

    if _want("cwl"):
        n_trials = max((len(t) for t in by_subject.values()), default=0)
        for group_key, group_label, fname_suffix in [
            ("adaptive", "Adaptive", "adaptive"),
            ("non_adaptive", "Non-Adaptive", "non_adaptive"),
        ]:
            group_subjects = sorted(sid for sid, g in groups.items() if g == group_key)
            if not group_subjects or n_trials == 0:
                continue
            fig_w = max(14, 3.5 * n_trials)
            fig_h = max(4, 2.2 * len(group_subjects))
            fig_g, axes_g = plt.subplots(
                1,
                n_trials,
                figsize=(fig_w, fig_h),
                sharey=False,
                gridspec_kw={"wspace": 0.05},
            )
            if n_trials == 1:
                axes_g = [axes_g]
            fig_g.suptitle(
                f"Racing Experiment [{group_label}] — CWL Estimate & Limit per Segment",
                fontsize=13,
                fontweight="bold",
            )
            _plot_experiment_segment_cwl_limits(axes_g, by_subject, group_subjects)
            fig_g.tight_layout(rect=[0, 0, 1, 0.95])
            figs.append(
                (
                    fig_g,
                    output_dir
                    / f"racing_experiment_segment_cwl_limits_{fname_suffix}.png",
                )
            )

    if _want("crash"):
        ref_gates = next(
            (
                tr["gates"]
                for trials in by_subject.values()
                for tr in trials
                if tr["gates"] is not None
            ),
            None,
        )
        if ref_gates is not None:
            sorted_gate_ids = _get_sorted_gate_ids(ref_gates)
            a_ids_r, b_ids_r, seg_names_r, seg_diffs_r = _segment_metadata(ref_gates)
            n_segs_r = len(seg_names_r)
            if n_segs_r > 0:
                n_subj_cr = len(by_subject)
                heatmap_h = max(1.2, n_subj_cr * 0.28)
                window_h = (
                    max(1.6, 2 * 0.65)
                    if WINDOW_AVERAGED
                    else max(2.5, n_subj_cr * 0.38)
                )
                fig_cr = plt.figure(
                    figsize=(16, max(14, 3.5 + 2 * heatmap_h + 2 * window_h))
                )
                gs_cr = fig_cr.add_gridspec(
                    5,
                    1,
                    height_ratios=[2, heatmap_h, heatmap_h, window_h, window_h],
                    hspace=0.62,
                )
                ax_mock = fig_cr.add_subplot(gs_cr[0])
                ax_heat = fig_cr.add_subplot(gs_cr[1], sharex=ax_mock)
                ax_outside = fig_cr.add_subplot(gs_cr[2], sharex=ax_mock)
                ax_cwl = fig_cr.add_subplot(gs_cr[3])
                ax_stp = fig_cr.add_subplot(gs_cr[4])
                fig_cr.suptitle(
                    f"{suptitle} — Crash Analysis",
                    fontsize=13,
                    fontweight="bold",
                )
                _build_mock_course_ax(ax_mock, seg_names_r, seg_diffs_r)
                _plot_crash_heatmap(
                    ax_heat,
                    by_subject,
                    groups,
                    sorted_gate_ids,
                    n_segs_r,
                    seg_diffs_r,
                    seg_names_r,
                    metric="crashes",
                    averaged=False,
                    cmap_name=CRASH_HEATMAP_CMAP,
                    show_xlabels=False,
                    show_legend=True,
                )
                _plot_crash_heatmap(
                    ax_outside,
                    by_subject,
                    groups,
                    sorted_gate_ids,
                    n_segs_r,
                    seg_diffs_r,
                    seg_names_r,
                    metric="outside",
                    averaged=False,
                    cmap_name=OUTSIDE_HEATMAP_CMAP,
                    show_xlabels=True,
                )
                ax_mock.tick_params(bottom=False, labelbottom=False)
                _plot_crash_window_heatmap(
                    ax_cwl,
                    ax_stp,
                    by_subject,
                    groups,
                    delta=WINDOW_USE_DELTA,
                    averaged=WINDOW_AVERAGED,
                    cwl_cmap_name=WINDOW_CWL_CMAP,
                    step_cmap_name=WINDOW_STEP_CMAP,
                )
                fig_cr.tight_layout(rect=[0, 0, 0.96, 0.97])
                figs.append(
                    (
                        fig_cr,
                        output_dir / "racing_experiment_crash_analysis.png",
                    )
                )

    if _want("profile"):
        n_trials_max = max(len(t) for t in by_subject.values())
        n_step_rows = n_trials_max + 1  # per trial + global
        n_cwl_rows = n_trials_max + 1

        # Row layout: mock | step rows | spacer | cwl rows
        mock_h, row_h, spacer_h = 1.6, 0.72, 0.4
        # 2-column grid: col 0 = data (wide), col 1 = shared colorbar strip.
        # Mock course is in col 0 only so it has the same width as the data rows
        # — colorbars don't steal space from any axes.
        n_data_rows = n_step_rows + 1 + n_cwl_rows  # +1 for spacer
        n_rows = 1 + n_data_rows
        height_ratios = (
            [mock_h] + [row_h] * n_step_rows + [spacer_h] + [row_h] * n_cwl_rows
        )
        fig_h = mock_h + row_h * (n_step_rows + n_cwl_rows) + spacer_h + 0.7
        spacer_row = 1 + n_step_rows  # index of the blank spacer row
        cwl_row0 = spacer_row + 1  # first CWL data row index

        for cmap_name in _PROFILE_STEP_CMAPS:
            fig_p = plt.figure(figsize=(14, fig_h))
            gs_p = fig_p.add_gridspec(
                n_rows,
                2,
                height_ratios=height_ratios,
                width_ratios=[35, 1],
                hspace=0.06,
                wspace=0.03,
            )
            # Mock course: col 0 only → same width as data rows
            ax_mock_p = fig_p.add_subplot(gs_p[0, 0])
            # Step data rows
            step_axes_p = [
                fig_p.add_subplot(gs_p[1 + i, 0], sharex=ax_mock_p)
                for i in range(n_step_rows)
            ]
            # CWL data rows (skip spacer_row)
            cwl_axes_p = [
                fig_p.add_subplot(gs_p[cwl_row0 + i, 0], sharex=ax_mock_p)
                for i in range(n_cwl_rows)
            ]
            # Dedicated colorbar axes — span their section's rows in col 1
            cbar_step_ax = fig_p.add_subplot(gs_p[1 : 1 + n_step_rows, 1])
            cbar_cwl_ax = fig_p.add_subplot(gs_p[cwl_row0 : cwl_row0 + n_cwl_rows, 1])

            fig_p.suptitle(
                f"{suptitle} — Adaptive Profile  [{cmap_name}]",
                fontsize=12,
                fontweight="bold",
            )
            _plot_adaptive_profile_summary(
                ax_mock_p,
                step_axes_p,
                cwl_axes_p,
                cbar_step_ax,
                cbar_cwl_ax,
                by_subject,
                groups,
                step_cmap_name=cmap_name,
                cwl_cmap_name=cmap_name,
            )
            fig_p.tight_layout(rect=[0.07, 0.01, 1.0, 0.96])
            figs.append(
                (
                    fig_p,
                    output_dir / f"racing_experiment_adaptive_profile_{cmap_name}.png",
                )
            )

    _save_or_show(figs, show)
