#!/usr/bin/env python3
"""Distribution and learning-curve plots for racing gate experiment metrics.

Generates four figures saved under data/results/metrics_analysis/:
  fig1_completion_time.png    — completion time strip plot per trial x group (2 col x 5 row)
  fig2_dead_drones.png        — dead-drone count strip plot per trial x group
  fig3_outside_gate.png       — sum of outside-drones per trial, grouped boxplot
  fig4_learning_curve.png     — per-subject learning curves, one per group subplot
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

_SERVICE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_SERVICE_ROOT / "src"))

from workload_inference.plot_results._common import (
    _load_experiment_racing,
    _compute_gate_breakdown,
    _gate_passage_times,
    _count_crashes,
    GROUP_COLORS,
    GROUP_LABELS,
)
from workload_inference.plot_results._experiment import (
    _read_subject_adaptive_flag,
)

# ── Constants ────────────────────────────────────────────────────────────────

_DATA_DIR = _SERVICE_ROOT / "data" / "experiments" / "experiment_racing_gates"
_OUT_DIR = _SERVICE_ROOT / "data" / "results" / "metrics_analysis"

N_TRIALS = 5
SWARM_SIZE = 9
RNG = np.random.default_rng(42)

# Colorblind-safe Okabe-Ito palette (up to 12 subjects per group)
_OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
    "#AA4499", "#44AA99", "#332288", "#DDCC77",
]

_INCOMPLETE_COLOR = "#AAAAAA"
_INCOMPLETE_MARKER = "X"
_COMPLETE_MARKER = "o"


# ── Metric extraction ────────────────────────────────────────────────────────

def _build_trial_row(subject_id: str, trial_num: int, group: str, tr: dict) -> dict | None:
    gates = tr["gates"]
    gs = tr["gate_status"]
    drones = tr["drones"]
    if gates is None or gs is None:
        return None

    passed = _gate_passage_times(gs)
    if len(passed) < 2:
        return None

    raw_elapsed_s = (
        passed["first_pass_timestamp"].max()
        - passed["first_pass_timestamp"].min()
    ) / 1000.0

    gates_reached = int((gs["pass_count"] > 0).sum())
    gates_total = len(gates)

    # A trial is incomplete if the last gate was never reached.
    # Checking pass_count > 0 would wrongly flag trials where a gate was skipped
    # but the course was still finished. Using first_pass_timestamp on the last
    # gate is the correct signal: 0 means no drone ever got there.
    last_gate_id = int(gates["id"].max())
    last_gate_ts = gs.loc[gs["id"] == last_gate_id, "first_pass_timestamp"]
    is_incomplete = last_gate_ts.empty or int(last_gate_ts.iloc[0]) == 0

    breakdown = _compute_gate_breakdown(gates, gs, drones)

    # n_crashes counts alive 1->0 transitions per unique drone ID — always 0-9
    n_crashes = _count_crashes(drones)

    sum_outside = sum(
        b.get("outside", 0) for b in breakdown.values() if b.get("reached")
    )

    return {
        "subject_id": subject_id,
        "trial_num": trial_num,
        "group": group,
        "raw_elapsed_s": raw_elapsed_s,
        "gates_reached": gates_reached,
        "gates_total": gates_total,
        "is_incomplete": is_incomplete,
        "n_crashes": n_crashes,
        "sum_outside": sum_outside,
    }


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    by_subject = _load_experiment_racing(data_dir)
    rows = []
    for sid, trials in by_subject.items():
        group = _read_subject_adaptive_flag(data_dir / sid)
        for i, tr in enumerate(trials):
            row = _build_trial_row(sid, i + 1, group, tr)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


# ── Strip plot helper ────────────────────────────────────────────────────────

def _strip_plot(ax, complete_vals, incomplete_vals, color, jitter_width=0.15):
    """
    Vertical strip plot at x=0.
    Complete trials: filled circles in group color.
    Incomplete trials: X markers in grey.
    Median line across complete values.
    """
    if len(complete_vals) > 0:
        jitter = RNG.uniform(-jitter_width, jitter_width, len(complete_vals))
        ax.scatter(
            jitter, complete_vals,
            color=color, s=55, alpha=0.8,
            marker=_COMPLETE_MARKER, zorder=3,
            edgecolors="white", linewidths=0.5,
        )
        med = np.median(complete_vals)
        ax.axhline(med, color=color, lw=2, ls="--", alpha=0.85, zorder=2)
        ax.text(
            jitter_width + 0.02, med,
            f"median={med:.0f}",
            va="center", ha="left", fontsize=8, color=color,
        )

    if len(incomplete_vals) > 0:
        jitter = RNG.uniform(-jitter_width, jitter_width, len(incomplete_vals))
        ax.scatter(
            jitter, incomplete_vals,
            color=_INCOMPLETE_COLOR, s=70, alpha=0.9,
            marker=_INCOMPLETE_MARKER, zorder=4,
            edgecolors="#666666", linewidths=0.8,
        )


# ── Figure 1 & 2: per-trial strip plot grid ───────────────────────────────────

def _make_per_trial_grid(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    fname: str,
    out_dir: Path,
    ylim: tuple | None = None,
):
    """2-column (Adaptive | Control) x 5-row (Trial 1–5) strip plot grid."""
    group_order = ["adaptive", "non_adaptive"]
    col_titles = [GROUP_LABELS[g] for g in group_order]
    col_colors = [GROUP_COLORS[g] for g in group_order]

    fig, axes = plt.subplots(
        N_TRIALS, 2,
        figsize=(10, 3.0 * N_TRIALS),
        sharex=False,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for trial_num in range(1, N_TRIALS + 1):
        row = trial_num - 1
        for col, (group, color) in enumerate(zip(group_order, col_colors)):
            ax = axes[row, col]
            sub = df[(df["trial_num"] == trial_num) & (df["group"] == group)]

            complete_vals = sub[~sub["is_incomplete"]][value_col].dropna().values
            incomplete_vals = sub[sub["is_incomplete"]][value_col].dropna().values

            _strip_plot(ax, complete_vals, incomplete_vals, color)

            n_complete = len(complete_vals)
            n_incomplete = len(incomplete_vals)

            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_ylabel(ylabel, fontsize=9)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(
                f"{col_titles[col]} — Trial {trial_num}  "
                f"(n={n_complete} complete, {n_incomplete} incomplete)",
                fontsize=9,
            )
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="y", labelsize=8)

    # Legend
    complete_h = mlines.Line2D([], [], color="gray", marker=_COMPLETE_MARKER,
                               ls="None", markersize=8, label="Complete trial")
    incomplete_h = mlines.Line2D([], [], color=_INCOMPLETE_COLOR, marker=_INCOMPLETE_MARKER,
                                 ls="None", markersize=9,
                                 markeredgecolor="#666666", label="Incomplete trial")
    median_h = mlines.Line2D([], [], color="gray", ls="--", lw=2, label="Median (complete)")
    fig.legend(
        handles=[complete_h, incomplete_h, median_h],
        loc="upper right", fontsize=9,
        bbox_to_anchor=(1.02, 1.0),
    )

    fig.tight_layout()
    path = out_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Figure 3: outside-gate box plot ──────────────────────────────────────────

def _make_outside_boxplot(df: pd.DataFrame, out_dir: Path):
    group_order = ["adaptive", "non_adaptive"]
    colors = [GROUP_COLORS[g] for g in group_order]
    labels = [GROUP_LABELS[g] for g in group_order]

    trial_nums = sorted(df["trial_num"].unique())
    n_groups = len(group_order)
    width = 0.32
    gap = 0.06

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "Sum of Drones Outside Gates per Trial",
        fontsize=14, fontweight="bold",
    )

    for g_idx, (group, color) in enumerate(zip(group_order, colors)):
        offsets = []
        data_lists = []
        for t_idx, tn in enumerate(trial_nums):
            sub = df[(df["trial_num"] == tn) & (df["group"] == group)]["sum_outside"]
            x_pos = t_idx + (g_idx - (n_groups - 1) / 2) * (width + gap)
            offsets.append(x_pos)
            data_lists.append(sub.dropna().values)

        ax.boxplot(
            data_lists,
            positions=offsets,
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.55),
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color=color, lw=1.5),
            capprops=dict(color=color, lw=1.5),
            flierprops=dict(marker="o", color=color, alpha=0.4, markersize=5),
            zorder=2,
        )

        for x_pos, vals in zip(offsets, data_lists):
            jitter = RNG.uniform(-0.07, 0.07, len(vals))
            ax.scatter(
                np.full(len(vals), x_pos) + jitter, vals,
                color=color, s=35, alpha=0.75, zorder=3,
                edgecolors="white", linewidths=0.4,
            )

    ax.set_xticks(range(len(trial_nums)))
    ax.set_xticklabels([f"Trial {tn}" for tn in trial_nums], fontsize=11)
    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("Sum of Drones Outside Gates", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    handles = [
        mpatches.Patch(facecolor=colors[i], alpha=0.7, label=labels[i])
        for i in range(n_groups)
    ]
    ax.legend(handles=handles, fontsize=10)
    fig.tight_layout()

    path = out_dir / "fig3_outside_gate_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Figure 4: per-subject learning curves ────────────────────────────────────

def _make_learning_curve(df: pd.DataFrame, out_dir: Path):
    group_order = ["adaptive", "non_adaptive"]
    group_titles = [GROUP_LABELS[g] for g in group_order]
    trial_nums = sorted(df["trial_num"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(
        "Learning Curve — Completion Time per Subject",
        fontsize=14, fontweight="bold",
    )

    for ax, group, title in zip(axes, group_order, group_titles):
        subjects = sorted(df[df["group"] == group]["subject_id"].unique())
        palette = _OKABE_ITO[: len(subjects)]

        for sid, color in zip(subjects, palette):
            sub = (
                df[(df["subject_id"] == sid) & (df["group"] == group)]
                .sort_values("trial_num")
            )

            # Draw segment-by-segment so each segment knows if it touches an incomplete point
            x_all = sub["trial_num"].values
            y_all = sub["raw_elapsed_s"].values
            incomplete_mask = sub["is_incomplete"].values

            for i in range(len(x_all) - 1):
                is_dashed = incomplete_mask[i] or incomplete_mask[i + 1]
                ax.plot(
                    [x_all[i], x_all[i + 1]],
                    [y_all[i], y_all[i + 1]],
                    color=color,
                    lw=2,
                    ls="--" if is_dashed else "-",
                    alpha=0.85 if not is_dashed else 0.5,
                    zorder=2,
                )

            # Plot markers — complete: circle, incomplete: X
            for x, y, inc in zip(x_all, y_all, incomplete_mask):
                ax.scatter(
                    x, y,
                    color=color if not inc else _INCOMPLETE_COLOR,
                    s=70 if not inc else 90,
                    marker=_COMPLETE_MARKER if not inc else _INCOMPLETE_MARKER,
                    zorder=4,
                    edgecolors="white" if not inc else "#666666",
                    linewidths=0.6 if not inc else 1.0,
                    label=sid if (x == x_all[0] and not inc) else "_",
                )

            # Add subject label next to the last point
            last_x, last_y = x_all[-1], y_all[-1]
            ax.text(
                last_x + 0.07, last_y, sid,
                fontsize=7, color=color, va="center",
            )

        ax.set_title(f"{title} (n={len(subjects)} subjects)", fontsize=12)
        ax.set_xlabel("Trial Number", fontsize=11)
        ax.set_ylabel("Completion Time (s)", fontsize=11)
        ax.set_xticks(trial_nums)
        ax.set_xlim(trial_nums[0] - 0.3, trial_nums[-1] + 0.6)
        ax.grid(alpha=0.25)

        # Subject legend via color patches
        handles = [
            mlines.Line2D([], [], color=_OKABE_ITO[i], marker="o", lw=2,
                          markersize=6, label=sid)
            for i, sid in enumerate(subjects)
        ]
        ax.legend(handles=handles, title="Subject", fontsize=7,
                  title_fontsize=8, loc="upper right", ncol=2)

    # Shared marker-type legend at bottom
    complete_h = mlines.Line2D([], [], color="gray", marker=_COMPLETE_MARKER,
                               lw=2, markersize=7, ls="-", label="Complete trial")
    incomplete_h = mlines.Line2D([], [], color=_INCOMPLETE_COLOR,
                                 marker=_INCOMPLETE_MARKER, lw=2, markersize=8,
                                 ls="--", markeredgecolor="#666666",
                                 label="Incomplete trial (dashed segment)")
    fig.legend(
        handles=[complete_h, incomplete_h],
        loc="lower center", ncol=2, fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout()
    path = out_dir / "fig4_learning_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Figure 5: per-subject completion time bar plots ──────────────────────────

def _make_per_subject_bars(df: pd.DataFrame, out_dir: Path):
    """2-column (Adaptive | Control) × N-row (one per subject) bar chart grid.

    Each cell shows completion time per trial (1–5).
    Complete trials: filled bar in group color.
    Incomplete trials: grey hatched bar.
    """
    group_order = ["adaptive", "non_adaptive"]
    col_titles = [GROUP_LABELS[g] for g in group_order]
    col_colors = [GROUP_COLORS[g] for g in group_order]

    subjects_by_group = [
        sorted(df[df["group"] == g]["subject_id"].unique())
        for g in group_order
    ]
    n_rows = max(len(s) for s in subjects_by_group)
    trial_nums = sorted(df["trial_num"].unique())

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(12, 2.6 * n_rows),
        sharex=True,
    )
    # Ensure axes is always 2-D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Completion Time per Subject and Trial",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # Global y-max for shared scale
    y_max = df["raw_elapsed_s"].max() * 1.12

    for col, (group, color, subjects) in enumerate(
        zip(group_order, col_colors, subjects_by_group)
    ):
        for row in range(n_rows):
            ax = axes[row, col]

            if row >= len(subjects):
                ax.axis("off")
                continue

            sid = subjects[row]
            sub = (
                df[(df["subject_id"] == sid) & (df["group"] == group)]
                .sort_values("trial_num")
            )

            bar_colors = [
                _INCOMPLETE_COLOR if inc else color
                for inc in sub["is_incomplete"]
            ]
            hatches = [
                "//" if inc else ""
                for inc in sub["is_incomplete"]
            ]

            bars = ax.bar(
                sub["trial_num"],
                sub["raw_elapsed_s"],
                color=bar_colors,
                edgecolor="white",
                linewidth=0.6,
                width=0.65,
                zorder=2,
            )
            # Apply hatching per bar
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
                bar.set_edgecolor("#888888" if hatch else "white")

            # Value labels on top of each bar
            for bar, val, inc in zip(bars, sub["raw_elapsed_s"], sub["is_incomplete"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + y_max * 0.01,
                    f"{val:.0f}s",
                    ha="center", va="bottom",
                    fontsize=7,
                    color="#555555" if inc else color,
                )

            ax.set_ylim(0, y_max)
            ax.set_xlim(trial_nums[0] - 0.6, trial_nums[-1] + 0.6)
            ax.set_xticks(trial_nums)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_ylabel("Time (s)", fontsize=8)
            ax.grid(axis="y", alpha=0.25, zorder=1)

            # Subject ID + group label in title only on first row
            if row == 0:
                ax.set_title(
                    f"{col_titles[col]}\n{sid}",
                    fontsize=9, fontweight="bold", color=color,
                )
            else:
                ax.set_title(sid, fontsize=9, color=color)

    # X-axis label on the bottom row only
    for col in range(2):
        axes[-1, col].set_xlabel("Trial Number", fontsize=9)

    # Legend
    complete_h = mpatches.Patch(facecolor="gray", label="Complete trial")
    incomplete_h = mpatches.Patch(
        facecolor=_INCOMPLETE_COLOR, hatch="//",
        edgecolor="#888888", label="Incomplete trial",
    )
    fig.legend(
        handles=[complete_h, incomplete_h],
        loc="upper right", fontsize=9,
        bbox_to_anchor=(1.02, 1.0),
    )

    fig.tight_layout()
    path = out_dir / "fig5_per_subject_completion_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nLoading data from {_DATA_DIR}")
    df = build_dataframe(_DATA_DIR)
    print(f"  Built {len(df)} trial rows across {df['subject_id'].nunique()} subjects")

    for g in ["adaptive", "non_adaptive"]:
        sub = df[df["group"] == g]
        print(
            f"  {GROUP_LABELS.get(g, g)}: "
            f"{sub['subject_id'].nunique()} subjects, "
            f"{len(sub)} trials, "
            f"{sub['is_incomplete'].sum()} incomplete"
        )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figures to {_OUT_DIR}\n")

    print("  Fig 1 — completion time strip plots...")
    _make_per_trial_grid(
        df,
        value_col="raw_elapsed_s",
        ylabel="Elapsed Time (s)",
        title="Completion Time per Trial",
        fname="fig1_completion_time.png",
        out_dir=_OUT_DIR,
    )

    print("  Fig 2 — dead drones strip plots...")
    _make_per_trial_grid(
        df,
        value_col="n_crashes",
        ylabel="# Crashed Drones",
        title="Crashed Drones per Trial  (0–9)",
        fname="fig2_dead_drones.png",
        out_dir=_OUT_DIR,
        ylim=(-0.5, 9.5),
    )

    print("  Fig 3 — outside-gate box plots...")
    _make_outside_boxplot(df, _OUT_DIR)

    print("  Fig 4 — per-subject learning curves...")
    _make_learning_curve(df, _OUT_DIR)

    print("  Fig 5 — per-subject completion time bars...")
    _make_per_subject_bars(df, _OUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
