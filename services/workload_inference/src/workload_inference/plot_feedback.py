"""Plot subject durations and feedback score distributions for an experiment.

Reads extra_info.yaml and feedback_*.csv files from each subject folder under
a given experiment directory and produces two output figures:
1. Subject durations (bar chart, colored by adaptive vs control group)
2. Feedback distribution (per-trial box plots + group comparison violins)

Usage
-----
    plot_feedback [--data DIR] [--output DIR] [--experiment NAME] [--show]
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_SERVICE_ROOT = Path(__file__).parents[2]
_DEFAULT_DATA = _SERVICE_ROOT / "data" / "experiments"
_DEFAULT_OUTPUT = _SERVICE_ROOT / "data" / "results"
_DEFAULT_EXPERIMENT = "experiment_racing_gates"

_SUBJECT_RE = re.compile(r"^[A-Z0-9]{4}$")
_FEEDBACK_GLOB = "feedback_*.csv"

_COLOR_ADAPTIVE = "#FF821B"
_COLOR_CONTROL = "#1976D2"
_COLOR_NEUTRAL = "#607D8B"

_BOUNDARY_LO = -0.33
_BOUNDARY_HI = 0.33
_JITTER_SEED = 42


@dataclasses.dataclass
class SubjectRecord:
    subject_id: str
    duration_sec: float
    adaptive: bool
    feedback: pd.DataFrame | None


def _find_subject_dirs(experiment_dir: Path) -> list[Path]:
    """Return sorted list of subject directories matching _SUBJECT_RE."""
    dirs = [
        d for d in experiment_dir.iterdir() if d.is_dir() and _SUBJECT_RE.match(d.name)
    ]
    dirs.sort(key=lambda d: d.name.lstrip("_"))
    return dirs


def _extract_subject_id(folder: Path) -> str:
    """Return the canonical subject ID (folder name)."""
    return folder.name


def _load_extra_info(subject_dir: Path) -> dict:
    """Load extra_info.yaml, return dict. Warn and return {} on missing."""
    info_file = subject_dir / "extra_info.yaml"
    if not info_file.exists():
        print(f"  Warning: {info_file} not found")
        return {}
    try:
        with open(info_file) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  Warning: Failed to load {info_file}: {e}")
        return {}


def _load_feedback_csvs(subject_dir: Path, subject_id: str) -> pd.DataFrame | None:
    """Glob feedback_*.csv, concat if multiple, return None if none found."""
    files = sorted(subject_dir.glob(_FEEDBACK_GLOB))
    if not files:
        return None
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["subject_id"] = subject_id
    return df


def load_experiment_data(experiment_dir: Path) -> list[SubjectRecord]:
    """Load all subjects from an experiment directory."""
    subject_dirs = _find_subject_dirs(experiment_dir)
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directories found in {experiment_dir}")

    records = []
    for subject_dir in subject_dirs:
        subject_id = _extract_subject_id(subject_dir)
        info = _load_extra_info(subject_dir)

        if not info:
            continue

        duration_sec = info.get("duration_sec")
        if duration_sec is None:
            print(f"  Warning: {subject_id} has no duration_sec")
            continue

        adaptive = bool(info.get("adaptive", True))
        feedback = _load_feedback_csvs(subject_dir, subject_id)

        records.append(
            SubjectRecord(
                subject_id=subject_id,
                duration_sec=duration_sec,
                adaptive=adaptive,
                feedback=feedback,
            )
        )

    records.sort(key=lambda r: r.duration_sec, reverse=True)

    n_with_fb = sum(1 for r in records if r.feedback is not None)
    n_adaptive = sum(1 for r in records if r.adaptive)
    print(
        f"Found {len(records)} subjects ({n_with_fb} with feedback, "
        f"{n_adaptive} adaptive)"
    )

    return records


def plot_subject_durations(records: list[SubjectRecord]) -> plt.Figure:
    """Create horizontal bar chart of subject durations."""
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(records))))

    durations_min = [r.duration_sec / 60 for r in records]
    subject_ids = [r.subject_id for r in records]
    colors = [_COLOR_ADAPTIVE if r.adaptive else _COLOR_CONTROL for r in records]

    ax.barh(
        range(len(records)),
        durations_min,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(records)))
    ax.set_yticklabels(subject_ids, fontsize=9)
    ax.set_xlabel("Duration (minutes)")
    ax.set_title(f"Subject Durations — {len(records)} subjects", fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    adaptive_patch = mpatches.Patch(color=_COLOR_ADAPTIVE, label="Adaptive")
    control_patch = mpatches.Patch(color=_COLOR_CONTROL, label="Control")
    ax.legend(handles=[adaptive_patch, control_patch], loc="lower right")

    fig.tight_layout()
    return fig


def _draw_region_shading(ax: plt.Axes) -> None:
    """Draw horizontal region bands for score interpretation."""
    ax.axhspan(-1.0, _BOUNDARY_LO, color="#F44336", alpha=0.05)
    ax.axhspan(_BOUNDARY_LO, _BOUNDARY_HI, color="#9E9E9E", alpha=0.05)
    ax.axhspan(_BOUNDARY_HI, 1.0, color="#4CAF50", alpha=0.05)

    for y in (_BOUNDARY_LO, 0, _BOUNDARY_HI):
        ax.axhline(y, linestyle="--", linewidth=0.8, color="#9E9E9E", alpha=0.6)


def _style_score_axis(ax: plt.Axes) -> None:
    """Apply consistent y-axis styling for feedback score panels."""
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([-1, -0.33, 0, 0.33, 1])
    ax.set_yticklabels(
        ["-1\n(Not helpful)", "-0.33", "0\n(Neutral)", "+0.33", "+1\n(Helpful)"],
        fontsize=8,
    )
    ax.set_ylabel("Perceived Adaptation Score")
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _build_feedback_df(records: list[SubjectRecord]) -> pd.DataFrame:
    """Concatenate all feedback frames, add adaptive group column."""
    frames = []
    for r in records:
        if r.feedback is not None:
            df = r.feedback.copy()
            df["adaptive"] = r.adaptive
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def plot_feedback_distribution(
    records: list[SubjectRecord], experiment_name: str
) -> plt.Figure:
    """Create feedback distribution figure with per-trial box plots and group comparison."""
    all_fb = _build_feedback_df(records)

    if all_fb.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(
            0.5,
            0.5,
            "No feedback data available",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        fig.suptitle(f"Feedback Distribution", fontweight="bold")
        return fig

    show_groups = any(not r.adaptive for r in records)

    if show_groups:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[2, 1], width_ratios=[1, 1], hspace=0.35, wspace=0.3
        )
        ax_adaptive = fig.add_subplot(gs[0, 0])
        ax_control = fig.add_subplot(gs[0, 1])
        ax_group = fig.add_subplot(gs[1, :])
        ax_main = None
    else:
        fig, ax_main = plt.subplots(figsize=(12, 6))
        ax_adaptive = None
        ax_control = None
        ax_group = None

    np.random.seed(_JITTER_SEED)

    trials = sorted(all_fb["trial"].unique())

    if show_groups:
        for ax, group_name, is_adaptive_group in [
            (ax_adaptive, "Adaptive", True),
            (ax_control, "Control", False),
        ]:
            group_data = all_fb[all_fb["adaptive"] == is_adaptive_group]
            box_data = [
                group_data[group_data["trial"] == t]["normalised_score"].dropna().values
                for t in trials
            ]

            bp = ax.boxplot(
                box_data,
                positions=range(1, len(trials) + 1),
                widths=0.5,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                showfliers=True,
            )

            for patch in bp["boxes"]:
                patch.set_facecolor(
                    _COLOR_ADAPTIVE if is_adaptive_group else _COLOR_CONTROL
                )
                patch.set_alpha(0.4)

            # for i, t in enumerate(trials):
            #     trial_scores = (
            #         group_data[group_data["trial"] == t]["normalised_score"]
            #         .dropna()
            #         .values
            #     )
            #     if len(trial_scores) > 0:
            #         x_pos = i + 1
            #         jitter = np.random.uniform(-0.15, 0.15, size=len(trial_scores))
            #         ax.scatter(
            #             x_pos + jitter,
            #             trial_scores,
            #             s=18,
            #             alpha=0.55,
            #             color=_COLOR_ADAPTIVE if is_adaptive_group else _COLOR_CONTROL,
            #             edgecolors="none",
            #             zorder=3,
            #         )

            _draw_region_shading(ax)
            if ax == ax_adaptive:
                _style_score_axis(ax)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")

            ax.set_xticks(range(1, len(trials) + 1))
            ax.set_xticklabels([f"Trial {t}" for t in trials])
            ax.set_xlabel("Trial")
            ax.set_title(f"{group_name} Group", fontweight="bold")

            n_group_subjects = len(
                all_fb[all_fb["adaptive"] == is_adaptive_group].groupby("subject_id")
            )
            ax.text(
                0.98,
                0.97,
                f"n={n_group_subjects} subjects",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="gray",
            )
    else:
        box_data = [
            all_fb[all_fb["trial"] == t]["normalised_score"].dropna().values
            for t in trials
        ]

        bp = ax_main.boxplot(
            box_data,
            positions=range(1, len(trials) + 1),
            widths=0.5,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            showfliers=False,
        )

        for patch in bp["boxes"]:
            patch.set_facecolor("#90CAF9")
            patch.set_alpha(0.4)

        # for i, t in enumerate(trials):
        #     trial_fb = all_fb[all_fb["trial"] == t]
        #     adaptive_mask = trial_fb["adaptive"]

        #     x_pos = i + 1

        #     adaptive_scores = (
        #         trial_fb[adaptive_mask]["normalised_score"].dropna().values
        #     )
        #     if len(adaptive_scores) > 0:
        #         jitter = np.random.uniform(-0.15, 0.15, size=len(adaptive_scores))
        #         ax_main.scatter(
        #             x_pos + jitter,
        #             adaptive_scores,
        #             s=18,
        #             alpha=0.55,
        #             color=_COLOR_ADAPTIVE,
        #             edgecolors="none",
        #             zorder=3,
        #         )

        #     control_scores = (
        #         trial_fb[~adaptive_mask]["normalised_score"].dropna().values
        #     )
        #     if len(control_scores) > 0:
        #         jitter = np.random.uniform(-0.15, 0.15, size=len(control_scores))
        #         ax_main.scatter(
        #             x_pos + jitter,
        #             control_scores,
        #             s=18,
        #             alpha=0.55,
        #             color=_COLOR_CONTROL,
        #             edgecolors="none",
        #             zorder=3,
        #         )

        _draw_region_shading(ax_main)
        _style_score_axis(ax_main)

        ax_main.set_xticks(range(1, len(trials) + 1))
        ax_main.set_xticklabels([f"Trial {t}" for t in trials])
        ax_main.set_xlabel("Trial")
        ax_main.set_title("Feedback Score Distribution by Trial", fontweight="bold")

        n_subjects_with_fb = sum(1 for r in records if r.feedback is not None)
        ax_main.text(
            0.98,
            0.97,
            f"n={n_subjects_with_fb} subjects",
            transform=ax_main.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="gray",
        )

    if ax_group is not None:
        np.random.seed(_JITTER_SEED)

        adaptive_scores = all_fb[all_fb["adaptive"]]["normalised_score"].dropna().values
        control_scores = all_fb[~all_fb["adaptive"]]["normalised_score"].dropna().values

        group_positions = [1, 2]
        group_data = [adaptive_scores, control_scores]
        group_labels = ["Adaptive", "Control"]
        group_colors = [_COLOR_ADAPTIVE, _COLOR_CONTROL]

        non_empty_pos = [p for p, d in zip(group_positions, group_data) if len(d) > 0]
        non_empty_data = [d for d in group_data if len(d) > 0]

        if non_empty_data:
            parts = ax_group.violinplot(
                non_empty_data,
                positions=non_empty_pos,
                widths=0.5,
                showmedians=True,
                showextrema=True,
            )

            for pc, color in zip(
                parts["bodies"],
                [c for c, d in zip(group_colors, group_data) if len(d) > 0],
            ):
                pc.set_facecolor(color)
                pc.set_alpha(0.4)

        for pos, scores, color in zip(group_positions, group_data, group_colors):
            if len(scores) == 0:
                continue
            jitter = np.random.uniform(-0.08, 0.08, size=len(scores))
            ax_group.scatter(
                pos + jitter,
                scores,
                s=18,
                alpha=0.6,
                color=color,
                edgecolors="none",
                zorder=3,
            )

        _draw_region_shading(ax_group)
        _style_score_axis(ax_group)

        ax_group.set_xticks(group_positions)
        ax_group.set_xticklabels(
            [
                f"Adaptive\n(n={len(adaptive_scores)})",
                f"Control\n(n={len(control_scores)})",
            ]
        )
        ax_group.set_title("Score Distribution by Group", fontweight="bold")

    fig.suptitle(
        f"Feedback Distribution — {experiment_name}", fontsize=13, fontweight="bold"
    )

    return fig


def _save_or_show(figs: list[tuple[plt.Figure, Path]], show: bool) -> None:
    """Save figures and optionally show them."""
    for fig, path in figs:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show:
        plt.show()


def run_feedback(data_dir: Path, output_dir: Path, experiment: str, show: bool) -> None:
    """Main pipeline: load data, generate figures, save/show."""
    experiment_dir = data_dir / experiment
    print(f"\nLoading feedback data from: {experiment_dir}")

    records = load_experiment_data(experiment_dir)

    output_subdir = output_dir / "feedback"
    output_subdir.mkdir(parents=True, exist_ok=True)

    figs: list[tuple[plt.Figure, Path]] = [
        (plot_subject_durations(records), output_subdir / "subject_durations.svg"),
        (
            plot_feedback_distribution(records, experiment),
            output_subdir / "feedback_distribution.svg",
        ),
    ]

    _save_or_show(figs, show)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_DATA,
        metavar="DIR",
        help="Root experiments folder (default: data/experiments/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="DIR",
        help="Root results folder (default: data/results/).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=_DEFAULT_EXPERIMENT,
        metavar="NAME",
        help="Experiment subfolder name (default: experiment_racing_gates).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Open an interactive matplotlib window after saving.",
    )
    args = parser.parse_args()
    run_feedback(
        data_dir=args.data,
        output_dir=args.output,
        experiment=args.experiment,
        show=args.show,
    )


if __name__ == "__main__":
    main()
