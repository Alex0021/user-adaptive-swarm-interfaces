"""Plot and visualize workload inference experiment results.

Usage:
    plot_results racing  [--show] [--data DIR] [--output DIR] [--type {inference,adaptive}] [--no-penalty]
    plot_results inference [--show] [--data DIR] [--output DIR] [--cwl {0,1,2}]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ._common import (
    _DEFAULT_DATA,
    _DEFAULT_OUTPUT,
    STATE_LABELS,
    _detect_mode,
    _detect_racing_mode,
    _draw_spline_background,
    _load_spline,
    _save_or_show,
    _task_trials_only,
    load_inference_data,
    plot_inference_time_series,
)
from ._experiment import (
    _collect_merged_frames,
    _plot_aggregate_task_trajectory,
    _plot_spline_accuracy_ribbon,
    _run_racing_experiment,
    plot_inference_accuracy_summary,
)
from ._subject import (
    _plot_subject_task_trajectory,
    _run_racing_subject,
    plot_subject_accuracy_summary,
)
from ._trial import _make_time_series_figure, _run_racing_trial


def run_inference(
    show: bool,
    output_dir: Path,
    data_dir: Path,
    cwl: int | None = None,
    **_kw,
):
    mode = _detect_mode(data_dir)
    print(f"Loading inference data from: {data_dir}  [{mode} mode]")

    output_dir.mkdir(parents=True, exist_ok=True)
    subject = data_dir.name
    figs: list[tuple[plt.Figure, Path]] = []

    # ── CWL trajectory / ribbon mode ────────────────────────────────────────
    if cwl is not None:
        spline_df = _load_spline()
        if spline_df is None:
            print("  Cannot produce trajectory plots without spline_trajectory.csv")
            return

        cwl_label = STATE_LABELS.get(cwl, str(cwl)).lower()

        if mode == "subject":
            fig, (ax_traj, ax_acc) = plt.subplots(
                1,
                2,
                figsize=(16, 7),
                gridspec_kw={"width_ratios": [2, 1], "wspace": 0.25},
            )
            _plot_subject_task_trajectory(data_dir, cwl, spline_df, ax_traj, ax_acc)
            fig.tight_layout()
            figs.append((fig, output_dir / f"trajectory_cwl_{cwl_label}.png"))

        elif mode == "experiment":
            fig, (ax_traj, ax_acc) = plt.subplots(
                1,
                2,
                figsize=(16, 7),
                gridspec_kw={"width_ratios": [2, 1], "wspace": 0.25},
            )
            _plot_aggregate_task_trajectory(data_dir, cwl, spline_df, ax_traj, ax_acc)
            fig.tight_layout()
            figs.append((fig, output_dir / f"trajectory_cwl_aggregate_{cwl_label}.png"))

        # Load all CWL data once, then render all 3 levels side-by-side.
        merged_df = _collect_merged_frames(data_dir)
        if merged_df.empty:
            print("  No merged drone+inference data found.")
            _save_or_show(figs, show)
            return

        fig_r, axes = plt.subplots(
            2,
            3,
            figsize=(18, 7),
            gridspec_kw={"height_ratios": [9, 1], "wspace": 0.08},
        )
        for col, cwl_plot in enumerate((0, 1, 2)):
            ax_r = axes[0, col]
            ax_summary = axes[1, col]
            _draw_spline_background(ax_r, spline_df)
            _plot_spline_accuracy_ribbon(
                ax_r,
                spline_df,
                merged_df,
                cwl_plot,
                ax_summary=ax_summary,
                show_legend=(col == 0),
            )
            if col > 0:
                ax_r.set_ylabel("")
                ax_r.tick_params(labelleft=False)
        figs.append((fig_r, output_dir / "trajectory_cwl_ribbon_all.png"))

        _save_or_show(figs, show)
        return

    # ── Standard mode (no --task) ────────────────────────────────────────────
    data = load_inference_data(data_dir)
    n_sources = data["_source"].nunique()
    print(f"  Loaded {len(data)} rows from {n_sources} session(s).")

    if mode == "trial":
        fig = _make_time_series_figure(data, f"Workload Inference — {subject}")
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
                fontsize=13,
                fontweight="bold",
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
            fontsize=13,
            fontweight="bold",
        )
        plot_inference_accuracy_summary(data, ax_overall, ax_per_class)
        fig2.tight_layout()
        figs.append((fig2, output_dir / "inference_accuracy_summary.png"))

    _save_or_show(figs, show)


def run_racing(
    show: bool,
    output_dir: Path,
    data_dir: Path,
    traj_type: str = "inference",
    apply_penalty: bool = False,
    debug: bool = False,
    figures: list[str] | None = None,
    **_kw,
):
    racing_mode = _detect_racing_mode(data_dir)
    print(f"Loading racing data from: {data_dir}  [{racing_mode} mode]")
    output_dir.mkdir(parents=True, exist_ok=True)
    if racing_mode == "experiment":
        _run_racing_experiment(
            show,
            output_dir,
            data_dir,
            apply_penalty=apply_penalty,
            debug=debug,
            figures=figures or ["all"],
        )
    elif racing_mode == "subject":
        _run_racing_subject(show, output_dir, data_dir, traj_type)
    else:
        _run_racing_trial(show, output_dir, data_dir, traj_type)


RESULT_TYPES = {
    "inference": run_inference,
    "racing": run_racing,
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
        "--cwl",
        type=int,
        default=None,
        choices=[0, 1, 2],
        metavar="CWL",
        help="CWL level to visualize as a trajectory plot: 0=Low, 1=Medium, "
        "2=High.  The corresponding task is resolved automatically per subject.",
    )
    parser.add_argument(
        "--type",
        dest="traj_type",
        default="inference",
        choices=["inference", "adaptive"],
        help="Trajectory colormap for racing plot: 'inference' colors by CWL state "
        "(green/orange/red), 'adaptive' colors by adaptation step "
        "(red=Soft/0 -> green=Racing/max).",
    )
    parser.add_argument(
        "--with-penalty",
        dest="apply_penalty",
        action="store_true",
        default=False,
        help="Apply the dead/miss penalty algorithm to completion times.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Show extra diagnostic info (e.g. raw p-values on all trials in completion time plot).",
    )
    _FIGURE_CHOICES = [
        "all",
        "completion",  # completion time boxplot
        "completion-points",  # completion time + per-subject dots sized by adapt. step
        "drones",  # dead drones & outside-gate counts
        "controls",  # normalised control inputs per segment
        "cwl",  # CWL estimate + limits per segment (adaptive & non-adaptive)
        "crash",  # crash analysis: course schematic + heatmap + CWL window
        "profile",  # adaptive profile summary: mock course + per-trial bars
    ]
    parser.add_argument(
        "--figure",
        dest="figures",
        nargs="+",
        choices=_FIGURE_CHOICES,
        default=["all"],
        metavar="FIG",
        help=(
            "Figure(s) to generate. Choices: "
            + ", ".join(_FIGURE_CHOICES[1:])
            + ". Default: all."
        ),
    )

    args = parser.parse_args()
    RESULT_TYPES[args.result_type](
        show=args.show,
        output_dir=args.output,
        data_dir=args.data,
        cwl=args.cwl,
        traj_type=args.traj_type,
        apply_penalty=args.apply_penalty,
        debug=args.debug,
        figures=args.figures,
    )


if __name__ == "__main__":
    main()
