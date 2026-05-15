#!/usr/bin/env python3
"""Generate diagnostic metrics and plots for racing gate experiment comparison."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_SERVICE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_SERVICE_ROOT / "src"))

from workload_inference.plot_results import (
    _load_experiment_racing,
    _classify_subject_adaptation,
    _build_experiment_metrics_enhanced,
    _plot_raw_distributions,
    _plot_correlation_scatter,
    _plot_penalized_time_distribution,
    _plot_learning_curve,
    _plot_gate_level_heatmap,
    GROUP_COLORS,
    GROUP_LABELS,
)


def run_diagnostic_metrics(data_dir: Path, output_dir: Path, show: bool = False):
    """Load racing experiment and generate comprehensive diagnostic plots."""
    print(f"\nLoading racing experiment from {data_dir}")
    by_subject = _load_experiment_racing(data_dir)
    if not by_subject:
        print("  No subject folders with racing trials found.")
        return

    groups = {
        sid: _classify_subject_adaptation(trials)
        for sid, trials in by_subject.items()
    }
    n_total_trials = sum(len(t) for t in by_subject.values())
    print(f"  Loaded {len(by_subject)} subject(s), {n_total_trials} trial(s)")
    print("\n  Group classification:")
    for sid in sorted(by_subject):
        print(f"    {sid}  ->  {GROUP_LABELS[groups[sid]]}")

    # Build metrics with default penalty weights
    df = _build_experiment_metrics_enhanced(
        by_subject, groups,
        p_crash=1.0,
        w_dead=1.0,
        p_outside=1.0,
        p_skip=2.0,
    )
    if df.empty:
        print("  No trial metrics could be computed.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    figs = []

    # Figure 1: Raw distributions (5 subplots)
    print("\n  Generating raw distributions plot...")
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle(
        f"Racing Experiment — Raw Metric Distributions "
        f"({df['subject_id'].nunique()} subjects, {len(df)} trials)",
        fontsize=14, fontweight="bold",
    )
    axes_flat = axes1.flatten()
    _plot_raw_distributions(axes_flat[:5], df)
    axes_flat[-1].axis("off")
    fig1.tight_layout()
    figs.append((fig1, output_dir / "01_racing_raw_distributions.png"))

    # Figure 2: Correlation matrix (6 subplots)
    print("  Generating correlation scatter matrix...")
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle(
        "Racing Experiment — Metric Correlations",
        fontsize=14, fontweight="bold",
    )
    _plot_correlation_scatter(axes2, df)
    fig2.tight_layout()
    figs.append((fig2, output_dir / "02_racing_correlation_scatter.png"))

    # Figure 3: Penalized time distribution
    print("  Generating penalized time distribution...")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.suptitle(
        "Racing Experiment — Penalized Time Distribution",
        fontsize=14, fontweight="bold",
    )
    _plot_penalized_time_distribution(ax3, df)
    fig3.tight_layout()
    figs.append((fig3, output_dir / "03_racing_penalized_time_dist.png"))

    # Figure 4: Learning curve
    print("  Generating learning curve...")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    fig4.suptitle(
        "Racing Experiment — Learning Curve (Penalized Time)",
        fontsize=14, fontweight="bold",
    )
    _plot_learning_curve(ax4, df)
    fig4.tight_layout()
    figs.append((fig4, output_dir / "04_racing_learning_curve.png"))

    # Figure 5: Gate-level heatmap
    print("  Generating gate-level heatmap...")
    fig5, ax5 = plt.subplots(figsize=(14, 5))
    fig5.suptitle(
        "Racing Experiment — Pass Rate per Gate",
        fontsize=14, fontweight="bold",
    )
    _plot_gate_level_heatmap(ax5, by_subject, groups)
    fig5.tight_layout()
    figs.append((fig5, output_dir / "05_racing_gate_heatmap.png"))

    # Figure 6: Bar plot of mean metrics by group
    print("  Generating group comparison bar plot...")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    fig6.suptitle(
        "Racing Experiment — Mean Metrics by Group",
        fontsize=14, fontweight="bold",
    )

    # Bar plot: mean metrics by group
    metrics_to_plot = [
        ("completion_s", "Completion\nTime (s)"),
        ("n_crashes", "# Crashes"),
        ("sum_dead_at_gates", "Σ Dead Drones\n@ Gates"),
        ("sum_outside_at_gates", "Σ Drones\nOutside Gates"),
    ]
    x_pos = np.arange(len(metrics_to_plot))
    width = 0.35

    for i, group in enumerate(["adaptive", "non_adaptive"]):
        sub = df[df["group"] == group]
        means = []
        for col, _ in metrics_to_plot:
            val = sub[col].mean()
            means.append(val)

        ax6.bar(
            x_pos + i * width,
            means,
            width,
            label=GROUP_LABELS[group],
            color=GROUP_COLORS[group],
            alpha=0.7,
        )

    ax6.set_ylabel("Mean Value")
    ax6.set_title("Mean Metrics by Group")
    ax6.set_xticks(x_pos + width / 2)
    ax6.set_xticklabels([label for _, label in metrics_to_plot], fontsize=9)
    ax6.legend(fontsize=9)
    ax6.grid(axis="y", alpha=0.3)

    fig6.tight_layout()
    figs.append((fig6, output_dir / "06_racing_metrics_comparison.png"))

    # Save all figures
    for fig, fpath in figs:
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"    Saved {fpath.name}")
        if not show:
            plt.close(fig)

    # Print summary statistics
    print("\n  Summary Statistics by Group:")
    print(f"  {'Adaptive':<20} | {'Control (Non-Adaptive)':<20}")
    print("  " + "-" * 45)

    for col, label in [
        ("completion_s", "Completion Time (s)"),
        ("n_crashes", "# Crashes"),
        ("penalized_s", "Penalized Time (s)"),
    ]:
        for group in ["adaptive", "non_adaptive"]:
            sub = df[df["group"] == group][col]
            mean_v = sub.mean()
            std_v = sub.std()
            print(f"  {label} ({group}): {mean_v:.2f} ± {std_v:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic metrics and plots for racing gate experiment."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "experiments" / "experiment_racing_gates",
        help="Experiment root directory or racing gates experiment folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "results",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )

    args = parser.parse_args()
    run_diagnostic_metrics(args.data, args.output, args.show)


if __name__ == "__main__":
    main()
