#!/usr/bin/env python3
"""Quick diagnostic analysis for racing gate experiment metrics."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_SERVICE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_SERVICE_ROOT / "src"))

from workload_inference.plot_results._common import (
    _load_experiment_racing,
    GROUP_COLORS,
    GROUP_LABELS,
)
from workload_inference.plot_results._experiment import (
    _read_subject_adaptive_flag,
    _build_experiment_metrics_enhanced,
)


def main():
    parser = argparse.ArgumentParser(
        description="Quick diagnostic metrics for racing gate experiment."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "experiments" / "experiment_racing_gates",
        help="Racing gates experiment folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "results",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    print(f"\nLoading racing experiment from {args.data}")
    by_subject = _load_experiment_racing(args.data)
    if not by_subject:
        print("  No trials found!")
        return

    groups = {
        sid: _read_subject_adaptive_flag(args.data / sid)
        for sid in by_subject
    }
    n_total_trials = sum(len(t) for t in by_subject.values())
    print(f"  Loaded {len(by_subject)} subject(s), {n_total_trials} trial(s)")

    # Build metrics once with default weights
    print("\n  Computing enhanced metrics...")
    df = _build_experiment_metrics_enhanced(by_subject, groups)
    if df.empty:
        print("  No metrics computed!")
        return

    print(f"  Built metrics for {len(df)} trials")

    # Quick summary statistics
    print("\n  ========================================================")
    print("  SUMMARY STATISTICS BY GROUP")
    print("  ========================================================")

    for group in ["adaptive", "non_adaptive"]:
        sub = df[df["group"] == group]
        n = len(sub)
        print(f"\n  {GROUP_LABELS[group]} (n={n} trials):")
        print(f"    Completion Time:     {sub['completion_s'].mean():7.1f} +- {sub['completion_s'].std():5.1f} s")
        print(f"    # Crashes:           {sub['n_crashes'].mean():7.2f} +- {sub['n_crashes'].std():5.2f}")
        print(f"    Sum Dead @ Gates:    {sub['sum_dead_at_gates'].mean():7.1f} +- {sub['sum_dead_at_gates'].std():5.1f}")
        print(f"    Sum Outside Gates:   {sub['sum_outside_at_gates'].mean():7.1f} +- {sub['sum_outside_at_gates'].std():5.1f}")
        print(f"    Penalized Time:      {sub['penalized_s'].mean():7.1f} +- {sub['penalized_s'].std():5.1f} s")

    # Effect sizes
    print("\n  ========================================================")
    print("  EFFECT SIZES (Cohen's d)")
    print("  ========================================================")

    adaptive = df[df["group"] == "adaptive"]
    control = df[df["group"] == "non_adaptive"]

    for col, label in [
        ("completion_s", "Completion Time"),
        ("n_crashes", "# Crashes"),
        ("penalized_s", "Penalized Time"),
    ]:
        a_vals = adaptive[col].dropna()
        c_vals = control[col].dropna()
        if len(a_vals) > 0 and len(c_vals) > 0:
            d = (a_vals.mean() - c_vals.mean()) / np.sqrt((a_vals.var() + c_vals.var()) / 2)
            t_stat, p_val = sp_stats.ttest_ind(a_vals, c_vals)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {label:20s}:  d={d:6.2f}  p={p_val:.4f}  {sig}")

    # Create minimal diagnostic plots
    args.output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Racing Experiment Diagnostics ({len(by_subject)} subjects, {len(df)} trials)",
        fontsize=14, fontweight="bold",
    )

    # Plot 1: Completion time
    for group in ["adaptive", "non_adaptive"]:
        sub = df[df["group"] == group]["completion_s"]
        axes[0, 0].hist(
            sub,
            alpha=0.6,
            bins=10,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
        )
    axes[0, 0].set_xlabel("Completion Time (s)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Completion Time Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Crashes
    for group in ["adaptive", "non_adaptive"]:
        sub = df[df["group"] == group]["n_crashes"]
        axes[0, 1].hist(
            sub,
            alpha=0.6,
            bins=10,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
        )
    axes[0, 1].set_xlabel("# Crashes")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Crash Count Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Scatter completion vs crashes
    for group in ["adaptive", "non_adaptive"]:
        sub = df[df["group"] == group]
        axes[1, 0].scatter(
            sub["n_crashes"],
            sub["completion_s"],
            s=60,
            alpha=0.6,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
        )
    axes[1, 0].set_xlabel("# Crashes")
    axes[1, 0].set_ylabel("Completion Time (s)")
    axes[1, 0].set_title("Completion Time vs Crashes")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Learning curve (trial by trial)
    for group in ["adaptive", "non_adaptive"]:
        sub = df[df["group"] == group]
        trial_nums = sorted(sub["trial_num"].unique())
        means = []
        for tn in trial_nums:
            means.append(sub[sub["trial_num"] == tn]["completion_s"].mean())
        axes[1, 1].plot(
            trial_nums,
            means,
            marker="o",
            linewidth=2,
            markersize=8,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
        )
    axes[1, 1].set_xlabel("Trial Number")
    axes[1, 1].set_ylabel("Mean Completion Time (s)")
    axes[1, 1].set_title("Learning Curve")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    outfile = args.output / "racing_metrics_diagnostic.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n  Saved plot to: {outfile.name}")
    plt.close()

    print("\n  Done!\n")


if __name__ == "__main__":
    main()
