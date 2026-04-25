"""Analyse the average and median CWL step across an experiment.

Reads all command_data.csv files found under a given experiment folder,
extracts cwl_current_step and cwl_total_steps, and computes aggregate statistics.

The intended workflow:
  1. Run this script to get average/median step numbers.
  2. Manually define limit constants in the FLIGHT_PROFILE_LIMITS dict below.
  3. Use those constants to configure control-group experiments without real-time
     CWL adaptation.

Usage
-----
    plot_command_limits [--data DIR] [--output DIR] [--show]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Max limits at step N-1 (defined manually because of bugs in recorded limit columns).
FLIGHT_PROFILE_LIMITS: dict[str, float] = {
    "max_pitch": 0.45,
    "max_roll": 0.45,
    "max_yaw_rate": 1.5,
    "max_speed": 15.0,
    "max_altitude_rate": 5.0,
    "max_alpha": 12.0,
}

# Min limits at step 0. If left empty, step 0 is assumed to map to 0 for all params.
FLIGHT_PROFILE_MIN_LIMITS: dict[str, float] = {
    "max_pitch": 0.2,
    "max_roll": 0.2,
    "max_yaw_rate": 0.6,
    "max_speed": 3.0,
    "max_altitude_rate": 1.5,
    "max_alpha": 5.0,
}

_SERVICE_ROOT = Path(__file__).parents[2]
_DEFAULT_DATA = _SERVICE_ROOT / "data" / "experiments"
_DEFAULT_OUTPUT = _SERVICE_ROOT / "data" / "results"

COMMAND_FILE_NAME = "command_data.csv"
GATE_STATUS_FILE_NAME = "gate_status.csv"

_SUBJECT_RE = re.compile(r"^[A-Z0-9]{4}$")
_TASK_RE = re.compile(r"^task_\d+$")
_TRIAL_RE = re.compile(r"^trial_\d+$")

_REQUIRED_COLS = frozenset({"cwl_current_step", "cwl_total_steps"})


def _detect_mode(data_dir: Path) -> str:
    """Return 'trial', 'subject', or 'experiment'."""
    if (data_dir / COMMAND_FILE_NAME).exists():
        return "trial"
    if _SUBJECT_RE.match(data_dir.name):
        return "subject"
    return "experiment"


def _find_command_files(data_dir: Path) -> list[Path]:
    """Return all command_data.csv files under data_dir, sorted by path."""
    return sorted(data_dir.rglob(COMMAND_FILE_NAME))


def _parse_metadata(csv_path: Path, data_dir: Path) -> dict[str, str]:
    """Extract subject_id, task, and trial from a csv path relative to data_dir."""
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


def _first_gate_timestamp(trial_dir: Path) -> int | None:
    """Return the earliest first_pass_timestamp from gate_status.csv, or None."""
    gate_file = trial_dir / GATE_STATUS_FILE_NAME
    if not gate_file.exists():
        return None
    try:
        gs = pd.read_csv(gate_file, usecols=["first_pass_timestamp"])
        valid = gs["first_pass_timestamp"]
        valid = valid[valid > 0]
        if valid.empty:
            return None
        return int(valid.min())
    except Exception:
        return None


def load_step_data(data_dir: Path) -> pd.DataFrame:
    """Discover all command_data.csv files and extract cwl_current_step values.

    For trials that have a gate_status.csv, rows recorded before the first gate
    crossing are excluded — that pre-flight window reflects the default step
    value rather than the user's actual flight behaviour.

    Subjects whose folder name starts with '_' are excluded (bad runs).
    Practice/warmup folders (e.g. FlyingPractice) are excluded.

    Returns a DataFrame with columns: subject_id, task, trial, cwl_current_step,
    cwl_total_steps. Files missing the step columns are skipped with a warning.
    Raises FileNotFoundError if no valid files are found.
    """
    frames: list[pd.DataFrame] = []
    trimmed_count = 0

    for csv_path in _find_command_files(data_dir):
        try:
            df = pd.read_csv(
                csv_path,
                usecols=lambda c: c in (_REQUIRED_COLS | {"timestamp"}),
            )
        except Exception:
            continue

        if not _REQUIRED_COLS.issubset(df.columns):
            rel = (
                csv_path.relative_to(data_dir)
                if data_dir in csv_path.parents
                else csv_path
            )
            print(f"  Skipping (missing step columns): {rel}")
            continue

        if df.empty:
            continue

        meta = _parse_metadata(csv_path, data_dir)
        if not meta["subject_id"] or not meta["task"] or not meta["trial"]:
            # Skip subjects prefixed with '_' (excluded runs), practice/warmup
            # folders (e.g. FlyingPractice), and any path that doesn't resolve
            # to a full subject/task/trial triple.
            continue

        # Trim rows before the first gate crossing if gate_status.csv exists.
        if "timestamp" in df.columns:
            first_gate_ts = _first_gate_timestamp(csv_path.parent)
            if first_gate_ts is not None:
                before = len(df)
                df = df[df["timestamp"] >= first_gate_ts].copy()
                trimmed = before - len(df)
                if trimmed > 0:
                    trimmed_count += trimmed

            df = df.drop(columns=["timestamp"])

        if df.empty:
            continue

        for key, val in meta.items():
            df[key] = val

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No command_data.csv with step columns found under {data_dir}"
        )

    if trimmed_count:
        print(f"  Trimmed {trimmed_count} pre-gate rows (before first gate crossing)")

    return pd.concat(frames, ignore_index=True)


# ── Statistics computation ────────────────────────────────────────────────────


def compute_step_statistics(step_data: pd.DataFrame) -> dict:
    """Compute average and median cwl_current_step across the entire dataset."""
    if step_data.empty:
        return {}

    global_mean = step_data["cwl_current_step"].mean()
    global_median = step_data["cwl_current_step"].median()
    global_min = step_data["cwl_current_step"].min()
    global_max = step_data["cwl_current_step"].max()
    total_steps = step_data["cwl_total_steps"].iloc[0]

    per_subject: dict[str, dict] = {}
    for sid in sorted(step_data["subject_id"].unique()):
        subject_data = step_data[step_data["subject_id"] == sid]
        per_subject[sid] = {
            "mean_step": subject_data["cwl_current_step"].mean(),
            "median_step": subject_data["cwl_current_step"].median(),
            "min_step": subject_data["cwl_current_step"].min(),
            "max_step": subject_data["cwl_current_step"].max(),
            "n_rows": len(subject_data),
            "n_trials": subject_data[["task", "trial"]].drop_duplicates().shape[0],
            "steps": subject_data["cwl_current_step"].to_numpy(),
        }

    return {
        "mean_step": global_mean,
        "median_step": global_median,
        "min_step": global_min,
        "max_step": global_max,
        "total_steps": int(total_steps),
        "n_rows": len(step_data),
        "per_subject": per_subject,
    }


# ── Limit interpolation ────────────────────────────────────────────────────────


def _step_to_limits(step: float, total_steps: int) -> dict[str, float]:
    """Linearly interpolate each limit between step 0 (min) and step N-1 (max).

    Uses FLIGHT_PROFILE_MIN_LIMITS for the step-0 anchor when defined;
    falls back to 0 for any parameter not listed there.
    """
    if not FLIGHT_PROFILE_LIMITS or total_steps <= 1:
        return {}
    ratio = step / (total_steps - 1)
    return {
        param: (
            FLIGHT_PROFILE_MIN_LIMITS.get(param, 0.0)
            + (max_val - FLIGHT_PROFILE_MIN_LIMITS.get(param, 0.0)) * ratio
        )
        for param, max_val in FLIGHT_PROFILE_LIMITS.items()
    }


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_step_distribution(
    step_data: pd.DataFrame,
    stats: dict,
) -> plt.Figure:
    """Two-panel figure: per-subject box plot (left) + overall histogram (right).

    The overall histogram shows the full step distribution with vertical lines for
    the global mean and median. The box plot shows per-subject spread with the
    global mean overlaid as a dashed line.
    """
    total_steps = stats["total_steps"]
    subject_ids = sorted(stats["per_subject"].keys())
    n_subjects = len(subject_ids)
    cmap = plt.get_cmap("tab10")
    colors = {sid: cmap(i % 10) for i, sid in enumerate(subject_ids)}

    fig, (ax_box, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(14, max(5, 1 + n_subjects * 0.55)),
        constrained_layout=True,
    )

    # ── Left: per-subject box plot ────────────────────────────────────────────
    box_data = [stats["per_subject"][sid]["steps"] for sid in subject_ids]
    bp = ax_box.boxplot(
        box_data,
        vert=False,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.3},
    )
    for patch, sid in zip(bp["boxes"], subject_ids, strict=True):
        patch.set_facecolor((*colors[sid][:3], 0.55))

    # Overlay mean dot per subject
    for i, sid in enumerate(subject_ids):
        ax_box.scatter(
            stats["per_subject"][sid]["mean_step"],
            i + 1,
            color=colors[sid],
            zorder=5,
            s=50,
            marker="D",
            edgecolors="black",
            linewidths=0.6,
            label="_nolegend_",
        )

    ax_box.axvline(
        stats["mean_step"],
        color="black",
        linewidth=1.5,
        linestyle="--",
        label=f"Global mean ({stats['mean_step']:.1f})",
    )
    ax_box.axvline(
        stats["median_step"],
        color="dimgray",
        linewidth=1.5,
        linestyle=":",
        label=f"Global median ({stats['median_step']:.1f})",
    )

    ax_box.set_yticks(range(1, n_subjects + 1))
    ax_box.set_yticklabels(subject_ids, fontsize=9)
    ax_box.set_xlabel("CWL step", fontsize=10)
    ax_box.set_xlim(-0.5, total_steps - 0.5)
    ax_box.set_title("Step distribution per subject", fontsize=11, fontweight="bold")
    ax_box.grid(axis="x", linestyle=":", alpha=0.5)
    ax_box.legend(fontsize=8, loc="lower right")

    # Diamond = mean marker legend entry
    ax_box.scatter(
        [],
        [],
        color="gray",
        marker="D",
        s=50,
        edgecolors="black",
        linewidths=0.6,
        label="Subject mean",
    )
    ax_box.legend(fontsize=8, loc="lower right")

    # ── Right: overall histogram ──────────────────────────────────────────────
    all_steps = step_data["cwl_current_step"].values
    bins = np.arange(-0.5, total_steps + 0.5, 1)
    ax_hist.hist(
        all_steps,
        bins=bins,
        color="steelblue",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
    )

    ax_hist.axvline(
        stats["mean_step"],
        color="tomato",
        linewidth=2.0,
        linestyle="--",
        label=f"Mean ({stats['mean_step']:.1f})",
    )
    ax_hist.axvline(
        stats["median_step"],
        color="darkorange",
        linewidth=2.0,
        linestyle=":",
        label=f"Median ({stats['median_step']:.1f})",
    )

    ax_hist.set_xlabel("CWL step", fontsize=10)
    ax_hist.set_ylabel("Row count", fontsize=10)
    ax_hist.set_xlim(-0.5, total_steps - 0.5)
    ax_hist.set_title("Overall step distribution", fontsize=11, fontweight="bold")
    ax_hist.grid(axis="y", linestyle=":", alpha=0.5)
    ax_hist.legend(fontsize=9)

    # Annotate step=0 and step=N-1 fractions
    n_total = len(all_steps)
    pct0 = (all_steps == 0).sum() / n_total * 100
    pct_max = (all_steps == total_steps - 1).sum() / n_total * 100
    ax_hist.text(
        0,
        ax_hist.get_ylim()[1] * 0.95,
        f"{pct0:.1f}%",
        ha="center",
        va="top",
        fontsize=8,
        color="gray",
    )
    ax_hist.text(
        total_steps - 1,
        ax_hist.get_ylim()[1] * 0.95,
        f"{pct_max:.1f}%",
        ha="center",
        va="top",
        fontsize=8,
        color="gray",
    )

    n_subjects_label = step_data["subject_id"].nunique()
    n_trials_label = (
        step_data[["subject_id", "task", "trial"]].drop_duplicates().shape[0]
    )
    fig.suptitle(
        f"CWL Step Distribution — {n_subjects_label} subject(s), "
        f"{n_trials_label} trial(s), {n_total:,} rows",
        fontsize=12,
        fontweight="bold",
    )
    return fig


# ── Save / show ────────────────────────────────────────────────────────────────


def _save_or_show(figs: list[tuple[plt.Figure, Path]], show: bool) -> None:
    for fig, path in figs:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show:
        plt.show()


# ── Orchestration ──────────────────────────────────────────────────────────────


def run_command_limits(
    data_dir: Path,
    output_dir: Path,
    show: bool,
) -> None:
    """Discover data, compute CWL step statistics, print results, and save plots."""
    mode = _detect_mode(data_dir)
    print(f"\nLoading command data from: {data_dir}  [{mode} mode]")

    step_data = load_step_data(data_dir)

    n_subjects = step_data["subject_id"].nunique()
    n_trials = step_data[["subject_id", "task", "trial"]].drop_duplicates().shape[0]
    n_rows = len(step_data)
    print(f"  Found {n_subjects} subject(s), {n_trials} trial(s), {n_rows} data rows\n")

    stats = compute_step_statistics(step_data)
    total_steps = stats["total_steps"]

    # ── Per-subject statistics ─────────────────────────────────────────────────
    print("=" * 70)
    print("PER-SUBJECT CWL STEP STATISTICS")
    print("=" * 70)
    for sid, subj_stats in stats["per_subject"].items():
        n_t = subj_stats["n_trials"]
        n_r = subj_stats["n_rows"]
        print(f"\n  {sid}  ({n_t} trial(s), {n_r} rows):")
        print(f"    Mean step:              {subj_stats['mean_step']:.2f}")
        print(f"    Median step:            {subj_stats['median_step']:.2f}")
        lo = subj_stats["min_step"]
        hi = subj_stats["max_step"]
        print(f"    Step range:             {lo:.0f} — {hi:.0f}")

    # ── Global summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GLOBAL CWL STEP SUMMARY")
    print("=" * 70)
    print(f"  Subjects / trials / rows:       {n_subjects} / {n_trials} / {n_rows}")
    print(f"  Total CWL steps (range 0–N):    {total_steps}")
    print(f"  Mean step   (all rows):         {stats['mean_step']:.2f}")
    print(f"  Median step (all rows):         {stats['median_step']:.2f}")
    g_lo = stats["min_step"]
    g_hi = stats["max_step"]
    print(f"  Observed range:                 {g_lo:.0f} — {g_hi:.0f}")

    # ── Flight profile ─────────────────────────────────────────────────────────
    if not FLIGHT_PROFILE_LIMITS:
        print("\n" + "=" * 70)
        print("FLIGHT PROFILE LIMITS — not yet defined")
        print("=" * 70)
        print("\n  Update FLIGHT_PROFILE_LIMITS at the top of this script with")
        print("  the maximum limit values for each control parameter, then re-run.")
        print("  The script will automatically compute the mean/median setpoints.")
    else:
        mean_limits = _step_to_limits(stats["mean_step"], total_steps)
        median_limits = _step_to_limits(stats["median_step"], total_steps)

        col_w = 22
        print("\n" + "=" * 70)
        print(
            "RECOMMENDED FLIGHT PROFILE"
            "  (linear interpolation from FLIGHT_PROFILE_LIMITS)"
        )
        print("=" * 70)
        mean_s = stats["mean_step"]
        med_s = stats["median_step"]
        max_step = total_steps - 1
        h_mean = f"@ mean ({mean_s:.2f})"
        h_med = f"@ median ({med_s:.2f})"
        h_max = f"Max (step {max_step})"
        print(f"  {'Parameter':{col_w}}  {h_mean:>18}  {h_med:>20}  {h_max:>14}")
        print(
            f"  {'-' * col_w}  {'------------------':>18}"
            f"  {'--------------------':>20}  {'--------------':>14}"
        )
        for param, max_val in FLIGHT_PROFILE_LIMITS.items():
            mean_val = mean_limits.get(param, float("nan"))
            med_val = median_limits.get(param, float("nan"))
            print(
                f"  {param:{col_w}}  {mean_val:>18.4f}"
                f"  {med_val:>20.4f}  {max_val:>14.4f}"
            )

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plot_step_distribution(step_data, stats)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_or_show([(fig, output_dir / "command_limits_step_distribution.png")], show)


# ── CLI ────────────────────────────────────────────────────────────────────────


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
        help="Experiment data folder (trial / subject / experiment mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="DIR",
        help="Directory where the plot PNG is saved (default: data/results/).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Open an interactive matplotlib window after saving.",
    )
    args = parser.parse_args()
    run_command_limits(
        data_dir=args.data,
        output_dir=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
