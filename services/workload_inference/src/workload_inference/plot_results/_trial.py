from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._common import (
    _save_or_show, _no_data_placeholder, _load_racing_trials,
    _shade_difficulty_z, _draw_gate_lines_z, _compute_gate_breakdown,
    load_inference_data, _detect_mode, STATE_COLORS, STATE_LABELS,
    INFERENCE_FILE_NAME, DRONE_FILE_NAME, GATE_LAYOUT_FILE, DIFFICULTY_COL,
    GATE_STATUS_FILE, COMMAND_DATA_FILE, SWARM_SIZE,
    RACING_DEAD_PENALTY_S, RACING_MISS_PENALTY_S,
    _find_t0, _gate_passage_times, _compute_centroid, _load_csv,
)

# ── Trial-specific constants ──────────────────────────────────────────────────

COMMAND_INPUTS = {
    "pitch_rate": ("#1976D2", "Pitch"),
    "yaw_rate": ("#E91E63", "Yaw"),
    "roll_rate": ("#FF9800", "Roll"),
    "altitude_rate": ("#4CAF50", "Altitude"),
}

ADAPTATION_PARAMS = {
    "max_speed": ("Max Speed", 3.0, 15.0),
    "max_yaw_rate": ("Max Yaw Rate", 0.6, 1.5),
    "max_pitch": ("Max Pitch", 0.15, 0.4),
    "max_roll": ("Max Roll", 0.15, 0.4),
    "max_ascent_rate": ("Max Ascent", 1.5, 3.0),
    "max_descent_rate": ("Max Descent", 1.5, 2.5),
    "max_altitude_rate": ("Max Altitude Rate", 1.5, 3.0),
    "max_alpha": ("Max Alpha", 6.0, 15.0),
}

_FLIGHT_PROFILE_LIMITS: dict[str, float] = {
    "max_pitch": 0.45,
    "max_roll": 0.45,
    "max_yaw_rate": 1.5,
    "max_speed": 15.0,
    "max_altitude_rate": 5.0,
    "max_alpha": 12.0,
}
_FLIGHT_PROFILE_MIN_LIMITS: dict[str, float] = {
    "max_pitch": 0.2,
    "max_roll": 0.2,
    "max_yaw_rate": 0.6,
    "max_speed": 3.0,
    "max_altitude_rate": 1.5,
    "max_alpha": 5.0,
}
_FLIGHT_PARAM_LABELS: dict[str, tuple[str, str]] = {
    "max_pitch": ("Max Pitch", "rad"),
    "max_roll": ("Max Roll", "rad"),
    "max_yaw_rate": ("Max Yaw Rate", "rad/s"),
    "max_speed": ("Max Speed", "m/s"),
    "max_altitude_rate": ("Max Altitude Rate", "m/s"),
    "max_alpha": ("Max Alpha", "deg"),
}

PROFILE_TOTAL_STEPS = 24
PROFILE_AVERAGE_STEP = 12
PROFILE_MIN_RATIO = 0.25


# ── Trial-specific helpers ────────────────────────────────────────────────────

def _shade_difficulty_time(ax, gates, gate_status, t0_ms):
    if DIFFICULTY_COL not in gates.columns or gate_status is None:
        return
    passed = _gate_passage_times(gate_status)
    if len(passed) < 2:
        return
    gate_info = gates.set_index("id")
    pts = []
    for _, row in passed.iterrows():
        gid = int(row["id"])
        if gid in gate_info.index:
            hard = bool(gate_info.loc[gid, DIFFICULTY_COL])
            t_s = (row["first_pass_timestamp"] - t0_ms) / 1000.0
            pts.append((t_s, hard))
    for i in range(len(pts) - 1):
        t0_s, _ = pts[i]
        t1_s, hard = pts[i + 1]
        color = "#F44336" if hard else "#4CAF50"
        ax.axvspan(t0_s, t1_s, alpha=0.08, color=color, linewidth=0)


def _draw_gate_lines(ax, gate_status, t0_ms):
    passed = _gate_passage_times(gate_status)
    for _, row in passed.iterrows():
        t_s = (row["first_pass_timestamp"] - t0_ms) / 1000.0
        ax.axvline(t_s, color="#aaa", linewidth=0.5, linestyle=":", alpha=0.5)


def _time_to_z(timestamps_ms, centroid: pd.DataFrame) -> np.ndarray:
    return np.interp(
        np.asarray(timestamps_ms, dtype=float),
        centroid["timestamp"].values.astype(float),
        centroid["z"].values.astype(float),
    )


def _find_dead_drones(drones: pd.DataFrame) -> pd.DataFrame:
    if drones is None or drones.empty or "alive" not in drones.columns:
        return pd.DataFrame(columns=["id", "position_x", "position_z", "timestamp"])
    deaths = []
    for drone_id, grp in drones.groupby("id"):
        grp = grp.sort_values("timestamp")
        alive = grp["alive"].values
        if len(alive) == 0 or alive.min() != 0 or alive.max() == 0:
            continue
        dead_idx = np.argmax(alive == 0)
        if dead_idx == 0:
            continue
        last_alive = grp.iloc[dead_idx - 1]
        deaths.append({
            "id": int(drone_id),
            "position_x": float(last_alive["position_x"]),
            "position_z": float(last_alive["position_z"]),
            "timestamp": int(last_alive["timestamp"]),
        })
    return pd.DataFrame(deaths)

def _plot_racing_combined(
    ax,
    gates,
    centroid,
    traj_merged,
    gate_breakdown: dict,
    dead_drones=None,
    traj_type: str = "inference",
    gate_status=None,
):
    from matplotlib.collections import LineCollection as _LC
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap

    _shade_difficulty_z(ax, gates, alpha=0.10)

    # Gate bars — colored by pass status
    for _, g in gates.iterrows():
        gid = int(g["id"])
        half_w = g["width"] / 2
        bd = gate_breakdown.get(gid, {})
        reached = bd.get("reached", False)
        outside = bd.get("outside", 0)
        if not reached:
            bar_color = "#BDBDBD"  # gray = unreached
        elif outside > 0:
            bar_color = "#FFC107"  # yellow = some outside
        else:
            bar_color = "#4CAF50"  # green = all inside
        ax.plot(
            [g["center_z"], g["center_z"]],
            [g["center_x"] - half_w, g["center_x"] + half_w],
            color=bar_color,
            linewidth=2.5,
            alpha=0.85,
            solid_capstyle="round",
        )

    # Split times
    split_map = {}
    if gate_status is not None:
        passed = _gate_passage_times(gate_status)
        # Deduplicate: keep only first occurrence of each gate ID
        passed_unique = passed.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
        ts_list = passed_unique["first_pass_timestamp"].values
        for i, (_, row) in enumerate(passed_unique.iterrows()):
            if i > 0:
                # Check if timestamps are in ms or seconds
                diff = ts_list[i] - ts_list[i - 1]
                if diff > 1000:  # Likely in milliseconds
                    split_map[int(row["id"])] = diff / 1000.0
                else:  # Already in seconds or very small
                    split_map[int(row["id"])] = diff

    # Gate labels — positioned above gates with margin
    for _, g in gates.iterrows():
        gid = int(g["id"])
        half_w = g["width"] / 2
        label_parts = [f"G{gid}"]
        if gid in split_map:
            label_parts.append(f"{split_map[gid]:.1f}s")

        ax.text(
            g["center_z"],
            g["center_x"] + half_w + 4,
            "\n".join(label_parts),
            fontsize=6,
            ha="center",
            va="bottom",
            color="#444",
            fontweight="bold" if gid in split_map else "normal",
        )

        bd = gate_breakdown.get(gid, {})
        outside = bd.get("outside", 0)
        if outside > 0 and bd.get("reached", False):
            ax.text(
                g["center_z"],
                g["center_x"] - half_w - 2,
                f"⚠{outside}",
                fontsize=7,
                ha="center",
                va="top",
                color="#E65100",
                fontweight="bold",
            )

    # Completion time annotation on end marker
    completion_str = None
    if gate_status is not None:
        passed_gates = _gate_passage_times(gate_status)
        if len(passed_gates) >= 2:
            t_start = passed_gates["first_pass_timestamp"].min()
            t_end = passed_gates["first_pass_timestamp"].max()
            elapsed_s = (t_end - t_start) / 1000.0
            mm = int(elapsed_s // 60)
            ss = elapsed_s % 60
            completion_str = f"{mm:02d}:{ss:04.1f}"

    # Trajectory
    if traj_merged is not None and not traj_merged.empty:
        z_vals = traj_merged["z"].values
        x_vals = traj_merged["x"].values

        if traj_type == "adaptive" and "cwl_current_step" in traj_merged.columns:
            total = traj_merged["cwl_total_steps"].replace(0, np.nan).fillna(24)
            step_norm = (traj_merged["cwl_current_step"] / total).clip(0, 1).values
            cmap = LinearSegmentedColormap.from_list(
                "adaptive", ["#F44336", "#FFC107", "#4CAF50"]
            )
            seg_colors = [cmap(float((step_norm[i] + step_norm[i + 1]) / 2)) for i in range(len(step_norm) - 1)]
        else:
            states = traj_merged["filtered_state"].values.astype(int)
            seg_colors = [STATE_COLORS.get(s, "#999") for s in states[:-1]]

        points = np.column_stack([z_vals, x_vals]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = _LC(segments, colors=seg_colors, linewidths=3, alpha=0.85, zorder=2)
        ax.add_collection(lc)

        ax.scatter(z_vals[0], x_vals[0], marker="o", s=80, color="#333", zorder=5, edgecolors="white", linewidths=1.5)
        ax.scatter(z_vals[-1], x_vals[-1], marker="s", s=80, color="#333", zorder=5, edgecolors="white", linewidths=1.5)

        if completion_str:
            ax.annotate(
                f"Finish\n{completion_str}",
                xy=(z_vals[-1], x_vals[-1]),
                xytext=(6, -14),
                textcoords="offset points",
                fontsize=7,
                color="#333",
                fontweight="bold",
            )
    elif centroid is not None:
        ax.plot(centroid["z"], centroid["x"], color="#555", linewidth=2, alpha=0.7, zorder=2)

    # Dead drone markers
    has_dead = dead_drones is not None and not dead_drones.empty
    if has_dead:
        for _, row in dead_drones.iterrows():
            ax.scatter(row["position_z"], row["position_x"], marker="x", color="#222", s=140, linewidths=2.8, zorder=6)

    # Legend
    if traj_type == "adaptive":
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize, LinearSegmentedColormap as _LSC
        cmap = _LSC.from_list("adaptive", ["#F44336", "#FFC107", "#4CAF50"])
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, location="right", fraction=0.02, pad=0.02,
                     label="Adaptation (Soft → Racing)")
        handles = []
    else:
        handles = [
            Line2D([0], [0], color=STATE_COLORS[i], linewidth=3, label=STATE_LABELS[i])
            for i in range(3)
        ]
    handles += [
        Line2D([0], [0], color="#4CAF50", linewidth=2.5, alpha=0.85, label="All inside"),
        Line2D([0], [0], color="#FFC107", linewidth=2.5, alpha=0.85, label="Some outside"),
        Line2D([0], [0], color="#BDBDBD", linewidth=2.5, alpha=0.85, label="Unreached gate"),
        Line2D([0], [0], marker="o", color="#333", linewidth=0, markersize=7, markeredgecolor="white", label="Start"),
        Line2D([0], [0], marker="s", color="#333", linewidth=0, markersize=7, markeredgecolor="white", label="End"),
    ]
    if has_dead:
        handles.append(
            Line2D([0], [0], marker="x", color="#222", linewidth=0, markersize=9, markeredgewidth=2.5, label="Dead drone")
        )
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    ax.set_ylabel("X — Lateral (m)")
    title_type = "Adaptation Step" if traj_type == "adaptive" else "CWL"
    ax.set_title(f"Course Overview — Trajectory colored by {title_type}, split times at gates")
    ax.margins(y=0.15)
    ax.grid(alpha=0.2)


def _plot_racing_cwl_probability(ax, inference, centroid, gates):
    if (
        inference is None
        or len(inference) <= 1
        or centroid is None
        or centroid.empty
    ):
        _no_data_placeholder(ax, "CWL Probabilities")
        return

    inf = inference[inference["timestamp"] > 0].copy()
    required = {"prob_low", "prob_medium", "prob_high"}
    if inf.empty or not required.issubset(inf.columns):
        _no_data_placeholder(ax, "CWL Probabilities")
        return

    inf = inf.sort_values("timestamp")
    z_raw = _time_to_z(inf["timestamp"].values, centroid)

    # Resample to uniform Z grid to avoid temporal-density distortion
    z_grid = np.linspace(z_raw.min(), z_raw.max(), 500)
    prob_low = np.interp(z_grid, z_raw, inf["prob_low"].values)
    prob_med = np.interp(z_grid, z_raw, inf["prob_medium"].values)
    prob_high = np.interp(z_grid, z_raw, inf["prob_high"].values)

    _shade_difficulty_z(ax, gates)
    ax.stackplot(
        z_grid,
        prob_low,
        prob_med,
        prob_high,
        colors=[STATE_COLORS[0], STATE_COLORS[1], STATE_COLORS[2]],
        alpha=0.85,
        labels=[STATE_LABELS[0], STATE_LABELS[1], STATE_LABELS[2]],
    )
    _draw_gate_lines_z(ax, gates)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("CWL Probability")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(axis="y", linestyle=":", alpha=0.3)


def _plot_racing_adaptation_z(ax, commands, centroid, gates):
    if commands is None or centroid is None or centroid.empty:
        _no_data_placeholder(ax, f"Adaptation Profile ({PROFILE_TOTAL_STEPS} steps)")
        return

    cmd = commands[commands["timestamp"] > 0].copy()
    if (
        cmd.empty
        or "cwl_current_step" not in cmd.columns
        or "cwl_total_steps" not in cmd.columns
    ):
        _no_data_placeholder(ax, f"Adaptation Profile ({PROFILE_TOTAL_STEPS} steps)")
        return

    cmd = cmd.sort_values("timestamp")
    total = cmd["cwl_total_steps"].replace(0, np.nan).fillna(PROFILE_TOTAL_STEPS)
    # Map steps (0-23) to profile range (PROFILE_MIN_RATIO to 1.0)
    step_ratio_raw = (cmd["cwl_current_step"] / total).fillna(0).clip(0, 1).values
    step_norm_raw = PROFILE_MIN_RATIO + (1.0 - PROFILE_MIN_RATIO) * step_ratio_raw
    z_raw = _time_to_z(cmd["timestamp"].values, centroid)

    # Resample to uniform Z grid
    z_grid = np.linspace(z_raw.min(), z_raw.max(), 500)
    step_norm = np.interp(z_grid, z_raw, step_norm_raw)

    _shade_difficulty_z(ax, gates)
    ax.fill_between(z_grid, 0, step_norm, alpha=0.25, color="#5C6BC0", zorder=2)
    ax.plot(z_grid, step_norm, color="#3949AB", linewidth=1.8, zorder=3)

    # Add average profile step as dotted line
    avg_step_ratio = PROFILE_AVERAGE_STEP / PROFILE_TOTAL_STEPS
    avg_step_norm = PROFILE_MIN_RATIO + (1.0 - PROFILE_MIN_RATIO) * avg_step_ratio
    ax.axhline(avg_step_norm, color="#FF9800", linewidth=1.2, linestyle="--", alpha=0.6, label=f"Avg step ({PROFILE_AVERAGE_STEP})")

    _draw_gate_lines_z(ax, gates)
    ax.set_ylim(-0.05, 1.10)
    ax.set_yticks([PROFILE_MIN_RATIO, 1.0])
    ax.set_yticklabels(["Soft", "Racing"])
    ax.set_ylabel(f"Profile ({PROFILE_TOTAL_STEPS} steps)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.3)


def _plot_racing_drone_misses(ax, gates, gate_breakdown: dict):
    if gates is None or gates.empty:
        _no_data_placeholder(ax, "Drones Through Gate")
        return

    sorted_gates = gates.sort_values("center_z").reset_index(drop=True)
    z_centers = sorted_gates["center_z"].values

    mean_span = float(np.mean(np.diff(z_centers))) if len(z_centers) >= 2 else 1.0
    bar_width = 0.6 * mean_span if mean_span > 0 else 1.0

    _shade_difficulty_z(ax, gates)

    legend_labels: set[str] = set()

    for _, g in sorted_gates.iterrows():
        gid = int(g["id"])
        z = float(g["center_z"])
        bd = gate_breakdown.get(gid, {})
        reached = bd.get("reached", False)

        if not reached:
            ax.bar(
                z, SWARM_SIZE, width=bar_width * 0.25,
                color="#9E9E9E", alpha=0.25, edgecolor="none", zorder=2,
                label="Unreached" if "Unreached" not in legend_labels else None,
            )
            legend_labels.add("Unreached")
            continue

        inside = bd.get("inside", 0)
        outside = bd.get("outside", 0)
        dead = bd.get("dead", 0)
        bottom = 0

        if inside > 0:
            ax.bar(z, inside, width=bar_width, bottom=bottom, color="#4CAF50",
                   edgecolor="white", linewidth=0.4, zorder=3,
                   label="Inside" if "Inside" not in legend_labels else None)
            legend_labels.add("Inside")
            bottom += inside

        if outside > 0:
            ax.bar(z, outside, width=bar_width, bottom=bottom, color="#FFC107",
                   edgecolor="white", linewidth=0.4, zorder=3,
                   label="Outside" if "Outside" not in legend_labels else None)
            legend_labels.add("Outside")
            bottom += outside

        if dead > 0:
            ax.bar(z, dead, width=bar_width, bottom=bottom, color="#212121",
                   edgecolor="white", linewidth=0.4, zorder=3,
                   label="Dead" if "Dead" not in legend_labels else None)
            legend_labels.add("Dead")

    _draw_gate_lines_z(ax, gates)
    ax.set_ylim(0, SWARM_SIZE + 0.5)
    ax.set_yticks([0, 3, 6, 9])
    ax.set_ylabel(f"Drones (/{SWARM_SIZE})")
    ax.set_xlabel("Course Progress Z (m)")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax.grid(axis="y", linestyle=":", alpha=0.3)


def _plot_racing_cwl(ax, inference, gates, gate_status, t0_ms):
    if inference is None or len(inference) <= 1:
        _no_data_placeholder(ax, "CWL Inference")
        return

    inf = inference[inference["timestamp"] > 0].copy()
    if inf.empty:
        _no_data_placeholder(ax, "CWL Inference")
        return

    t = (inf["timestamp"] - t0_ms) / 1000.0

    if gates is not None and gate_status is not None:
        _shade_difficulty_time(ax, gates, gate_status, t0_ms)

    ax.step(
        t,
        inf["raw_state"],
        where="post",
        color="#1976D2",
        linewidth=1,
        alpha=0.5,
        label="Raw",
    )
    ax.step(
        t,
        inf["filtered_state"],
        where="post",
        color="#E91E63",
        linewidth=1.5,
        label="Filtered",
    )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([STATE_LABELS[i] for i in range(3)])
    ax.set_ylabel("CWL Level")
    ax.set_ylim(-0.3, 2.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("CWL Inference")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    if gate_status is not None:
        _draw_gate_lines(ax, gate_status, t0_ms)


def _plot_racing_commands(ax, commands, gate_status, t0_ms):
    if commands is None:
        _no_data_placeholder(ax, "Control Inputs")
        return

    cmd = commands[commands["timestamp"] > 0].copy()
    if cmd.empty:
        _no_data_placeholder(ax, "Control Inputs")
        return

    t = (cmd["timestamp"] - t0_ms) / 1000.0

    for col, (color, label) in COMMAND_INPUTS.items():
        if col in cmd.columns:
            ax.plot(t, cmd[col], color=color, linewidth=0.8, alpha=0.8, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Input Rate")
    ax.set_title("Control Inputs")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.grid(alpha=0.3)

    if gate_status is not None:
        _draw_gate_lines(ax, gate_status, t0_ms)


def _plot_racing_profile_with_controls(axes, commands, centroid, gates, gate_status=None, t0_ms=0):
    """4×1 layout: Adaptation Profile + Normalized Control Commands.

    Each subplot shows over time:
    - Profile as limit envelope (normalized to current step's allowed range)
    - Commanded rate normalized by current limit (shows % of authority used)
    Easier interpretation: see if user commands match available authority at each step.
    """
    # Map control inputs to their corresponding flight profile parameters
    control_map = [
        ("roll_rate", "max_roll", "Roll", "#FF9800"),
        ("pitch_rate", "max_pitch", "Pitch", "#1976D2"),
        ("yaw_rate", "max_yaw_rate", "Yaw", "#E91E63"),
        ("altitude_rate", "max_altitude_rate", "Altitude", "#4CAF50"),
    ]

    if commands is None:
        for ax, (_, _, label, _) in zip(axes, control_map):
            _no_data_placeholder(ax, label)
        return

    cmd = commands[commands["timestamp"] > 0].copy()
    if cmd.empty or "cwl_current_step" not in cmd.columns or "cwl_total_steps" not in cmd.columns:
        for ax, (_, _, label, _) in zip(axes, control_map):
            _no_data_placeholder(ax, label)
        return

    cmd = cmd.sort_values("timestamp")
    t = (cmd["timestamp"] - t0_ms) / 1000.0
    total_steps = cmd["cwl_total_steps"].replace(0, np.nan).fillna(PROFILE_TOTAL_STEPS)
    # Map steps (0-23) to profile range (PROFILE_MIN_RATIO to 1.0)
    step_ratio = (cmd["cwl_current_step"] / total_steps).fillna(0).clip(0, 1).values
    step_norm = PROFILE_MIN_RATIO + (1.0 - PROFILE_MIN_RATIO) * step_ratio

    # Plot each control channel
    for i, (ax, (ctrl_col, param, label, color)) in enumerate(zip(axes, control_map)):
        if ctrl_col not in cmd.columns or param not in _FLIGHT_PROFILE_LIMITS:
            _no_data_placeholder(ax, label)
            continue

        # Get flight parameter limits at min and max adaptation
        vmin = _FLIGHT_PROFILE_MIN_LIMITS.get(param, 0.0)
        vmax = _FLIGHT_PROFILE_LIMITS[param]

        # Compute current limit for each timestep based on current step
        current_limits = vmin + (vmax - vmin) * step_ratio

        # Get raw control input and compute commanded rate normalized by current limit
        ctrl_raw = cmd[ctrl_col].values

        # Commanded rate as percentage of current limit (0 to step_norm when input is -1 to 1)
        commanded_norm = np.abs(ctrl_raw) * step_norm

        # Preserve sign for visualization
        commanded_signed = np.sign(ctrl_raw) * commanded_norm

        # Plot limit envelope as symmetric fill (profile defines the bounds) - label only on first plot
        envelope_label = "Limit envelope" if i == 0 else None
        ax.fill_between(t, -step_norm, step_norm, alpha=0.15, color="#3949AB", zorder=1, label=envelope_label)

        # Plot normalized commanded rate (shows % of current limit)
        ax.plot(t, commanded_signed, color=color, linewidth=1.8, alpha=0.9, zorder=3,
                label=f"{label}: {vmin:.2f}→{vmax:.2f}")
        ax.fill_between(t, 0, commanded_signed, where=(commanded_signed >= 0), alpha=0.25, color=color, zorder=2)
        ax.fill_between(t, 0, commanded_signed, where=(commanded_signed < 0), alpha=0.25, color=color, zorder=2)

        # Draw gate lines
        if gate_status is not None:
            _draw_gate_lines(ax, gate_status, t0_ms)

        # Center line
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4, zorder=0)

        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
        ax.set_yticklabels(["-1.0", "-0.5", "0", "0.5", "1.0"], fontsize=8)
        ax.set_ylabel(label, fontsize=9, fontweight="bold")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

        # Add right y-axis with same scale
        ax_right = ax.twinx()
        ax_right.set_ylim(ax.get_ylim())
        ax_right.set_yticks(ax.get_yticks())
        ax_right.set_yticklabels([f"{v:.1f}" for v in ax.get_yticks()], fontsize=8)
        ax_right.set_ylabel("")


def _plot_racing_command_limits(axes, commands, gate_status, t0_ms):
    """2×3 small-multiple panels: per-parameter flight profile limit over time.

    For each parameter the theoretical limit is linearly interpolated between
    FLIGHT_PROFILE_MIN_LIMITS (step 0) and FLIGHT_PROFILE_LIMITS (step N-1)
    using cwl_current_step from command_data.csv.

    Three filled bands make the operating envelope immediately readable:
      • green  — always-on minimum base
      • blue   — range unlocked by the current CWL step
      • red    — remaining locked headroom
    """
    params = list(_FLIGHT_PROFILE_LIMITS.keys())
    flat_axes = [ax for row in axes for ax in row]

    no_data = commands is None
    if not no_data:
        cmd = commands[commands["timestamp"] > 0].copy()
        no_data = (
            cmd.empty
            or "cwl_current_step" not in cmd.columns
            or "cwl_total_steps" not in cmd.columns
        )

    if no_data:
        for ax, param in zip(flat_axes, params):
            _no_data_placeholder(ax, _FLIGHT_PARAM_LABELS.get(param, (param, ""))[0])
        return

    cmd = cmd.sort_values("timestamp")
    t = (cmd["timestamp"] - t0_ms) / 1000.0
    steps = cmd["cwl_current_step"].values
    totals = cmd["cwl_total_steps"].replace(0, np.nan).fillna(24).values
    ratio = np.clip(steps / np.maximum(totals - 1, 1), 0.0, 1.0)

    for ax, param in zip(flat_axes, params):
        vmin = _FLIGHT_PROFILE_MIN_LIMITS.get(param, 0.0)
        vmax = _FLIGHT_PROFILE_LIMITS[param]
        label, unit = _FLIGHT_PARAM_LABELS.get(param, (param, ""))
        limit_vals = vmin + (vmax - vmin) * ratio

        ax.fill_between(t, 0, vmin, alpha=0.18, color="#4CAF50", linewidth=0)
        ax.fill_between(t, vmin, limit_vals, alpha=0.32, color="#3949AB", linewidth=0)
        ax.fill_between(t, limit_vals, vmax, alpha=0.10, color="#F44336", linewidth=0)
        ax.plot(t, limit_vals, color="#3949AB", linewidth=1.5, label="Current limit")
        ax.axhline(
            vmin, color="#4CAF50", linewidth=1.0, linestyle="--", alpha=0.9,
            label=f"Min ({vmin})",
        )
        ax.axhline(
            vmax, color="#F44336", linewidth=1.0, linestyle="--", alpha=0.9,
            label=f"Max ({vmax})",
        )

        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_ylabel(unit, fontsize=8)
        ax.set_ylim(min(0, vmin * 0.9), vmax * 1.08)
        ax.legend(fontsize=7, loc="upper left", handlelength=1.2)
        ax.grid(linestyle=":", alpha=0.3)

        if gate_status is not None:
            _draw_gate_lines(ax, gate_status, t0_ms)


def _plot_racing_adaptation(ax, commands, inference, t0_ms):
    if commands is None:
        _no_data_placeholder(ax, "Adaptation Parameters")
        return

    cmd = commands[commands["timestamp"] > 0].copy()
    available = [col for col in ADAPTATION_PARAMS if col in cmd.columns]

    if not available or cmd.empty:
        _no_data_placeholder(ax, "Adaptation Parameters")
        return

    t = (cmd["timestamp"] - t0_ms) / 1000.0

    # CWL background shading
    if inference is not None and len(inference) > 1:
        inf = inference[inference["timestamp"] > 0].copy()
        if not inf.empty:
            inf_t = (inf["timestamp"] - t0_ms) / 1000.0
            for i in range(len(inf) - 1):
                level = int(inf.iloc[i]["filtered_state"])
                ax.axvspan(
                    inf_t.iloc[i],
                    inf_t.iloc[i + 1],
                    alpha=0.08,
                    color=STATE_COLORS.get(level, "#999"),
                    linewidth=0,
                )

    for col in available:
        label, _, _ = ADAPTATION_PARAMS[col]
        vals = cmd[col]
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin < 1e-9:
            normalized = pd.Series(0.5, index=vals.index)
        else:
            normalized = (vals - vmin) / (vmax - vmin)
        ax.plot(
            t,
            normalized,
            linewidth=1.2,
            label=f"{label} [{vmin:.2f}–{vmax:.2f}]",
            alpha=0.85,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized (0 = restricted, 1 = full)")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Adaptation Parameters — CWL background")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)


def _plot_racing_splits(ax, gates, gate_status):
    if gate_status is None:
        _no_data_placeholder(ax, "Gate Split Times")
        return

    passed = _gate_passage_times(gate_status)
    if len(passed) < 2:
        _no_data_placeholder(ax, "Gate Split Times")
        return

    # Deduplicate: keep only first occurrence of each gate ID
    passed_unique = passed.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    if len(passed_unique) < 2:
        _no_data_placeholder(ax, "Gate Split Times")
        return

    gate_ids = passed_unique["id"].values
    timestamps = passed_unique["first_pass_timestamp"].values

    # Calculate splits — check if timestamps are in ms or seconds
    diffs = np.diff(timestamps)
    if len(diffs) > 0 and np.max(diffs) > 1000:  # Likely in milliseconds
        splits = diffs / 1000.0
    else:  # Already in seconds or very small
        splits = diffs

    gate_info = gates.set_index("id")
    colors = []
    for i in range(1, len(gate_ids)):
        gid = gate_ids[i]
        if DIFFICULTY_COL in gate_info.columns and gid in gate_info.index:
            hard = bool(gate_info.loc[gid, DIFFICULTY_COL])
            colors.append("#F44336" if hard else "#4CAF50")
        else:
            colors.append("#999")

    x = np.arange(len(splits))
    labels = [str(int(gate_ids[i + 1])) for i in range(len(splits))]

    ax.bar(x, splits, color=colors, edgecolor="white", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("Split Time (s)")
    ax.set_title("Gate Split Times")
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _plot_racing_stats(ax, gates, gate_status, drones, centroid):
    ax.axis("off")

    lines = []

    if gate_status is not None:
        # Deduplicate gate_status (keep first occurrence of each gate ID)
        gate_status_unique = gate_status.drop_duplicates(subset="id", keep="first")

        passed = _gate_passage_times(gate_status)
        if len(passed) >= 2:
            total_s = (
                passed["first_pass_timestamp"].max()
                - passed["first_pass_timestamp"].min()
            ) / 1000.0
            mm = int(total_s // 60)
            ss = total_s % 60
            lines.append(f"Completion time (first→last gate):  {mm}m {ss:.1f}s")

        n_passed = int((gate_status_unique["pass_count"] > 0).sum())
        alive_at_last = centroid['n_alive'].iloc[-1] if (centroid is not None and "n_alive" in centroid.columns) else SWARM_SIZE
        n_full = int((gate_status_unique["pass_count"] >= alive_at_last).sum())
        lines.append(f"Gates reached:                   {n_passed} / {len(gates)}")
        lines.append(f"Gates fully passed:              {n_full} / {len(gates)}")

    if centroid is not None and "n_alive" in centroid.columns:
        lines.append(
            f"Drones alive:                    "
            f"{centroid['n_alive'].iloc[-1]} / {SWARM_SIZE}"
        )

    if not lines:
        lines.append("No performance data available")

    ax.text(
        0.05,
        0.9,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8),
    )
    ax.set_title("Performance Summary", fontsize=11, fontweight="bold")




def _make_time_series_figure(data: pd.DataFrame, title: str) -> plt.Figure:
    fig, (ax_states, ax_rolling) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plot_inference_time_series(data, ax_states, ax_rolling)
    fig.tight_layout()
    return fig




def _run_racing_trial(show: bool, output_dir: Path, data_dir: Path, traj_type: str = "inference"):

    gates = _load_csv(data_dir, GATE_LAYOUT_FILE)
    gate_status = _load_csv(data_dir, GATE_STATUS_FILE)
    inference = _load_csv(data_dir, INFERENCE_FILE_NAME)
    commands = _load_csv(data_dir, COMMAND_DATA_FILE)
    drones = _load_csv(data_dir, DRONE_FILE_NAME)

    if gates is None:
        print("  ERROR: gate_layout.csv not found")
        return

    centroid = _compute_centroid(drones) if drones is not None else None

    # Build trajectory merged dataset (inference or adaptive coloring)
    traj_merged = None
    if centroid is not None:
        if traj_type == "adaptive" and commands is not None:
            cmd_cols = commands[
                ["timestamp", "cwl_current_step", "cwl_total_steps"]
            ].sort_values("timestamp")
            traj_merged = pd.merge_asof(
                centroid.sort_values("timestamp"),
                cmd_cols,
                on="timestamp",
                direction="backward",
            ).dropna(subset=["cwl_current_step"])
        elif inference is not None and len(inference) > 1:
            inf_cols = inference[["timestamp", "filtered_state", "raw_state"]].sort_values("timestamp")
            traj_merged = pd.merge_asof(
                centroid.sort_values("timestamp"),
                inf_cols,
                on="timestamp",
                direction="backward",
            ).dropna(subset=["filtered_state"])

    gate_breakdown = _compute_gate_breakdown(gates, gate_status, drones)
    dead_drones = _find_dead_drones(drones) if drones is not None else pd.DataFrame()
    t0_ms = _find_t0(inference, centroid, commands, gate_status)
    figs: list[tuple[plt.Figure, Path]] = []

    # ── Figure 1: Course Analysis ──
    fig1 = plt.figure(figsize=(18, 16))
    gs1 = fig1.add_gridspec(
        4,
        1,
        height_ratios=[3, 1.8, 1.2, 1.2],
        hspace=0.10,
    )
    fig1.suptitle("Racing Trial — Course Analysis", fontsize=14, fontweight="bold")

    ax_traj = fig1.add_subplot(gs1[0])
    ax_prob = fig1.add_subplot(gs1[1], sharex=ax_traj)
    ax_step = fig1.add_subplot(gs1[2], sharex=ax_traj)
    ax_miss = fig1.add_subplot(gs1[3], sharex=ax_traj)

    _plot_racing_combined(
        ax_traj, gates, centroid, traj_merged, gate_breakdown,
        dead_drones=dead_drones, traj_type=traj_type, gate_status=gate_status,
    )
    _plot_racing_cwl_probability(ax_prob, inference, centroid, gates)
    _plot_racing_adaptation_z(ax_step, commands, centroid, gates)
    _plot_racing_drone_misses(ax_miss, gates, gate_breakdown)

    for ax in (ax_traj, ax_prob, ax_step):
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")

    fig1.tight_layout()
    figs.append((fig1, output_dir / "racing_course_analysis.png"))

    # ── Figure 2: Profile & Control Response Analysis ──
    fig2 = plt.figure(figsize=(16, 13))
    fig2.suptitle(
        "Racing Trial — Profile & Control Response", fontsize=14, fontweight="bold"
    )
    outer_gs = fig2.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.45)
    control_gs = outer_gs[0].subgridspec(4, 1, hspace=0.45)
    # Bottom: 2/3 split times, 1/3 stats
    bottom_gs = outer_gs[1].subgridspec(1, 2, width_ratios=[2, 1], wspace=0.35)

    # Build 4×1 control profile axes
    axes_control = [fig2.add_subplot(control_gs[i]) for i in range(4)]
    for i, ax in enumerate(axes_control[:-1]):
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")
    axes_control[-1].set_xlabel("Time (s)", fontsize=8)

    ax_splits = fig2.add_subplot(bottom_gs[0])
    ax_stats = fig2.add_subplot(bottom_gs[1])

    _plot_racing_profile_with_controls(axes_control, commands, centroid, gates, gate_status=gate_status, t0_ms=t0_ms)
    _plot_racing_splits(ax_splits, gates, gate_status)
    _plot_racing_stats(ax_stats, gates, gate_status, drones, centroid)
    fig2.tight_layout()
    figs.append((fig2, output_dir / "racing_adaptation_performance.png"))

    _save_or_show(figs, show)

