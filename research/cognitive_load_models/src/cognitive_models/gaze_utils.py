import pandas as pd
import numpy as np
from scipy.signal import firwin, filtfilt

CUTOFF_FREQ = 10  # Hz
FS = 60  # Hz
TAPS = 5


def detect_gaps_and_blinks(
    df, confidence_threshold=0.95, blink_threshold_range=(100, 300)
):
    if not all([col in df.columns for col in ["timestamp", "pupil_confidence"]]):
        raise ValueError("DataFrame must contain 'pupil_confidence' column")

    low_confidence_df = df[["timestamp", "pupil_confidence"]].copy()
    low_confidence_df["low_confidence"] = (
        low_confidence_df["pupil_confidence"] < confidence_threshold
    )
    low_confidence_df["transition"] = low_confidence_df[
        "low_confidence"
    ] != low_confidence_df["low_confidence"].shift(1)
    low_confidence_df["group"] = low_confidence_df["transition"].cumsum()
    low_confidence_df["id"] = low_confidence_df.index
    low_confidence_df_gfoup = low_confidence_df.groupby("group").agg(
        {
            "low_confidence": ["first", "count"],
            "timestamp": ["min", "max"],
            "id": ["min", "max"],
        }
    )
    low_confidence_df_gfoup["duration_ms"] = (
        low_confidence_df_gfoup["timestamp"]["max"]
        - low_confidence_df_gfoup["timestamp"]["min"]
    ) * 1000

    # Gaps
    gaps_to_fill_df = low_confidence_df_gfoup[
        low_confidence_df_gfoup["low_confidence"]["first"]
        & (low_confidence_df_gfoup["duration_ms"] < blink_threshold_range[0])
    ]
    gaps_to_fill_df["start_timestamp"] = gaps_to_fill_df["timestamp"]["min"]
    gaps_to_fill_df["stop_timestamp"] = gaps_to_fill_df["timestamp"]["max"]
    gaps_to_fill_df["start_id"] = gaps_to_fill_df["id"]["min"].astype(int)
    gaps_to_fill_df["stop_id"] = gaps_to_fill_df["id"]["max"].astype(int)
    gaps_to_fill_df = gaps_to_fill_df[
        ["start_id", "stop_id", "start_timestamp", "stop_timestamp", "duration_ms"]
    ].reset_index(drop=True)
    gaps_to_fill_df = gaps_to_fill_df.droplevel(level=1, axis=1)

    # Blinks
    custom_blinks_df = low_confidence_df_gfoup[
        (
            low_confidence_df_gfoup["low_confidence"]["first"]
            & (low_confidence_df_gfoup["duration_ms"] >= blink_threshold_range[0])
            & (low_confidence_df_gfoup["duration_ms"] <= blink_threshold_range[1])
        )
    ].copy()
    custom_blinks_df["start_timestamp"] = custom_blinks_df["timestamp"]["min"]
    custom_blinks_df["stop_timestamp"] = custom_blinks_df["timestamp"]["max"]
    custom_blinks_df["start_id"] = custom_blinks_df["id"]["min"]
    custom_blinks_df["stop_id"] = custom_blinks_df["id"]["max"]
    custom_blinks_df = custom_blinks_df[
        ["start_id", "stop_id", "start_timestamp", "stop_timestamp", "duration_ms"]
    ].reset_index(drop=True)
    custom_blinks_df = custom_blinks_df.droplevel(level=1, axis=1)

    return gaps_to_fill_df, custom_blinks_df


def calculate_gaze_angular_delta(df):
    """
    Make sur the dataframe has the following columns:
    'timestamp', 'gaze_point_3d_x', 'gaze_point_3d_y', 'gaze_point_3d_z'

    """
    if not all(
        col in df.columns
        for col in [
            "timestamp",
            "gaze_point_3d_x",
            "gaze_point_3d_y",
            "gaze_point_3d_z",
        ]
    ):
        raise ValueError(
            "DataFrame must contain the following columns: "
            "'timestamp', 'gaze_point_3d_x', 'gaze_point_3d_y', 'gaze_point_3d_z'"
        )
    gaze_angular_data = df[
        ["timestamp", "gaze_point_3d_x", "gaze_point_3d_y", "gaze_point_3d_z"]
    ].copy()
    gaze_angular_data["prev_gaze_point_3d_x"] = gaze_angular_data[
        "gaze_point_3d_x"
    ].shift(1)
    gaze_angular_data["prev_gaze_point_3d_y"] = gaze_angular_data[
        "gaze_point_3d_y"
    ].shift(1)
    gaze_angular_data["prev_gaze_point_3d_z"] = gaze_angular_data[
        "gaze_point_3d_z"
    ].shift(1)

    def calculate_angle(row):
        if (
            pd.isna(row["prev_gaze_point_3d_x"])
            or pd.isna(row["prev_gaze_point_3d_y"])
            or pd.isna(row["prev_gaze_point_3d_z"])
        ):
            return np.nan
        v1 = np.array(
            [row["gaze_point_3d_x"], row["gaze_point_3d_y"], row["gaze_point_3d_z"]]
        )
        v2 = np.array(
            [
                row["prev_gaze_point_3d_x"],
                row["prev_gaze_point_3d_y"],
                row["prev_gaze_point_3d_z"],
            ]
        )
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return np.nan
        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clip to avoid numerical issues
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    return gaze_angular_data.apply(calculate_angle, axis=1)


def calculate_angular_velocity(
    gaze_df, absolute=True, LIMITS=(-1000, 1000), filtered=True
):
    if "gaze_angle_delta_deg" not in gaze_df.columns:
        raise ValueError("gaze_df must contain 'gaze_angle_delta_deg' column")

    gaze_angular_data = gaze_df[["timestamp", "gaze_angle_delta_deg"]].copy()
    gaze_angular_data["prev_gaze_angle_delta_deg"] = gaze_angular_data[
        "gaze_angle_delta_deg"
    ].shift(1)
    gaze_angular_data["delta_time"] = gaze_angular_data[
        "timestamp"
    ] - gaze_angular_data["timestamp"].shift(1)

    def angular_velocity(row):
        if pd.isna(row["prev_gaze_angle_delta_deg"]) or pd.isna(
            row["gaze_angle_delta_deg"]
        ):
            return np.nan
        delta_angle = row["gaze_angle_delta_deg"] - row["prev_gaze_angle_delta_deg"]
        delta_time = row["delta_time"]
        if delta_time == 0:
            return np.nan
        angular_velocity = delta_angle / delta_time
        angular_velocity = np.clip(angular_velocity, LIMITS[0], LIMITS[1])
        return abs(angular_velocity) if absolute else angular_velocity

    gaze_angular_data["gaze_angular_velocity"] = gaze_angular_data.apply(
        angular_velocity, axis=1
    )

    # Apply filter
    def apply_fir_filter(data, column, cutoff_freq, fs, numtaps=5):
        fir_coefficients = firwin(numtaps, cutoff_freq, fs=fs)
        filtered_data = filtfilt(fir_coefficients, [1.0], data[column])
        return filtered_data

    if filtered:
        gaze_angular_data["gaze_angular_velocity_filtered"] = apply_fir_filter(
            gaze_angular_data, "gaze_angular_velocity", CUTOFF_FREQ, FS, numtaps=TAPS
        )

    return gaze_angular_data["gaze_angular_velocity"]


def calculate_fixations_saccades(
    eye_df, ivt_threshold, min_fixation_duration=55, min_datapoints=2
):
    eye_df = eye_df.copy()
    eye_df["saccade"] = eye_df["gaze_angular_velocity_filtered"] > ivt_threshold
    eye_df["fixation"] = ~eye_df["saccade"]
    eye_df["transition"] = eye_df["saccade"] != eye_df["saccade"].shift(1)
    print(
        f"Found {eye_df['transition'].sum()} transitions between saccades and fixations"
    )

    # Min duration of fixations
    min_fixation_n_datapoints = int(np.ceil(min_fixation_duration / (1000 / FS)))
    eye_df["transition"] = eye_df["saccade"] != eye_df["saccade"].shift(1)
    eye_df["id"] = eye_df.index
    grouped_transitions = eye_df.groupby(eye_df["transition"].cumsum())
    # find groups of fixations with a duration less than 55 ms and mark them as saccades
    fixation_groups = grouped_transitions.agg(
        {"fixation": ["first", "count"], "id": ["min", "max"]}
    )
    print(
        f"Will mark {((fixation_groups[('fixation', 'count')] < min_fixation_n_datapoints) & (fixation_groups[('fixation', 'first')])).sum()} short fixations as saccades"
    )
    for idx, group in fixation_groups.iterrows():
        if (
            group["fixation"]["count"] < min_fixation_n_datapoints
            and group["fixation"]["first"]
        ):  # 55 ms at 250 Hz sampling rate
            # Mark the entire group as saccade
            eye_df.loc[group["id"]["min"] : group["id"]["max"], "saccade"] = True
            eye_df.loc[group["id"]["min"] : group["id"]["max"], "fixation"] = False

    if min_datapoints > 1:
        eye_df["saccade"] = eye_df["saccade"].astype(np.int32)
        eye_df["fixation"] = eye_df["fixation"].astype(np.int32)
        eye_df["transition"] = eye_df["saccade"] != eye_df["saccade"].shift(1)
        grouped_transitions = eye_df.groupby(eye_df["transition"].cumsum())
        single_samples = grouped_transitions.agg(
            {"saccade": "count", "id": "first", "fixation": "count"}
        )
        single_samples = single_samples[
            (single_samples["saccade"] <= min_datapoints - 1)
            | (single_samples["fixation"] <= min_datapoints - 1)
        ]["id"]
        eye_df.loc[single_samples, "saccade"] = pd.NA
        eye_df.loc[single_samples, "fixation"] = pd.NA
        eye_df["saccade"] = eye_df["saccade"].interpolate(method="nearest")
        eye_df["fixation"] = eye_df["fixation"].interpolate(method="nearest")
        eye_df["saccade"] = eye_df["saccade"].astype(np.bool)
        eye_df["fixation"] = eye_df["fixation"].astype(np.bool)

        # Update transitions after interpolation
        eye_df["transition"] = eye_df["saccade"] != eye_df["saccade"].shift(1)

    # From this df, build exclusive saccade and fixation infos df
    rows_saccades = []
    rows_fixations = []

    grouped_transitions = eye_df.groupby(eye_df["transition"].cumsum())
    for idx, group in grouped_transitions:
        if group["saccade"].iloc[0]:
            rows_saccades.append(
                {
                    "start_id": group["id"].iloc[0],
                    "stop_id": group["id"].iloc[-1],
                    "start_timestamp": group["timestamp"].iloc[0],
                    "stop_timestamp": group["timestamp"].iloc[-1],
                    "duration_ms": (
                        group["timestamp"].iloc[-1] - group["timestamp"].iloc[0]
                    )
                    * 1000,
                }
            )
        else:
            mean_x, mean_y = group["norm_pos_x"].mean(), group["norm_pos_y"].mean()
            radius = np.sqrt(
                (
                    (group["norm_pos_x"] - mean_x) ** 2
                    + (group["norm_pos_y"] - mean_y) ** 2
                ).max()
            )
            rows_fixations.append(
                {
                    "start_id": group["id"].iloc[0],
                    "stop_id": group["id"].iloc[-1],
                    "start_timestamp": group["timestamp"].iloc[0],
                    "stop_timestamp": group["timestamp"].iloc[-1],
                    "duration_ms": (
                        group["timestamp"].iloc[-1] - group["timestamp"].iloc[0]
                    )
                    * 1000,
                    "x": mean_x,
                    "y": mean_y,
                    "radius": radius,
                }
            )
    saccades_df = pd.DataFrame(rows_saccades)
    fixations_df = pd.DataFrame(rows_fixations)

    return eye_df, fixations_df, saccades_df
