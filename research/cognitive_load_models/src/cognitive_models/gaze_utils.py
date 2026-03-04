import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin

CUTOFF_FREQ = 10  # Hz
FS = 60  # Hz
TAPS = 5


def detect_gaps_and_blinks(
    df: pd.DataFrame,
    confidence_threshold: float = 0.95,
    blink_threshold_range: tuple[int, int] = (100, 300),
    eye_openness_column: str = None,
    openness_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Detect low confidence gaps and blinks in the eye-tracking data.

    :param df: DataFrame containing eye-tracking data with 'timestamp_sec' and 'confidence' columns.
    :param confidence_threshold: The threshold below which confidence is considered low.
    :param blink_threshold_range: The range of durations (in milliseconds) that are considered blinks.
    :param eye_openness_column: Optional column name for eye openness to help identify blinks.
    :raises ValueError: If the DataFrame does not contain the required columns.
    :return gaps: A DataFrame with information about detected gaps and blinks.
    """
    if not all([col in df.columns for col in ["timestamp_sec", "confidence"]]):
        raise ValueError("DataFrame must contain 'confidence' column")

    low_confidence_df = df[["timestamp_sec", "confidence"]].copy()
    low_confidence_df["openness"] = 0  # Initialize openness column with NaN
    normal_openness_value = np.inf
    if eye_openness_column and eye_openness_column in df.columns:
        normal_openness_value = df[eye_openness_column].mean(skipna=True)
        low_confidence_df["openness"] = df[eye_openness_column].copy().fillna(0)

    low_confidence_df["low_confidence"] = (
        low_confidence_df["confidence"] < confidence_threshold
    )
    low_confidence_df["transition"] = low_confidence_df[
        "low_confidence"
    ] != low_confidence_df["low_confidence"].shift(1)
    low_confidence_df["group"] = low_confidence_df["transition"].cumsum()
    low_confidence_df["id"] = low_confidence_df.index
    low_confidence_df_gfoup = low_confidence_df.groupby("group").agg(
        {
            "low_confidence": ["first", "count"],
            "timestamp_sec": ["min", "max"],
            "id": ["min", "max"],
            "openness": "mean",
        }
    )
    low_confidence_df_gfoup["duration_ms"] = (
        low_confidence_df_gfoup["timestamp_sec"]["max"]
        - low_confidence_df_gfoup["timestamp_sec"]["min"]
    ) * 1000

    # Gaps
    gaps_to_fill_df = low_confidence_df_gfoup[
        low_confidence_df_gfoup["low_confidence"]["first"]
    ]
    gaps_to_fill_df["start_timestamp"] = gaps_to_fill_df["timestamp_sec"]["min"]
    gaps_to_fill_df["stop_timestamp"] = gaps_to_fill_df["timestamp_sec"]["max"]
    gaps_to_fill_df["start_id"] = gaps_to_fill_df["id"]["min"].astype(int)
    gaps_to_fill_df["stop_id"] = gaps_to_fill_df["id"]["max"].astype(int)
    gaps_to_fill_df["openness_mean"] = gaps_to_fill_df["openness"]["mean"]
    gaps_to_fill_df = gaps_to_fill_df[
        [
            "start_id",
            "stop_id",
            "start_timestamp",
            "stop_timestamp",
            "duration_ms",
            "openness_mean",
        ]
    ].reset_index(drop=True)
    gaps_to_fill_df = gaps_to_fill_df.droplevel(level=1, axis=1)

    # Blinks
    gaps_to_fill_df["is_blink"] = (
        gaps_to_fill_df["duration_ms"] >= blink_threshold_range[0]
    )
    gaps_to_fill_df["is_blink"] &= (
        gaps_to_fill_df["duration_ms"] <= blink_threshold_range[1]
    )
    gaps_to_fill_df["is_blink"] &= gaps_to_fill_df["openness_mean"] < (
        normal_openness_value * openness_threshold
    )
    gaps_to_fill_df.drop(columns=["openness_mean"], inplace=True)

    return gaps_to_fill_df


def calculate_gaze_angular_delta(
    df: pd.DataFrame, gaze_point_columns_prefix: str
) -> pd.Series:
    """
    Make sur the dataframe has the following columns:
    'timestamp_sec', 'gaze_point_3d_x', 'gaze_point_3d_y', 'gaze_point_3d_z'

    Returns:
        pd.Series: A series containing the gaze angular delta in degrees.
    """
    gaze_columns = [
        f"{gaze_point_columns_prefix}_x",
        f"{gaze_point_columns_prefix}_y",
        f"{gaze_point_columns_prefix}_z",
    ]
    if not all(col in df.columns for col in ["timestamp_sec"] + gaze_columns):
        raise ValueError(
            f"DataFrame must contain the following columns: "
            f"'timestamp_sec', {', '.join(gaze_columns)}"
        )
    gaze_angular_data = df[["timestamp_sec"] + gaze_columns].copy()
    pre_gaze_columns = [f"prev_{col}" for col in gaze_columns]
    gaze_angular_data[pre_gaze_columns] = gaze_angular_data[gaze_columns].shift(1)

    def calculate_angle(row):
        if any(pd.isna(row[col]) for col in pre_gaze_columns):
            return np.nan
        v1 = np.array([row[col] for col in gaze_columns])
        v2 = np.array([row[col] for col in pre_gaze_columns])
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
    gaze_df: pd.DataFrame,
    absolute: bool = True,
    LIMITS: tuple[int, int] = (-1000, 1000),
    filtered: bool = True,
):
    """
    Make sure the dataframe has the following columns: 'timestamp_sec', 'gaze_angle_delta_deg'

    :param gaze_df: DataFrame containing gaze angle delta in degrees with timestamps.
    :param absolute: If True, return absolute angular velocity values.
    :param LIMITS: Tuple of (min, max) limits for angular velocity values.
    :param filtered: If True, apply a FIR filter to the angular velocity data.
    :raises ValueError: If the required columns are not present in gaze_df.
    :return: Series containing angular velocity values.
    """
    if "gaze_angle_delta_deg" not in gaze_df.columns:
        raise ValueError("gaze_df must contain 'gaze_angle_delta_deg' column")

    gaze_angular_data = gaze_df[["timestamp_sec", "gaze_angle_delta_deg"]].copy()
    gaze_angular_data["prev_gaze_angle_delta_deg"] = gaze_angular_data[
        "gaze_angle_delta_deg"
    ].shift(1)
    gaze_angular_data["delta_time"] = gaze_angular_data[
        "timestamp_sec"
    ] - gaze_angular_data["timestamp_sec"].shift(1)

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
        gaze_angular_data["gaze_angular_velocity"] = apply_fir_filter(
            gaze_angular_data, "gaze_angular_velocity", CUTOFF_FREQ, FS, numtaps=TAPS
        )

    return gaze_angular_data["gaze_angular_velocity"]


def calculate_fixations_saccades(
    eye_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    ivt_threshold: float,
    min_fixation_duration: int = 55,
    min_datapoints: int = 2,
    verbose: bool = True,
):
    """
    Make sure the dataframe has the following columns: 'timestamp_sec', 'gaze_angular_velocity'

    :param eye_df: DataFrame containing eye-tracking data with required columns.
    :param gaps_df: DataFrame containing information about gaps in the data.
    :param ivt_threshold: Threshold for identifying saccades based on angular velocity.
    :param min_fixation_duration: Minimum duration in milliseconds for a fixation to be valid.
    :param min_datapoints: Minimum number of consecutive samples for a fixation or saccade to be valid.

    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    :returns eye_df, fixations_df, saccades_df:
    """
    eye_df = eye_df.copy()
    eye_df["saccade"] = eye_df["gaze_angular_velocity"] > ivt_threshold
    eye_df["fixation"] = ~eye_df["saccade"]
    eye_df["transition"] = eye_df["saccade"] != eye_df["saccade"].shift(1)
    if verbose:
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
    if verbose:
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
                    "start_timestamp": group["timestamp_sec"].iloc[0],
                    "stop_timestamp": group["timestamp_sec"].iloc[-1],
                    "duration_ms": (
                        group["timestamp_sec"].iloc[-1] - group["timestamp_sec"].iloc[0]
                    )
                    * 1000,
                    "amplitude_deg": abs(
                        group["gaze_angle_delta_deg"].iloc[-1]
                        - group["gaze_angle_delta_deg"].iloc[0]
                    ),
                    "peak_velocity": group["gaze_angular_velocity"].max(),
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
                    "start_timestamp": group["timestamp_sec"].iloc[0],
                    "stop_timestamp": group["timestamp_sec"].iloc[-1],
                    "duration_ms": (
                        group["timestamp_sec"].iloc[-1] - group["timestamp_sec"].iloc[0]
                    )
                    * 1000,
                    "x": mean_x,
                    "y": mean_y,
                    "radius": radius,
                }
            )
    saccades_df = pd.DataFrame(rows_saccades)
    fixations_df = pd.DataFrame(rows_fixations)

    # for _, row in gaps_df[gaps_df["is_blink"]].iterrows():
    #     condition = (saccades_df["start_timestamp"] >= row["start_timestamp"]) & (
    #         saccades_df["stop_timestamp"] <= row["start_timestamp"]
    #     )
    #     if condition.any():
    #         saccades_df.loc[condition, "stop_timestamp"] = row["start_timestamp"]
    #         saccades_df.loc[condition, "duration_ms"] = (
    #             saccades_df.loc[condition, "stop_timestamp"]
    #             - saccades_df.loc[condition, "start_timestamp"]
    #         ) * 1000
    #     condition = (fixations_df["start_timestamp"] >= row["start_timestamp"]) & (
    #         fixations_df["stop_timestamp"] <= row["start_timestamp"]
    #     )
    #     if condition.any():
    #         fixations_df.loc[condition, "stop_timestamp"] = row["start_timestamp"]
    #         fixations_df.loc[condition, "duration_ms"] = (
    #             fixations_df.loc[condition, "stop_timestamp"]
    #             - fixations_df.loc[condition, "start_timestamp"]
    #         ) * 1000

    return eye_df, fixations_df, saccades_df
