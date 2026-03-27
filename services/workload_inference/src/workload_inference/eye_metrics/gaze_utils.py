import numpy as np
import pandas as pd


def detect_gaps_and_blinks(
    df: pd.DataFrame,
    confidence_threshold: float = 0.5,
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


def calculate_fixations_saccades_idt(
    eye_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    idt_duration_threshold: int = 100,
    idt_dispersion_threshold: float = 0.05,
    fs: int = 60,
    verbose: bool = True,
):
    """
    Make sure the dataframe has the following columns: 'timestamp_sec', 'gaze_point_screen_x', 'gaze_point_screen_y'

     :param eye_df: DataFrame containing eye-tracking data with required columns.
    :param gaps_df: DataFrame containing information about gaps in the data.
    :param idt_duration_threshold: Duration threshold in milliseconds for the IDT algorithm.
    :param idt_dispersion_threshold: Dispersion threshold in degrees for the IDT algorithm.
    :param fs: Sampling frequency of the eye-tracking data in Hz.
    :param verbose: If True, print verbose information about the detection process.
    """

    def dispersion(x_points: pd.Series, y_points: pd.Series):
        return np.sqrt(
            (x_points.max() - x_points.min()) ** 2
            + (y_points.max() - y_points.min()) ** 2
        )

    window_size = int(idt_duration_threshold / 1000 * fs)
    sliding_window_start = 0
    gaps_to_skip = gaps_df[(gaps_df["duration_ms"] >= idt_duration_threshold)]
    # Make sure initial window does not contain a gap
    fixations_rows = []
    if (gaps_to_skip["start_timestamp"] < idt_duration_threshold / 1000).any():
        sliding_window_start = window_size

    while sliding_window_start < eye_df.shape[0]:
        end_idx = min(sliding_window_start + window_size, eye_df.shape[0])
        sliding_window = eye_df.iloc[sliding_window_start:end_idx]
        disp_value = dispersion(
            sliding_window["gaze_point_screen_x"], sliding_window["gaze_point_screen_y"]
        )
        if disp_value <= idt_dispersion_threshold:
            while disp_value <= idt_dispersion_threshold:
                end_idx += 1
                if end_idx >= eye_df.shape[0]:
                    break
                sliding_window = eye_df.iloc[sliding_window_start:end_idx]
                disp_value = dispersion(
                    sliding_window["gaze_point_screen_x"],
                    sliding_window["gaze_point_screen_y"],
                )
                if disp_value > idt_dispersion_threshold:
                    break
            mean_x = sliding_window["gaze_point_screen_x"].mean()
            mean_y = sliding_window["gaze_point_screen_y"].mean()
            radius = np.sqrt(
                (sliding_window["gaze_point_screen_x"] - mean_x) ** 2
                + (sliding_window["gaze_point_screen_y"] - mean_y) ** 2
            ).max()
            duration = (
                sliding_window["timestamp_sec"].iloc[-1]
                - sliding_window["timestamp_sec"].iloc[0]
            ) * 1000
            fixations_rows.append(
                {
                    "start_timestamp": sliding_window["timestamp_sec"].iloc[0],
                    "stop_timestamp": sliding_window["timestamp_sec"].iloc[-1],
                    "duration_ms": duration,
                    "x": mean_x,
                    "y": mean_y,
                    "radius": radius,
                    "start_idx": sliding_window.index[0],
                    "stop_idx": sliding_window.index[-1],
                }
            )
            sliding_window_start = end_idx + 1
        else:
            sliding_window_start += 1

    fixations_df_idt = pd.DataFrame(fixations_rows)

    # Saccades are the non-fixation periods between fixations that are not gaps
    saccades_rows = []
    gaps_too_long = gaps_df[gaps_df["duration_ms"] >= idt_duration_threshold]
    for i in range(len(fixations_df_idt) - 1):
        saccade_start_time = fixations_df_idt["stop_timestamp"].iloc[i]
        saccade_end_time = fixations_df_idt["start_timestamp"].iloc[i + 1]
        overlapping_gaps = gaps_too_long[
            (gaps_too_long["start_timestamp"] < saccade_end_time)
            & (gaps_too_long["stop_timestamp"] > saccade_start_time)
        ]
        if not overlapping_gaps.empty:
            # Make sure to identify saccades around the gaps
            for _, gap in overlapping_gaps.iterrows():
                if gap["start_timestamp"] > saccade_start_time:
                    saccade_end_time = gap["start_timestamp"]
                if gap["stop_timestamp"] < saccade_end_time:
                    saccade_start_time = gap["stop_timestamp"]
                start_idx = eye_df[eye_df["timestamp_sec"] >= saccade_start_time][
                    "timestamp_sec"
                ].idxmin()
                stop_idx = eye_df[eye_df["timestamp_sec"] <= saccade_end_time][
                    "timestamp_sec"
                ].idxmax()
                amplitude = dispersion(
                    eye_df.loc[start_idx:stop_idx, "gaze_point_screen_x"],
                    eye_df.loc[start_idx:stop_idx, "gaze_point_screen_y"],
                )
                saccades_rows.append(
                    {
                        "start_timestamp": saccade_start_time,
                        "stop_timestamp": saccade_end_time,
                        "duration_ms": (saccade_end_time - saccade_start_time) * 1000,
                        "start_idx": start_idx,
                        "stop_idx": stop_idx,
                        "amplitude": amplitude,
                        "peak_velocity": eye_df["gaze_angular_velocity"]
                        .iloc[start_idx:stop_idx]
                        .max(),
                        "velocity": eye_df["gaze_angular_velocity"]
                        .iloc[start_idx:stop_idx]
                        .mean(),
                    }
                )
        else:
            start_idx = fixations_df_idt["stop_idx"].iloc[i]
            stop_idx = fixations_df_idt["start_idx"].iloc[i + 1]
            amplitude = np.sqrt(
                (
                    eye_df.loc[start_idx:stop_idx, "gaze_point_screen_x"].max()
                    - eye_df.loc[start_idx:stop_idx, "gaze_point_screen_x"].min()
                )
                ** 2
                + (
                    eye_df.loc[start_idx:stop_idx, "gaze_point_screen_y"].max()
                    - eye_df.loc[start_idx:stop_idx, "gaze_point_screen_y"].min()
                )
                ** 2
            )
            saccades_rows.append(
                {
                    "start_timestamp": saccade_start_time,
                    "stop_timestamp": saccade_end_time,
                    "duration_ms": (saccade_end_time - saccade_start_time) * 1000,
                    "start_idx": start_idx,
                    "stop_idx": stop_idx,
                    "amplitude": amplitude,
                    "peak_velocity": eye_df["gaze_angular_velocity"]
                    .iloc[start_idx:stop_idx]
                    .max(),
                    "velocity": eye_df["gaze_angular_velocity"]
                    .iloc[start_idx:stop_idx]
                    .mean(),
                }
            )
    saccades_df_idt = pd.DataFrame(saccades_rows)

    if verbose:
        print(
            f"Detected {len(fixations_df_idt)} fixations and {len(saccades_df_idt)} saccades with IDT algorithm"
        )

    return fixations_df_idt, saccades_df_idt
