import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def merge_colet_eye_data(raw_gaze_df, raw_eye_df, f_des=60):
    """
    Merges the gaze and pupil dataframes on the timestamp column and checks if the frequency of the merged dataframe is consistent with the desired frequency.

    """
    # First keep only the best pupil meausurement for each timestamp
    eye_df_best = raw_eye_df.groupby(
        ["pupil_timestamp", "eye_id"], as_index=False
    ).apply(lambda x: x.loc[x["confidence"].idxmax()])
    eye_df_best.reset_index(drop=True, inplace=True)
    eye_df_best.drop(columns=["norm_pos_x", "norm_pos_y"], inplace=True)
    eye_df_best.rename(
        columns={"diameter": "pupil_diameter_mm", "confidence": "pupil_confidence"},
        inplace=True,
    )

    # Merge the gaze and pupil dataframes on the timestamp column
    gaze_start_time = raw_gaze_df["gaze_timestamp"].min()
    merged_df = pd.merge_asof(
        raw_gaze_df.sort_values("gaze_timestamp"),
        eye_df_best.sort_values("pupil_timestamp"),
        left_on="gaze_timestamp",
        right_on="pupil_timestamp",
        direction="nearest",
        tolerance=1.0 / 240 * 2,
    )  # tolerance in ms
    print(
        f"There are {merged_df[merged_df.pupil_confidence.isna()].index.size} non matching timestamp within the limits"
    )
    # Since only one value, drop the row
    merged_df_clean = merged_df.dropna()
    # Let's reindex the dataframe using timestamp starting from 0 and dropping the original timestamp columns
    merged_df_clean["timestamp"] = merged_df_clean["gaze_timestamp"] - gaze_start_time
    merged_df_clean.drop(columns=["gaze_timestamp", "pupil_timestamp"], inplace=True)
    merged_df_clean = (
        merged_df_clean.groupby(
            (merged_df_clean["timestamp"] // (1.0 / 240 / 2)).astype(int)
        )
        .mean()
        .reset_index(drop=True)
    )
    print(f"Final merged dataset has {merged_df_clean.shape[0]} records at 240 Hz")

    # Then adjust sampling frequency to the desired frequency
    T = f"{1.0/f_des:.2f}ms"
    merged_df_clean["timestamp"] = pd.to_timedelta(
        merged_df_clean["timestamp"], unit="s"
    )
    merged_df_clean.set_index("timestamp", inplace=True)
    merged_df_clean = (
        merged_df_clean.resample(T).nearest().interpolate()
    )  # Resample to 60 Hz and interpolate missing values
    merged_df_clean.reset_index(inplace=True)
    merged_df_clean["timestamp"] = merged_df_clean[
        "timestamp"
    ].dt.total_seconds()  # Convert back to seconds
    print(
        f"Final merged and resampled dataset has {merged_df_clean.shape[0]} records at 60 Hz"
    )

    return merged_df_clean


def interpolate_blinks(eye_df, blink_df, N=100, MIN_SAMPLES=5, MARGIN=5):
    """
    Interpolate pupil diameter during blinks using linear interpolation.

    Args:
        eye_df (pd.DataFrame): Dataframe containing all eye data
        blink_df (pd.DataFrame): DataFrame containing 'start_id' and 'stop_id' columns for blinks.
        N (int): Total number of samples to consider around the blink for interpolation.
        MIN_SAMPLES (int): Minimum number of samples required on either side of the blink for interpolation.
        MARGIN (int): Number of original samples to exclude around the blinks to be interpolated.
    """
    if eye_df is None or blink_df is None:
        raise ValueError("eye_df and blink_df cannot be None")

    if not {"start_id", "stop_id"}.issubset(blink_df.columns):
        raise ValueError("blink_df must contain 'start_id' and 'stop_id' columns")

    if "pupil_diameter_mm" in eye_df.columns:
        inter_eye_df = inter_eye_df.copy()
        if "interpolated" not in inter_eye_df.columns:
            inter_eye_df["interpolated"] = False
        inter_data_df = pd.DataFrame(
            columns=[
                "blink_id",
                "original_pupil_diameter_mm",
                "inter_pupil_diameter_mm",
            ]
        )
        for idx, blink in blink_df.iterrows():
            start_id = blink["start_id"].astype(int)
            stop_id = blink["stop_id"].astype(int)

            if start_id < 0 or stop_id > inter_eye_df.index.max():
                print(
                    f"Skipping invalid blink with start_id {start_id} and stop_id {stop_id}"
                )
                continue

            # Check if window around the data fits within the dataframe
            window_start = max(0, start_id - N // 2)
            window_end = min(len(inter_eye_df) - 1, stop_id + N // 2)
            pre_samples_count = (start_id - MARGIN) - window_start
            post_samples_count = window_end - (stop_id + MARGIN)
            if pre_samples_count < MIN_SAMPLES or post_samples_count < MIN_SAMPLES:
                print(
                    f"Skipping blink with insufficient data around start_id {start_id} and stop_id {stop_id}"
                )
                continue

            # For interpolation, using timestamps outside the blinks
            pre_blink_data = inter_eye_df.loc[
                window_start : start_id - MARGIN - 1, "pupil_diameter_mm"
            ]
            post_blink_data = inter_eye_df.loc[
                stop_id + MARGIN + 1 : window_end, "pupil_diameter_mm"
            ]
            x = pd.concat([pre_blink_data, post_blink_data]).index
            y = pd.concat([pre_blink_data, post_blink_data]).values
            interp_func = interp1d(x, y, kind="slinear")
            blink_indices = range(start_id - MARGIN, stop_id + MARGIN + 1)
            interpolated_values = interp_func(blink_indices)
            original_values = (
                inter_eye_df.loc[blink_indices, "pupil_diameter_mm"].copy().values
            )
            inter_eye_df.loc[blink_indices, "pupil_diameter_mm"] = interpolated_values
            inter_eye_df.loc[blink_indices, "interpolated"] = True
            inter_data_df = pd.concat(
                [
                    inter_data_df,
                    pd.DataFrame(
                        {
                            "blink_id": idx,
                            "original_pupil_diameter_mm": original_values,
                            "inter_pupil_diameter_mm": interpolated_values,
                        }
                    ),
                ],
                ignore_index=True,
            )
    if "gaze_angular_delta_deg" in inter_eye_df.columns:
        for _, row in blink_df.iterrows():
            # Set gaze points to NaN
            columns = ["gaze_angular_delta_deg"]
            inter_eye_df.loc[
                (inter_eye_df.index >= row["start_id"] - MARGIN)
                & (inter_eye_df.index <= row["stop_id"] + MARGIN),
                columns,
            ] = np.nan
        inter_eye_df["gaze_angle_delta_deg"].isna().sum()

    return inter_eye_df, inter_data_df


def interpolate_missing_gaze(eye_data_df, gaps_df, N=100, MIN_SAMPLES=10, MARGIN=5):
    if "gaze_angle_delta_deg" not in eye_data_df.columns:
        raise ValueError("eye_data_df must contain 'gaze_angle_delta_deg' column")
    if "pupil_diameter_mm" not in eye_data_df.columns:
        raise ValueError("eye_data_df must contain 'pupil_diameter_mm' column")

    if "id" not in gaps_df.columns:
        raise ValueError("gaps_df must contain 'id' column with 'min' and 'max' keys")

    eye_data_df = eye_data_df.copy()
    eye_data_df["interpolated"] = False
    inter_data_df = pd.DataFrame(
        columns=[
            "gap_id",
            "original_gaze_angle_delta_deg",
            "original_pupil_diameter_mm",
            "inter_gaze_angle_delta_deg",
            "inter_pupil_diameter_mm",
        ]
    )

    for _, gap in gaps_df.iterrows():
        start_id = gap[("id", "min")].astype(int)
        stop_id = gap[("id", "max")].astype(int)

        if start_id < 0 or stop_id >= len(eye_data_df):
            print(
                f"Skipping invalid gap with start_id {start_id} and stop_id {stop_id}"
            )
            continue

        # Check if window around the data fits within the dataframe
        window_start = max(0, start_id - N // 2)
        window_end = min(len(eye_data_df) - 1, stop_id + N // 2)
        pre_samples_count = (start_id - MARGIN) - window_start
        post_samples_count = window_end - (stop_id + MARGIN)
        if pre_samples_count < MIN_SAMPLES or post_samples_count < MIN_SAMPLES:
            print(
                f"Skipping gap with insufficient data around start_id {start_id} and stop_id {stop_id}"
            )
            continue

        # Discard other low confidence samples (gaps) in the currrent window
        # to avoid using them for interpolation
        gaps_in_window = gaps_df[
            (gaps_df[("id", "min")] >= window_start)
            & (gaps_df[("id", "max")] <= window_end)
        ]
        if (
            gaps_in_window[("id", "max")] - gaps_in_window[("id", "min")]
        ).sum() > 0.5 * N:
            print(
                f"Skipping gap with too many other gaps in the window around start_id {start_id} and stop_id {stop_id}"
            )
            continue

        # For interpolation, using timestamps outside the gaps
        indices = list(range(window_start, window_end + 1))
        for _, other_gap in gaps_in_window.iterrows():
            other_start_id = other_gap[("id", "min")].astype(int)
            other_stop_id = other_gap[("id", "max")].astype(int)
            for i in range(other_start_id, other_stop_id + 1):
                indices.remove(i)
        x = eye_data_df.loc[indices, "timestamp"]
        y_angle = eye_data_df.loc[indices, "gaze_angle_delta_deg"]
        y_pupil = eye_data_df.loc[indices, "pupil_diameter_mm"]

        # Create interpolation functions
        interp_angle = interp1d(x, y_angle, kind="linear")
        interp_pupil = interp1d(x, y_pupil, kind="linear")

        # Interpolate values for the gap
        gap_timestamps = eye_data_df.loc[start_id:stop_id, "timestamp"]
        interpolated_angles = interp_angle(gap_timestamps)
        interpolated_pupils = interp_pupil(gap_timestamps)

        original_angles = (
            eye_data_df.loc[start_id:stop_id, "gaze_angle_delta_deg"].copy().values
        )
        original_pupils = (
            eye_data_df.loc[start_id:stop_id, "pupil_diameter_mm"].copy().values
        )
        # Fill in the interpolated values
        eye_data_df.loc[start_id:stop_id, "gaze_angle_delta_deg"] = interpolated_angles
        eye_data_df.loc[start_id:stop_id, "pupil_diameter_mm"] = interpolated_pupils
        eye_data_df.loc[start_id:stop_id, "interpolated"] = True

        # Store interpolation data for analysis
        inter_data_df = pd.concat(
            [
                inter_data_df,
                pd.DataFrame(
                    {
                        "gap_id": start_id,
                        "original_gaze_angle_delta_deg": original_angles,
                        "original_pupil_diameter_mm": original_pupils,
                        "inter_gaze_angle_delta_deg": interpolated_angles,
                        "inter_pupil_diameter_mm": interpolated_pupils,
                    }
                ),
            ],
            ignore_index=True,
        )

    return eye_data_df, inter_data_df
