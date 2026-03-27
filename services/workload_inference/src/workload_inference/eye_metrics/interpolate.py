import pandas as pd


def interpolate_pupil_data(
    eye_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    column: str = "pupil_diameter",
    max_gap: int = 300,
) -> pd.DataFrame:
    return interpolate_eye_data(eye_df, gaps_df, columns=[column], max_gap=max_gap)


def interpolate_eye_data(
    eye_df: pd.DataFrame, gaps_df: pd.DataFrame, columns: list[str], max_gap: int
) -> pd.DataFrame:
    interpolated_df = eye_df[["timestamp_sec"] + columns].copy()
    interpolated_df["timestamp_sec"] = pd.to_timedelta(
        interpolated_df["timestamp_sec"], unit="s"
    )
    interpolated_df.set_index("timestamp_sec", inplace=True)
    interpolated_df.drop_duplicates(inplace=True)
    interpolated_df = interpolated_df.resample("16.67ms").interpolate(method="slinear")
    interpolated_df.reset_index(inplace=True)
    interpolated_df["timestamp_sec"] = interpolated_df[
        "timestamp_sec"
    ].dt.total_seconds()
    # Remove zones that exceed the interpolation threshold
    for _, row in gaps_df[gaps_df["duration_ms"] >= max_gap].iterrows():
        interpolated_df = interpolated_df[
            (interpolated_df["timestamp_sec"] < row["start_timestamp"])
            | (interpolated_df["timestamp_sec"] > row["stop_timestamp"])
        ]
    interpolated_df["is_interpolated"] = ~interpolated_df["timestamp_sec"].isin(
        eye_df["timestamp_sec"]
    )
    interpolated_df.reset_index(drop=True, inplace=True)

    return interpolated_df


def interpolate_gaze(
    eye_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    columns: list[str] = ["gaze_angle_delta_deg"],
    max_gap: int = 300,
) -> pd.DataFrame:
    interpolated_df = interpolate_eye_data(
        eye_df, gaps_df, columns=columns, max_gap=max_gap
    )
    # Also remove blinks from the interpolated gaze angle dataframe
    for _, row in gaps_df[gaps_df["is_blink"]].iterrows():
        interpolated_df = interpolated_df[
            (interpolated_df["timestamp_sec"] < row["start_timestamp"])
            | (interpolated_df["timestamp_sec"] > row["stop_timestamp"])
        ]
    interpolated_df.reset_index(drop=True, inplace=True)

    return interpolated_df
