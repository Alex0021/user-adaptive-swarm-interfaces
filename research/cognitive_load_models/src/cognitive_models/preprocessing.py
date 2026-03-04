import glob
from pathlib import Path

import pandas as pd
import tqdm

from cognitive_models.pupil_utils import detect_outliers

from .gaze_utils import (
    calculate_angular_velocity,
    calculate_gaze_angular_delta,
    detect_gaps_and_blinks,
)
from .interpolate import (
    interpolate_gaze,
    interpolate_pupil_data,
    merge_colet_eye_data,
)

# COLET dataset file patterns
PARTICIPANT_FODLER_PATTERN = "participant_{:02d}"
TASK_FOLDER_PATTERN = "Task_{:01d}"
GAZE_FILE_PATTERN = "*gaze.csv"
PUPIL_FILE_PATTERN = "*pupil.csv"
BLINK_FILE_PATTERN = "*blinks.csv"
ANNOTATION_FILENAME = "annotations.csv"

CONFIDENCE_THRESHOLD = 0.5
DURATION_THRESHOLD = 75 / 1000  # 75 ms in seconds
INTERPOLATION_THRESHOLD = 300 / 1000


def load_colet_data(dataset_dir: str, subject_ids: list[int], task_ids: list[int]):
    """
    Loads the Colet dataset for the specified subjects and task IDs
    Args:
        dataset_dir (str): The base directory where the dataset is stored.
        subject_ids (list[int]): A list of subject IDs to load.
        task_ids (list[int]): A list of task IDs to load.
    Returns:
        dataframe: A pandas DataFrame containing the loaded data.
    """
    if not Path(dataset_dir).exists():
        raise FileNotFoundError(
            f"The specified dataset directory '{dataset_dir}' does not exist."
        )

    all_eye_df = pd.DataFrame()
    for subject_id in subject_ids:
        subject_path = Path(dataset_dir) / PARTICIPANT_FODLER_PATTERN.format(subject_id)
        # In every subject folder, there is a "annotations.csv" file containing
        # the NASA TLX scores for each task. Add it also to the dataframe.
        # Only keep mean score for each task
        annotation_file = subject_path / ANNOTATION_FILENAME
        annotation_df = None
        if annotation_file.exists():
            annotation_df = pd.read_csv(annotation_file, header=0)
            annotation_df["mean_score"] = annotation_df.mean(axis=1, numeric_only=True)

        for task_id in task_ids:
            task_path = subject_path / TASK_FOLDER_PATTERN.format(task_id)
            gaze_file = glob.glob(str(subject_path / task_path / GAZE_FILE_PATTERN))
            pupil_file = glob.glob(str(subject_path / task_path / PUPIL_FILE_PATTERN))
            blink_file = glob.glob(str(subject_path / task_path / BLINK_FILE_PATTERN))

            if not gaze_file or not pupil_file or not blink_file:
                print(
                    f"Warning: Missing data for participant {subject_id}, task {task_id}. Skipping."
                )
                continue

            # Load the data files into DataFrames
            gaze_df = pd.read_csv(gaze_file[0])
            pupil_df = pd.read_csv(pupil_file[0])

            # Merge gaze and pupil data, keep blink data separate for now
            merged_df = merge_colet_eye_data(gaze_df, pupil_df, f_des=60)

            # Add subject and task identifiers to the DataFrame
            merged_df["subject_id"] = subject_id
            merged_df["task_id"] = task_id

            # Add CL estimation from NASA RTX annotations
            if annotation_df is not None:
                mean_score = annotation_df.iloc[task_id - 1]["mean_score"]
                merged_df["mean_score"] = mean_score

            # Append the merged DataFrame to the overall DataFrame
            all_eye_df = pd.concat([all_eye_df, merged_df], ignore_index=True)

    return all_eye_df


def load_nback_dataset(
    dataset_dir: str,
    subject_ids: list[int],
    task_ids: list[int],
    trial_ids: list[int],
    f_des: int = 60,
):
    """
    Loads the n-back dataset for the specified subjects, task IDs, and trial IDs.

    Also tries to load the n-back task level

    :param dataset_dir: The base directory where the dataset is stored.
    :param subject_ids: A list of subject UIDs to load.
    :param task_ids: A list of task IDs to load.
    :param trial_ids: A list of trial IDs to load.
    :return: A pandas DataFrame containing the loaded eye-tracking data with annotations.
    """
    all_eye_df = pd.DataFrame()
    for subject_id in tqdm.tqdm(subject_ids, desc="Loading subjects"):
        subject_dir = dataset_dir / subject_id
        if not subject_dir.exists():
            print(f"Warning: Subject directory {subject_dir} does not exist. Skipping.")
            continue
        for task_id in task_ids:
            for trial_id in trial_ids:
                data_path = subject_dir / f"task_{task_id}" / f"trial_{trial_id}"
                # Load eye-tracking data
                eye_data_path = data_path / "gaze_data.csv"
                if not eye_data_path.exists():
                    print(
                        f"    Warning: Eye data file {eye_data_path} does not exist. Skipping."
                    )
                    continue
                eye_df = pd.read_csv(eye_data_path)

                # Substract the start time of the trial from the timestamps to align them
                # And also convert timestamps to pd timedelta in seconds
                if "timestamp" in eye_df.columns:
                    trial_start_time = eye_df["timestamp"].min()
                    eye_df["timestamp"] = (
                        eye_df["timestamp"] - trial_start_time
                    ) / 1000.0  # Convert to seconds

                # Verify sampling rate and interpolate missing data if necessary
                if (
                    eye_df["timestamp"].diff().median() > 1.0 / f_des
                ):  # If median sampling interval is greater than 50 ms
                    print(
                        f"    Warning: Detected irregular sampling in {eye_data_path}. Interpolating missing data."
                    )
                    eye_df["timestamp"] = pd.to_timedelta(eye_df["timestamp"], unit="s")
                    eye_df.set_index("timestamp", inplace=True)
                    eye_df = eye_df.resample(f"{1000 / f_des:.2f}ms").interpolate(
                        method="time"
                    )  # Resample at desired frequency
                    eye_df.reset_index(inplace=True)
                    eye_df["timestamp"] = eye_df[
                        "timestamp"
                    ].dt.total_seconds()  # Convert back to seconds

                # Load nback test results to get the "n" level for this trial
                nback_results_path = data_path / "nback_data.csv"
                if nback_results_path.exists():
                    nback_df = pd.read_csv(nback_results_path)
                    nback_level = (
                        nback_df["nback_level"].iloc[0] if not nback_df.empty else None
                    )
                else:
                    nback_level = None

                # Merge eye-tracking data with annotations (if available)
                eye_df["subject_id"] = subject_id
                eye_df["task_id"] = task_id
                eye_df["trial_id"] = trial_id
                eye_df["nback_level"] = nback_level

                all_eye_df = pd.concat([all_eye_df, eye_df], ignore_index=True)

    all_eye_df.rename(columns={"timestamp": "timestamp_sec"}, inplace=True)
    all_eye_df.drop(columns=["openness_validity"], inplace=True, errors="ignore")

    return all_eye_df


def select_best_eye(df: pd.DataFrame, threshold: float = 0.05):
    """
    Selects the best eye (left, right, or average) based on validity and confidence.

    :param df: DataFrame containing the eye-tracking data with columns for left and right eye features and validity.
    :param threshold: The maximum allowed ratio of valid samples between the two eyes to consider them both valid
    """
    left_cols = [c for c in df.columns if c.startswith("left_")]
    other_cols = [
        c
        for c in df.columns
        if not c.startswith("left_") and not c.startswith("right_")
    ]
    feature_names = [c.removeprefix("left_") for c in left_cols]

    left_valid = df["left_validity"].sum()
    right_valid = df["right_validity"].sum()
    ratio = abs(left_valid / right_valid - 1) if right_valid > 0 else float("inf")

    result = df[other_cols].copy()
    method = ""

    if ratio <= threshold:  # both eyes within 5%
        for feat in feature_names:
            result[feat] = (df[f"left_{feat}"] + df[f"right_{feat}"]) / 2
        method = "mean"
    elif left_valid >= right_valid:  # left eye more confident
        for feat in feature_names:
            result[feat] = df[f"left_{feat}"].values
        method = "left"
    else:  # right eye more confident
        for feat in feature_names:
            result[feat] = df[f"right_{feat}"].values
        method = "right"

    result.rename(
        columns={"validity": "confidence", "pupil_diameter": "pupil_diameter_mm"},
        inplace=True,
    )

    return result, method


def preprocess_colet_data(
    eye_df: pd.DataFrame,
    max_confidence_percentage: int = 30,
    margins: int = 50 / 1000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the Colet dataset by interpolating blinks and low condifence zones.

    Extract blinks & gaps --> calculate gaze angles --> interpolate blinks
      --> interpolate low confidence gaps --> calculate gaze angular velocity

    :param eye_df: DataFrame containing the raw eye-tracking data with columns 'timestamp_sec', 'confidence', and 'blink'.
    :param inter_window_N: The number of samples to consider for fitting interpolation function.
    :param min_num_samples: The minimum number of samples required on either side of a blink for interpolation.
    :param margins: The number of original samples to also inteprolate around the gaps / blinks (avoid edge effects).
    :param verbose: Whether to print progress information during preprocessing.
    :return inter_eye_df, custom_gaze_df, custom_pupil_df, gaps_to_fill_df: The preprocessed DataFrame with interpolated values and the custom blinks DataFrame.
    """
    eye_df = eye_df.copy()
    # Identify blinks and low confidence gaps
    gaps_to_fill_df = detect_gaps_and_blinks(
        eye_df, confidence_threshold=CONFIDENCE_THRESHOLD
    )

    if verbose:
        print(
            f"Identified {len(gaps_to_fill_df[gaps_to_fill_df['is_blink']])} blinks and {len(gaps_to_fill_df)} low confidence gaps to fill."
        )

    # Add percentage of low confidence samples w.r.t total samples
    total_samples = len(eye_df)
    low_confidence_samples = (
        gaps_to_fill_df.loc[~gaps_to_fill_df["is_blink"], "stop_id"]
        - gaps_to_fill_df.loc[~gaps_to_fill_df["is_blink"], "start_id"]
        + 1
    ).sum()
    low_confidence_percent = low_confidence_samples / (total_samples) * 100
    eye_df["low_confidence_percentage"] = low_confidence_percent
    if low_confidence_percent > max_confidence_percentage:
        return eye_df, pd.DataFrame(), pd.DataFrame(), gaps_to_fill_df

    # Remove low confidence samples
    n_to_remove = eye_df[eye_df["confidence"] < CONFIDENCE_THRESHOLD].shape[0]
    eye_df = eye_df[eye_df["confidence"] >= CONFIDENCE_THRESHOLD]
    if verbose:
        print(f"Removed {n_to_remove} low confidence samples from the window.")

    # Remove outliers
    outliers_df = detect_outliers(eye_df, column="pupil_diameter", n_multiplier=10)
    eye_df = eye_df[~eye_df["timestamp_sec"].isin(outliers_df["timestamp_sec"])]
    if verbose:
        print(
            f"Removed {outliers_df.shape[0]} pupil diameter outliers from the window."
        )

    # Remove samples that are within the margins of detected blinks and gaps
    size_before = eye_df.shape[0]
    for _, row in gaps_to_fill_df[
        gaps_to_fill_df["duration_ms"] >= DURATION_THRESHOLD
    ].iterrows():
        idx_to_drop = eye_df[
            (eye_df["timestamp_sec"] >= row["start_timestamp"] - margins)
            & (eye_df["timestamp_sec"] <= row["stop_timestamp"] + margins)
        ].index
        eye_df.drop(idx_to_drop, inplace=True)
    size_after = eye_df.shape[0]
    if verbose:
        print(
            f"Removed {size_before - size_after} samples due to low confidence and proximity to detected blinks/gaps."
        )

    # Calculate gaze angles
    eye_df["gaze_angle_delta_deg"] = calculate_gaze_angular_delta(
        eye_df, gaze_point_columns_prefix="gaze_point_3d"
    )

    # Interpolate data
    pupil_df_inter = interpolate_pupil_data(
        eye_df,
        gaps_to_fill_df,
        column="pupil_diameter",
        max_gap=INTERPOLATION_THRESHOLD,
    )
    gaze_df_inter = interpolate_gaze(
        eye_df,
        gaps_to_fill_df,
        columns=["gaze_angle_delta_deg", "gaze_point_screen_x", "gaze_point_screen_y"],
        max_gap=INTERPOLATION_THRESHOLD,
    )

    # Finally, calculate gaze angular velocity
    gaze_df_inter["gaze_angular_velocity"] = calculate_angular_velocity(gaze_df_inter)

    return eye_df, gaze_df_inter, pupil_df_inter, gaps_to_fill_df


def preprocess_nback_data(
    eye_df: pd.DataFrame,
    max_confidence_percentage: int = 30,
    margins: int = 50 / 1000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the n-back dataset by interpolating blinks and low condifence zones.

    Extract blinks & gaps --> calculate gaze angles --> interpolate blinks
      --> interpolate low confidence gaps --> calculate gaze angular velocity

    :param eye_df: DataFrame containing the raw eye-tracking data with columns 'timestamp_sec', 'confidence', and 'blink'.
    :param inter_window_N: The number of samples to consider for fitting interpolation function.
    :param min_num_samples: The minimum number of samples required on either side of a blink for interpolation.
    :param margins: The number of original samples to also inteprolate around the gaps / blinks (avoid edge effects).
    :param verbose: Whether to print progress information during preprocessing.
    :return inter_eye_df, custom_gaze_df, custom_pupil_df, gaps_to_fill_df: The preprocessed DataFrame with interpolated values and the custom blinks DataFrame.
    """
    eye_df = eye_df.copy()

    eye_df, best_eye = select_best_eye(eye_df, threshold=0.05)
    eye_df["confidence_method"] = best_eye

    # Identify blinks and low confidence gaps
    gaps_to_fill_df = detect_gaps_and_blinks(
        eye_df,
        confidence_threshold=0.95,
        blink_threshold_range=(100, 300),
        eye_openness_column="openness",
        openness_threshold=0.5,
    )

    if verbose:
        print(
            f"Identified {len(gaps_to_fill_df[gaps_to_fill_df['is_blink']])} blinks and {len(gaps_to_fill_df)} low confidence gaps to fill."
        )

    # Add percentage of low confidence samples w.r.t total samples
    total_samples = len(eye_df)
    low_confidence_samples = (
        gaps_to_fill_df.loc[~gaps_to_fill_df["is_blink"], "stop_id"]
        - gaps_to_fill_df.loc[~gaps_to_fill_df["is_blink"], "start_id"]
        + 1
    ).sum()
    low_confidence_percent = low_confidence_samples / (total_samples) * 100
    eye_df["low_confidence_percentage"] = low_confidence_percent
    if low_confidence_percent > max_confidence_percentage:
        return eye_df, pd.DataFrame(), pd.DataFrame(), gaps_to_fill_df

    # Remove low confidence samples
    n_to_remove = eye_df[eye_df["confidence"] < CONFIDENCE_THRESHOLD].shape[0]
    eye_df = eye_df[eye_df["confidence"] >= CONFIDENCE_THRESHOLD]
    if verbose:
        print(f"Removed {n_to_remove} low confidence samples from the window.")

    # Remove outliers
    outliers_df = detect_outliers(eye_df, column="pupil_diameter_mm", n_multiplier=10)
    eye_df = eye_df[~eye_df["timestamp_sec"].isin(outliers_df["timestamp_sec"])]
    if verbose:
        print(
            f"Removed {outliers_df.shape[0]} pupil diameter outliers from the window."
        )

    # Remove samples that are within the margins of detected blinks and gaps
    size_before = eye_df.shape[0]
    for _, row in gaps_to_fill_df[
        gaps_to_fill_df["duration_ms"] >= DURATION_THRESHOLD
    ].iterrows():
        idx_to_drop = eye_df[
            (eye_df["timestamp_sec"] >= row["start_timestamp"] - margins)
            & (eye_df["timestamp_sec"] <= row["stop_timestamp"] + margins)
        ].index
        eye_df.drop(idx_to_drop, inplace=True)
    size_after = eye_df.shape[0]
    if verbose:
        print(
            f"Removed {size_before - size_after} samples due to low confidence and proximity to detected blinks/gaps."
        )

    # TEMP FIX
    M_WIDTH_MM, M_HEIGHT_MM = 582, 363
    # Replace screen point x,y by gaze_point x,y using the screen size
    eye_df["gaze_point_x"] = eye_df["point_screen_x"] * M_WIDTH_MM
    eye_df["gaze_point_y"] = eye_df["point_screen_y"] * M_HEIGHT_MM

    # Calculate gaze angles
    eye_df["gaze_angle_delta_deg"] = calculate_gaze_angular_delta(
        eye_df, gaze_point_columns_prefix="gaze_point"
    )

    # Interpolate data
    pupil_df_inter = interpolate_pupil_data(
        eye_df,
        gaps_to_fill_df,
        column="pupil_diameter_mm",
        max_gap=INTERPOLATION_THRESHOLD,
    )
    gaze_df_inter = interpolate_gaze(
        eye_df,
        gaps_to_fill_df,
        columns=["gaze_angle_delta_deg", "point_screen_x", "point_screen_y"],
        max_gap=INTERPOLATION_THRESHOLD,
    )

    # Finally, calculate gaze angular velocity
    gaze_df_inter["gaze_angular_velocity"] = calculate_angular_velocity(gaze_df_inter)

    # Rename some columns to be consistent with Colet dataset
    gaze_df_inter.rename(
        columns={
            "point_screen_x": "norm_pos_x",
            "point_screen_y": "norm_pos_y",
        },
        inplace=True,
    )

    return eye_df, gaze_df_inter, pupil_df_inter, gaps_to_fill_df
