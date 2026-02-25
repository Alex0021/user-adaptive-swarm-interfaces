import glob
from pathlib import Path

import pandas as pd

from .gaze_utils import (
    calculate_angular_velocity,
    calculate_gaze_angular_delta,
    detect_gaps_and_blinks,
)
from .interpolate import (
    interpolate_blinks,
    interpolate_missing_gaze,
    merge_colet_eye_data,
)

# COLET dataset file patterns
PARTICIPANT_FODLER_PATTERN = "participant_{:02d}"
TASK_FOLDER_PATTERN = "Task_{:01d}"
GAZE_FILE_PATTERN = "*gaze.csv"
PUPIL_FILE_PATTERN = "*pupil.csv"
BLINK_FILE_PATTERN = "*blinks.csv"
ANNOTATION_FILENAME = "annotations.csv"


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
                merged_df["cl_class"] = (
                    "high"
                    if mean_score > 49
                    else ("low" if mean_score < 30 else "medium")
                )

            # Append the merged DataFrame to the overall DataFrame
            all_eye_df = pd.concat([all_eye_df, merged_df], ignore_index=True)

            # Drop some unnecessary columns from the dataframes
            all_eye_df.drop(
                columns=["confidence", "eye_id"]
                + [c for c in all_eye_df.columns if c.startswith("ellipse_")],
                inplace=True,
                errors="ignore",
            )

    return all_eye_df


def preprocess_colet_data(
    eye_df: pd.DataFrame,
    inter_window_N: int = 100,
    min_num_samples: int = 5,
    margins: int = 5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the Colet dataset by interpolating blinks and low condifence zones.

    Extract blinks & gaps --> calculate gaze angles --> interpolate blinks
      --> interpolate low confidence gaps --> calculate gaze angular velocity

    :param eye_df: DataFrame containing the raw eye-tracking data with columns 'timestamp_sec', 'confidence', and 'blink'.
    :param inter_window_N: The number of samples to consider for fitting interpolation function.
    :param min_num_samples: The minimum number of samples required on either side of a blink for interpolation.
    :param margins: The number of original samples to also inteprolate around the gaps / blinks (avoid edge effects).
    :param verbose: Whether to print progress information during preprocessing.
    :return inter_eye_df, custom_blinks_df: The preprocessed DataFrame with interpolated values and the custom blinks DataFrame.
    """
    eye_df = eye_df.copy()
    # Identify blinks and low confidence gaps
    gaps_to_fill_df, custom_blinks_df = detect_gaps_and_blinks(eye_df)

    if verbose:
        print(
            f"Identified {len(custom_blinks_df)} blinks and {len(gaps_to_fill_df)} low confidence gaps to fill."
        )

    # Add percentage of low confidence samples w.r.t total samples
    total_samples = len(eye_df)
    low_confidence_samples = (
        gaps_to_fill_df["stop_id"] - gaps_to_fill_df["start_id"] + 1
    ).sum()
    blink_samples = (
        custom_blinks_df["stop_id"] - custom_blinks_df["start_id"] + 1
    ).sum()
    eye_df["low_confidence_percentage"] = (
        low_confidence_samples / (total_samples - blink_samples) * 100
    )
    # Calculate gaze angles
    eye_df["gaze_angle_delta_deg"] = calculate_gaze_angular_delta(eye_df)

    # Interpolate blinks
    eye_df, blink_interpolation_info = interpolate_blinks(
        eye_df,
        custom_blinks_df,
        inter_window_N,
        min_num_samples,
        margins,
        verbose=verbose,
    )
    if verbose:
        print(
            f"Succesfully interpolated {len(blink_interpolation_info)} pupil samples within blinks."
        )

    # Interpolate missing gaze data
    eye_df, missing_gaze_interpolation_info = interpolate_missing_gaze(
        eye_df,
        gaps_to_fill_df,
        inter_window_N,
        min_num_samples,
        margins,
        verbose=verbose,
    )
    if verbose:
        print(
            f"Successfully interpolated {len(missing_gaze_interpolation_info)} "
            "low confidence samples during gaps."
        )

    # Finally, calculate gaze angular velocity
    eye_df["gaze_angular_velocity"] = calculate_angular_velocity(eye_df)

    return eye_df, custom_blinks_df
