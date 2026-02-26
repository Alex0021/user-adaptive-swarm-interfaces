from collections import defaultdict
from typing import Any

import pandas as pd

from .gaze_utils import calculate_fixations_saccades
from .pupil_utils import lhipa, ripa2


def extract_window_features(
    window_df: pd.DataFrame,
    window_gaze_df: pd.DataFrame,
    window_pupil_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    ivt_threshold: int,
    min_fixation_duration: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Extracts features from the eye-tracking window data, including fixations, saccades, and blinks.

    :param window_df:With columns 'timestamp_sec', 'confidence', 'blink', 'gaze_angle_delta_deg', and 'gaze_angular_velocity'.
    :param window_gaze_df: DataFrame containing gaze data for the window.
    :param window_pupil_df: DataFrame containing pupil data for the window.
    :param gaps_df: DataFrame containing gap information with columns 'start_timestamp', 'stop_timestamp', and 'is_blink'.
    :param ivt_threshold: The velocity threshold (in deg/s) for identifying saccades.
    :param min_fixation_duration: The minimum duration (in milliseconds) for a fixation to be considered valid.
    :return features: A dictionary containing extracted features.
    """
    # 1- Extract fixations and saccades
    _, fixations, saccades = calculate_fixations_saccades(
        window_gaze_df, gaps_df, ivt_threshold, min_fixation_duration, verbose=verbose
    )

    features = defaultdict(lambda: 0)
    # 2- Fixations: count, duration mean/max/min/std
    features["fixations_count"] = len(fixations)
    if not fixations.empty:
        features["fixations_duration_mean"] = fixations["duration_ms"].mean()
        features["fixations_duration_max"] = fixations["duration_ms"].max()
        features["fixations_duration_min"] = fixations["duration_ms"].min()
        features["fixations_duration_skew"] = fixations["duration_ms"].skew()
        features["fixations_duration_kurt"] = fixations["duration_ms"].kurtosis()
        features["fixations_duration_std"] = (
            fixations["duration_ms"].std() if len(fixations) > 1 else 0
        )

    # 3- Saccades: count, peak_velocity, amplitude mean/max/min/std, duration mean/max/min/std
    features["saccades_count"] = len(saccades)
    if not saccades.empty:
        features["saccades_peak_velocity_min"] = saccades["peak_velocity"].min()
        features["saccades_peak_velocity_max"] = saccades["peak_velocity"].max()
        features["saccades_peak_velocity_skew"] = saccades["peak_velocity"].skew()
        features["saccades_peak_velocity_kurt"] = saccades["peak_velocity"].kurtosis()
        features["saccades_amplitude_mean"] = saccades["amplitude_deg"].mean()
        features["saccades_amplitude_max"] = saccades["amplitude_deg"].max()
        features["saccades_amplitude_min"] = saccades["amplitude_deg"].min()
        features["saccades_amplitude_skew"] = saccades["amplitude_deg"].skew()
        features["saccades_amplitude_kurt"] = saccades["amplitude_deg"].kurtosis()
        features["saccades_amplitude_std"] = (
            saccades["amplitude_deg"].std() if len(saccades) > 1 else 0
        )
        features["saccades_duration_mean"] = saccades["duration_ms"].mean()
        features["saccades_duration_max"] = saccades["duration_ms"].max()
        features["saccades_duration_min"] = saccades["duration_ms"].min()
        features["saccades_duration_skew"] = saccades["duration_ms"].skew()
        features["saccades_duration_kurt"] = saccades["duration_ms"].kurtosis()
        features["saccades_duration_std"] = (
            saccades["duration_ms"].std() if len(saccades) > 1 else 0
        )

    # 4- Blinks: count, duration mean
    blink_df = gaps_df[gaps_df["is_blink"]]
    if blink_df.empty:
        features["blinks_count"] = 0
        features["blinks_duration_max"] = 0
        features["blinks_duration_min"] = 0
    else:
        features["blinks_count"] = len(blink_df)
        features["blinks_duration_max"] = blink_df["duration_ms"].max()
        features["blinks_duration_min"] = blink_df["duration_ms"].min()

    # 5- Pupil related features
    features["pupil_lhipa"] = lhipa(window_pupil_df, wavelet_type="sym8")
    features["pupil_ripa2"] = ripa2(window_pupil_df, VLF=(98, 2), LF=(13, 4), D=1)

    return features
