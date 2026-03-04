from collections import defaultdict
from typing import Any

import pandas as pd

from .gaze_utils import (
    calculate_fixations_saccades_idt,
    calculate_fixations_saccades_ivt,
)
from .pupil_utils import lhipa, ripa2


def extract_window_features(
    window_pupil_df: pd.DataFrame,
    fixations_df: pd.DataFrame,
    saccades_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Extracts features from the eye-tracking window data, including fixations_df, saccades_df, and blinks.

    :param window_pupil_df: DataFrame containing pupil data for the window.
    :param fixations_df: DataFrame containing fixations_df data for the window.
    :param saccades_df: DataFrame containing saccades_df data for the window.
    :param gaps_df: DataFrame containing gap information with columns 'start_timestamp', 'stop_timestamp', and 'is_blink'.
    :param ivt_threshold: The velocity threshold (in deg/s) for identifying saccades_df.
    :param min_fixation_duration: The minimum duration (in milliseconds) for a fixation to be considered valid.
    :return features: A dictionary containing extracted features.
    """
    features = defaultdict(lambda: 0)
    # 1- Fixations_df: count, duration mean/max/min/std
    features["fixations_count"] = len(fixations_df)
    if not fixations_df.empty:
        features["fixations_duration_mean"] = fixations_df["duration_ms"].mean()
        features["fixations_duration_max"] = fixations_df["duration_ms"].max()
        features["fixations_duration_min"] = fixations_df["duration_ms"].min()
        features["fixations_duration_skew"] = fixations_df["duration_ms"].skew()
        features["fixations_duration_kurt"] = fixations_df["duration_ms"].kurtosis()
        features["fixations_duration_std"] = (
            fixations_df["duration_ms"].std() if len(fixations_df) > 1 else 0
        )

    # 2- Saccades_df: count, peak_velocity, amplitude mean/max/min/std, duration mean/max/min/std
    features["saccades_count"] = len(saccades_df)
    if not saccades_df.empty:
        features["saccades_peak_velocity_min"] = saccades_df["peak_velocity"].min()
        features["saccades_peak_velocity_max"] = saccades_df["peak_velocity"].max()
        features["saccades_peak_velocity_skew"] = saccades_df["peak_velocity"].skew()
        features["saccades_peak_velocity_kurt"] = saccades_df[
            "peak_velocity"
        ].kurtosis()
        features["saccades_amplitude_mean"] = saccades_df["amplitude"].mean()
        features["saccades_amplitude_max"] = saccades_df["amplitude"].max()
        features["saccades_amplitude_min"] = saccades_df["amplitude"].min()
        features["saccades_amplitude_skew"] = saccades_df["amplitude"].skew()
        features["saccades_amplitude_kurt"] = saccades_df["amplitude"].kurtosis()
        features["saccades_amplitude_std"] = (
            saccades_df["amplitude"].std() if len(saccades_df) > 1 else 0
        )
        features["saccades_duration_mean"] = saccades_df["duration_ms"].mean()
        features["saccades_duration_max"] = saccades_df["duration_ms"].max()
        features["saccades_duration_min"] = saccades_df["duration_ms"].min()
        features["saccades_duration_skew"] = saccades_df["duration_ms"].skew()
        features["saccades_duration_kurt"] = saccades_df["duration_ms"].kurtosis()
        features["saccades_duration_std"] = (
            saccades_df["duration_ms"].std() if len(saccades_df) > 1 else 0
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
