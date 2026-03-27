import logging
from typing import Any

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

FEATURE_SETS = {
    "colet": [r"pupil_diameter", r"saccades_", r"fixations_", r"blinks_"],
    "all": [
        r"pupil_.*wv",
        r"pupil_.*ipa",
        r"saccades_.*_(m|(std))",
        r"blinks_",
        r"fixations_.*_(m|(std))",
    ],
    "no_fixations": [r"pupil_.*wv", r"pupil_.*ipa", r"saccades_", r"blinks_"],
    "no_blinks": [r"pupil_.*wv", r"pupil_.*ipa", r"saccades_", r"fixations_"],
    "no_fix_no_blinks": [r"pupil_.*wv", r"pupil_.*ipa", r"saccades_"],
    "no_wavelets": [r"pupil_.*ipa", r"saccades_", r"fixations_"],
    "ipa_wavelets": [r"pupil_.*wv", r"pupil_.*ipa"],
    "pupil": [r"pupil_"],
}

_DEFAULT_FEATURES: dict[str, float] = {
    "fixations_count": 0.0,
    "fixations_frequency": 0.0,
    "fixations_duration_total": 0.0,
    "fixations_duration_mean": 0.0,
    "fixations_duration_median": 0.0,
    "fixations_duration_max": 0.0,
    "fixations_duration_std": 0.0,
    "fixations_duration_skew": 0.0,
    "fixations_duration_kurt": 0.0,
    "saccades_count": 0.0,
    "saccades_frequency": 0.0,
    "saccades_peak_velocity_min": 0.0,
    "saccades_peak_velocity_max": 0.0,
    "saccades_peak_velocity_skew": 0.0,
    "saccades_peak_velocity_kurt": 0.0,
    "saccades_amplitude_mean": 0.0,
    "saccades_amplitude_median": 0.0,
    "saccades_amplitude_max": 0.0,
    "saccades_amplitude_min": 0.0,
    "saccades_amplitude_std": 0.0,
    "saccades_amplitude_skew": 0.0,
    "saccades_amplitude_kurt": 0.0,
    "saccades_duration_mean": 0.0,
    "saccades_duration_median": 0.0,
    "saccades_duration_max": 0.0,
    "saccades_duration_std": 0.0,
    "saccades_duration_skew": 0.0,
    "saccades_duration_kurt": 0.0,
    "saccades_velocity_mean": 0.0,
    "saccades_velocity_std": 0.0,
    "saccades_velocity_skew": 0.0,
    "saccades_velocity_kurt": 0.0,
    "fixations_saccades_duration_ratio": 0.0,
    "blinks_count": 0.0,
    "blinks_frequency": 0.0,
    "blinks_duration_max": 0.0,
    "blinks_duration_min": 0.0,
    "blinks_duration_mean": 0.0,
    "pupil_diameter_mean": 0.0,
    "pupil_diameter_std": 0.0,
    "pupil_diameter_skewness": 0.0,
    "pupil_diameter_kurtosis": 0.0,
    "pupil_lhipa": 0.0,
    "pupil_ripa2": 0.0,
}


def extract_window_features(
    window_pupil_df: pd.DataFrame,
    fixations_df: pd.DataFrame,
    saccades_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
) -> dict[str, Any]:
    """Extract features from the eye-tracking window data.

    Computes gaze-based features including fixations, saccades, and blinks statistics.

    :param window_pupil_df: DataFrame containing pupil data for the window.
    :param fixations_df: DataFrame containing fixations data for the window.
    :param saccades_df: DataFrame containing saccades data for the window.
    :param gaps_df: DataFrame containing gap information with columns 'start_timestamp',
                    'stop_timestamp', and 'is_blink'.
    :returns: Dictionary containing extracted features with keys matching _DEFAULT_FEATURES.
    """
    features = dict(_DEFAULT_FEATURES)  # Start with default values
    duration_sec = (
        window_pupil_df["timestamp_sec"].iloc[-1]
        - window_pupil_df["timestamp_sec"].iloc[0]
    )

    # 1- Fixations_df: count, duration mean/max/min/std
    features["fixations_count"] = len(fixations_df)
    features["fixations_frequency"] = len(fixations_df) / duration_sec
    features["fixations_duration_total"] = fixations_df["duration_ms"].sum()
    if not fixations_df.empty:
        features["fixations_duration_mean"] = fixations_df["duration_ms"].mean()
        features["fixations_duration_median"] = fixations_df["duration_ms"].median()
        features["fixations_duration_max"] = fixations_df["duration_ms"].max()
        features["fixations_duration_skew"] = fixations_df["duration_ms"].skew()
        features["fixations_duration_kurt"] = fixations_df["duration_ms"].kurtosis()
        features["fixations_duration_std"] = (
            fixations_df["duration_ms"].std() if len(fixations_df) > 1 else 0
        )

    # 2- Saccades_df: count, peak_velocity, amplitude mean/max/min/std, duration mean/max/min/std
    features["saccades_count"] = len(saccades_df)
    features["saccades_frequency"] = len(saccades_df) / duration_sec
    if not saccades_df.empty:
        features["saccades_peak_velocity_min"] = saccades_df["peak_velocity"].min()
        features["saccades_peak_velocity_max"] = saccades_df["peak_velocity"].max()
        features["saccades_peak_velocity_skew"] = saccades_df["peak_velocity"].skew()
        features["saccades_peak_velocity_kurt"] = saccades_df[
            "peak_velocity"
        ].kurtosis()
        features["saccades_amplitude_mean"] = saccades_df["amplitude"].mean()
        features["saccades_amplitude_median"] = saccades_df["amplitude"].median()
        features["saccades_amplitude_max"] = saccades_df["amplitude"].max()
        features["saccades_amplitude_min"] = saccades_df["amplitude"].min()
        features["saccades_amplitude_skew"] = saccades_df["amplitude"].skew()
        features["saccades_amplitude_kurt"] = saccades_df["amplitude"].kurtosis()
        features["saccades_amplitude_std"] = (
            saccades_df["amplitude"].std() if len(saccades_df) > 1 else 0
        )
        features["saccades_duration_mean"] = saccades_df["duration_ms"].mean()
        features["saccades_duration_median"] = saccades_df["duration_ms"].median()
        features["saccades_duration_max"] = saccades_df["duration_ms"].max()
        features["saccades_duration_skew"] = saccades_df["duration_ms"].skew()
        features["saccades_duration_kurt"] = saccades_df["duration_ms"].kurtosis()
        features["saccades_duration_std"] = (
            saccades_df["duration_ms"].std() if len(saccades_df) > 1 else 0
        )
        features["saccades_velocity_mean"] = saccades_df["velocity"].mean()
        features["saccades_velocity_skew"] = saccades_df["velocity"].skew()
        features["saccades_velocity_kurt"] = saccades_df["velocity"].kurtosis()
        features["saccades_velocity_std"] = (
            saccades_df["velocity"].std() if len(saccades_df) > 1 else 0
        )

    saccades_duration_total = saccades_df["duration_ms"].sum()
    features["fixations_saccades_duration_ratio"] = (
        features["fixations_duration_total"] / saccades_duration_total
        if saccades_duration_total > 0
        else 0
    )

    # 4- Blinks: count, duration mean
    blink_df = gaps_df[gaps_df["is_blink"]]
    features["blinks_frequency"] = len(blink_df) / duration_sec
    if not blink_df.empty:
        features["blinks_count"] = len(blink_df)
        features["blinks_duration_max"] = blink_df["duration_ms"].max()
        features["blinks_duration_min"] = blink_df["duration_ms"].min()
        features["blinks_duration_mean"] = blink_df["duration_ms"].mean()

    # 5- Basic pupil diameter features
    features["pupil_diameter_mean"] = window_pupil_df["pupil_diameter"].mean()
    features["pupil_diameter_std"] = window_pupil_df["pupil_diameter"].std()
    features["pupil_diameter_skewness"] = window_pupil_df["pupil_diameter"].skew()
    features["pupil_diameter_kurtosis"] = window_pupil_df["pupil_diameter"].kurtosis()

    return features
