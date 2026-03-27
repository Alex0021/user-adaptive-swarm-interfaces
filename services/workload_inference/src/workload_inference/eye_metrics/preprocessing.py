import pandas as pd


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
