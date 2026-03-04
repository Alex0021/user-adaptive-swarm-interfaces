import numpy as np
import pandas as pd
import pywt
from scipy.signal import savgol_coeffs

EPSILON = 1e-10

LF_COEFFS = None
VLF_COEFFS = None


def detect_outliers(
    eye_df: pd.DataFrame, column: str, n_multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers in a given column of a DataFrame the MAD method

    :param eye_df: The DataFrame to analyze.
    :param column: The name of the column to analyze for outliers.
    :param n_multiplier: The multiplier used for the MAD threshold.
    :return outliers_df: A DataFrame containing the detected outliers.
    """

    def dilation_speed(data, time_col="timestamp_sec", diameter_col=column):
        speed_before = data[diameter_col].diff(1) / data[time_col].diff(1)
        speed_after = data[diameter_col].diff(-1) / data[time_col].diff(-1)
        return pd.concat([speed_before.abs(), speed_after.abs()], axis=1).max(axis=1)

    pupil_df = eye_df[[column, "timestamp_sec"]].copy()
    pupil_df["dilation_speed"] = dilation_speed(pupil_df)
    # Find the threshold
    MAD = (
        (pupil_df["dilation_speed"] - pupil_df["dilation_speed"].median())
        .abs()
        .median()
    )
    threshold = pupil_df["dilation_speed"].median() + n_multiplier * MAD

    # Marked as outliers the points where dilation speed is above the threshold
    pupil_df["is_outlier"] = pupil_df["dilation_speed"] > threshold

    return pupil_df[pupil_df["is_outlier"]]


def lhipa(eye_df: pd.DataFrame, wavelet_type: str = "sym16") -> float:
    """
    Computes the LHIPA (Low-High Index of Pupillary Activity) for a given eye-tracking DataFrame.

    :param eye_df: Should have a 'pupil_diameter_px' column with the pupil diameter measurements.
    :param wavelet_type: The type of wavelet to use for decomposition.
    :return lhipa_value: The computed LHIPA value.
    """
    pupil_column = [c for c in eye_df.columns if "pupil_diameter" in c]
    if len(pupil_column) != 1:
        print(
            "Error: The input DataFrame must contain exactly one column with 'pupil_diameter' in its name."
        )
        return np.nan
    data = eye_df[pupil_column[0]].to_numpy().copy()
    w = pywt.Wavelet(wavelet_type)
    max_level = pywt.dwt_max_level(len(data), w.dec_len)

    hif, lof = 1, int(max_level / 2)

    if hif == lof:
        print(
            "Warning: hif and lof are equal, which may lead to issues in LHIPA computation"
        )
        return np.nan

    cD_H = pywt.downcoef("d", data, w, level=hif, mode="per")
    cD_L = pywt.downcoef("d", data, w, level=lof, mode="per")

    # Normalize
    cD_H /= np.sqrt(2**hif)
    cD_L /= np.sqrt(2**lof)

    # Check for zero values in cD_H to avoid division by zero
    cD_H[cD_H == 0.0] = EPSILON

    cD_LH = cD_L / cD_H[[i for i in range(len(cD_H)) if i % (2 ** (lof - hif)) == 0]]

    # Modmax
    cD_LHm = modmax(cD_LH)

    # Universal threshold for noise estimation
    lambda_univ = np.std(cD_LHm) * np.sqrt(2.0 * np.log2(len(cD_LHm)))
    cD_LHt = pywt.threshold(cD_LHm, lambda_univ, mode="less")

    duration = eye_df["timestamp_sec"].max() - eye_df["timestamp_sec"].min()

    return (cD_LHt > 0).sum() / duration


def modmax(coeffs):
    abs_coeffs = np.abs(np.array(coeffs))
    lcoeffs = np.roll(abs_coeffs, -1)
    lcoeffs[-1] = 0.0
    rcoeffs = np.roll(abs_coeffs, 1)
    rcoeffs[0] = 0.0
    return np.multiply(
        abs_coeffs,
        ((abs_coeffs >= lcoeffs) & (abs_coeffs >= rcoeffs))
        & ((abs_coeffs > lcoeffs) | (abs_coeffs > rcoeffs)),
    )


def ripa2(
    window_df: pd.DataFrame, VLF: tuple[int, int], LF: tuple[int, int], D: int = 1
) -> float:
    """
    Compute the RIPA2 value which is the newest IPA measure to be used online.

    Refer to this paper for explainations: https://doi.org/10.3390/jemr18060070

    :param window_df: Must contain a 'pupil_diameter_px' column
    :param (M_VLF,N_VLF): The filter length (M) and polynomial order (N) for the VLF filter.
    :param (M_LF,N_LF): The filter length (M) and polynomial order (N) for the LF filter.
    :param D: The order of the derivative to compute (default is 1 for first derivative).
    :return ripa2_mean: The computed RIPA2 value (mean over the valid window).
    """
    global LF_COEFFS, VLF_COEFFS
    # 1- Compute the S-G filter coefficients if needed
    if LF_COEFFS is None or VLF_COEFFS is None:
        VLF_window_length = VLF[0] * 2 + 1
        LF_window_length = LF[0] * 2 + 1
        VLF_COEFFS = savgol_coeffs(
            window_length=VLF_window_length, polyorder=VLF[1], deriv=D
        )
        LF_COEFFS = savgol_coeffs(
            window_length=LF_window_length, polyorder=LF[1], deriv=D
        )

    # 2- Get the filtered samples that fit the window length (centered)
    pupil_column = [c for c in window_df.columns if "pupil_diameter" in c]
    if len(pupil_column) != 1:
        print(
            "Error: The input DataFrame must contain exactly one column with 'pupil_diameter' in its name."
        )
        return np.nan
    pupil_data = window_df[pupil_column[0]].to_numpy()
    VLF_filtered = np.convolve(pupil_data, VLF_COEFFS, mode="valid")
    LF_filtered = np.convolve(pupil_data, LF_COEFFS, mode="valid")

    # for LF filtered, only keep matching samples with VLF filtered (centered)
    delta = VLF[0] - LF[0]
    LF_filtered = LF_filtered[delta : len(LF_filtered) - delta]

    # 3- Compute the RIPA2 value
    if LF_filtered.shape[0] != VLF_filtered.shape[0]:
        print(
            "Warning: Shape mismatch between LF and VLF signals. Returning NaN for RIPA2."
        )
        return np.nan
    ripa2 = np.clip(LF_filtered**2 - VLF_filtered**2, 0, 1.5)

    # 4- For now, return the mean value of the RIPA2
    return np.mean(ripa2)
