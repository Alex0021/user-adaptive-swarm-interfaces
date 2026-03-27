from collections import deque

import numpy as np
import pandas as pd
import pywt
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.signal import savgol_coeffs

EPSILON = 1e-10

LF_COEFFS = None
VLF_COEFFS = None
RIPA_CLIP_VALUE = 1.5


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


class RealtimeFeatures:
    """
    Base class for real-time feature extraction using ring buffers (deques).
    Maintains a sliding window of the N most recent raw samples.
    """

    def __init__(self, buffer_size: int, sample_rate: float):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.raw: deque[float] = deque(maxlen=buffer_size)

    def push(self, sample: float) -> None:
        self.raw.append(sample)
        self._on_new_sample(sample)

    def _on_new_sample(self, sample: float) -> None:
        pass

    def is_ready(self) -> bool:
        return len(self.raw) == self.buffer_size

    def as_array(self) -> np.ndarray:
        return np.array(self.raw)

    def flush(self) -> None:
        self.raw.clear()


class RIPA2(RealtimeFeatures):
    """
    Real-time RIPA2 implementation using two Savitzky-Golay first-order
    derivative filters.

    RIPA2 index from paper: 10.3390/jemr18060070
    """

    # SG filter parameters from the paper (tuned for 300 Hz)
    M_VLF: int = 98  # half-window → full window = 2*M+1 = 973
    N_VLF: int = 2  # polynomial order

    M_LF: int = 13  # half-window → full window = 121
    N_LF: int = 4

    RIPA2_MIN: float = 0.0
    RIPA2_MAX: float = 1.5

    def __init__(
        self,
        sample_rate: float = 60.0,
        buffer_size: int = 300,
        smoothing_window_s: float = -1.0,
    ):
        """ """
        if buffer_size is None:
            buffer_size = int(5.0 * sample_rate)

        self._vlf_window_size = 2 * self.M_VLF + 1
        self._lf_window_size = 2 * self.M_LF + 1
        if buffer_size < self._vlf_window_size:
            raise ValueError(
                f"buffer_size ({buffer_size}) must be ≥ 2*M_VLF+1 = {self._vlf_window_size}"
            )

        super().__init__(buffer_size, sample_rate)

        # ---- pre-compute SG first-order derivative coefficients -----------
        self._h_vlf: np.ndarray = savgol_coeffs(
            2 * self.M_VLF + 1, self.N_VLF, deriv=1, delta=1.0 / sample_rate, use="conv"
        )
        self._h_lf: np.ndarray = savgol_coeffs(
            2 * self.M_LF + 1, self.N_LF, deriv=1, delta=1.0 / sample_rate, use="conv"
        )

        # Temporal alignment offset
        self._delta: int = self.M_VLF - self.M_LF

        # Output buffers
        self._initialize_buffers(buffer_size)

        # Moving-average smoothing kernel
        self._smooth_kernel = None
        if smoothing_window_s > 0:
            smooth_n = max(1, int(smoothing_window_s * sample_rate))
            self._smooth_kernel: np.ndarray = np.ones(smooth_n) / smooth_n

        # Internal sample counter (gates output until VLF window is full)
        self._n_samples: int = 0

    def _initialize_buffers(self, buffer_size: int) -> None:
        self._usable_range = buffer_size - self._vlf_window_size
        self.vlf_signal: deque[float] = deque(maxlen=buffer_size)
        self.lf_signal: deque[float] = deque(maxlen=buffer_size)
        self.ripa2_raw: deque[float] = deque(maxlen=self._usable_range)
        self.ripa2_smooth: deque[float] = deque(maxlen=self._usable_range)

    def _on_new_sample(self) -> None:
        self._n_samples += 1

        # Need a full VLF window before any output is meaningful
        if self._n_samples < 2 * self.M_VLF + 1:
            return

        buf = self.as_array()  # snapshot of the current ring buffer

        vlf_window = buf[-(2 * self.M_VLF + 1) :]
        # Coefficients were created for convolution, so they must me reversed for dot
        value_vlf = float(np.dot(self._h_vlf[::-1], vlf_window))
        self.vlf_signal.append(value_vlf)

        lf_start = -self._delta - 2 * self.M_LF - 1
        lf_window = buf[lf_start : -self._delta]
        value_lf = float(np.dot(self._h_lf[::-1], lf_window))
        self.lf_signal.append(value_lf)

        ripa2 = float(
            np.clip(
                value_lf**2 - value_vlf**2,
                self.RIPA2_MIN,
                self.RIPA2_MAX,
            )
        )
        self.ripa2_raw.append(ripa2)

        raw_arr = np.array(self.ripa2_raw)
        k = len(self._smooth_kernel)
        if len(raw_arr) >= k and self._smooth_kernel is not None:
            smoothed = float(
                np.convolve(raw_arr, self._smooth_kernel, mode="valid")[-1]
            )
        else:
            smoothed = float(np.mean(raw_arr))
        self.ripa2_smooth.append(smoothed)

    def push_batch(self, samples: NDArray[np.float32]) -> None:
        # For a batch, convert the deque to an array and concatenate the new samples,
        # then apply the filter using a convolution to be more efficient
        buf = self.as_array()
        buf = np.concatenate([buf[-(2 * self.M_VLF + 1) :], samples])

        self._n_samples += len(samples)
        if len(buf) < 2 * self.M_VLF + 1:
            return

        vlf_values = np.convolve(buf, self._h_vlf, mode="valid")
        lf_values = np.convolve(
            buf[self._delta : -self._delta], self._h_lf, mode="valid"
        )

        ripa2_values = np.clip(
            lf_values**2 - vlf_values**2, self.RIPA2_MIN, self.RIPA2_MAX
        )

        if self._smooth_kernel is not None:
            smoothed_values = np.convolve(
                ripa2_values, self._smooth_kernel, mode="valid"
            )
        else:
            smoothed_values = [np.mean(ripa2_values)]

        # Update the buffers with the new values
        self.raw.extend(samples)
        self.vlf_signal.extend(vlf_values)
        self.lf_signal.extend(lf_values)
        self.ripa2_raw.extend(ripa2_values)
        self.ripa2_smooth.extend(smoothed_values)

    def is_ready(self) -> bool:
        """True once the VLF window is satisfied (not just buffer full)."""
        return self._n_samples >= 2 * self.M_VLF + 1

    def current_ripa2(self) -> float:
        """Latest raw RIPA2 value, or NaN if not yet ready."""
        if not self.ripa2_raw:
            return np.nan
        return self.ripa2_raw[-1]

    def current_ripa2_smooth(self) -> float:
        """Latest smoothed RIPA2 value, or NaN if not yet ready."""
        if not self.ripa2_smooth:
            return np.nan
        return self.ripa2_smooth[-1]

    def get_smoothed_ripa2_series(self) -> np.ndarray:
        """Returns the current smoothed RIPA2 values as a numpy array."""
        return np.array(self.ripa2_smooth)

    def flush(self) -> None:
        super().flush()
        self.vlf_signal.clear()
        self.lf_signal.clear()
        self.raw.clear()
        self.ripa2_raw.clear()
        self.ripa2_smooth.clear()
        self._n_samples = 0

    def set_new_buffer_size(self, new_buffer_size: int) -> None:
        """Flushes the current buffers and sets a new buffer size."""
        self.flush()
        self.raw = deque(maxlen=new_buffer_size)
        self._initialize_buffers(new_buffer_size)


class WaveletFeature(RealtimeFeatures):
    """
    Real-time wavelet-based feature extraction using a ring buffer (deque).
    Maintains a sliding window of the N most recent raw samples and computes
    features based on wavelet decomposition.
    """

    WAVELET_TYPE = "db8"

    def __init__(
        self,
        buffer_size: int,
        sample_rate: float,
        level: int = 2,
        update_rate_samples: int = None,
    ):
        super().__init__(buffer_size, sample_rate)
        self._level = level
        self._total_samples = 0
        self._update_rate_samples = update_rate_samples or buffer_size
        self._last_update_index_top_level = 0
        self._sample_count_since_last_update = 0

        self._wv = pywt.Wavelet(self.WAVELET_TYPE)
        self._min_valid_samples = 2 ** (level - 1) * self._wv.dec_len
        # Initialize additional buffers
        self._initialize_buffers(buffer_size, level)

    def _initialize_buffers(self, buffer_size, level) -> None:
        max_lev = pywt.dwt_max_level(buffer_size, self._wv.dec_len)
        if level > max_lev:
            print(
                f"Warning: buffer_size ({buffer_size}) is too small for the chosen wavelet and level."
            )
            level = max_lev
        # Level 1 --> Level N wavelet coefficients (with Nth level having the CA coeffs)
        # Should be adapted to the length of every level
        self._num_valid_samples_per_level = [
            buffer_size // (2**i) - self._wv.dec_len for i in range(level, -1, -1)
        ]
        self._buffers = [deque(maxlen=self._num_valid_samples_per_level[0])] + [
            deque(maxlen=self._num_valid_samples_per_level[i]) for i in range(level)
        ]

    def _on_new_sample(self) -> None:
        self._total_samples += 1
        self._sample_count_since_last_update += 1

        # Only update features when we have enough new samples to fill the buffer
        if self._total_samples % self._update_rate_samples == 0:
            self._update_features()

    def _update_features(self) -> None:
        if len(self.raw) < self._min_valid_samples:
            return

        data = self.as_array()
        coeffs = pywt.wavedec(data, self._wv, level=self._level, mode="periodization")

        for i, coeff in enumerate(coeffs):
            margin = self._wv.dec_len // 2
            self._buffers[i].extend(
                coeff[margin:-margin]
            )  # Skip the first and last dec_len//2 samples which are not valid for convolution

        self._sample_count_since_last_update = 0

    def push_batch(self, samples: NDArray[np.float32]) -> None:
        self._total_samples += len(samples)
        self._sample_count_since_last_update += len(samples)
        self.raw.extend(samples)

        # Always update if batch is added
        self._update_features()

    def get_latest_coefficients(self) -> list[np.ndarray]:
        return [np.array(buf) for buf in self._buffers]

    def get_last_coefficients(self) -> list[float | None]:
        return [buf[-1] if buf else None for buf in self._buffers]

    def get_all_smoothed_coefficients(
        self, smoothing_window_s: float
    ) -> list[np.ndarray[float]]:
        smoothed_coeffs = []
        k = max(1, int(smoothing_window_s * self.sample_rate))
        kernel = np.ones(k) / k
        for buf in self._buffers:
            buf = np.array(buf)
            if len(buf) >= k:
                smoothed = np.convolve(buf, kernel, mode="valid")
                smoothed_coeffs.append(smoothed)
            elif len(buf) > 0:
                smoothed_coeffs.append(np.array([float(np.mean(buf))]))
            else:
                smoothed_coeffs.append(np.array([np.nan]))
        return smoothed_coeffs

    def get_last_smoothed_coefficients(self, smoothing_size: int) -> list[float | None]:
        smoothed_coeffs = []
        for buf in self._buffers:
            buf = np.array(buf)
            coeff = np.nan
            if len(buf) >= smoothing_size:
                coeff = np.mean(np.abs(buf[-smoothing_size:]))
            elif len(buf) > 0:
                coeff = np.mean(np.abs(buf))
            smoothed_coeffs.append(coeff)

        return smoothed_coeffs

    def flush(self) -> None:
        super().flush()
        for buf in self._buffers:
            buf.clear()
        self._total_samples = 0

    def set_new_buffer_size(self, new_buffer_size: int) -> None:
        """Flushes the current buffers and sets a new buffer size."""
        self.flush()
        self.raw = deque(maxlen=new_buffer_size)
        self._initialize_buffers(new_buffer_size, self._level)


class LHIPA(RealtimeFeatures):
    """
    Real-time LHIPA implementation using wavelet decomposition and modmax.
    """

    WAVELET_TYPE = "sym16"

    def __init__(self, sample_rate: float = 60.0, buffer_size: int = 300):
        super().__init__(buffer_size, sample_rate)
        self._wv = pywt.Wavelet(self.WAVELET_TYPE)
        self._max_level = pywt.dwt_max_level(buffer_size, self._wv.dec_len)
        if self._max_level < 2:
            raise ValueError(
                f"buffer_size ({buffer_size}) is too small for the chosen wavelet."
            )
        self._total_samples = 0
        self._latest_lhipa: float | None = None

    def _on_new_sample(self) -> None:
        self._total_samples += 1
        if self._total_samples % self.buffer_size == 0:
            self._latest_lhipa = self._compute_lhipa()

    def push_batch(self, samples: NDArray[np.float32]) -> None:
        self._total_samples += len(samples)
        self.raw.extend(samples)
        self._latest_lhipa = self._compute_lhipa()

    def _compute_lhipa(self) -> float | None:
        data = self.as_array()
        hif, lof = 1, np.ceil(self._max_level / 2)

        if len(self.raw) < self._wv.dec_len * (2**lof):
            return np.nan

        if hif == lof:
            print(
                "Warning: hif and lof are equal, which may lead to issues in LHIPA computation"
            )
            return np.nan

        cD_H = pywt.downcoef("d", data, self._wv, level=hif, mode="periodization")
        cD_L = pywt.downcoef("d", data, self._wv, level=lof, mode="periodization")

        # Normalize
        cD_H /= np.sqrt(2**hif)
        cD_L /= np.sqrt(2**lof)

        # Check for zero values in cD_H to avoid division by zero
        cD_H[cD_H == 0.0] = EPSILON

        cD_LH = (
            cD_L / cD_H[[i for i in range(len(cD_H)) if i % (2 ** (lof - hif)) == 0]]
        )

        # Modmax
        cD_LHm = modmax(cD_LH)

        # Universal threshold for noise estimation
        lambda_univ = np.std(cD_LHm) * np.sqrt(2.0 * np.log2(len(cD_LHm)))
        cD_LHt = pywt.threshold(cD_LHm, lambda_univ, mode="less")

        duration = len(self.raw) / self.sample_rate

        return (cD_LHt > 0).sum() / duration

    def current_lhipa(self) -> float | None:
        """Latest LHIPA value, or None if not yet ready."""
        return self._latest_lhipa

    def set_new_buffer_size(self, new_buffer_size: int) -> None:
        """Flushes the current buffers and sets a new buffer size."""
        self.flush()
        self.raw = deque(maxlen=new_buffer_size)
        self._max_level = pywt.dwt_max_level(new_buffer_size, self._wv.dec_len)
