"""Online statistics helpers for the inference engine."""

import numpy as np


class OnlinePupilStats:
    """Running statistics for pupil diameter used for fast outlier rejection.

    Tracks an EMA of dilation speed median and MAD so that each inference
    window can reject outliers without recomputing global statistics.
    """

    def __init__(self, ema_alpha: float = 0.3):
        self._alpha = ema_alpha
        self.median_speed: float | None = None
        self.mad_speed: float | None = None

    def update_from_speeds(self, speeds: np.ndarray) -> None:
        speeds = speeds[np.isfinite(speeds)]
        if len(speeds) < 5:
            return
        med = float(np.median(speeds))
        mad = float(np.median(np.abs(speeds - med)))
        if self.median_speed is None:
            self.median_speed = med
            self.mad_speed = mad
        else:
            a = self._alpha
            self.median_speed = a * med + (1 - a) * self.median_speed
            self.mad_speed = a * mad + (1 - a) * self.mad_speed

    def outlier_mask(
        self, speeds: np.ndarray, n_multiplier: float = 10.0
    ) -> np.ndarray:
        if self.median_speed is None or self.mad_speed is None:
            return np.zeros(len(speeds), dtype=bool)
        threshold = self.median_speed + n_multiplier * self.mad_speed
        return speeds > threshold


class WelfordNormalizer:
    """Online z-normalization using Welford's algorithm."""

    def __init__(self, n_features: int):
        self.n = 0
        self.mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self._m2 / (self.n - 1))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        s = self.std.copy()
        s[s < 1e-10] = 1.0
        return (x - self.mean) / s
