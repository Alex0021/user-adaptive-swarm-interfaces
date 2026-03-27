"""Online cognitive workload inference engine.

Accumulates streaming GazeData, preprocesses sliding windows, extracts
pupil-based features, and runs a trained classifier to estimate workload
level (low / medium / high).
"""

import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from workload_inference.data_structures import GazeData
from workload_inference.eye_metrics.features import FEATURE_SETS
from workload_inference.eye_metrics.gaze_utils import detect_gaps_and_blinks
from workload_inference.eye_metrics.interpolate import interpolate_pupil_data
from workload_inference.eye_metrics.preprocessing import select_best_eye
from workload_inference.eye_metrics.pupil_utils import (
    LHIPA,
    RIPA2,
    WaveletFeature,
)

from .filters import SmoothingSchmittFilter, WorkloadFilter
from .online_stats import OnlinePupilStats, WelfordNormalizer
from .settings import InferenceSettings

logger = logging.getLogger(__name__)

# Preprocessing thresholds (matching offline pipeline)
CONFIDENCE_THRESHOLD = 0.5
DURATION_THRESHOLD_SEC = 100 / 1000
INTERPOLATION_THRESHOLD_SEC = 300 / 1000


class WorkloadInferenceEngine:
    """Online cognitive workload inference from streaming gaze data.

    Accumulates GazeData samples, periodically preprocesses a sliding window,
    extracts pupil-based features, and runs a trained classifier.

    Usage::

        settings = InferenceSettings.from_yaml("settings.yml")
        engine = WorkloadInferenceEngine(
            model_path="models/my_model.zip",
            settings=settings,
        )
        engine.register_listener(my_display_callback)
        gaze_receiver.register_listener(engine.gaze_datas_callback)
    """

    # Sentinel value indicating no valid prediction has been made yet.
    UNKNOWN_CLASS: int = -1

    def __init__(
        self,
        model_path: str | Path | None = None,
        settings: InferenceSettings | None = None,
        filter: WorkloadFilter | None = None,
    ):
        if settings is None:
            settings = InferenceSettings()
        self._settings = settings

        self._sample_rate = settings.sample_rate
        self._window_size_samples = settings.window_size_samples
        self._inference_interval_samples = settings.inference_interval_samples
        self._wavelet_level = settings.wavelet_level
        self._min_valid_ratio = settings.min_valid_ratio
        self._feature_set_name = settings.feature_set
        self._feature_patterns = FEATURE_SETS.get(
            settings.feature_set, FEATURE_SETS["all"]
        )

        # Filter
        if filter is None:
            filter = SmoothingSchmittFilter(
                smoothing_predictions=settings.smoothing_predictions,
                min_fraction=settings.schmitt_min_fraction,
                min_consecutive=settings.schmitt_min_consecutive,
            )
        self._filter = filter

        # Model
        self._model = None
        self._model_type: str | None = None
        if model_path is not None:
            self._load_model(Path(model_path))

        # Data buffer
        self._raw_buffer: deque[GazeData] = deque(
            maxlen=self._window_size_samples * 2
        )
        self._samples_since_last_inference = 0

        # Running pupil stats for online outlier rejection
        self._pupil_stats = OnlinePupilStats()

        # Persistent pupil feature extractors
        buffer_size = settings.rolling_buffer_size
        self._ripa2 = RIPA2(
            buffer_size=buffer_size,
            smoothing_window_s=settings.window_size_sec,
            sample_rate=settings.sample_rate,
        )
        self._lhipa = LHIPA(
            buffer_size=buffer_size, sample_rate=settings.sample_rate
        )
        self._wavelet = WaveletFeature(
            level=settings.wavelet_level,
            buffer_size=buffer_size,
            sample_rate=settings.sample_rate,
        )

        # Feature columns: if settings specifies them use those directly,
        # otherwise resolve on first successful extraction.
        if settings.feature_columns:
            self._feature_columns: list[str] | None = list(settings.feature_columns)
            self._normalizer: WelfordNormalizer | None = WelfordNormalizer(
                len(self._feature_columns)
            )
        else:
            self._feature_columns = None
            self._normalizer = None

        # Prediction state
        self._prediction_history: deque[tuple[int, np.ndarray, float]] = deque(
            maxlen=200
        )
        self._current_workload: int = self.UNKNOWN_CLASS
        self._current_probabilities: np.ndarray = np.zeros(3)
        self._listeners: list[Callable[[int, int, np.ndarray], None]] = []

        # Threading
        self._inference_lock = threading.Lock()
        self._inference_thread: threading.Thread | None = None
        self._last_inference_timestamp: float | None = None

        logger.info(
            "WorkloadInferenceEngine initialized: window=%.1fs, interval=%.1fs, "
            "features=%s, model=%s",
            settings.window_size_sec,
            settings.inference_interval_sec,
            settings.feature_set,
            model_path or "none",
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: Path) -> None:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = model_path.resolve()
        suffix = model_path.suffix.lower()

        if suffix == ".zip" or model_path.with_suffix(".zip").exists():
            try:
                from pytorch_tabnet.tab_model import TabNetClassifier

                self._model = TabNetClassifier(device_name="cpu")
                self._model.load_model(model_path)
                self._model_type = "tabnet"
                logger.info("Loaded TabNet model from %s", model_path)
            except ImportError:
                logger.error(
                    "pytorch_tabnet not installed. Cannot load TabNet model."
                )
                raise
        elif suffix in (".pkl", ".joblib"):
            import joblib

            self._model = joblib.load(model_path)
            self._model_type = "sklearn"
            logger.info("Loaded sklearn model from %s", model_path)
        else:
            raise ValueError(
                f"Unsupported model format: '{suffix}'. "
                "Expected .zip (TabNet) or .pkl/.joblib (sklearn)."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_listener(
        self, listener: Callable[[int, int, np.ndarray], None]
    ) -> None:
        """Register a callback: ``listener(raw_class, filtered_class, probabilities)``."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def gaze_datas_callback(
        self, datas: Sequence[GazeData], batch_update: bool = False
    ) -> None:
        """Listener callback for streaming GazeData."""
        if batch_update:
            self._raw_buffer.clear()
            self._samples_since_last_inference = 0

        for gaze in datas:
            self._raw_buffer.append(gaze)
            self._samples_since_last_inference += 1

        if self._samples_since_last_inference >= self._inference_interval_samples:
            self._samples_since_last_inference = 0
            if (
                self._inference_thread is not None
                and self._inference_thread.is_alive()
            ):
                return
            snapshot = list(self._raw_buffer)[-self._window_size_samples :]
            self._inference_thread = threading.Thread(
                target=self._run_inference_safe,
                args=(snapshot,),
                daemon=True,
            )
            self._inference_thread.start()

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------

    def _run_inference_safe(self, snapshot: list[GazeData]) -> None:
        """Top-level wrapper that ensures exceptions never escape the thread."""
        try:
            self._run_inference(snapshot)
        except Exception:
            logger.exception("Inference thread crashed")

    def _run_inference(self, snapshot: list[GazeData]) -> None:
        """Run the full inference pipeline on a snapshot of gaze data."""
        # Use only the portion of the buffer that actually contains data.
        # At startup the buffer may have fewer samples than the full window.
        n_available = len(snapshot)
        min_samples = int(self._window_size_samples * self._min_valid_ratio)
        if n_available < min_samples:
            logger.debug(
                "Skipping inference: only %d samples available (need %d)",
                n_available,
                min_samples,
            )
            return

        df = self._build_dataframe(snapshot)

        pupil_df, gaps_df = self._preprocess_online(df)
        if pupil_df.empty or len(pupil_df) < 50:
            logger.debug(
                "Skipping inference: insufficient samples after preprocessing"
            )
            return

        # Extract features
        duration = (
            pupil_df["timestamp_sec"].iloc[-1] - pupil_df["timestamp_sec"].iloc[0]
        )
        if duration <= 0:
            logger.debug("Skipping inference: zero-length window")
            return

        features = self._extract_blink_features(gaps_df, duration)
        self._extract_pupil_realtime_features(pupil_df, features)

        if not features:
            logger.debug("Skipping inference: no features extracted")
            return

        if self._feature_columns is None:
            self._resolve_feature_columns(features)

        # Build feature vector -- missing features get NaN, not 0.0
        feature_vector = np.array(
            [features.get(col, np.nan) for col in self._feature_columns],
            dtype=np.float64,
        )

        # If any feature is NaN, skip this window rather than silently
        # substituting zeros which would distort the model.
        nan_mask = np.isnan(feature_vector)
        if nan_mask.any():
            nan_cols = [
                c
                for c, is_nan in zip(self._feature_columns, nan_mask)
                if is_nan
            ]
            logger.debug(
                "Skipping inference: %d NaN features: %s",
                len(nan_cols),
                nan_cols,
            )
            return

        # Replace inf with NaN then check again
        feature_vector = np.where(
            np.isinf(feature_vector), np.nan, feature_vector
        )
        if np.isnan(feature_vector).any():
            logger.debug("Skipping inference: inf values in feature vector")
            return

        self._normalizer.update(feature_vector)
        if self._normalizer.n >= 3:
            feature_vector = self._normalizer.normalize(feature_vector)

        if self._model is not None:
            X = feature_vector.reshape(1, -1).astype(np.float32)
            proba = self._model.predict_proba(X)[0]
            pred_class = int(np.argmax(proba))
        else:
            # No model loaded -- report unknown rather than a fake class
            proba = np.array([1 / 3, 1 / 3, 1 / 3])
            pred_class = self.UNKNOWN_CLASS

        with self._inference_lock:
            self._prediction_history.append((pred_class, proba, time.time()))
        self._smooth_and_notify()
        self._last_inference_timestamp = time.time()

    # ------------------------------------------------------------------
    # DataFrame construction
    # ------------------------------------------------------------------

    def _build_dataframe(self, samples: list[GazeData]) -> pd.DataFrame:
        """Convert GazeData samples to a DataFrame for the preprocessing pipeline.

        Only includes columns needed for pupil-based features.
        Gaze point columns are kept structurally for future use but are not
        required by the current feature set.
        """
        rows = []
        start_ts = None
        for gaze in samples:
            ts_sec = float(gaze.timestamp) / 1000.0
            if start_ts is None:
                start_ts = ts_sec
            rows.append(
                {
                    "timestamp_sec": ts_sec - start_ts,
                    # Gaze points -- kept for architecture, not used by
                    # current pupil-only feature set
                    "left_gaze_point_x": float(gaze.left_gaze_point_x),
                    "left_gaze_point_y": float(gaze.left_gaze_point_y),
                    "left_gaze_point_z": float(gaze.left_gaze_point_z),
                    "right_gaze_point_x": float(gaze.right_gaze_point_x),
                    "right_gaze_point_y": float(gaze.right_gaze_point_y),
                    "right_gaze_point_z": float(gaze.right_gaze_point_z),
                    # Screen points
                    "left_point_screen_x": float(gaze.left_point_screen_x),
                    "left_point_screen_y": float(gaze.left_point_screen_y),
                    "right_point_screen_x": float(gaze.right_point_screen_x),
                    "right_point_screen_y": float(gaze.right_point_screen_y),
                    # Validity & pupil
                    "left_validity": int(gaze.left_validity),
                    "right_validity": int(gaze.right_validity),
                    "left_pupil_diameter": float(gaze.left_pupil_diameter),
                    "right_pupil_diameter": float(gaze.right_pupil_diameter),
                    "left_openness": float(gaze.left_openness),
                    "right_openness": float(gaze.right_openness),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_online(
        self, eye_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Online preprocessing pipeline for pupil data."""
        empty = pd.DataFrame()

        eye_df, _best_eye = select_best_eye(eye_df.copy(), threshold=0.05)

        gaps_df = detect_gaps_and_blinks(
            eye_df,
            confidence_threshold=0.95,
            blink_threshold_range=(100, 300),
            eye_openness_column="openness",
            openness_threshold=0.5,
        )

        eye_df.rename(
            columns={"pupil_diameter_mm": "pupil_diameter"}, inplace=True
        )

        total_samples = len(eye_df)
        if total_samples == 0:
            return empty, gaps_df

        # Check low-confidence ratio
        if not gaps_df.empty:
            non_blink = gaps_df[~gaps_df["is_blink"]]
            if not non_blink.empty:
                lc_count = (
                    non_blink["stop_id"] - non_blink["start_id"] + 1
                ).sum()
                if lc_count / total_samples > 0.30:
                    return empty, gaps_df

        eye_df = eye_df[eye_df["confidence"] >= CONFIDENCE_THRESHOLD]

        # Outlier rejection via running pupil dilation speed stats
        if "pupil_diameter" in eye_df.columns and len(eye_df) > 10:
            ts = eye_df["timestamp_sec"].values
            pd_vals = eye_df["pupil_diameter"].values
            dt = np.diff(ts)
            dt[dt == 0] = 1e-6
            speed_fwd = np.abs(np.diff(pd_vals) / dt)
            speeds = np.empty(len(pd_vals))
            speeds[0] = speed_fwd[0] if len(speed_fwd) > 0 else 0.0
            speeds[-1] = speed_fwd[-1] if len(speed_fwd) > 0 else 0.0
            speeds[1:-1] = np.maximum(speed_fwd[:-1], speed_fwd[1:])

            self._pupil_stats.update_from_speeds(speeds)
            outlier_mask = self._pupil_stats.outlier_mask(speeds)
            eye_df = eye_df[~outlier_mask]

        # Vectorized gap-margin removal
        margins = 50 / 1000
        sig_gaps = gaps_df[gaps_df["duration_ms"] >= DURATION_THRESHOLD_SEC * 1000]
        if not sig_gaps.empty:
            ts_arr = eye_df["timestamp_sec"].values
            keep = np.ones(len(ts_arr), dtype=bool)
            for start_t, stop_t in zip(
                sig_gaps["start_timestamp"].values,
                sig_gaps["stop_timestamp"].values,
            ):
                keep &= (ts_arr < start_t - margins) | (
                    ts_arr > stop_t + margins
                )
            eye_df = eye_df[keep]

        if len(eye_df) < 50:
            return empty, gaps_df

        pupil_df = interpolate_pupil_data(
            eye_df,
            gaps_df,
            column="pupil_diameter",
            max_gap=INTERPOLATION_THRESHOLD_SEC,
        )

        return pupil_df, gaps_df

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_blink_features(
        self, gaps_df: pd.DataFrame, duration_sec: float
    ) -> dict:
        features = {}
        blink_df = gaps_df[gaps_df["is_blink"]]
        features["blinks_count"] = len(blink_df)
        features["blinks_frequency"] = (
            len(blink_df) / duration_sec if duration_sec > 0 else 0.0
        )
        if not blink_df.empty:
            features["blinks_duration_max"] = blink_df["duration_ms"].max()
            features["blinks_duration_min"] = blink_df["duration_ms"].min()
            features["blinks_duration_mean"] = blink_df["duration_ms"].mean()
        else:
            features["blinks_duration_max"] = 0.0
            features["blinks_duration_min"] = 0.0
            features["blinks_duration_mean"] = 0.0
        return features

    def _extract_pupil_realtime_features(
        self, pupil_df: pd.DataFrame, features: dict
    ) -> None:
        pupil_values = pupil_df["pupil_diameter"].values.astype("float32")
        if len(pupil_values) < 50:
            return

        self._ripa2.push_batch(pupil_values)
        self._lhipa.push_batch(pupil_values)
        self._wavelet.push_batch(pupil_values)

        lhipa_val = self._lhipa.current_lhipa()
        ripa2_val = self._ripa2.current_ripa2_smooth()

        # Only set feature if the extractor returned a valid value
        if lhipa_val is not None:
            features["pupil_lhipa"] = lhipa_val
        if ripa2_val is not None:
            features["pupil_ripa2"] = ripa2_val

        wv_coeffs = self._wavelet.get_last_smoothed_coefficients(
            len(pupil_values)
        )
        for i, coeff in enumerate(wv_coeffs):
            if coeff is not None:
                features[f"pupil_wv_coeff_{i + 1}"] = coeff

    def _resolve_feature_columns(self, features: dict) -> None:
        all_columns = sorted(features.keys())
        selected = []
        for col in all_columns:
            for pattern in self._feature_patterns:
                if re.search(pattern, col):
                    selected.append(col)
                    break
        self._feature_columns = selected if selected else all_columns
        self._normalizer = WelfordNormalizer(len(self._feature_columns))
        logger.info(
            "Resolved %d feature columns from set '%s': %s",
            len(self._feature_columns),
            self._feature_set_name,
            self._feature_columns,
        )

    # ------------------------------------------------------------------
    # Smoothing & notification
    # ------------------------------------------------------------------

    def _smooth_and_notify(self) -> None:
        if not self._prediction_history:
            return

        raw_class, proba, _ = self._prediction_history[-1]

        # Don't feed unknown predictions into the filter
        if raw_class == self.UNKNOWN_CLASS:
            return

        with self._inference_lock:
            self._current_workload, self._current_probabilities = (
                self._filter.update(raw_class, proba)
            )

        for listener in self._listeners:
            try:
                listener(
                    raw_class,
                    self._current_workload,
                    self._current_probabilities,
                )
            except Exception:
                logger.exception("Listener callback failed")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_workload(self) -> int:
        return self._current_workload

    @property
    def current_probabilities(self) -> np.ndarray:
        return self._current_probabilities

    @property
    def prediction_history(self) -> list[tuple[int, np.ndarray, float]]:
        return list(self._prediction_history)

    @property
    def feature_columns(self) -> list[str] | None:
        return self._feature_columns

    @property
    def last_inference_timestamp(self) -> float | None:
        return self._last_inference_timestamp

    @property
    def has_model(self) -> bool:
        return self._model is not None

    @property
    def settings(self) -> InferenceSettings:
        return self._settings
