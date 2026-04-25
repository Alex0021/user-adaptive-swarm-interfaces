"""Online cognitive workload inference engine.

Accumulates streaming GazeData, preprocesses sliding windows, and runs a
trained classifier to estimate workload level (low / medium / high).

Architecture
------------
``WorkloadInferenceEngine`` is an abstract base class that owns all shared
infrastructure: gaze buffering, preprocessing, smoothing, and notification.
Two abstract methods must be implemented by each concrete subclass:

- :meth:`_load_model`                – load the model artefact from disk.
- :meth:`_extract_features_and_predict` – build the model input and return a
  ``(pred_class, proba)`` pair, or ``None`` to skip the window.

Concrete subclasses:

- :class:`TabNetInferenceEngine`  – PyTorch-TabNet (``.zip``)
- :class:`SklearnInferenceEngine` – scikit-learn compatible (``.pkl`` / ``.joblib``)
- :class:`TCNInferenceEngine`     – Temporal Convolutional Network (``.pt`` / ``.pth``)

Use :meth:`WorkloadInferenceEngine.create` to pick the right subclass from
``settings.model_type`` automatically.
"""

import logging
import re
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from eye_metrics.config import EyeMetricsConfig
from eye_metrics.features.definitions import FEATURE_SETS
from eye_metrics.features.normalization import WelfordNormalizer
from eye_metrics.features.pupil import LHIPA, RIPA2, WaveletFeature
from eye_metrics.preprocessing.eye_selection import select_best_eye
from eye_metrics.preprocessing.gaps import detect_gaps_and_blinks
from eye_metrics.preprocessing.interpolation import interpolate_pupil_data
from eye_metrics.preprocessing.outliers import OnlinePupilStats

from workload_inference.constants import DATA_DIR
from workload_inference.experiments.data_structures import GazeData

from .filters import FILTER_REGISTRY, WorkloadFilter
from .settings import InferenceSettings

logger = logging.getLogger(__name__)

DEFAULT_EYE_METRICS_CONFIG_FILENAME = "eye_metrics.yml"


def _build_filter_from_settings(settings: InferenceSettings) -> WorkloadFilter:
    """Build a filter instance from settings.filter dict with 'type' and params."""
    filter_config = settings.filter or {}
    if "type" not in filter_config:
        logger.warning(
            "No filter type specified in settings. Using RawFilter (no smoothing)."
        )
        return FILTER_REGISTRY["RawFilter"]()
    filter_type = filter_config["type"]

    filter_cls = FILTER_REGISTRY.get(filter_type)
    if filter_cls is None:
        raise ValueError(
            f"Unknown filter type '{filter_type}'. Available: {list(FILTER_REGISTRY)}"
        )

    # Extract all keys except 'type' as constructor parameters
    params = {k: v for k, v in filter_config.items() if k != "type"}
    return filter_cls(**params)


class WorkloadInferenceEngine(ABC):
    """Abstract base for online cognitive workload inference from streaming gaze data.

    Usage::

        settings = InferenceSettings.from_yaml("settings.yml")
        engine = WorkloadInferenceEngine.create("models/my_model.zip", settings)
        engine.register_listener(my_callback)
        gaze_receiver.register_listener(engine.gaze_datas_callback)
    """

    UNKNOWN_CLASS: int = -1

    def __init__(
        self,
        model_path: str | Path | None = None,
        settings: InferenceSettings | None = None,
        filter: WorkloadFilter | None = None,
        eye_metrics_config: EyeMetricsConfig | None = None,
    ):
        if settings is None:
            logger.warning(
                "No inference settings provided. Using defaults for all parameters."
            )
            settings = InferenceSettings()
        self._settings = settings

        if eye_metrics_config is not None:
            self._eye_metrics_config = eye_metrics_config
        elif (DATA_DIR / DEFAULT_EYE_METRICS_CONFIG_FILENAME).exists():
            self._eye_metrics_config = EyeMetricsConfig.from_yaml(
                DATA_DIR / DEFAULT_EYE_METRICS_CONFIG_FILENAME
            )
        else:
            self._eye_metrics_config = EyeMetricsConfig()

        self._sample_rate = settings.sample_rate
        self._window_size_samples = settings.window_size_samples
        self._inference_interval_samples = settings.inference_interval_samples
        self._wavelet_level = settings.wavelet_level
        self._min_valid_ratio = settings.min_valid_ratio
        self._feature_set_name = settings.feature_set
        self._feature_patterns = FEATURE_SETS.get(
            settings.feature_set, FEATURE_SETS["all"]
        )

        if filter is None:
            filter = _build_filter_from_settings(settings)
        self._filter = filter

        self._model = None
        self._model_type: str | None = None
        if model_path is not None:
            self._load_model(Path(model_path))

        self._raw_buffer: deque[GazeData] = deque(maxlen=self._window_size_samples * 2)
        self._samples_since_last_inference = 0

        self._pupil_stats = OnlinePupilStats(
            self._eye_metrics_config.preprocessing.outlier_rejection.ema_alpha
        )

        buffer_size = settings.rolling_buffer_samples
        self._ripa2 = RIPA2(
            buffer_size=buffer_size,
            smoothing_window_s=settings.pupil_ripa2_smoothing,
            sample_rate=settings.sample_rate,
        )
        self._lhipa = LHIPA(buffer_size=buffer_size, sample_rate=settings.sample_rate)
        self._wavelet = WaveletFeature(
            level=settings.wavelet_level,
            buffer_size=buffer_size,
            sample_rate=settings.sample_rate,
        )

        # Resolved lazily on first extraction (feature-based engines only)
        if settings.feature_columns:
            self._feature_columns: list[str] | None = list(settings.feature_columns)
            self._normalizer: WelfordNormalizer | None = WelfordNormalizer(
                len(self._feature_columns)
            )
        else:
            self._feature_columns = None
            self._normalizer = None

        # Last valid feature vector used to impute bad values (NaN/inf)
        self._last_feature_vector: np.ndarray | None = None

        self._prediction_history: deque[tuple[int, np.ndarray, float]] = deque(
            maxlen=200
        )
        self._current_workload: int = self.UNKNOWN_CLASS
        self._current_probabilities: np.ndarray = np.zeros(3)
        self._listeners: list[Callable[[int, int, np.ndarray], None]] = []

        self._inference_lock = threading.Lock()
        self._inference_thread: threading.Thread | None = None
        self._last_inference_timestamp: float | None = None

        logger.info(
            "%s initialized: window=%.1fs, interval=%.1fs, features=%s, model=%s",
            type(self).__name__,
            settings.window_size_sec,
            settings.inference_interval_sec,
            settings.feature_set,
            model_path or "none",
        )

    @classmethod
    def create(
        cls,
        model_path: str | Path | None = None,
        settings: InferenceSettings | None = None,
        filter: WorkloadFilter | None = None,
        eye_metrics_config: EyeMetricsConfig | None = None,
    ) -> WorkloadInferenceEngine:
        """Instantiate the engine subclass specified by ``settings.model_type``."""
        if settings is None:
            logger.warning(
                "No inference settings provided. Using defaults for all parameters."
            )
            settings = InferenceSettings()
        engines = {
            "tabnet": TabNetInferenceEngine,
            "sklearn": SklearnInferenceEngine,
            "tcn": TCNInferenceEngine,
        }
        engine_cls = engines.get(settings.model_type.lower())
        if engine_cls is None:
            raise ValueError(
                f"Unknown model_type '{settings.model_type}'. "
                f"Expected one of: {list(engines)}"
            )
        return engine_cls(model_path, settings, filter, eye_metrics_config)

    @abstractmethod
    def _load_model(self, model_path: Path) -> None:
        """Load the model artefact from *model_path* into ``self._model``."""

    @abstractmethod
    def _extract_features_and_predict(
        self,
        pupil_df: pd.DataFrame,
        gaps_df: pd.DataFrame,
        duration: float,
    ) -> tuple[int, np.ndarray] | None:
        """Build model input and return ``(pred_class, proba)``, or ``None`` to skip."""

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
            if self._inference_thread is not None and self._inference_thread.is_alive():
                return
            snapshot = list(self._raw_buffer)[-self._window_size_samples :]
            self._inference_thread = threading.Thread(
                target=self._run_inference_safe,
                args=(snapshot,),
                daemon=True,
            )
            self._inference_thread.start()

    def _run_inference_safe(self, snapshot: list[GazeData]) -> None:
        try:
            self._run_inference(snapshot)
        except Exception:
            logger.exception("Inference thread crashed")

    def _run_inference(self, snapshot: list[GazeData]) -> None:
        n_available = len(snapshot)
        min_samples = int(self._window_size_samples * self._min_valid_ratio)
        if n_available < min_samples:
            logger.info(
                "Skipping inference: only %d samples available (need %d)",
                n_available,
                min_samples,
            )
            return

        df = self._build_dataframe(snapshot)
        pupil_df, gaps_df = self._preprocess_online(df)
        if pupil_df.empty or len(pupil_df) < 50:
            logger.info("Skipping inference: insufficient samples after preprocessing")
            self._repeat_last_prediction()
            return

        duration = (
            pupil_df["timestamp_sec"].iloc[-1] - pupil_df["timestamp_sec"].iloc[0]
        )
        if duration <= 0:
            logger.info("Skipping inference: zero-length window")
            self._repeat_last_prediction()
            return

        result = self._extract_features_and_predict(pupil_df, gaps_df, duration)
        if result is None:
            if self._prediction_history:
                self._repeat_last_prediction()
            else:
                neutral = np.array([1.0 / 3.0, 1.0 / 3.0 + 1e-9, 1.0 / 3.0])
                with self._inference_lock:
                    self._prediction_history.append((1, neutral, time.time()))
                self._smooth_and_notify()
                self._last_inference_timestamp = time.time()
            return

        pred_class, proba = result
        with self._inference_lock:
            self._prediction_history.append((pred_class, proba, time.time()))
        self._smooth_and_notify()
        self._last_inference_timestamp = time.time()

    def _repeat_last_prediction(self) -> None:
        """Re-emit the last known prediction when the current window is unusable."""
        if self._prediction_history:
            self._smooth_and_notify()
            self._last_inference_timestamp = time.time()

    # ------------------------------------------------------------------
    # DataFrame construction
    # ------------------------------------------------------------------

    def _build_dataframe(self, samples: list[GazeData]) -> pd.DataFrame:
        rows = []
        start_ts = None
        for gaze in samples:
            ts_sec = float(gaze.timestamp) / 1000.0
            if start_ts is None:
                start_ts = ts_sec
            rows.append(
                {
                    "timestamp_sec": ts_sec - start_ts,
                    "left_gaze_point_x": float(gaze.left_gaze_point_x),
                    "left_gaze_point_y": float(gaze.left_gaze_point_y),
                    "left_gaze_point_z": float(gaze.left_gaze_point_z),
                    "right_gaze_point_x": float(gaze.right_gaze_point_x),
                    "right_gaze_point_y": float(gaze.right_gaze_point_y),
                    "right_gaze_point_z": float(gaze.right_gaze_point_z),
                    "left_point_screen_x": float(gaze.left_point_screen_x),
                    "left_point_screen_y": float(gaze.left_point_screen_y),
                    "right_point_screen_x": float(gaze.right_point_screen_x),
                    "right_point_screen_y": float(gaze.right_point_screen_y),
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
        empty = pd.DataFrame()
        pre = self._eye_metrics_config.preprocessing

        eye_df, _best_eye = select_best_eye(
            eye_df.copy(),
            threshold=pre.eye_selection.validity_difference_threshold,
        )

        gaps_df = detect_gaps_and_blinks(
            eye_df,
            confidence_threshold=pre.gaps_and_blinks.confidence_threshold,
            blink_threshold_range=(
                pre.gaps_and_blinks.blink_duration_min_ms,
                pre.gaps_and_blinks.blink_duration_max_ms,
            ),
            eye_openness_column="openness",
            openness_threshold=pre.gaps_and_blinks.openness_threshold,
        )

        eye_df.rename(columns={"pupil_diameter_mm": "pupil_diameter"}, inplace=True)

        total_samples = len(eye_df)
        if total_samples == 0:
            return empty, gaps_df

        if not gaps_df.empty:
            non_blink = gaps_df[~gaps_df["is_blink"]]
            if not non_blink.empty:
                lc_count = (non_blink["stop_id"] - non_blink["start_id"] + 1).sum()
                if lc_count / total_samples > pre.validation.min_non_blink_gap_ratio:
                    return empty, gaps_df

        min_conf = pre.gaps_and_blinks.confidence_threshold
        eye_df = eye_df[eye_df["confidence"] >= min_conf]

        if "pupil_diameter" in eye_df.columns:
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
            outlier_mask = self._pupil_stats.outlier_mask(
                speeds,
                n_multiplier=pre.outlier_rejection.n_mad_multiplier,
            )
            eye_df = eye_df[~outlier_mask]

        # Remove blinks with a margin on either side to avoid edge artefacts
        margins = pre.gaps_and_blinks.blink_margin_ms / 1000.0
        for _, row in gaps_df[
            gaps_df["duration_ms"] >= pre.gaps_and_blinks.blink_duration_min_ms
        ].iterrows():
            idx_to_drop = eye_df[
                (eye_df["timestamp_sec"] >= row["start_timestamp"] - margins)
                & (eye_df["timestamp_sec"] <= row["stop_timestamp"] + margins)
            ].index
            eye_df.drop(idx_to_drop, inplace=True)

        if len(eye_df) < pre.interpolation.min_samples:
            return empty, gaps_df

        pupil_df = interpolate_pupil_data(
            eye_df,
            gaps_df,
            column="pupil_diameter",
            max_gap_ms=pre.interpolation.max_gap_ms,
            resample_period_ms=round(1000.0 / self._sample_rate, 2),
        )
        return pupil_df, gaps_df

    # ------------------------------------------------------------------
    # Feature extraction helpers (used by feature-based subclasses)
    # ------------------------------------------------------------------

    def _run_feature_pipeline(
        self,
        pupil_df: pd.DataFrame,
        gaps_df: pd.DataFrame,
        duration: float,
    ) -> np.ndarray | None:
        """Extract scalar engineered features and return a normalised vector.

        NaN/inf values are imputed from the last valid window; if no prior
        window exists the window is skipped (returns ``None``).  Windows where
        the normalizer has not yet seen enough observations to produce reliable
        z-scores also return ``None`` so the model never receives raw features.
        """
        features = self._extract_blink_features(gaps_df, duration)
        self._extract_pupil_realtime_features(pupil_df, features)

        if not features:
            logger.info("Skipping inference: no features extracted")
            return None

        if self._feature_columns is None:
            self._resolve_feature_columns(features)

        feature_vector = np.array(
            [features.get(col, np.nan) for col in self._feature_columns],
            dtype=np.float64,
        )

        # Treat inf as NaN so both are handled uniformly
        feature_vector = np.where(np.isinf(feature_vector), np.nan, feature_vector)

        # Use latest valid feature value if one or more features are bad (NaN or inf).
        # This allows engine to at least try its best to estimate cwl on current window
        bad_mask = np.isnan(feature_vector)
        if bad_mask.any():
            if self._last_feature_vector is not None:
                feature_vector = np.where(
                    bad_mask, self._last_feature_vector, feature_vector
                )
                logger.info(
                    "Imputed %d bad features from last valid window", bad_mask.sum()
                )
            else:
                logger.warning(
                    "Skipping inference: %d bad features, no prior window",
                    bad_mask.sum(),
                )
                return None

        # Save raw (pre-normalisation) vector for future imputation
        self._last_feature_vector = feature_vector.copy()

        # Only update the normalizer on clean windows (no imputation) and only
        # during warmup.  After warmup the mean/std are frozen so late-session
        # workload drift doesn't distort z-scores.
        warmup = self._settings.normalization_warmup_windows
        if not bad_mask.any() and (warmup <= 0 or self._normalizer.n < warmup):
            self._normalizer.update(feature_vector)

        min_obs = self._eye_metrics_config.normalization.min_observations
        if self._normalizer.n < min_obs:
            logger.warning(
                "Skipping inference: normalizer has %d/%d observations",
                self._normalizer.n,
                min_obs,
            )
            return None
        feature_vector = self._normalizer.normalize(feature_vector)

        return feature_vector

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

        if lhipa_val is not None:
            features["pupil_lhipa"] = lhipa_val
        if ripa2_val is not None:
            features["pupil_ripa2"] = ripa2_val

        wv_coeffs = self._wavelet.get_last_smoothed_coefficients(len(pupil_values))
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
        if raw_class == self.UNKNOWN_CLASS:
            return

        with self._inference_lock:
            self._current_workload, self._current_probabilities = self._filter.update(
                raw_class, proba
            )

        for listener in self._listeners:
            try:
                listener(raw_class, self._current_workload, self._current_probabilities)
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

    def reset_pupil_buffers(self) -> None:
        """Flush per-session buffers between tasks.

        Clears the raw gaze buffer, pupil feature extractors, and the Schmitt
        filter warmup state.  The WelfordNormalizer and OnlinePupilStats are
        left intact — they carry subject-level statistics that should stay warm
        across tasks.
        """
        self._ripa2.flush()
        self._lhipa.flush()
        self._wavelet.flush()
        self._raw_buffer.clear()
        self._samples_since_last_inference = 0
        self._last_feature_vector = None
        if hasattr(self, "_last_raw_sequence"):
            self._last_raw_sequence = None
        self._filter.reset()
        logger.debug("Pupil feature buffers and Schmitt filter reset for new task")


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------


class TabNetInferenceEngine(WorkloadInferenceEngine):
    """Inference engine backed by a PyTorch-TabNet model (``.zip``)."""

    def _load_model(self, model_path: Path) -> None:
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError:
            logger.error("pytorch_tabnet not installed. Cannot load TabNet model.")
            raise

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Device used",
                category=UserWarning,
                module="pytorch_tabnet",
            )
            self._model = TabNetClassifier(device_name="cpu")
        self._model.load_model(model_path)
        self._model_type = "tabnet"
        logger.info("Loaded TabNet model from %s", model_path)

    def _extract_features_and_predict(
        self,
        pupil_df: pd.DataFrame,
        gaps_df: pd.DataFrame,
        duration: float,
    ) -> tuple[int, np.ndarray] | None:
        feature_vector = self._run_feature_pipeline(pupil_df, gaps_df, duration)
        if feature_vector is None:
            return None
        X = feature_vector.reshape(1, -1).astype(np.float32)
        proba = self._model.predict_proba(X)[0]
        return int(np.argmax(proba)), proba


class SklearnInferenceEngine(WorkloadInferenceEngine):
    """Inference engine backed by a scikit-learn compatible model.

    Supports ``.pkl`` and ``.joblib`` serialised estimators.
    """

    def _load_model(self, model_path: Path) -> None:
        import joblib

        self._model = joblib.load(model_path)
        self._model_type = "sklearn"
        logger.info("Loaded sklearn model from %s", model_path)

    def _extract_features_and_predict(
        self,
        pupil_df: pd.DataFrame,
        gaps_df: pd.DataFrame,
        duration: float,
    ) -> tuple[int, np.ndarray] | None:
        feature_vector = self._run_feature_pipeline(pupil_df, gaps_df, duration)
        if feature_vector is None:
            return None
        X = feature_vector.reshape(1, -1).astype(np.float32)
        proba = self._model.predict_proba(X)[0]
        return int(np.argmax(proba)), proba


class TCNInferenceEngine(WorkloadInferenceEngine):
    """Inference engine backed by a Temporal Convolutional Network (``.pt``/``.pth``).

    The model receives a raw time-series tensor of shape
    ``(1, n_channels, seq_len)`` where *n_channels* corresponds to
    ``settings.raw_feature_columns`` (default: ``["pupil_diameter"]``) and
    *seq_len* is resampled to ``settings.window_size_samples``.

    Both TorchScript (``.pt`` saved with ``torch.jit.save``) and standard
    ``.pth`` checkpoints (``torch.save``) are supported; TorchScript is
    attempted first.
    """

    _DEFAULT_RAW_FEATURES: list[str] = ["pupil_diameter"]

    def _load_model(self, model_path: Path) -> None:
        try:
            import torch
        except ImportError:
            logger.error("torch not installed. Cannot load TCN model.")
            raise

        try:
            self._model = torch.jit.load(str(model_path), map_location="cpu")
            logger.info("Loaded TorchScript TCN model from %s", model_path)
        except Exception:
            self._model = torch.load(
                str(model_path), map_location="cpu", weights_only=False
            )
            if hasattr(self._model, "eval"):
                self._model.eval()
            logger.info("Loaded PyTorch TCN model from %s", model_path)

        self._model_type = "tcn"

    def _extract_features_and_predict(
        self,
        pupil_df: pd.DataFrame,
        gaps_df: pd.DataFrame,
        duration: float,
    ) -> tuple[int, np.ndarray] | None:
        import torch

        raw_cols = self._settings.raw_feature_columns or self._DEFAULT_RAW_FEATURES
        target_len = self._window_size_samples
        n_channels = len(raw_cols)

        # Lazily initialise the last-sequence cache
        if not hasattr(self, "_last_raw_sequence"):
            self._last_raw_sequence: np.ndarray | None = None

        # Build full (current_len, n_channels) array, channel by channel.
        # Columns present in pupil_df are used directly; missing ones (e.g. gaze
        # columns that don't survive pupil preprocessing) are filled from the
        # last known sequence for that channel, or zeros on the first window.
        n_rows = len(pupil_df)
        seq = np.zeros((n_rows, n_channels), dtype=np.float32)
        for i, col in enumerate(raw_cols):
            if col in pupil_df.columns:
                seq[:, i] = pupil_df[col].values.astype(np.float32)
            elif self._last_raw_sequence is not None:
                # Repeat the last known channel at its resampled length
                seq[:, i] = np.interp(
                    np.linspace(0, target_len - 1, n_rows),
                    np.arange(target_len),
                    self._last_raw_sequence[:, i],
                ).astype(np.float32)
                logger.debug("TCN: imputed missing column '%s' from last window", col)
            # else: column stays zero (first window with no history)

        # Resample to the fixed window length the model was trained on
        if n_rows != target_len:
            src = np.arange(n_rows)
            dst = np.linspace(0, n_rows - 1, target_len)
            seq = np.stack(
                [np.interp(dst, src, seq[:, i]) for i in range(n_channels)],
                axis=1,
            ).astype(np.float32)

        # Save for future imputation of missing channels
        self._last_raw_sequence = seq.copy()

        # (1, n_channels, seq_len) — standard conv1d layout
        X = torch.from_numpy(seq.T).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(X)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        return int(np.argmax(proba)), proba
