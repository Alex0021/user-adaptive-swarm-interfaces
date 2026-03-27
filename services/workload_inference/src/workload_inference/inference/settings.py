"""Inference settings loaded from a YAML file generated during training.

The settings file captures the exact configuration used to train the model
so that the online inference pipeline uses matching parameters.

All window/interval sizes are expressed in **samples** so that they stay
consistent with the training pipeline.  Convenience properties convert to
seconds when needed.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class InferenceSettings:
    """Configuration for the online inference engine.

    Generated during training and saved alongside the model so that inference
    uses the exact same feature set and window parameters.
    """

    feature_set: str = "ipa_wavelets"
    sample_rate: float = 60.0
    window_size_samples: int = 300  # 5s at 60 Hz
    inference_interval_samples: int = 60  # 1s at 60 Hz
    wavelet_level: int = 4
    min_valid_ratio: float = 0.5
    rolling_buffer_multiplier: int = 2
    smoothing_predictions: int = 5
    schmitt_min_fraction: float = 0.6
    schmitt_min_consecutive: int = 3
    feature_columns: list[str] = field(default_factory=list)

    # -- derived helpers (seconds) -----------------------------------------

    @property
    def window_size_sec(self) -> float:
        return self.window_size_samples / self.sample_rate

    @property
    def inference_interval_sec(self) -> float:
        return self.inference_interval_samples / self.sample_rate

    @property
    def rolling_buffer_size(self) -> int:
        return self.window_size_samples * self.rolling_buffer_multiplier

    # -- persistence --------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InferenceSettings":
        """Load settings from a YAML file.

        Unknown keys are silently ignored so the file can carry extra metadata.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Settings file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in known_fields}
        settings = cls(**filtered)
        logger.info("Loaded inference settings from %s", path)
        return settings

    def save_yaml(self, path: str | Path) -> None:
        """Persist settings to a YAML file (typically called after training)."""
        from dataclasses import asdict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
        logger.info("Saved inference settings to %s", path)
