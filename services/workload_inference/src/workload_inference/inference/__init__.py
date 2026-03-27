"""Online workload inference engine package."""

from .engine import WorkloadInferenceEngine
from .filters import SmoothingSchmittFilter, WorkloadFilter
from .online_stats import OnlinePupilStats, WelfordNormalizer
from .settings import InferenceSettings

WORKLOAD_LABELS = {-1: "Unknown", 0: "Low", 1: "Medium", 2: "High"}
WORKLOAD_COLORS = {-1: "#888888", 0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}

__all__ = [
    "InferenceSettings",
    "OnlinePupilStats",
    "SmoothingSchmittFilter",
    "WelfordNormalizer",
    "WORKLOAD_COLORS",
    "WORKLOAD_LABELS",
    "WorkloadFilter",
    "WorkloadInferenceEngine",
]
