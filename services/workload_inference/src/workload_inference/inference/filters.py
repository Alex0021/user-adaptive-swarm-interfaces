"""Workload prediction filtering strategies.

Filters smooth raw per-window predictions to produce stable workload estimates.
"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class WorkloadFilter(ABC):
    """Abstract base class for workload prediction filtering strategies."""

    @abstractmethod
    def update(self, pred_class: int, proba: np.ndarray) -> tuple[int, np.ndarray]:
        """Accept a new prediction and return (filtered_class, filtered_proba)."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state (called between tasks)."""


class ProbabilitySmoothingFilter(WorkloadFilter):
    """Rolling-window probability average filter.

    Averages class probabilities over the last `window` predictions and returns
    the argmax as the smoothed class.  During the optional warmup period the raw
    input class is passed through unchanged (while history still accumulates).

    :param window: Number of recent predictions to average (including the current one).
    :param warmup_windows: Number of initial predictions to skip smoothing, allowing the filter to
    """

    def __init__(self, window: int = 5, warmup_windows: int = 0, **kwargs):
        self._warmup_windows = warmup_windows
        self._history: deque[tuple[int, np.ndarray]] = deque(maxlen=window)
        self._n_updates: int = 0

    def update(self, pred_class: int, proba: np.ndarray) -> tuple[int, np.ndarray]:
        self._n_updates += 1
        self._history.append((pred_class, proba))
        avg_proba = np.mean([p[1] for p in self._history], axis=0)
        if self._n_updates <= self._warmup_windows:
            return pred_class, avg_proba
        return int(np.argmax(avg_proba)), avg_proba

    def reset(self) -> None:
        self._history.clear()
        self._n_updates = 0


class SchmittTriggerFilter(WorkloadFilter):
    """Hysteresis filter that only switches class when the candidate is stable.

    Maintains a short history of recent input classes and only latches onto a
    new class when *both* conditions hold:
      - The candidate appears in at least `min_fraction` of the history window.
      - The last `min_consecutive` samples are all the candidate class.

    :param min_fraction: Minimum fraction of samples in the history window that must match the candidate class.
    :param min_consecutive: Minimum number of consecutive samples at the end of the history that must match the candidate class.  Set to 0 to disable this condition.
    :param window: Number of recent predictions to consider in the history window (including the current one).
    :param warmup_windows: Number of initial predictions to skip before applying the Schmitt trigger logic, allowing the
    """

    def __init__(
        self,
        min_fraction: float = 0.6,
        min_consecutive: int = 3,
        window: int = 5,
        warmup_windows: int = 0,
        **kwargs,
    ):
        self._min_fraction = min_fraction
        self._min_consecutive = min_consecutive
        self._warmup_windows = warmup_windows
        self._history: deque[int] = deque(maxlen=window)
        self._stable_class: int = -1
        self._n_updates: int = 0

    def update(self, pred_class: int, proba: np.ndarray) -> tuple[int, np.ndarray]:
        self._n_updates += 1
        self._history.append(pred_class)

        if self._n_updates <= self._warmup_windows:
            return pred_class, proba

        recent = np.array(self._history)
        candidate = pred_class
        if candidate != self._stable_class:
            fraction = np.count_nonzero(recent == candidate) / len(recent)
            passes_fraction = fraction >= self._min_fraction

            passes_consecutive = True
            if self._min_consecutive > 0:
                tail = recent[-self._min_consecutive :]
                passes_consecutive = len(tail) == self._min_consecutive and np.all(
                    tail == candidate
                )

            if passes_fraction and passes_consecutive:
                self._stable_class = candidate

        # First post-warmup prediction: latch immediately onto input class
        if self._stable_class == -1:
            self._stable_class = pred_class

        return self._stable_class, proba

    def reset(self) -> None:
        self._history.clear()
        self._n_updates = 0
        self._stable_class = -1


class FilterPipeline(WorkloadFilter):
    """Cascades multiple WorkloadFilters, feeding each output into the next."""

    def __init__(self, *filters: WorkloadFilter, **kwargs):
        self._filters = list(filters)

    def update(self, pred_class: int, proba: np.ndarray) -> tuple[int, np.ndarray]:
        for f in self._filters:
            pred_class, proba = f.update(pred_class, proba)
        return pred_class, proba

    def reset(self) -> None:
        for f in self._filters:
            f.reset()


class SmoothingSchmittFilter(FilterPipeline):
    """Convenience pipeline: ProbabilitySmoothingFilter -> SchmittTriggerFilter.

    Equivalent to the original combined filter.  Warmup is applied only to the
    smoothing stage so the Schmitt trigger always sees a valid input class.

    :param smoothing_predictions: Number of predictions to average in the smoothing stage.
    :param min_fraction: Schmitt trigger candidate class fraction threshold.
    :param min_consecutive: Schmitt trigger candidate class consecutive count threshold.
    :param warmup_windows: Number of initial predictions to skip smoothing (but not Schmitt
    """

    def __init__(
        self,
        smoothing_predictions: int = 5,
        min_fraction: float = 0.6,
        min_consecutive: int = 3,
        warmup_windows: int = 0,
        **kwargs,
    ):
        super().__init__(
            ProbabilitySmoothingFilter(
                window=smoothing_predictions,
                warmup_windows=warmup_windows,
            ),
            SchmittTriggerFilter(
                min_fraction=min_fraction,
                min_consecutive=min_consecutive,
                window=smoothing_predictions,
            ),
        )


class RawFilter(WorkloadFilter):
    """Pass-through filter: returns raw predictions unchanged."""

    def __init__(self, **kwargs):
        pass

    def update(self, pred_class: int, proba: np.ndarray) -> tuple[int, np.ndarray]:
        return pred_class, proba

    def reset(self) -> None:
        pass


# Registry maps class name strings (as used in settings.yml) to filter classes
FILTER_REGISTRY: dict[str, type[WorkloadFilter]] = {
    "RawFilter": RawFilter,
    "ProbabilitySmoothingFilter": ProbabilitySmoothingFilter,
    "SchmittTriggerFilter": SchmittTriggerFilter,
    "SmoothingSchmittFilter": SmoothingSchmittFilter,
}
