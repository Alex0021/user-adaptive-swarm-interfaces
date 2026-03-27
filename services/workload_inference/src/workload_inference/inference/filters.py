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


class SmoothingSchmittFilter(WorkloadFilter):
    """Two-stage filter: probability smoothing + Schmitt trigger for class stability.

    Stage 1: Averages probabilities over a rolling window of recent predictions.
    Stage 2: Gates the final class via hysteresis -- only switches state when the
             smoothed class is "stable enough" (passes both a fraction and
             consecutive-sample threshold).
    """

    def __init__(
        self,
        smoothing_predictions: int = 5,
        min_fraction: float = 0.6,
        min_consecutive: int = 3,
    ):
        self._smoothing_predictions = smoothing_predictions
        self._min_fraction = min_fraction
        self._min_consecutive = min_consecutive
        self._history: deque[tuple[int, np.ndarray]] = deque(
            maxlen=smoothing_predictions
        )
        self._stable_class: int = -1

    def update(self, pred_class: int, proba: np.ndarray) -> tuple[int, np.ndarray]:
        self._history.append((pred_class, proba))
        recent = list(self._history)

        # Stage 1: smoothed probabilities
        avg_proba = np.mean([p[1] for p in recent], axis=0)
        smooth_class = int(np.argmax(avg_proba))

        # Stage 2: Schmitt trigger
        candidate = smooth_class
        if candidate != self._stable_class:
            classes = [p[0] for p in recent]
            fraction = classes.count(candidate) / len(classes)
            passes_fraction = fraction >= self._min_fraction

            passes_consecutive = True
            if self._min_consecutive > 0:
                tail = classes[-self._min_consecutive :]
                passes_consecutive = len(tail) == self._min_consecutive and all(
                    c == candidate for c in tail
                )

            if passes_fraction and passes_consecutive:
                self._stable_class = candidate

        # On first prediction, accept whatever the smooth class says
        if self._stable_class == -1:
            self._stable_class = smooth_class

        return self._stable_class, avg_proba
