"""Overlay implementations."""

from video_sync.overlay.base import BAR_HEIGHT, BaseOverlay
from video_sync.overlay.cwl import AdaptationStepOverlay, CWLStepOverlay
from video_sync.overlay.drones import DroneCountOverlay
from video_sync.overlay.gates import GateProgressOverlay
from video_sync.overlay.layout import HorizontalTopBar
from video_sync.overlay.timeline import ElapsedTimeOverlay
from video_sync.overlay.workload import WorkloadStateOverlay

OVERLAY_REGISTRY: dict[str, type[BaseOverlay]] = {
    "workload": WorkloadStateOverlay,
    "drones": DroneCountOverlay,
    "adaptation": AdaptationStepOverlay,
    "gates": GateProgressOverlay,
    "timeline": ElapsedTimeOverlay,
}

__all__ = [
    "BAR_HEIGHT",
    "BaseOverlay",
    "WorkloadStateOverlay",
    "DroneCountOverlay",
    "AdaptationStepOverlay",
    "CWLStepOverlay",
    "GateProgressOverlay",
    "ElapsedTimeOverlay",
    "HorizontalTopBar",
    "OVERLAY_REGISTRY",
]
