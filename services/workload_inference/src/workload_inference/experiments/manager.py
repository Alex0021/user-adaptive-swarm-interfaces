"""Concrete ExperimentManager subclasses for each experiment type."""

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from workload_inference.experiments.base import (
    COMMAND_DATA_FILE_NAME,
    DRONE_DATA_FILE_NAME,
    GAZE_DATA_FILE_NAME,
    INFERENCE_DATA_FILE_NAME,
    ExperimentManager,
)
from workload_inference.experiments.data_structures import (
    ExperimentState,
    ExperimentStatus,
    GateLayoutEntry,
    GateStatusEntry,
    InferenceRecord,
    NBackData,
)
from workload_inference.py_receiver import SMReceiver
from workload_inference.utilities import ExperimentDataWriter

# ══════════════════════════════════════════════════════════════════════════════
# N-back experiment
# ══════════════════════════════════════════════════════════════════════════════

NBACK_DATA_FILE_NAME = "nback_data.csv"
NBACK_CSV_HEADER = NBackData.__annotations__.keys()
NBACK_DATA_BLOCK_NAME = "ExperimentUnityNBackData"
NBACK_SEQUENCE_LEN = 20

logger_nback = logging.getLogger("NBackExperimentManager")


class NBackExperimentManager(ExperimentManager):
    """ExperimentManager for the N-back cognitive load experiment."""

    def __init__(self, base_folder: str = "experiments", queue_size: int = 1000):
        # N-back-specific state (must be set before super().__init__ which starts the thread)
        self._nback_receiver: SMReceiver | None = None
        self.nback_latest_datas: list[NBackData] | None = None
        self._request_nback_dump = False
        super().__init__(base_folder=base_folder, queue_size=queue_size)

    # ── Receivers ────────────────────────────────────────────────────────────

    def initialize_receivers(self) -> None:
        super().initialize_receivers()
        if self._nback_receiver is None:
            self._nback_receiver = SMReceiver(
                NBACK_DATA_BLOCK_NAME, NBackData, 15, NBACK_SEQUENCE_LEN
            )

    def start_receivers(self) -> None:
        super().start_receivers()
        if self._nback_receiver is not None and not self._nback_receiver.is_alive():
            self._nback_receiver.start()

    def stop_receivers(self) -> None:
        super().stop_receivers()
        if self._nback_receiver is not None and self._nback_receiver.is_alive():
            self._nback_receiver.stop()

    # ── Listeners ────────────────────────────────────────────────────────────

    def initialize_listeners(self) -> None:
        super().initialize_listeners()
        if self._nback_receiver is not None:
            self._nback_receiver.register_listener(self.nback_datas_callback)

    # ── Structure ────────────────────────────────────────────────────────────

    def _initialize_structure(
        self, extra_file_names: list[str] | None = None, overwrite: bool = False
    ) -> None:
        super()._initialize_structure(
            extra_file_names=(extra_file_names or []) + [NBACK_DATA_FILE_NAME],
            overwrite=overwrite,
        )

    # ── State-change hook ────────────────────────────────────────────────────

    def _on_state_change_hook(
        self, new_status: ExperimentStatus, last_status: ExperimentStatus | None
    ) -> None:
        """Handle N-back-specific state transitions."""
        if self._last_status is None:
            self._last_status = new_status
            return

        # Check for critical states (for data writing)
        if new_status.current_state != self._last_status.current_state:
            if (
                new_status.current_state == ExperimentState.Finished
                and self._start_time is not None
            ):
                self._duration = time.time() - self._start_time
                self._write_extra_experiment_info()
                # Stop the N-back receiver so Unity cannot overwrite nback_latest_datas
                # after the experiment ends (which would blank the score display).
                if self._nback_receiver is not None and self._nback_receiver.is_alive():
                    self._nback_receiver.stop()
            if new_status.current_state == ExperimentState.FlyingPractice:
                # Set folder path
                self._current_folder = (
                    self.base_folder
                    / self.experiment_config["name"]
                    / self.experiment_config["participant"]["uid"]
                    / new_status.current_state.name
                )
                # Set file to data writers
                if (
                    self._gaze_data_writer is not None
                    and self._drone_data_writer is not None
                ):
                    self._gaze_data_writer.new_file(
                        self._current_folder / GAZE_DATA_FILE_NAME
                    )
                    self._drone_data_writer.new_file(
                        self._current_folder / DRONE_DATA_FILE_NAME
                    )
                    if self._user_input_data_writer is not None:
                        self._user_input_data_writer.new_file(
                            self._current_folder / COMMAND_DATA_FILE_NAME
                        )
                    if self._inference_data_writer is not None:
                        self._inference_data_writer.new_file(
                            self._current_folder / INFERENCE_DATA_FILE_NAME
                        )
                    # Start recording
                    self._gaze_data_writer.start()
                    self._drone_data_writer.start()
                    self._user_input_data_writer.start()
                    if self._inference_data_writer is not None:
                        self._inference_data_writer.start()
                    self.start_receivers()
            elif new_status.current_state == ExperimentState.NBackPractice:
                # Set folder
                self._current_folder = (
                    self.base_folder
                    / self.experiment_config["name"]
                    / self.experiment_config["participant"]["uid"]
                    / new_status.current_state.name
                    / f"NBack{new_status.current_nback_level}"
                )
                if self._gaze_data_writer is not None:
                    self._gaze_data_writer.new_file(
                        self._current_folder / GAZE_DATA_FILE_NAME
                    )
                    self._gaze_data_writer.start()
                if self._inference_data_writer is not None:
                    self._inference_data_writer.new_file(
                        self._current_folder / INFERENCE_DATA_FILE_NAME
                    )
                    self._inference_data_writer.start()
                self.start_receivers()
                self._request_nback_dump = True
            elif new_status.current_state == ExperimentState.Trial:
                # Set folder
                self._current_folder = (
                    self.base_folder
                    / self.experiment_config["name"]
                    / self.experiment_config["participant"]["uid"]
                    / f"task_{new_status.current_task}"
                    / f"trial_{new_status.current_trial}"
                )
                # Set file to data writers
                if self._gaze_data_writer is not None:
                    self._gaze_data_writer.new_file(
                        self._current_folder / GAZE_DATA_FILE_NAME
                    )
                    self._gaze_data_writer.start()
                if self._drone_data_writer is not None:
                    self._drone_data_writer.new_file(
                        self._current_folder / DRONE_DATA_FILE_NAME
                    )
                    self._drone_data_writer.start()
                if self._user_input_data_writer is not None:
                    self._user_input_data_writer.new_file(
                        self._current_folder / COMMAND_DATA_FILE_NAME
                    )
                    self._user_input_data_writer.start()
                if self._inference_data_writer is not None:
                    self._inference_data_writer.new_file(
                        self._current_folder / INFERENCE_DATA_FILE_NAME
                    )
                    self._inference_data_writer.start()
                self.start_receivers()
                self._request_nback_dump = True
            else:
                # Stop recording of writers
                if self._gaze_data_writer is not None:
                    self._gaze_data_writer.stop()
                if self._drone_data_writer is not None:
                    self._drone_data_writer.stop()
                if self._user_input_data_writer is not None:
                    self._user_input_data_writer.stop()
                if self._inference_data_writer is not None:
                    self._inference_data_writer.stop()
                self.dump_latest_nback_data()

        self._last_status = new_status

    # ── N-back callbacks ─────────────────────────────────────────────────────

    def nback_datas_callback(
        self, datas: Sequence[NBackData], batch_update: bool = False
    ) -> None:
        """Store incoming N-back data for later display and CSV dump."""
        if not isinstance(datas, list):
            logger_nback.warning(
                "Received N-back data is not a list. Got type '%s'. Ignoring.",
                type(datas),
            )
            return
        self.nback_latest_datas = datas

    def dump_latest_nback_data(self) -> None:
        """Write accumulated N-back sequence to a CSV file."""
        if not self._request_nback_dump:
            return
        if self.nback_latest_datas is None:
            logger_nback.warning("No N-back data available to dump.")
            return
        if self._current_folder is None:
            logger_nback.warning("Current folder is not set. Cannot dump N-back data.")
            return
        with open(
            self._current_folder / NBACK_DATA_FILE_NAME, "w", encoding="utf-8"
        ) as f:
            f.write(",".join(NBACK_CSV_HEADER) + "\n")
            for nback_data in self.nback_latest_datas:
                f.write(
                    ",".join(
                        str(getattr(nback_data, field)) for field in NBACK_CSV_HEADER
                    )
                    + "\n"
                )
        logger_nback.info(
            "Dumped latest N-back data to '%s'",
            self._current_folder / NBACK_DATA_FILE_NAME,
        )
        self._request_nback_dump = False

    # ── Inference callback ────────────────────────────────────────────────────

    def inference_callback(
        self, raw_class: int, filtered_class: int, probabilities: np.ndarray
    ) -> None:
        """Write inference record including the current N-back level."""
        if self._inference_data_writer is None:
            return
        nback_level = (
            self._current_status.current_nback_level
            if self._current_status is not None
            else -1
        )
        record = InferenceRecord(
            timestamp=int(time.time() * 1000),
            prob_low=float(probabilities[0]),
            prob_medium=float(probabilities[1]),
            prob_high=float(probabilities[2]),
            raw_state=raw_class,
            filtered_state=filtered_class,
            nback_level=nback_level,
        )
        self._inference_data_writer.datas_callback([record])

    # ── Close ────────────────────────────────────────────────────────────────

    def close(self) -> None:
        super().close()
        if self._nback_receiver is not None and self._nback_receiver.is_alive():
            self._nback_receiver.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Gate racing experiment
# ══════════════════════════════════════════════════════════════════════════════

GATE_LAYOUT_FILE_NAME = "gate_layout.csv"
GATE_STATUS_FILE_NAME = "gate_status.csv"
GATE_LAYOUT_BLOCK_NAME = "ExperimentUnityGateLayout"
GATE_STATUS_BLOCK_NAME = "ExperimentUnityGateStatus"
GATE_LAYOUT_CSV_HEADER = GateLayoutEntry.__annotations__.keys()
GATE_STATUS_CSV_HEADER = GateStatusEntry.__annotations__.keys()
GATE_BLOCK_COUNT = 40

logger_gates = logging.getLogger("GateRacingExperimentManager")


class GateRacingExperimentManager(ExperimentManager):
    """ExperimentManager for the gate racing drone experiment."""

    def __init__(self, base_folder: str = "experiments", queue_size: int = 1000):
        # Gate-specific state (must be set before super().__init__)
        self._gate_layout_receiver: SMReceiver | None = None
        self._gate_status_receiver: SMReceiver | None = None
        self._gate_layout_writer: ExperimentDataWriter | None = None
        self._gate_status_writer: ExperimentDataWriter | None = None
        self._gate_layout: list[GateLayoutEntry] = []
        self._latest_gate_statuses: dict[int, GateStatusEntry] = {}
        self.trial_start_timestamp: int | None = None
        self.trial_finish_times: dict[int, float] = {}
        self.trial_crashed_drones: dict[int, int] = {}
        super().__init__(base_folder=base_folder, queue_size=queue_size)

    # ── Receivers ────────────────────────────────────────────────────────────

    def initialize_receivers(self) -> None:
        super().initialize_receivers()
        if self._gate_layout_receiver is None:
            self._gate_layout_receiver = SMReceiver(
                GATE_LAYOUT_BLOCK_NAME, GateLayoutEntry, 15, GATE_BLOCK_COUNT
            )
        if self._gate_status_receiver is None:
            self._gate_status_receiver = SMReceiver(
                GATE_STATUS_BLOCK_NAME, GateStatusEntry, 15, GATE_BLOCK_COUNT
            )

    def start_receivers(self) -> None:
        super().start_receivers()
        if (
            self._gate_layout_receiver is not None
            and not self._gate_layout_receiver.is_alive()
        ):
            self._gate_layout_receiver.start()
        if (
            self._gate_status_receiver is not None
            and not self._gate_status_receiver.is_alive()
        ):
            self._gate_status_receiver.start()

    def stop_receivers(self) -> None:
        super().stop_receivers()
        if (
            self._gate_layout_receiver is not None
            and self._gate_layout_receiver.is_alive()
        ):
            self._gate_layout_receiver.stop()
        if (
            self._gate_status_receiver is not None
            and self._gate_status_receiver.is_alive()
        ):
            self._gate_status_receiver.stop()

    # ── Writers ───────────────────────────────────────────────────────────────

    def initialize_data_writers(self) -> None:
        super().initialize_data_writers()
        if self._gate_layout_writer is None:
            self._gate_layout_writer = ExperimentDataWriter(
                header=GATE_LAYOUT_CSV_HEADER,
                name=GATE_LAYOUT_FILE_NAME,
                block_size=GATE_BLOCK_COUNT,
                mode=ExperimentDataWriter.WriterMode.SNAPSHOT,
            )
        if self._gate_status_writer is None:
            self._gate_status_writer = ExperimentDataWriter(
                header=GATE_STATUS_CSV_HEADER,
                block_size=GATE_BLOCK_COUNT,
                name=GATE_STATUS_FILE_NAME,
                mode=ExperimentDataWriter.WriterMode.SNAPSHOT,
            )

    # ── Listeners ────────────────────────────────────────────────────────────

    def initialize_listeners(self) -> None:
        super().initialize_listeners()
        if self._gate_layout_receiver is not None:
            self._gate_layout_receiver.register_listener(self.gate_layout_callback)
        if self._gate_status_receiver is not None:
            self._gate_status_receiver.register_listener(self.gate_status_callback)
        if self._drone_receiver is not None:
            self._drone_receiver.register_listener(self.drone_data_callback)

    # ── Structure ────────────────────────────────────────────────────────────

    def _initialize_structure(
        self, extra_file_names: list[str] | None = None, overwrite: bool = False
    ) -> None:
        super()._initialize_structure(
            extra_file_names=(extra_file_names or [])
            + [GATE_LAYOUT_FILE_NAME, GATE_STATUS_FILE_NAME],
            overwrite=overwrite,
        )

    # ── State-change hook ────────────────────────────────────────────────────

    def _on_state_change_hook(
        self, new_status: ExperimentStatus, last_status: ExperimentStatus | None
    ) -> None:
        """Open/close gate writers alongside the shared base writers."""
        # Record trial lap time when exiting Trial state
        if (
            last_status is not None
            and last_status.current_state == ExperimentState.Trial
            and new_status.current_state != ExperimentState.Trial
            and self.trial_start_timestamp is not None
        ):
            elapsed_s = (time.time() * 1000 - self.trial_start_timestamp) / 1000
            self.trial_finish_times[last_status.current_trial] = elapsed_s
            logger_gates.info(
                "Trial %d completed in %.2fs",
                last_status.current_trial,
                elapsed_s,
            )

        # Only stop the timer on ReadyScreen (keep finish times for display)
        if new_status.current_state == ExperimentState.ReadyScreen:
            self.trial_start_timestamp = None

        if (
            new_status.current_state == ExperimentState.Finished
            and self._start_time is not None
        ):
            self._duration = time.time() - self._start_time
            self._write_extra_experiment_info()
        if new_status.current_state == ExperimentState.FlyingPractice:
            # Set folder path
            self._current_folder = (
                self.base_folder
                / self.experiment_config["name"]
                / self.experiment_config["participant"]["uid"]
                / new_status.current_state.name
            )
            # Set file to data writers
            if (
                self._gaze_data_writer is not None
                and self._drone_data_writer is not None
            ):
                self._open_and_start_base_writers(
                    gaze=True, drone=True, user_input=True, inference=True
                )
                # Start recording
                self._gaze_data_writer.start()
                self._drone_data_writer.start()
                self._user_input_data_writer.start()
                if self._inference_data_writer is not None:
                    self._inference_data_writer.start()
            self.start_receivers()
        elif new_status.current_state == ExperimentState.Trial:
            # Reset trial start timestamp (will be set when first gate activates)
            self.trial_start_timestamp = None
            # Initialize current trial's crash count if not present
            if new_status.current_trial not in self.trial_crashed_drones:
                self.trial_crashed_drones[new_status.current_trial] = 0
            # Set folder
            self._current_folder = (
                self.base_folder
                / self.experiment_config["name"]
                / self.experiment_config["participant"]["uid"]
                / f"task_{new_status.current_task}"
                / f"trial_{new_status.current_trial}"
            )
            # Set file to data writers
            self._open_and_start_base_writers(
                gaze=True, drone=True, user_input=True, inference=True
            )
            if self._gate_layout_writer is not None:
                self._gate_layout_writer.new_file(
                    self._current_folder / GATE_LAYOUT_FILE_NAME
                )
                self._gate_layout_writer.start()
            if self._gate_status_writer is not None:
                self._gate_status_writer.new_file(
                    self._current_folder / GATE_STATUS_FILE_NAME
                )
                self._gate_status_writer.start()
            self.start_receivers()
        elif new_status.current_state == ExperimentState.Countdown:
            self.stop_receivers()
        else:
            # Stop recording of writers
            self._stop_base_writers()
            if self._gate_layout_writer is not None:
                self._gate_layout_writer.stop()
            if self._gate_status_writer is not None:
                self._gate_status_writer.stop()

    def _on_api_ready_hook(self) -> None:
        """When API becomes ready, also dump gate layout if available."""
        super()._on_api_ready_hook()

    # ── Gate callbacks ────────────────────────────────────────────────────────

    def gate_layout_callback(
        self, datas: Sequence[GateLayoutEntry], batch_update: bool = False
    ) -> None:
        """Cache static gate layout and persist it to CSV."""
        self._gate_layout = list(datas)
        if self._gate_layout_writer is not None:
            self._gate_layout_writer.datas_callback(datas, batch_update)

    def gate_status_callback(
        self, datas: Sequence[GateStatusEntry], batch_update: bool = False
    ) -> None:
        """Cache latest gate status and persist real-time events to CSV."""
        for entry in datas:
            self._latest_gate_statuses[int(entry.id)] = entry
        # Only start the timer if the course started (first gate becomes "Next" = 1)
        if self.trial_start_timestamp is None and len(datas) > 0:
            first_gate = datas[0]
            if int(first_gate.gate_state) == 1:
                self.trial_start_timestamp = int(
                    time.time() * 1000
                )  # Convert to milliseconds
                logger_gates.debug(
                    "Trial started: first gate activated at %d ms",
                    self.trial_start_timestamp,
                )
        if self._gate_status_writer is not None:
            self._gate_status_writer.datas_callback(datas, batch_update)

    def drone_data_callback(self, datas: Sequence, batch_update: bool = False) -> None:
        """Track crashed drone count during current trial."""
        if self._current_status is None:
            return
        # Count drones that crashed (alive field == 0)
        crashed_count = sum(
            1 for drone in datas if hasattr(drone, "alive") and int(drone.alive) == 0
        )
        self.trial_crashed_drones[self._current_status.current_trial] = crashed_count
        if self._drone_data_writer is not None:
            self._drone_data_writer.datas_callback(datas, batch_update)

    # ── Extra info ────────────────────────────────────────────────────────────

    def _write_extra_experiment_info(self) -> None:
        import yaml

        super()._write_extra_experiment_info()
        folder = (
            self.base_folder
            / self.experiment_config["name"]
            / self.experiment_config["participant"]["uid"]
        )
        # Append gate layout to the same yaml file
        extra_info_path = folder / "extra_info.yaml"
        try:
            with open(extra_info_path) as f:
                extra_info = yaml.safe_load(f) or {}
        except FileNotFoundError:
            extra_info = {}
        extra_info["gates"] = [
            {
                "id": int(g.id),
                "center_x": float(g.center_x),
                "center_y": float(g.center_y),
                "center_z": float(g.center_z),
                "width": float(g.width),
                "height": float(g.height),
            }
            for g in self._gate_layout
        ]
        with open(extra_info_path, "w") as f:
            yaml.dump(extra_info, f)

    # ── Close ────────────────────────────────────────────────────────────────

    def close(self) -> None:
        super().close()
        for writer in (self._gate_layout_writer, self._gate_status_writer):
            if writer is not None:
                writer.stop()
        for receiver in (self._gate_layout_receiver, self._gate_status_receiver):
            if receiver is not None and receiver.is_alive():
                receiver.stop()
