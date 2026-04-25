"""Shared base ExperimentManager for all experiment types."""

import glob
import logging
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from workload_inference.api import ExperimentAPI, ExperimentAPIError
from workload_inference.constants import DATA_DIR
from workload_inference.experiments.data_structures import (
    DroneData,
    ExperimentState,
    ExperimentStatus,
    GazeData,
    InferenceRecord,
    UserInputData,
)
from workload_inference.inference import InferenceSettings, WorkloadInferenceEngine
from workload_inference.py_receiver import SMReceiver, SMReceiverCircularBuffer
from workload_inference.utilities import ExperimentDataWriter

# ── Shared constants ──────────────────────────────────────────────────────────
CONFIG_FILE_NAME = "experiment.yaml"
SAMPLE_CONFIG_FILE_NAME = "sample_experiment.yaml"
GAZE_DATA_FILE_NAME = "gaze_data.csv"
DRONE_DATA_FILE_NAME = "drone_data.csv"
COMMAND_DATA_FILE_NAME = "command_data.csv"
INFERENCE_DATA_FILE_NAME = "inference_data.csv"
GAZE_CSV_HEADER = GazeData.__annotations__.keys()
DRONE_CSV_HEADER = DroneData.__annotations__.keys()
INFERENCE_CSV_HEADER = InferenceRecord.__annotations__.keys()
EXPERIMENT_STATUS_UPDATE_RATE_MS = 500

METADATA_BLOCK_NAME = "TobiiUnityMetadata"
GAZE_DATA_BLOCK_NAME = "TobiiUnityGazeData"
DRONE_DATA_BLOCK_NAME = "ExperimentUnityDroneData"
USER_INPUT_DATA_BLOCK_NAME = "ExperimentUnityUserInputData"
GAZE_DATA_BLOCK_CNT = 100
DRONE_COUNT = 9

logger = logging.getLogger("ExperimentManager")


class ExperimentManager:
    """
    Shared base class for all experiment types.

    Manages shared SM receivers (gaze, drone, user_input), CSV data writers,
    and the API polling thread. Subclasses extend by overriding the
    initialize_* methods and _on_state_change_hook().
    """

    def __init__(self, base_folder: str = "experiments", queue_size: int = 1000):
        self.base_folder = DATA_DIR / base_folder
        self._api = ExperimentAPI()

        # ── Shared receivers ──
        self._gaze_receiver: SMReceiverCircularBuffer | None = None
        self._drone_receiver: SMReceiver | None = None
        self._user_input_receiver: SMReceiver | None = None

        # ── Shared data writers ──
        self._gaze_data_writer: ExperimentDataWriter | None = None
        self._drone_data_writer: ExperimentDataWriter | None = None
        self._user_input_data_writer: ExperimentDataWriter | None = None
        self._inference_data_writer: ExperimentDataWriter | None = None
        self._current_folder: Path | None = None

        # ── Experiment state ──
        self._current_status: ExperimentStatus | None = None
        self._last_status: ExperimentStatus | None = None
        self._start_time: float | None = None
        self._duration: float | None = None
        self._already_initialized = False

        # ── API thread ──
        self._api_thread: threading.Thread | None = None
        self._api_thread_running = False
        self._lock = threading.Lock()
        self._api_on_error: bool = True
        self._previous_api_on_error: bool = True

        # ── Listeners ──
        self._api_ready_listeners: list[Callable] = [self.initialize_all]

        # ── Load config ──
        self._load_experiment_config()
        self._initialize_structure(overwrite=True)

        logger.info("Initialized with queue size %d.", queue_size)
        self._api_thread = threading.Thread(
            target=self._experiment_status_querry, daemon=True
        )
        self._api_thread_running = True
        self._api_thread.start()

    def _load_experiment_config(self) -> None:
        """Load experiment config from yaml, falling back to sample config."""
        if not (self.base_folder / CONFIG_FILE_NAME).exists():
            logger.warning(
                "Experiment configuration file '%s' not found in '%s'."
                " Using sample configuration '%s'.",
                CONFIG_FILE_NAME,
                self.base_folder,
                SAMPLE_CONFIG_FILE_NAME,
            )
            if not (self.base_folder / SAMPLE_CONFIG_FILE_NAME).exists():
                raise FileNotFoundError(
                    f"Sample experiment configuration file '{SAMPLE_CONFIG_FILE_NAME}'"
                    f" not found in '{self.base_folder}'."
                    " Please create an experiment configuration file."
                )
            with open(self.base_folder / SAMPLE_CONFIG_FILE_NAME) as f:
                self.experiment_config = yaml.safe_load(f)
        else:
            with open(self.base_folder / CONFIG_FILE_NAME) as f:
                self.experiment_config = yaml.safe_load(f)

    # ── Receiver lifecycle ───────────────────────────────────────────────────

    def start_receivers(self) -> None:
        """Start shared data receiving threads."""
        if self._gaze_receiver is not None and not self._gaze_receiver.is_alive():
            self._gaze_receiver.start()
        if self._drone_receiver is not None and not self._drone_receiver.is_alive():
            self._drone_receiver.start()
        if (
            self._user_input_receiver is not None
            and not self._user_input_receiver.is_alive()
        ):
            self._user_input_receiver.start()

    def stop_receivers(self) -> None:
        """Stop shared data receiving threads."""
        if self._gaze_receiver is not None and self._gaze_receiver.is_alive():
            self._gaze_receiver.stop()
        if self._drone_receiver is not None and self._drone_receiver.is_alive():
            self._drone_receiver.stop()
        if (
            self._user_input_receiver is not None
            and self._user_input_receiver.is_alive()
        ):
            self._user_input_receiver.stop()

    # ── Initialization ───────────────────────────────────────────────────────

    def register_api_ready_listener(self, listener: Callable) -> None:
        """Register a listener to be called when the API shifts from error to ready."""
        with self._lock:
            if listener not in self._api_ready_listeners:
                self._api_ready_listeners.append(listener)
            if not self._api_on_error:
                # Call immediately if API is already ready (might have missed the initial call)
                listener()

    def initialize_all(self) -> None:
        """Initialize all components once: receivers, writers, listeners."""
        if self._already_initialized:
            return
        self._already_initialized = True
        self.initialize_receivers()
        self.initialize_data_writers()
        self.initialize_listeners()

    def initialize_receivers(self) -> None:
        """Create shared SM receivers. Subclasses call super() then add their own."""
        if self._gaze_receiver is None:
            self._gaze_receiver = SMReceiverCircularBuffer(
                GAZE_DATA_BLOCK_NAME, METADATA_BLOCK_NAME, GazeData, GAZE_DATA_BLOCK_CNT
            )
        if self._drone_receiver is None:
            self._drone_receiver = SMReceiver(
                DRONE_DATA_BLOCK_NAME, DroneData, 30, DRONE_COUNT
            )
        if self._user_input_receiver is None:
            self._user_input_receiver = SMReceiver(
                USER_INPUT_DATA_BLOCK_NAME, UserInputData, 30
            )

    def initialize_data_writers(self) -> None:
        """Create shared CSV writers. Subclasses call super() then add their own."""
        if self._gaze_data_writer is None:
            self._gaze_data_writer = ExperimentDataWriter(
                header=GAZE_CSV_HEADER, name=GAZE_DATA_FILE_NAME
            )
        if self._drone_data_writer is None:
            self._drone_data_writer = ExperimentDataWriter(
                header=DRONE_CSV_HEADER, name=DRONE_DATA_FILE_NAME
            )
        if self._user_input_data_writer is None:
            self._user_input_data_writer = ExperimentDataWriter(
                header=UserInputData.__annotations__.keys(), name=COMMAND_DATA_FILE_NAME
            )
        if self._inference_data_writer is None:
            self._inference_data_writer = ExperimentDataWriter(
                header=INFERENCE_CSV_HEADER, name=INFERENCE_DATA_FILE_NAME
            )

    def initialize_listeners(self) -> None:
        """Wire shared receivers to their writers. Subclasses call super() then add their own."""
        if self._gaze_receiver is not None and self._gaze_data_writer is not None:
            self._gaze_receiver.register_listener(self._gaze_data_writer.datas_callback)
        if self._drone_receiver is not None and self._drone_data_writer is not None:
            self._drone_receiver.register_listener(
                self._drone_data_writer.datas_callback
            )
        if (
            self._user_input_receiver is not None
            and self._user_input_data_writer is not None
        ):
            self._user_input_receiver.register_listener(
                self._user_input_data_writer.datas_callback
            )

    def _initialize_structure(
        self, extra_file_names: list[str] | None = None, overwrite: bool = False
    ) -> None:
        """
        Create the experiment folder tree and check for pre-existing data files.

        Args:
            extra_file_names: Additional experiment-specific file names to check.
            overwrite: Whether to silently allow pre-existing files.
        Raises:
            FileExistsError: If data files already exist and overwrite is False.
        """
        if "name" not in self.experiment_config:
            logger.warning(
                "Experiment name not found in configuration. Using 'anonymous'."
            )
        if (
            "participant" not in self.experiment_config
            or "uid" not in self.experiment_config.get("participant", {})
        ):
            logger.warning("Participant UID not found in configuration. Using 'X123'.")
        if "tasks" not in self.experiment_config:
            logger.error(
                "At least one task must be defined in the experiment configuration."
            )
            return

        exp_name = self.experiment_config.get("name", "anonymous")
        if not isinstance(exp_name, str):
            logger.error("Experiment name must be a string. Got '%s'.", exp_name)
            exp_name = "anonymous"

        participant_uid = self.experiment_config.get("participant", {}).get(
            "uid", "X123"
        )
        if not isinstance(participant_uid, str):
            logger.error("Participant UID must be a string. Got '%s'.", participant_uid)
            participant_uid = "X123"

        exp_folder = self.base_folder / exp_name / participant_uid
        exp_folder.mkdir(parents=True, exist_ok=True)

        all_file_names = [
            GAZE_DATA_FILE_NAME,
            DRONE_DATA_FILE_NAME,
            COMMAND_DATA_FILE_NAME,
            INFERENCE_DATA_FILE_NAME,
        ] + (extra_file_names or [])

        existing_files: list[str] = []
        for fname in all_file_names:
            existing_files += glob.glob(str(exp_folder / "**" / fname), recursive=True)

        if existing_files and not overwrite:
            raise FileExistsError(
                f"Some experiment data files already exist in {exp_folder}."
                " Please check carefully before overwriting."
                f" Found {len(existing_files)} existing files."
                " Set overwrite=True or delete existing files to proceed."
            )

    # ── State machine ────────────────────────────────────────────────────────

    def update_internal_state(self, new_status: ExperimentStatus) -> None:
        """Handle state transitions — open/close writers and call the subclass hook."""
        if self._last_status is None:
            self._last_status = new_status
            self._on_state_change_hook(new_status, new_status)
            return

        if new_status.current_state != self._last_status.current_state:
            self._on_state_change_hook(new_status, self._last_status)

        self._last_status = new_status

    def _open_and_start_base_writers(
        self,
        gaze: bool = False,
        drone: bool = False,
        user_input: bool = False,
        inference: bool = False,
    ) -> None:
        """Open new CSV files and start the requested shared writers."""
        assert self._current_folder is not None
        if gaze and self._gaze_data_writer is not None:
            self._gaze_data_writer.new_file(self._current_folder / GAZE_DATA_FILE_NAME)
            self._gaze_data_writer.start()
        if drone and self._drone_data_writer is not None:
            self._drone_data_writer.new_file(
                self._current_folder / DRONE_DATA_FILE_NAME
            )
            self._drone_data_writer.start()
        if user_input and self._user_input_data_writer is not None:
            self._user_input_data_writer.new_file(
                self._current_folder / COMMAND_DATA_FILE_NAME
            )
            self._user_input_data_writer.start()
        if inference and self._inference_data_writer is not None:
            self._inference_data_writer.new_file(
                self._current_folder / INFERENCE_DATA_FILE_NAME
            )
            self._inference_data_writer.start()

    def _stop_base_writers(self) -> None:
        """Stop all shared data writers."""
        for writer in (
            self._gaze_data_writer,
            self._drone_data_writer,
            self._user_input_data_writer,
            self._inference_data_writer,
        ):
            if writer is not None:
                writer.stop()

    def _on_state_change_hook(
        self, new_status: ExperimentStatus, last_status: ExperimentStatus | None
    ) -> None:
        """Called after each state transition. Override in subclasses to add custom behaviour."""

    def _on_api_ready_hook(self) -> None:
        """Called when the API shifts from error to ready. Override in subclasses to add custom behaviour."""

    # ── Inference callback ───────────────────────────────────────────────────

    def inference_callback(
        self, raw_class: int, filtered_class: int, probabilities: np.ndarray
    ) -> None:
        """Receive inference results and write them to CSV."""
        if self._inference_data_writer is None:
            return
        record = InferenceRecord(
            timestamp=int(time.time() * 1000),
            prob_low=float(probabilities[0]),
            prob_medium=float(probabilities[1]),
            prob_high=float(probabilities[2]),
            raw_state=raw_class,
            filtered_state=filtered_class,
            nback_level=-1,
        )
        self._inference_data_writer.datas_callback([record])
        # Send to Unity via API endpoint
        self._api.send_to(
            "cwl/level",
            {
                "level": int(filtered_class),
                "lowProb": float(probabilities[0]),
                "medProb": float(probabilities[1]),
                "highProb": float(probabilities[2]),
            },
        )

    # ── Background API polling ───────────────────────────────────────────────

    def _experiment_status_querry(self) -> None:
        """Background thread: polls the API and drives state transitions."""
        while self._api_thread_running:
            try:
                self._current_status = self._api.get_experiment_state()
                self._api_on_error = False
                if self._previous_api_on_error:
                    for listener in self._api_ready_listeners:
                        listener()
                    self._on_api_ready_hook()
                self.update_internal_state(self._current_status)
            except ExperimentAPIError as e:
                logger.warning("Failed to fetch experiment status: %s", e)
                self._api_on_error = True
            self._previous_api_on_error = self._api_on_error
            time.sleep(EXPERIMENT_STATUS_UPDATE_RATE_MS / 1000)

    def _write_extra_experiment_info(self) -> None:
        """Write duration and any extra info to extra_info.yaml."""
        folder = (
            self.base_folder
            / self.experiment_config["name"]
            / self.experiment_config["participant"]["uid"]
        )
        extra_info: dict[str, Any] = {"duration_sec": self._duration}
        with open(folder / "extra_info.yaml", "w") as f:
            yaml.dump(extra_info, f)
        logger.info("Wrote extra experiment info to '%s'", folder / "extra_info.yaml")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop shared receivers, writers, and the API thread."""
        self.stop_receivers()
        self._stop_base_writers()
        self._api_thread_running = False
        if self._api_thread is not None and self._api_thread.is_alive():
            self._api_thread.join(timeout=2)

    def request_next_state(self) -> None:
        """Ask the API to advance the experiment state machine."""
        try:
            self._api.trigger_next_state()
        except ExperimentAPIError as e:
            logger.error("Failed to trigger next state: %s", e)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def gaze_receiver(self) -> SMReceiverCircularBuffer | None:
        return self._gaze_receiver

    @property
    def drone_receiver(self) -> SMReceiver | None:
        return self._drone_receiver

    @property
    def user_input_receiver(self) -> SMReceiver | None:
        return self._user_input_receiver

    @property
    def api_on_error(self) -> bool:
        return self._api_on_error

    @api_on_error.setter
    def api_on_error(self, value: bool) -> None:
        with self._lock:
            self._api_on_error = value

    @property
    def experiment_status(self) -> ExperimentStatus | None:
        return self._current_status
