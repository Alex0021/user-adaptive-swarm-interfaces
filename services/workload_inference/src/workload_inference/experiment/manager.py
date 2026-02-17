import glob
import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import yaml

from workload_inference.utils.api import ExperimentAPI, ExperimentAPIError
from workload_inference.utils.constants import DATA_DIR
from workload_inference.data.data_structures import (
    DroneData,
    ExperimentState,
    ExperimentStatus,
    GazeData,
    NBackData,
)
from workload_inference.data.py_receiver import SMReceiver, SMReceiverCircularBuffer
from workload_inference.data.writers import ExperimentDataWriter

# Experiment specific constants
CONFIG_FILE_NAME = "experiment.yaml"
SAMPLE_CONFIG_FILE_NAME = "sample_experiment.yaml"
GAZE_DATA_FILE_NAME = "gaze_data.csv"
DRONE_DATA_FILE_NAME = "drone_data.csv"
NBACK_DATA_FILE_NAME = "nback_data.csv"
GAZE_CSV_HEADER = GazeData.__annotations__.keys()
DRONE_CSV_HEADER = DroneData.__annotations__.keys()
NBACK_CSV_HEADER = NBackData.__annotations__.keys()
EXPERIMENT_STATUS_UPDATE_RATE_MS = 500

# SM Block constants
METADATA_BLOCK_NAME = "TobiiUnityMetadata"
GAZE_DATA_BLOCK_NAME = "TobiiUnityGazeData"
NBACK_DATA_BLOCK_NAME = "ExperimentUnityNBackData"
DRONE_DATA_BLOCK_NAME = "ExperimentUnityDroneData"
GAZE_DATA_BLOCK_CNT = 100
NBACK_SEQUENCE_LEN = 20
DRONE_COUNT = 9

logger = logging.getLogger("ExperimentManager")


class ExperimentManager:
    """
    Manage the current experiment and store incoming data into csv files.
    """

    def __init__(self, base_folder: str = "experiments", queue_size: int = 1000):
        self.base_folder = DATA_DIR / base_folder
        self._api = ExperimentAPI()

        # Listeners
        self._gaze_receiver: SMReceiverCircularBuffer | None = None
        self._drone_receiver: SMReceiver | None = None
        self._nback_receiver: SMReceiver | None = None

        # Data writers
        self._gaze_data_writer: ExperimentDataWriter | None = None
        self._drone_data_writer: ExperimentDataWriter | None = None
        self._current_folder: Path | None = None

        # Experiment status
        self._current_status: ExperimentStatus | None = None
        self._last_status: ExperimentStatus | None = None
        self.nback_latest_datas: list[NBackData] | None = None
        self._request_nback_dump = False
        self._start_time: float | None = None
        self._duration: float | None = None

        # Threads
        self._api_thread: threading.Thread | None = None
        self._api_thread_running = False
        self._lock = threading.Lock()
        self._api_on_error: bool = False

        # Try to read experiment data from yaml file
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
                    f"Sample experiment configuration file '{SAMPLE_CONFIG_FILE_NAME}' "
                    f" not found in '{self.base_folder}'. "
                    "Please create an experiment configuration file."
                )
            with open(self.base_folder / SAMPLE_CONFIG_FILE_NAME) as f:
                self.experiment_config = yaml.safe_load(f)
        else:
            with open(self.base_folder / CONFIG_FILE_NAME) as f:
                self.experiment_config = yaml.safe_load(f)
        self._initialize_structure(overwrite=True)
        logger.info("Initialized with queue size %d.", queue_size)
        self._api_thread = threading.Thread(
            target=self._experiment_status_querry, daemon=True
        )
        self._api_thread_running = True
        self._api_thread.start()

    def start_receivers(self) -> None:
        """
        Start the data receiving threads.
        """
        if self._gaze_receiver is not None and not self._gaze_receiver.is_alive():
            self._gaze_receiver.start()
        if self._drone_receiver is not None and not self._drone_receiver.is_alive():
            self._drone_receiver.start()
        if self._nback_receiver is not None and not self._nback_receiver.is_alive():
            self._nback_receiver.start()

    def stop_receivers(self) -> None:
        """
        Stop the data receiving threads.
        """
        if self._gaze_receiver is not None and self._gaze_receiver.is_alive():
            self._gaze_receiver.stop()
        if self._drone_receiver is not None and self._drone_receiver.is_alive():
            self._drone_receiver.stop()
        if self._nback_receiver is not None and self._nback_receiver.is_alive():
            self._nback_receiver.stop()

    def initialize_receivers(self) -> None:
        if self._gaze_receiver is None:
            self._gaze_receiver = SMReceiverCircularBuffer(
                GAZE_DATA_BLOCK_NAME, METADATA_BLOCK_NAME, GazeData, GAZE_DATA_BLOCK_CNT
            )
        if self._drone_receiver is None:
            self._drone_receiver = SMReceiver(
                DRONE_DATA_BLOCK_NAME, DroneData, 30, DRONE_COUNT
            )
        if self._nback_receiver is None:
            self._nback_receiver = SMReceiver(
                NBACK_DATA_BLOCK_NAME, NBackData, 15, NBACK_SEQUENCE_LEN
            )

    def initialize_data_writers(self) -> None:
        if self._gaze_data_writer is None:
            self._gaze_data_writer = ExperimentDataWriter(
                header=GAZE_CSV_HEADER, name="gaze_data"
            )
        if self._drone_data_writer is None:
            self._drone_data_writer = ExperimentDataWriter(
                header=DRONE_CSV_HEADER, name="drone_data"
            )

    def initialize_listeners(self) -> None:
        if self._gaze_receiver is not None and self._gaze_data_writer is not None:
            self._gaze_receiver.register_listener(self._gaze_data_writer.datas_callback)
        if self._drone_receiver is not None and self._drone_data_writer is not None:
            self._drone_receiver.register_listener(
                self._drone_data_writer.datas_callback
            )
        if self._nback_receiver is not None:
            self._nback_receiver.register_listener(self.nback_datas_callback)

    def _initialize_structure(self, overwrite: bool = False) -> None:
        """
        Initialize the experiment data file structure.

        Args:
            overwrite (bool, optional): Whether to overwrite existing files.
        Raises:
            FileExistsError: If the experiment data file already exists and
            overwrite is False.
        """
        if "name" not in self.experiment_config:
            logger.warning(
                "Experiment name not found in configuration. Using 'anonymous'."
            )
        if "uid" not in self.experiment_config["participant"]:
            logger.warning("Participant UID not found in configuration. Using 'X123'.")
        if "tasks" not in self.experiment_config:
            logger.error(
                "At least one task must be defined in the experiment configuration."
            )
            return
        exp_name = self.experiment_config.get("name", "anonymous")
        if not isinstance(exp_name, str):
            logger.error(
                "Experiment name must be a string. Got '%s'.",
                exp_name,
            )  #
            exp_name = "anonymous"
        participant_uid = self.experiment_config["participant"].get("uid", "X123")
        if not isinstance(participant_uid, str):
            logger.error(
                "Participant UID must be a string. Got '%s'.",
                participant_uid,
            )
            participant_uid = "X123"
        exp_folder = self.base_folder / exp_name / participant_uid
        exp_folder.mkdir(parents=True, exist_ok=True)

        # Sanity check if any data file already exists
        existing_files = glob.glob(
            str(
                self.base_folder
                / exp_name
                / participant_uid
                / "**"
                / GAZE_DATA_FILE_NAME
            ),
            recursive=True,
        )
        existing_files += glob.glob(
            str(
                self.base_folder
                / exp_name
                / participant_uid
                / "**"
                / DRONE_DATA_FILE_NAME
            ),
            recursive=True,
        )
        existing_files += glob.glob(
            str(
                self.base_folder
                / exp_name
                / participant_uid
                / "**"
                / NBACK_DATA_FILE_NAME
            ),
            recursive=True,
        )

        if existing_files and not overwrite:
            raise FileExistsError(
                "Some experiment data files already exists in "
                f"{self.base_folder / exp_name / participant_uid}."
                " Please check carefully before overwriting. "
                f"Found {len(existing_files)} existing files."
                "Set overwrite=True or delete existing files to proceed."
            )

    def update_internal_state(self, new_status: ExperimentStatus) -> None:
        """Update the internal experiment state based on the provided status."""
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
                    # Start recording
                    self._gaze_data_writer.start()
                    self._drone_data_writer.start()
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
                self.start_receivers()
                self._request_nback_dump = True
            else:
                # Stop recording of writers
                if self._gaze_data_writer is not None:
                    self._gaze_data_writer.stop()
                if self._drone_data_writer is not None:
                    self._drone_data_writer.stop()
                self.dump_latest_nback_data()

        self._last_status = new_status

    def nback_datas_callback(
        self, datas: Sequence[NBackData], batch_update: bool = False
    ) -> None:
        """Callback to receive the latest N-back data."""
        if not isinstance(datas, list):
            logger.warning(
                "Received N-back data is not a list. Got type '%s'. Ignoring.",
                type(datas),
            )
            return
        self.nback_latest_datas = datas

    def dump_latest_nback_data(self) -> None:
        """Dump the latest N-back data into a csv file."""
        if not self._request_nback_dump:
            return
        if self.nback_latest_datas is None:
            logger.warning("No N-back data available to dump.")
            return
        if self._current_folder is None:
            logger.warning("Current folder is not set. Cannot dump N-back data.")
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
        logger.info(
            "Dumped latest N-back data to '%s'",
            self._current_folder / NBACK_DATA_FILE_NAME,
        )
        self._request_nback_dump = False

    def _experiment_status_querry(self) -> None:
        """Fetch the current experiment status from the API."""
        while self._api_thread_running:
            try:
                self._current_status = self._api.get_experiment_state()
                self.update_internal_state(self._current_status)
                self._api_on_error = False
            except ExperimentAPIError as e:
                logger.warning("Failed to fetch experiment status: %s", e)
                self._api_on_error = True
            time.sleep(EXPERIMENT_STATUS_UPDATE_RATE_MS / 1000)

    def _write_extra_experiment_info(self) -> None:
        """Write extra experiment info such as duration into a yaml file."""
        if self.base_folder is None:
            logger.warning(
                "Base folder is not set. Cannot write extra experiment info."
            )
            return
        extra_info = {
            "duration_sec": self._duration,
        }
        folder = (
            self.base_folder
            / self.experiment_config["name"]
            / self.experiment_config["participant"]["uid"]
        )
        with open(folder / "extra_info.yaml", "w") as f:
            yaml.dump(extra_info, f)
        logger.info(
            "Wrote extra experiment info to '%s'",
            folder / "extra_info.yaml",
        )

    def close(self):
        """Perform cleanup when closing the experiment manager."""
        self.stop_receivers()
        if self._gaze_data_writer is not None:
            self._gaze_data_writer.stop()
        if self._drone_data_writer is not None:
            self._drone_data_writer.stop()
        self._api_thread_running = False
        if self._api_thread is not None and self._api_thread.is_alive():
            self._api_thread.join(timeout=2)

    def request_next_state(self) -> None:
        """Request the API to move to the next state in the experiment."""
        try:
            self._api.trigger_next_state()
        except ExperimentAPIError as e:
            logger.error("Failed to trigger next state: %s", e)

    @property
    def gaze_receiver(self) -> SMReceiverCircularBuffer | None:
        return self._gaze_receiver

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
