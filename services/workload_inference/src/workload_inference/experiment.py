import glob
import logging
import threading
import time
from pathlib import Path

import yaml
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from workload_inference.api import ExperimentAPI, ExperimentAPIError
from workload_inference.constants import DATA_DIR
from workload_inference.data_structures import (
    DroneData,
    ExperimentState,
    ExperimentStatus,
    GazeData,
    NBackData,
)
from workload_inference.py_receiver import SMReceiver, SMReceiverCircularBuffer
from workload_inference.utilities import ExperimentDataWriter
from workload_inference.visualize import GazeDataCanvas

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
            if new_status.current_state == ExperimentState.FlyingPractice:
                # Set folder path
                self._current_folder = (
                    self.base_folder
                    / self.experiment_config["name"]
                    / self.experiment_config["participant"]["uid"]
                    / new_status.current_state.name
                )
                # Set file to data writers
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
                self._request_nback_dump = True
            else:
                # Stop recording of writers
                if self._gaze_data_writer is not None:
                    self._gaze_data_writer.stop()
                if self._drone_data_writer is not None:
                    self._drone_data_writer.stop()
                self.dump_latest_nback_data()

        self._last_status = new_status

    def nback_datas_callback(self, nback_datas: list[NBackData]) -> None:
        """Callback to receive the latest N-back data."""
        self.nback_latest_datas = nback_datas

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
                logger.error("Failed to fetch experiment status: %s", e)
                self._api_on_error = True
            time.sleep(EXPERIMENT_STATUS_UPDATE_RATE_MS / 1000)

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


class ExperimentManagerWindow:
    """
    PyQt application to manage the experiment and visualize realtime gaze data.
    """

    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self._is_status_error = True

        self._initialize_core_compoonents()
        self._initialize_widgets()

    def _initialize_core_compoonents(self):
        self._window = QMainWindow()
        self._window.setWindowTitle("Experiment Manager")
        self._window.setGeometry(100, 100, 1200, 800)
        self._window.closeEvent = self._on_window_close
        self._layout = QVBoxLayout()
        self._central_widget = QWidget()
        self._central_widget.setLayout(self._layout)
        self._window.setCentralWidget(self._central_widget)

    def _initialize_widgets(self):
        self._gaze_visualizer = GazeDataCanvas(
            parent=self._window,
            screen_width=1920,
            screen_height=1200,
            plotting_window=200,
        )
        # Experiment control and status widgets
        self._experiment_management_widget = QWidget()
        self._experiment_management_layout = QGridLayout()
        self._experiment_management_widget.setLayout(self._experiment_management_layout)
        self._layout.addWidget(self._experiment_management_widget, 0)
        self._layout.addWidget(self._gaze_visualizer, 1)

        # Title
        self._title_label = QLabel(
            "Experiment Management", alignment=Qt.AlignmentFlag.AlignCenter
        )
        self._title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self._experiment_management_layout.addWidget(self._title_label, 0, 0, 1, 3)
        # Panel for experiment info
        self._experiment_info_panel = QWidget()
        self._experiment_info_layout = QHBoxLayout()
        self._experiment_info_panel.setLayout(self._experiment_info_layout)
        self._experiment_management_layout.addWidget(self._experiment_info_panel, 1, 0)
        # Experiment name label
        self._experiment_info_layout.addWidget(
            QLabel("**Experiment:**", textFormat=Qt.TextFormat.MarkdownText),
        )
        self._experiment_name_value_label = QLabel(
            f"{self.experiment_manager.experiment_config.get('name', 'unknown')}"
        )
        self._experiment_info_layout.addWidget(self._experiment_name_value_label)
        # Vertical separator
        separator = QLabel("|")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("font-size: 18px; color: gray;")
        self._experiment_info_layout.addWidget(separator)
        # UID label
        self._experiment_info_layout.addWidget(
            QLabel("**PUID:**", textFormat=Qt.TextFormat.MarkdownText)
        )
        self._uid_value_label = QLabel(
            f"{
                self.experiment_manager.experiment_config['participant'].get(
                    'uid', '????'
                )
            }"
        )
        self._experiment_info_layout.addWidget(self._uid_value_label)
        # Vertical separator
        separator = QLabel("|")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("font-size: 18px; color: gray;")
        self._experiment_info_layout.addWidget(separator)
        # Task name label
        self._task_number_value_label = QLabel("Task #0")
        self._experiment_info_layout.addWidget(self._task_number_value_label)
        # Arrow separator
        separator = QLabel("→")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("font-size: 18px; color: gray;")
        self._experiment_info_layout.addWidget(separator)
        # Trial number label
        self._trial_number_value_label = QLabel("Trial #0")
        self._experiment_info_layout.addWidget(self._trial_number_value_label)

        # NBack info panel
        nback_info_layout = QHBoxLayout()
        self._experiment_management_layout.addLayout(nback_info_layout, 2, 0)
        # Nback sequence label
        nback_info_layout.addWidget(
            QLabel("**N-back order:**", textFormat=Qt.TextFormat.MarkdownText)
        )
        self._nback_levels_value_label = QLabel("N/A")
        nback_info_layout.addWidget(self._nback_levels_value_label)
        # Vertical separator
        separator = QLabel("|")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("font-size: 18px; color: gray;")
        nback_info_layout.addWidget(separator)
        # Current N-back label
        nback_info_layout.addWidget(
            QLabel("**Current Level:**", textFormat=Qt.TextFormat.MarkdownText)
        )
        self._current_nback_level_value_label = QLabel("N/A")
        nback_info_layout.addWidget(self._current_nback_level_value_label)
        # Vertical separator
        separator = QLabel("|")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("font-size: 18px; color: gray;")
        nback_info_layout.addWidget(separator)
        # NBack last two stimulus
        nback_info_layout.addWidget(
            QLabel("**Sequence:**", textFormat=Qt.TextFormat.MarkdownText)
        )
        self._nback_sequence_value_label = QLabel("[N/A]")
        nback_info_layout.addWidget(self._nback_sequence_value_label)

        # Create state boxes with arrows
        state_container = QWidget()
        state_layout = QGridLayout()
        state_container.setLayout(state_layout)
        state_label_stylesheet = (
            "border: 2px solid black; padding: 20px;font-size: 16px;"
        )

        # Previous state box
        self._previous_state_label = QLabel(
            "Previous", alignment=Qt.AlignmentFlag.AlignCenter
        )
        self._previous_state_label.setStyleSheet(state_label_stylesheet)
        state_layout.addWidget(self._previous_state_label, 0, 0)

        # Arrow
        label_arrow = QLabel("→")
        label_arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_arrow.setStyleSheet("font-weight: bold; font-size: 32px;")
        state_layout.addWidget(label_arrow, 0, 1)

        # Current state box
        self._current_state_label = QLabel(
            "Current", alignment=Qt.AlignmentFlag.AlignCenter
        )
        self._current_state_label.setStyleSheet(state_label_stylesheet)
        state_layout.addWidget(self._current_state_label, 0, 2)

        # Arrow
        label_arrow = QLabel("→")
        label_arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_arrow.setStyleSheet("font-weight: bold; font-size: 32px;")
        state_layout.addWidget(label_arrow, 0, 3)

        # Next state box
        self._next_state_label = QLabel("Next", alignment=Qt.AlignmentFlag.AlignCenter)
        self._next_state_label.setStyleSheet(state_label_stylesheet)
        state_layout.addWidget(self._next_state_label, 0, 4)

        self._experiment_management_layout.addWidget(state_container, 1, 1)

        # Experiment buttons
        buttons_panel = QWidget()
        buttons_layout = QGridLayout()
        buttons_panel.setLayout(buttons_layout)
        self._experiment_management_layout.addWidget(buttons_panel, 1, 2)

        self._next_state_btn = QPushButton("Next State")
        self._next_state_btn.setMinimumHeight(60)
        self._next_state_btn.clicked.connect(self.experiment_manager.request_next_state)
        self._next_state_btn.setEnabled(False)
        buttons_layout.addWidget(self._next_state_btn, 0, 0)

    def start(self):
        self._flash_visible = True
        self._experiment_status_update_timer = QTimer()
        self._experiment_status_update_timer.timeout.connect(
            self._update_experiment_status
        )
        self._experiment_status_update_timer.start(500)

    def _update_experiment_status(self):
        self._toggle_current_state_border()
        if self.experiment_manager.api_on_error:
            # Schedule next update attempt
            self._is_status_error = True
            self._next_state_btn.setEnabled(False)
            return

        status = self.experiment_manager.experiment_status
        if status is None:
            self._is_status_error = True
            self._next_state_btn.setEnabled(False)
            return
        self._is_status_error = False
        self._next_state_btn.setEnabled(True)
        # Update state labels
        self._previous_state_label.setText(status.previous_state.name or "None")
        self._current_state_label.setText(status.current_state.name or "None")
        self._next_state_label.setText(status.next_state.name or "None")

        # Update task and trial numbers
        self._task_number_value_label.setText(f"#{status.current_task}")
        self._trial_number_value_label.setText(f"#{status.current_trial}")
        # Update N-back sequence and level
        self._nback_levels_value_label.setText(
            " -> ".join(map(str, status.nback_levels_order))
            if status.nback_levels_order
            else "N/A"
        )
        self._current_nback_level_value_label.setText(
            f"{status.current_nback_level}"
            if status.current_nback_level >= 0
            else "N/A"
        )
        nback_data = self.experiment_manager.nback_latest_datas
        if nback_data is not None:
            stimuli = [str(data.stimulus) for data in nback_data]
            self._nback_sequence_value_label.setText(" -> ".join(map(str, stimuli)))

    def _toggle_current_state_border(self):
        if self._flash_visible:
            if self._is_status_error:
                self._title_label.setStyleSheet(
                    "font-size: 24px; font-weight: bold; background-color: red;"
                )
                self._current_state_label.setStyleSheet(
                    "border: 2px solid red; padding: 20px;font-size: 16px;"
                )
            else:
                self._current_state_label.setStyleSheet(
                    "border: 2px solid green; padding: 20px;font-size: 16px;"
                )
        else:
            self._current_state_label.setStyleSheet(
                "border: 2px solid black; padding: 20px;font-size: 16px;"
            )
            self._title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self._flash_visible = not self._flash_visible

    def attach_listeners(self):
        if self.experiment_manager.gaze_receiver is not None:
            self.experiment_manager.gaze_receiver.register_listener(
                self._gaze_visualizer.datas_callback
            )
        else:
            logger.warning(
                "Gaze receiver is not initialized. "
                "Cannot attach gaze visualizer listener."
            )

    def show(self):
        self._window.show()

    def close(self):
        # Programmatic close: trigger the window close which will call the
        # installed close-event handler (`_on_window_close`).
        self._window.close()

    def _on_window_close(self, event) -> None:
        """Handle QMainWindow close events and perform cleanup."""
        try:
            # Stop timers if running
            if self._experiment_status_update_timer:
                self._experiment_status_update_timer.stop()
            # Stop receivers and other cleanup
            try:
                self.experiment_manager.close()
            except Exception:
                logger.exception("Error while stopping receivers during close")
        finally:
            # Accept the close event so the window actually closes
            event.accept()
