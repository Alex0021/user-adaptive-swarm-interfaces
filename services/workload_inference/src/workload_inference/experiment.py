import logging
import threading
import time
from queue import Queue
from typing import TextIO

import yaml
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QGridLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from workload_inference.api import ExperimentAPI, ExperimentAPIError
from workload_inference.constants import DATA_DIR
from workload_inference.data_structures import ExperimentStatus, GazeData
from workload_inference.visualize import GazeDataCanvas

CONFIG_FILE_NAME = "experiment.yaml"
SAMPLE_CONFIG_FILE_NAME = "sample_experiment.yaml"
DATA_FILE_NAME = "gaze_data.csv"
CSV_HEADER = GazeData.__annotations__.keys()
EXPERIMENT_STATUS_UPDATE_RATE_MS = 500

logger = logging.getLogger("ExperimentManager")


class ExperimentManager:
    """
    Manage the current experiment and store incoming data into csv files.
    """

    def __init__(self, base_folder: str = "experiments", queue_size: int = 1000):
        self.base_folder = DATA_DIR / base_folder
        self.data_queue: Queue[GazeData] = Queue(maxsize=queue_size)
        # Try to read experiment data from yaml file
        if not (self.base_folder / CONFIG_FILE_NAME).exists():
            logger.warning(
                f"Experiment configuration file '{CONFIG_FILE_NAME}' not found in '{self.base_folder}'."
                f" Using sample configuration '{SAMPLE_CONFIG_FILE_NAME}'."
            )
            if not (self.base_folder / SAMPLE_CONFIG_FILE_NAME).exists():
                raise FileNotFoundError(
                    f"Sample experiment configuration file '{SAMPLE_CONFIG_FILE_NAME}' not found"
                    f" in '{self.base_folder}'. Please create an experiment configuration file."
                )
            with open(self.base_folder / SAMPLE_CONFIG_FILE_NAME, "r") as f:
                self.experiment_config = yaml.safe_load(f)
        else:
            with open(self.base_folder / CONFIG_FILE_NAME, "r") as f:
                self.experiment_config = yaml.safe_load(f)
        self._writer_thread: threading.Thread | None = None
        self._running: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._current_exp_filestream: TextIO | None = None
        self._block_size: int = 100
        self._data_cnt: int = 0
        self._api = ExperimentAPI()
        logger.info("Initialized with queue size %d.", queue_size)

    def datas_callback(self, datas: list[GazeData]) -> None:
        """
        Callback function to receive new data points and store them in the queue.

        Args:
            datas (list[GazeData]): The new data points to add.
        Raises:
            OverflowError: If the queue is full and cannot accept new data points.
        """
        if self.data_queue.qsize() + len(datas) > self.data_queue.maxsize:
            raise OverflowError("Data queue is full. Cannot add new data points.")
        for data in datas:
            self.data_queue.put(data)

    def start_recording(self) -> None:
        """
        Start the data recording thread.
        """
        # Create experiment data file
        if not self._running:
            self._initialize_structure()
            self._running = True
            self._data_cnt = 0
            self._writer_thread = threading.Thread(target=self._data_writer)
            self._writer_thread.start()
            logger.info(
                "Data recording started. Data blocks set to %d.", self._block_size
            )

    def stop_recording(self) -> None:
        """
        Stop the data recording thread.
        """
        if self._running:
            self._running = False
            if self._writer_thread is not None:
                self._writer_thread.join()
                self._writer_thread = None
            if self._current_exp_filestream is not None:
                self._current_exp_filestream.close()
                self._current_exp_filestream = None
            logger.info("Data recording stopped. %d lines recorded.", self._data_cnt)

    def _data_writer(self) -> None:
        """
        Thread function to write data from the queue to the experiment data file.
        """
        if self._current_exp_filestream is None:
            raise RuntimeError("Experiment file stream is not initialized.")

        while self._running:
            if self.data_queue.qsize() >= self._block_size:
                # Write block to file
                for _ in range(self._block_size):
                    gaze_data = self.data_queue.get()
                    line = (
                        ",".join(str(gaze_data.__dict__[field]) for field in CSV_HEADER)
                        + "\n"
                    )
                    self._current_exp_filestream.write(line)
                self._current_exp_filestream.flush()
                self._data_cnt += self._block_size
            time.sleep(0.01)

        # Write remaining data in the queue
        if self.data_queue.qsize() > 0:
            self._data_cnt += self.data_queue.qsize()
            for _ in range(self.data_queue.qsize()):
                gaze_data = self.data_queue.get()
                line = (
                    ",".join(str(getattr(gaze_data, field)) for field in CSV_HEADER)
                    + "\n"
                )
                self._current_exp_filestream.write(line)
            self._current_exp_filestream.flush()

    def _initialize_structure(self, overwrite: bool = True) -> None:
        """
        Initialize the experiment data file structure.

        Args:
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to True.
        Raises:
            FileExistsError: If the experiment data file already exists and overwrite is False.
        """
        if "name" not in self.experiment_config:
            logger.warning(
                "Experiment name not found in configuration. Using 'anonymous'."
            )
        if "name" not in self.experiment_config["participant"]:
            logger.warning(
                "Participant name not found in configuration. Using 'person'."
            )
        if "name" not in self.experiment_config["task"]:
            logger.warning(
                "Task name not found in configuration. Using 'unknown_task'."
            )
        exp_name = self.experiment_config.get("name", "anonymous")
        participant_name = self.experiment_config["participant"].get("name", "person")
        task_name = self.experiment_config["task"].get("name", "unknown_task")
        # Make sure folders are snake_case (especially for linux compatibility)
        formatted_str_path = (
            "/".join([exp_name, participant_name, task_name]).replace(" ", "_").lower()
        )
        exp_folder = self.base_folder / formatted_str_path
        exp_folder.mkdir(parents=True, exist_ok=True)
        if (exp_folder / DATA_FILE_NAME).exists() and not overwrite:
            raise FileExistsError(
                f"Experiment data file '{DATA_FILE_NAME}' already exists in '{exp_folder}'."
                " To bypass, set the 'overwrite' parameter to True."
            )
        self._current_exp_filestream = open(
            exp_folder / DATA_FILE_NAME, "w", encoding="utf-8"
        )
        # Write CSV header
        header = ",".join(CSV_HEADER) + "\n"
        self._current_exp_filestream.write(header)
        self._current_exp_filestream.flush()
        logger.info(f"Data structure ready at '{exp_folder}'.")

    def get_experiment_status(self) -> tuple[bool, ExperimentStatus | None]:
        """Fetch the current experiment status from the API."""
        try:
            status = self._api.get_experiment_state()
            return True, status
        except ExperimentAPIError as e:
            logger.error("Failed to fetch experiment status: %s", e)
            return False, None

    def request_next_state(self) -> None:
        """Request the API to move to the next state in the experiment."""
        try:
            self._api.trigger_next_state()
        except ExperimentAPIError as e:
            logger.error("Failed to trigger next state: %s", e)


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
        self._experiment_info_layout = QGridLayout()
        self._experiment_info_panel.setLayout(self._experiment_info_layout)
        self._experiment_management_layout.addWidget(self._experiment_info_panel, 1, 0)
        # Experiment name label
        self._experiment_info_layout.addWidget(
            QLabel("**Experiment:**", textFormat=Qt.TextFormat.MarkdownText),
            0,
            0,
        )
        self._experiment_name_value_label = QLabel(
            f"{self.experiment_manager.experiment_config.get('name', 'unknown')}"
        )
        self._experiment_info_layout.addWidget(self._experiment_name_value_label, 0, 1)
        # UID label
        self._experiment_info_layout.addWidget(
            QLabel("**Participant UID:**", textFormat=Qt.TextFormat.MarkdownText), 1, 0
        )
        self._uid_value_label = QLabel(
            f"{self.experiment_manager.experiment_config['participant'].get('uid', '????')}"
        )
        self._experiment_info_layout.addWidget(self._uid_value_label, 1, 1)
        # Task name label
        self._experiment_info_layout.addWidget(
            QLabel("**Task:**", textFormat=Qt.TextFormat.MarkdownText), 2, 0
        )
        self._task_number_value_label = QLabel("#0")
        self._experiment_info_layout.addWidget(self._task_number_value_label, 2, 1)
        # Trial number label
        self._experiment_info_layout.addWidget(
            QLabel("**Trial:**", textFormat=Qt.TextFormat.MarkdownText), 3, 0
        )
        self._trial_number_value_label = QLabel("#0")
        self._experiment_info_layout.addWidget(self._trial_number_value_label, 3, 1)
        # Nback sequence label
        self._experiment_info_layout.addWidget(
            QLabel("**N-back Sequence:**", textFormat=Qt.TextFormat.MarkdownText), 4, 0
        )
        self._nback_sequence_value_label = QLabel("N/A")
        self._experiment_info_layout.addWidget(self._nback_sequence_value_label, 4, 1)
        # Current N-back label
        self._experiment_info_layout.addWidget(
            QLabel("**Current N-back Level:**", textFormat=Qt.TextFormat.MarkdownText),
            5,
            0,
        )
        self._current_nback_level_value_label = QLabel("N/A")
        self._experiment_info_layout.addWidget(
            self._current_nback_level_value_label, 5, 1
        )

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

        # Setup flashing timer for current state
        self._flash_timer = QTimer()
        self._flash_timer.timeout.connect(self._toggle_current_state_border)
        self._flash_visible = True

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
        self._flash_timer.start(500)
        self._experiment_status_update_timer = QTimer()
        self._experiment_status_update_timer.timeout.connect(
            self._update_experiment_status
        )
        self._experiment_status_update_timer.setSingleShot(True)
        self._experiment_status_update_timer.setInterval(
            EXPERIMENT_STATUS_UPDATE_RATE_MS
        )
        self._experiment_status_update_timer.start()

    def _update_experiment_status(self):
        success, status = self.experiment_manager.get_experiment_status()
        if not success:
            # Schedule next update attempt
            self._is_status_error = True
            self._next_state_btn.setEnabled(False)
            self._experiment_status_update_timer.start(2000)
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
        self._nback_sequence_value_label.setText(
            " -> ".join(map(str, status.nback_levels_order))
            if status.nback_levels_order
            else "N/A"
        )
        self._current_nback_level_value_label.setText(
            f"#{status.current_nback_level}"
            if status.current_nback_level >= 0
            else "N/A"
        )

        # Schedule next update
        self._experiment_status_update_timer.start(EXPERIMENT_STATUS_UPDATE_RATE_MS)

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

    def show(self):
        self._window.show()
