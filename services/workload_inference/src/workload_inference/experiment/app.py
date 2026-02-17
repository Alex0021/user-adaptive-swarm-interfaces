from typing import Any
import time
import logging

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

from workload_inference.data.data_structures import NBackData
from workload_inference.visualizer.gaze import GazeDataCanvas
from workload_inference.experiment.manager import ExperimentManager

logger = logging.getLogger("ExperimentWindow")

class ExperimentManagerWindow(QMainWindow):
    """
    PyQt application to manage the experiment and visualize realtime gaze data.
    """

    def __init__(self, experiment_manager: ExperimentManager):
        super().__init__()
        self.experiment_manager = experiment_manager
        self._is_status_error = True

        self._initialize_core_compoonents()
        self._initialize_widgets()

    def _initialize_core_compoonents(self):
        self.setWindowTitle("Experiment Manager")
        self.setGeometry(100, 100, 1200, 800)
        self._layout = QVBoxLayout()
        self._central_widget = QWidget()
        self._central_widget.setLayout(self._layout)
        self.setCentralWidget(self._central_widget)

    def _initialize_widgets(self):
        self._gaze_visualizer = GazeDataCanvas(
            parent=self,
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
        self._title_label = QLabel("Experiment Management")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self._experiment_management_layout.addWidget(self._title_label, 0, 0, 1, 3)
        # Panel for experiment info
        self._experiment_info_panel = QWidget()
        self._experiment_info_layout = QHBoxLayout()
        self._experiment_info_panel.setLayout(self._experiment_info_layout)
        self._experiment_management_layout.addWidget(self._experiment_info_panel, 1, 0)
        # Experiment name label
        exp_label = QLabel("Experiment:")
        exp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        exp_label.setStyleSheet("font-weight: bold;")
        self._experiment_info_layout.addWidget(exp_label)
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
        uuid_label = QLabel("Participant UID:")
        uuid_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        uuid_label.setStyleSheet("font-weight: bold;")
        self._experiment_info_layout.addWidget(uuid_label)
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
        self._experiment_management_layout.addLayout(nback_info_layout, 2, 0, 1, 2)
        # Nback sequence label
        nback_order_label = QLabel("N-back order:")
        nback_order_label.setStyleSheet("font-weight: bold;")
        nback_info_layout.addWidget(nback_order_label)
        self._nback_levels_value_label = QLabel("N/A")
        nback_info_layout.addWidget(self._nback_levels_value_label)
        # Vertical separator
        separator = QLabel("|")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("font-size: 18px; color: gray;")
        nback_info_layout.addWidget(separator)
        # Current N-back label
        current_nback_label = QLabel("Current N-back:")
        current_nback_label.setStyleSheet("font-weight: bold;")
        nback_info_layout.addWidget(current_nback_label)
        self._current_nback_level_value_label = QLabel("N/A")
        nback_info_layout.addWidget(self._current_nback_level_value_label)
        # Vertical separator
        separator = QLabel("|")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("font-size: 18px; color: gray;")
        nback_info_layout.addWidget(separator)
        # NBack sequence label
        nback_sequece_label = QLabel("Sequence:")
        nback_sequece_label.setStyleSheet("font-weight: bold;")
        nback_info_layout.addWidget(nback_sequece_label)
        self._nback_sequence_value_label = QLabel("[N/A]")
        self._nback_sequence_value_label.setTextFormat(Qt.TextFormat.MarkdownText)
        nback_info_layout.addWidget(self._nback_sequence_value_label, 1)
        # NBack score
        separator = QLabel("|")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("font-size: 18px; color: gray;")
        nback_info_layout.addWidget(separator)
        self.nback_score_label = QLabel("Score: N/A")
        self.nback_score_label.setStyleSheet("font-weight: bold;")
        nback_info_layout.addWidget(self.nback_score_label)

        # Create state boxes with arrows
        state_container = QWidget()
        state_layout = QGridLayout()
        state_container.setLayout(state_layout)
        state_label_stylesheet = (
            "border: 2px solid black; padding: 20px;font-size: 16px;"
        )

        # Previous state box
        self._previous_state_label = QLabel("Previous")
        self._previous_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._previous_state_label.setStyleSheet(state_label_stylesheet)
        state_layout.addWidget(self._previous_state_label, 0, 0)

        # Arrow
        label_arrow = QLabel("→")
        label_arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_arrow.setStyleSheet("font-weight: bold; font-size: 32px;")
        state_layout.addWidget(label_arrow, 0, 1)

        # Current state box
        self._current_state_label = QLabel("Current")
        self._current_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._current_state_label.setStyleSheet(state_label_stylesheet)
        state_layout.addWidget(self._current_state_label, 0, 2)

        # Arrow
        label_arrow = QLabel("→")
        label_arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_arrow.setStyleSheet("font-weight: bold; font-size: 32px;")
        state_layout.addWidget(label_arrow, 0, 3)

        # Next state box
        self._next_state_label = QLabel("Next")
        self._next_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._next_state_label.setStyleSheet(state_label_stylesheet)
        state_layout.addWidget(self._next_state_label, 0, 4)

        self._experiment_management_layout.addWidget(state_container, 1, 1)

        # Ellapsed Time panel
        timer_panel = QWidget()
        timer_layout = QHBoxLayout()
        timer_panel.setLayout(timer_layout)
        self._ellapsed_time_label = QLabel("00:00")
        self._ellapsed_time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._ellapsed_time_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        timer_layout.addWidget(self._ellapsed_time_label, 1)
        self.start_ellapsed_time_button = QPushButton("Start timer")
        self.start_ellapsed_time_button.setMinimumHeight(30)
        self.start_ellapsed_time_button.clicked.connect(self._start_experiment_timer)
        timer_layout.addWidget(self.start_ellapsed_time_button, 0)
        self._experiment_management_layout.addWidget(timer_panel, 2, 2)
        self._ellapsed_timer = QTimer()
        self._ellapsed_timer.timeout.connect(self._update_ellapsed_time)

        # Experiment buttons
        buttons_panel = QWidget()
        buttons_layout = QHBoxLayout()
        buttons_panel.setLayout(buttons_layout)
        self._experiment_management_layout.addWidget(buttons_panel, 1, 2)

        self._next_state_btn = QPushButton("Next State")
        self._next_state_btn.setMinimumHeight(60)
        self._next_state_btn.clicked.connect(self.experiment_manager.request_next_state)
        self._next_state_btn.setEnabled(False)
        buttons_layout.addWidget(self._next_state_btn, 1)

    def start(self):
        self._flash_visible = True
        self._experiment_status_update_timer = QTimer()
        self._experiment_status_update_timer.timeout.connect(
            self._update_experiment_status
        )
        self._experiment_status_update_timer.start(500)

    # ================
    # Timer callbacks
    # ================

    def _update_ellapsed_time(self):
        if self.experiment_manager._duration is not None:
            self._ellapsed_timer.stop()
        if self.experiment_manager._start_time is None:
            self._ellapsed_time_label.setText("00:00")
            return
        ellapsed_seconds = int(time.time() - self.experiment_manager._start_time)
        minutes = ellapsed_seconds // 60
        seconds = ellapsed_seconds % 60
        self._ellapsed_time_label.setText(f"{minutes:02d}:{seconds:02d}")

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
            stimuli = list(self._generate_nback_stimulus_click_expected(nback_data))
            self._nback_sequence_value_label.setText(" -> ".join(stimuli))
            score = sum(1 for data in nback_data if data.is_correct)
            num_stimuli = sum(1 for data in nback_data if data.timestamp > 0)
            self.nback_score_label.setText(f"Score: {score}/{num_stimuli}")

    def _generate_nback_stimulus_click_expected(self, sequence: list[NBackData]):
        """Yields the stimulus in the sequence, with expected clicks marked with **."""
        nback_level = sequence[0].nback_level
        for idx, data in enumerate(sequence):
            if idx < nback_level:
                yield str(data.stimulus)
                continue
            expected_click = data.stimulus == sequence[idx - nback_level].stimulus
            if expected_click:
                yield f"<span style='color: red;'><b>{data.stimulus}</b></span>"
            else:
                yield str(data.stimulus)

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

    # ==================
    # BUTTONS callbacks
    # ==================

    def _start_experiment_timer(self):
        if self.experiment_manager._start_time is None:
            self.experiment_manager._start_time = time.time()
            self._ellapsed_timer.start(1000)
        self.start_ellapsed_time_button.setEnabled(False)

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

    def closeEvent(self, event: Any) -> None:
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