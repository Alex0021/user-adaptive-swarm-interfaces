import logging
import os

os.environ["QT_API"] = "PyQt6"  # Ensure PyQt6 is used for matplotlib backend
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider
from numpy.typing import NDArray
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from workload_inference.utils.constants import DATA_DIR
from workload_inference.data.data_structures import DroneData, GazeData, Listener
from workload_inference.visualizer.gaze import GazeDataCanvas

class ReplaySlider:
    """Placeholder for a replay slider widget to scrub through recorded data"""

    def __init__(
        self,
        parent: QMainWindow | None = None,
        min_value: int = 0,
        max_value: int = 100,
        initial_value: int = 0,
        step: int = 1,
        on_change: Callable | None = None,
    ):
        self.figure = Figure(figsize=(4, 1), dpi=100, tight_layout=True)
        self.widget = Slider(
            ax=self.figure.add_subplot(2, 4, (1, 4)),
            label="",
            valmin=min_value,
            valmax=max_value,
            valinit=initial_value,
            valstep=step,
        )
        self.play_btn = Button(ax=self.figure.add_subplot(246), label="Play")
        self.pause_btn = Button(ax=self.figure.add_subplot(247), label="Pause")
        self.step_back_btn = Button(ax=self.figure.add_subplot(245), label="<<")
        self.step_forward_btn = Button(ax=self.figure.add_subplot(248), label=">>")
        self.widget.on_changed(on_change if on_change else lambda val: None)
        self.canvas = FigureCanvas(self.figure)
        self.parent = parent
        self.playing = False


class ReplayData:
    """
    Load the data from the specified folder.
    Allows streaming of data to simulate real-time playback.
    Will stream using the usual callback mechanism to the visualizer.
    """

    def __init__(
        self,
        trial_folder: Path,
        gaze_callback: Listener[GazeData] | None = None,
        drone_callback: Listener[DroneData] | None = None,
        playback_window: int = 200,
        sampling_rate: float = 30.0,
    ):
        self.data_folder = trial_folder
        self._logger = logging.getLogger("ReplayData")
        self.drones_data: list[pd.DataFrame] = []
        self.gaze_data: pd.DataFrame = pd.DataFrame()
        self.drone_data: pd.DataFrame = pd.DataFrame()
        try:
            self.gaze_data = pd.read_csv(trial_folder / "gaze_data.csv")
        except FileNotFoundError:
            self._logger.error("Gaze data file not found in %s", trial_folder)
        try:
            self.drone_data = pd.read_csv(trial_folder / "drone_data.csv")
        except FileNotFoundError:
            self._logger.error("Drone data file not found in %s", trial_folder)

        self._playing = False
        self._running = True
        self.replay_thread: threading.Thread | None = threading.Thread(
            target=self._replay_loop, daemon=True
        )

        self._idx = 0
        self._playback_window = playback_window
        self._sampling_rate = sampling_rate
        self.gaze_callback = gaze_callback
        self.drone_callback = drone_callback
        self.timestamps = self._initialize_timestamps()
        self.slider = ReplaySlider(
            min_value=0,
            max_value=len(self.timestamps) - 1,
            initial_value=0,
            step=1,
            on_change=self.update_data_from_index,
        )
        self._initialize_dfs()
        self.replay_thread.start()

    @property
    def is_playing(self) -> bool:
        return self._playing

    @is_playing.setter
    def is_playing(self, value: bool):
        self._playing = value

    @property
    def index(self) -> int:
        return self._idx

    def _initialize_timestamps(self) -> NDArray[np.float64]:
        """
        Align gaze and drone data by closest timestamps at choosen frequency
        and store in a dict for easy streaming to visualizer
        """
        start_timestamp = max(
            self.gaze_data["timestamp"].min(), self.drone_data["timestamp"].min()
        )
        end_timestamp = min(
            self.gaze_data["timestamp"].max(), self.drone_data["timestamp"].max()
        )
        timestamps = np.arange(
            start_timestamp, end_timestamp, 1000 / self._sampling_rate, dtype=np.float64
        )
        return timestamps

    def _initialize_dfs(self) -> None:
        """Preprocess the dataframes to align with the common timestamps"""
        self.gaze_data = self._resample_df(self.gaze_data, "timestamp")
        # Find the number of drones from the id column
        num_drones = self.drone_data["id"].nunique()
        # Create a separate dataframe for each drone and resample
        for drone_id in range(num_drones):
            drone_df = self.drone_data[self.drone_data["id"] == drone_id]
            resampled_drone_df = self._resample_df(drone_df, "timestamp")
            self.drones_data.append(resampled_drone_df)

    def _resample_df(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """
        Resample a dataframe to the common timestamps using nearest neighbor

        Args:
            df: The dataframe to resample
            timestamp_col: The name of the timestamp column in the dataframe

        Returns:
            pd.DataFrame: The resampled dataframe aligned to the common timestamps
        """
        if df.empty:
            return df
        if not df[timestamp_col].is_unique:
            df.drop_duplicates(timestamp_col, inplace=True)
        df = df.set_index(timestamp_col)
        resampled = df.reindex(
            self.timestamps, method="nearest", tolerance=2 * 1000 / self._sampling_rate
        )
        resampled = resampled.reset_index()
        return resampled

    def _replay_loop(self):
        """Loop to stream data to visualizer at the specified frequency"""
        while self._running:
            if self._playing:
                self.update_data_from_index(self._idx)
                self._idx += 1
                # call the slider update every 15 samples to avoid excessive updates
                if self._idx % 15 == 0:
                    self.slider.widget.set_val(self._idx)
                if self._idx >= len(self.timestamps):
                    self._idx = 0  # Loop back to start
            time.sleep(1 / self._sampling_rate)

    def update_data_from_index(self, idx: int):
        """Update the data for the current index and call the callbacks

        Args:
            idx: The index of the timestamp to update data for
        """
        if idx < 0 or idx >= len(self.timestamps):
            self._logger.warning("Index %d out of range for timestamps", idx)
            return

        if self._playing:
            gazes, drones = self._get_single_idx_data(idx)
        else:
            gazes, drones = self._get_window_idx_data(idx)

        if self.gaze_callback:
            self.gaze_callback(gazes, batch_update=not self._playing)
        if self.drone_callback:
            self.drone_callback(drones, batch_update=not self._playing)

    def _get_single_idx_data(self, idx: int) -> tuple[list[GazeData], list[DroneData]]:
        """Update the data for a single index and call the callbacks

        Args:
            idx: The index of the timestamp to update data for
        """
        assert idx >= 0 and idx < len(self.timestamps), "Index out of range"

        gaze = GazeData(**self.gaze_data.iloc[idx])
        drones = [
            DroneData(**self.drones_data[i].iloc[idx])
            for i in range(len(self.drones_data))
        ]
        return [gaze], drones

    def _get_window_idx_data(self, idx: int) -> tuple[list[GazeData], list[DroneData]]:
        """
        Get the datapoints in the range [idx-N, idx] (window size N).
        Will automatically manage edge cases for the start of the data.

        Args:
            idx: The index of the timestamp to update data for
        """
        assert idx >= 0 and idx < len(self.timestamps), "Index out of range"
        # Also consider the window size to get a batch of the last N samples
        if idx < self._playback_window:
            gazes = [
                GazeData(**row) for _, row in self.gaze_data.iloc[: idx + 1].iterrows()
            ]
        else:
            gazes = [
                GazeData(**row)
                for _, row in self.gaze_data.iloc[
                    idx - self._playback_window + 1 : idx + 1
                ].iterrows()
            ]
        drones = [
            DroneData(**self.drones_data[i].iloc[idx])
            for i in range(len(self.drones_data))
            if idx < len(self.drones_data[i])
        ]

        return gazes, drones

    def get_range(self) -> tuple[float, float]:
        """Get the timestamp range of the data for slider limits"""
        return self.timestamps[0], self.timestamps[-1]

    def step_idx(self, step: int):
        """Step the index by a given amount and update data

        Args:
            step: The number of indices to step (positive or negative)
        """
        new_idx = self._idx + step
        if new_idx < 0:
            new_idx = 0
        elif new_idx >= len(self.timestamps):
            new_idx = len(self.timestamps) - 1
        self._idx = new_idx
        self.update_data_from_index(self._idx)

    def close(self):
        """Stop the replay loop and clean up resources"""
        self._running = False
        if self.replay_thread and self.replay_thread.is_alive():
            self.replay_thread.join(timeout=5)
            if self.replay_thread.is_alive():
                self._logger.warning("Replay thread did not finish in time")


class ExperimentDataReplayWindow(QMainWindow):
    """Placeholder for a window that would allow replaying recorded experiment
    data with the visualizer and slider"""

    def __init__(
        self, parent: QMainWindow | None = None, trial_folder: Path | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Experiment Data Replay")
        self.setGeometry(150, 150, 1000, 800)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.visualizer = GazeDataCanvas(parent=self)
        if trial_folder:
            self.replay_data = ReplayData(
                trial_folder=trial_folder, gaze_callback=self.visualizer.datas_callback
            )
            self.replay_slider = self.replay_data.slider
        else:
            self.replay_slider = ReplaySlider(parent=self)

        layout.addWidget(self.replay_slider.canvas)
        layout.addWidget(self.visualizer, 1)

        self.replay_slider.play_btn.on_clicked(self.play)
        self.replay_slider.pause_btn.on_clicked(self.pause)
        self.replay_slider.step_forward_btn.on_clicked(self.step_forward)
        self.replay_slider.step_back_btn.on_clicked(self.step_backward)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def play_pause(self, event):
        """Toggle play/pause state of the replay"""
        self.replay_data._playing = not self.replay_data._playing

    def play(self, event):
        """Start playing the replay"""
        self.replay_data._playing = True

    def step_forward(self, event):
        """Step forward by one index"""
        if self.replay_data.is_playing:
            self.replay_data.is_playing = False
        self.replay_data.step_idx(1)
        self.replay_slider.widget.set_val(self.replay_data.index)

    def step_backward(self, event):
        """Step backward by one index"""
        if self.replay_data.is_playing:
            self.replay_data.is_playing = False
        self.replay_data.step_idx(-1)
        self.replay_slider.widget.set_val(self.replay_data.index)

    def pause(self, event):
        """Pause the replay"""
        self.replay_data.is_playing = False
        # Make sure to update the slider to the current index when pausing
        self.replay_slider.widget.set_val(self.replay_data.index)

    def keyPressEvent(self, event):
        """Allow using spacebar to toggle play/pause"""
        if event.key() == Qt.Key.Key_Space:
            self.play_pause(None)

    def closeEvent(self, event):
        """Ensure replay thread is stopped when window is closed"""
        if hasattr(self, "replay_data"):
            self.replay_data.close()
        event.accept()


def main():
    import sys

    app = QApplication(sys.argv)

    replay_folder = DATA_DIR / "experiments/experiment_1/P001/task_1/trial_1"
    window = ExperimentDataReplayWindow(trial_folder=replay_folder)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()