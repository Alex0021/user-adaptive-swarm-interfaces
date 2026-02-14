import logging
import os

os.environ["QT_API"] = "PyQt6"  # Ensure PyQt6 is used for matplotlib backend
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import PathCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
from numpy.typing import NDArray
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from workload_inference.constants import DATA_DIR
from workload_inference.data_structures import DroneData, GazeData, Listener


class GazeDataCanvas(FigureCanvas):
    """Matplotlib canvas with 3 subplots for gaze visualization"""

    def __init__(
        self,
        parent: QMainWindow | None = None,
        screen_width: int = 1920,
        screen_height: int = 1200,
        max_history: int = 1000,
        plotting_window: int = 100,
        update_freq: int = 30,
    ):
        """
        Initialize the canvas and subplots for gaze visualization.
        Screen size in pixels is needed to scale gaze positions correctly.

        Args:
            parent (QMainWindow | None): Parent window for the canvas.
            screen_width (int): Width of the screen in pixels.
            screen_height (int): Height of the screen in pixels.
            max_history (int): Maximum number of data points to keep in history.
            plotting_window (int): Number of data points to display in the plots.
            update_freq (int): Frequency of plot updates in Hz.
        """
        self.fig = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        super().__init__(self.fig)
        self.parent = parent
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.window_size = plotting_window
        self.update_freq = update_freq
        self.data_cb_cnt = 0
        self._timer = QTimer(parent)
        self._timer.timeout.connect(self._update_all)

        # Initalize 3 plots
        ar = self.screen_height / self.screen_width
        self.ax_gaze = self.fig.add_subplot(3, 2, (1, 4), aspect=ar, adjustable="box")
        self.ax_validity = self.fig.add_subplot(3, 2, 5)
        self.ax_pupil = self.fig.add_subplot(3, 2, 6)

        # Data buffers
        self.gaze_hist: deque[tuple[float, float]] = deque(
            maxlen=plotting_window
        )  # (x, y) positions
        self.validity_hist: deque[tuple[int, int]] = deque(
            maxlen=plotting_window
        )  # (left_validity, right_validity)
        self.pupil_hist: deque[tuple[float, float]] = deque(
            maxlen=plotting_window
        )  # (left_diameter, right_diameter)

        # TEST data
        d = np.linspace(3.0, 4.0, plotting_window)
        for i in range(plotting_window):
            self.validity_hist.append(
                (i > plotting_window // 2, i < plotting_window // 2)
            )
            self.pupil_hist.append((d[i], np.random.rand() + 3.5))
        # Create a circle trace of gaze points counter clockwise starting from top
        for i in range(plotting_window):
            angle = 2 * np.pi * (i / plotting_window)
            x = (self.screen_width / 2) + (self.screen_width / 4) * np.sin(angle)
            y = (self.screen_height / 2) - (self.screen_height / 4) * np.cos(angle)
            self.gaze_hist.append((x, y))

        # Blit objects
        self.pupil_hist_lines: list[Line2D] = []
        self.validity_img: AxesImage | None = None
        self.gaze_scatter: PathCollection | None = None
        self.sizes = np.linspace(5, 50, plotting_window)
        self.colors = np.linspace(0.1, 1.0, plotting_window)

        # Blitting state
        self._background = None
        self._blit_ready = False

        self._init_plots()
        self.update_pupil_diameter()
        self.update_eye_validity()
        self.update_gaze_trace()

        # Blitting hooks
        self.mpl_connect("draw_event", self._on_draw)
        self.mpl_connect("resize_event", self._on_resize)
        self._init_blit()

        self._timer.start(1000 // self.update_freq)

    def _init_plots(self):
        """Initialize plot styling and labels"""
        # Gaze trace plot
        self.ax_gaze.set_title("Gaze Position Trace")
        self.ax_gaze.set_xlabel("X (pixels)")
        self.ax_gaze.set_ylabel("Y (pixels)")
        self.ax_gaze.set_xlim(0, self.screen_width)
        self.ax_gaze.set_ylim(0, self.screen_height)
        self.ax_gaze.invert_yaxis()  # Invert Y axis to match screen coordinates

        # Eye validity bar
        self.ax_validity.set_title("Eye Validity History")
        self.ax_validity.set_yticks([0, 1])
        self.ax_validity.set_yticklabels(["right", "left"])
        self.ax_validity.set_xlabel("Sample Index")
        self.ax_validity.set_xlim(-self.window_size, 0)

        # Pupil diameter plot
        self.ax_pupil.set_title("Pupil Diameter Trend")
        self.ax_pupil.set_xlabel("Sample Index")
        self.ax_pupil.set_ylabel("Diameter (mm)")
        self.ax_pupil.legend(["Left", "Right", "Mean"])
        self.ax_pupil.set_xlim(-self.window_size, 0)
        self.ax_pupil.set_ylim(2, 5)

    def _init_blit(self):
        """Initialize blitting by caching the background"""
        if self.pupil_hist_lines:
            for line in self.pupil_hist_lines:
                line.set_animated(True)
        if self.validity_img is not None:
            self.validity_img.set_animated(True)
        if self.gaze_scatter is not None:
            self.gaze_scatter.set_animated(True)

        self.draw()
        self._background = self.copy_from_bbox(self.fig.bbox)
        self._blit_ready = True

    def _on_draw(self, _event):
        """Cache background on draw events"""
        self._background = self.copy_from_bbox(self.fig.bbox)
        self._blit_ready = True

    def _on_resize(self, _event):
        """Reinitialize blitting on resize"""
        self._blit_ready = False
        self.draw()

    def _blit_update(self):
        """Fast redraw using blitting"""
        if not self._blit_ready or self._background is None:
            self.draw_idle()
            return

        self.restore_region(self._background)

        if self.pupil_hist_lines:
            for line in self.pupil_hist_lines:
                self.ax_pupil.draw_artist(line)

        if self.validity_img is not None:
            self.ax_validity.draw_artist(self.validity_img)

        if self.gaze_scatter is not None:
            self.ax_gaze.draw_artist(self.gaze_scatter)

        self.blit(self.fig.bbox)

    def _update_all(self):
        """Update all plots"""
        self.update_gaze_trace()
        self.update_eye_validity()
        self.update_pupil_diameter()
        self._blit_update()

    def update_pupil_diameter(self):
        """Update line plot for pupil diameter trends"""
        pupil_data = np.array(self.pupil_hist)
        if len(self.pupil_hist_lines) == 0:
            indices = np.arange(-len(pupil_data), 0)
            self.pupil_hist_lines = self.ax_pupil.plot(
                indices, pupil_data[:, 0], label="Left", color="blue"
            )
            self.pupil_hist_lines += self.ax_pupil.plot(
                indices, pupil_data[:, 1], label="Right", color="orange"
            )
            mean_diameter = np.mean(pupil_data, axis=1)
            self.pupil_hist_lines += self.ax_pupil.plot(
                indices, mean_diameter, label="Mean", linestyle="--", color="black"
            )
            self.ax_pupil.legend()
        else:
            # Update x data if needed (in case of window size change),
            # but only update y data for efficiency
            xdata = np.asarray(self.pupil_hist_lines[0].get_xdata())
            if pupil_data.shape[0] != xdata.shape[0]:
                indices = np.arange(-pupil_data.shape[0], 0)
                for line in self.pupil_hist_lines:
                    line.set_xdata(indices)

            for i, line in enumerate(self.pupil_hist_lines):
                if i < 2:
                    line.set_ydata([pupil_data[t][i] for t in range(len(pupil_data))])
                else:
                    mean_diameter = np.mean(pupil_data, axis=1)
                    line.set_ydata(mean_diameter)

    def update_eye_validity(self):
        """
        Update bar plot for eye validity history
        Using an image mapping to be efficient for plotting
        """
        validity_data = np.array(self.validity_hist).T  # Shape (2, N)
        # Pad to full window size with -1 (unknown) on the left if needed
        cur_len = validity_data.shape[1]
        if cur_len < self.window_size:
            pad = np.full((2, self.window_size - cur_len), -1, dtype=int)
            validity_padded = np.concatenate((pad, validity_data), axis=1)
        else:
            validity_padded = validity_data

        cmap = ListedColormap(["white", "red", "green"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

        if self.validity_img is None:
            self.validity_img = self.ax_validity.imshow(
                validity_padded,
                aspect="auto",
                cmap=cmap,
                norm=norm,
                extent=(-self.window_size, 0, -0.5, 1.5),
            )
        else:
            self.validity_img.set_data(validity_padded)

    def update_gaze_trace(self):
        """
        Update scatter plot for gaze position trace
        Only recalculate sizes/colors for NEW points
        """
        gaze_data = np.array(self.gaze_hist)

        if self.gaze_scatter is None:
            # Initial creation only
            self.gaze_scatter = self.ax_gaze.scatter(
                gaze_data[:, 0],
                gaze_data[:, 1],
                s=self.sizes,
                c=self.colors,
                cmap="Greys",
                alpha=0.7,
            )
        else:
            if len(gaze_data) < self.window_size:
                self.gaze_scatter.set_sizes(self.sizes[-len(gaze_data) :])
                self.gaze_scatter.set_array(self.colors[-len(gaze_data) :])
            self.gaze_scatter.set_offsets(gaze_data)

    def datas_callback(
        self, datas: Sequence[GazeData], batch_update: bool = False
    ) -> None:
        """Callback to only store gaze data (minimal processing)

        Args:
            datas: List of new gaze data points to add to the history
            batch_update: Wether to flush the history and only use the given datas
        """
        if batch_update:
            # empty the history to only use data from the batch
            self.gaze_hist.clear()
            self.validity_hist.clear()
            self.pupil_hist.clear()
        for gaze_data in datas:
            self.data_cb_cnt += 1
            x = gaze_data.left_point_screen_x * self.screen_width
            y = gaze_data.left_point_screen_y * self.screen_height
            left_validity = int(gaze_data.left_validity)
            right_validity = int(gaze_data.right_validity)
            left_diameter = float(gaze_data.left_pupil_diameter)
            right_diameter = float(gaze_data.right_pupil_diameter)

            self.gaze_hist.append((float(x), float(y)))
            self.validity_hist.append((left_validity, right_validity))
            self.pupil_hist.append((left_diameter, right_diameter))


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
