import logging
import os

os.environ["QT_API"] = "PyQt6"  # Ensure PyQt6 is used for matplotlib backend
import time
from collections import deque
from typing import List

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from workload_inference.constants import DATA_DIR
from workload_inference.data_structures import DroneData, GazeData


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
        Initialize the canvas and subplots for gaze visualization. Screen size in pixels is needed to scale gaze positions correctly.

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

        # Line objects
        self.pupil_hist_lines = None
        self.validity_img = None
        self.gaze_scatter = None

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
        if self.pupil_hist_lines is None:
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
        if self.validity_img is None:
            self.validity_img = self.ax_validity.imshow(
                validity_data,
                aspect="auto",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                extent=[-self.window_size, 0, -0.5, 1.5],
            )
        else:
            self.validity_img.set_data(validity_data)

    def update_gaze_trace(self):
        """
        Update scatter plot for gaze position trace
        Only recalculate sizes/colors for NEW points
        """
        gaze_data = np.array(self.gaze_hist)

        if self.gaze_scatter is None:
            # Initial creation only
            sizes = np.linspace(5, 50, len(gaze_data))
            colors = np.linspace(0.1, 1.0, len(gaze_data))
            self.gaze_scatter = self.ax_gaze.scatter(
                gaze_data[:, 0],
                gaze_data[:, 1],
                s=sizes,
                c=colors,
                cmap="Greys",
                alpha=0.7,
            )
        else:
            self.gaze_scatter.set_offsets(gaze_data)

    def datas_callback(self, gaze_datas: List[GazeData]):
        """Callback to only store gaze data (minimal processing)

        Args:
            gaze_datas (List[GazeData]): List of new gaze data points to add to the history
        """
        for gaze_data in gaze_datas:
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
        on_change: callable | None = None,
    ):
        self.figure = Figure(figsize=(4, 1), dpi=100, tight_layout=True)
        self.widget = Slider(
            ax=self.figure.add_subplot(2, 3, (1, 3)),
            label="",
            valmin=min_value,
            valmax=max_value,
            valinit=initial_value,
            valstep=step,
        )
        self.play_btn = Button(ax=self.figure.add_subplot(235), label="Play")
        self.step_back_btn = Button(ax=self.figure.add_subplot(234), label="<<")
        self.step_forward_btn = Button(ax=self.figure.add_subplot(236), label=">>")
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
        trial_folder: str,
        gaze_callback: callable | None = None,
        drone_callback: callable | None = None,
        playback_window: int = 200,
        sampling_rate: float = 30.0,
    ):
        self.data_folder = trial_folder
        self._logger = logging.getLogger("ReplayData")
        try:
            self.gaze_data = pd.read_csv(os.path.join(trial_folder, "gaze_data.csv"))
        except FileNotFoundError:
            self._logger.error("Gaze data file not found in %s", trial_folder)
            self.gaze_data = pd.DataFrame()
        try:
            self.drone_data = pd.read_csv(os.path.join(trial_folder, "drone_data.csv"))
        except FileNotFoundError:
            self._logger.error("Drone data file not found in %s", trial_folder)
            self.drone_data = pd.DataFrame()

        self._playing = False
        self._idx = 0
        self._playback_window = playback_window
        self._sampling_rate = sampling_rate
        self.gaze_callback = gaze_callback
        self.drone_callback = drone_callback
        self.timestamps = self._initializetime_stamps()

    def _initializetime_stamps(self) -> list[float]:
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
            start_timestamp, end_timestamp, 1000 / self._sampling_rate
        )
        return timestamps

    def get_closest_data(
        self, ts: float
    ) -> tuple[GazeData | None, list[DroneData] | None]:
        """Get the closest gaze and drone data for a given timestamp"""
        closest_gaze = self.gaze_data.iloc[
            (self.gaze_data["timestamp"] - ts).abs().argsort()[:1]
        ]
        if not closest_gaze.empty:
            gaze_row = closest_gaze.iloc[0]
            gaze = GazeData(**gaze_row)
        else:
            gaze = None

        closest_drone = self.drone_data.iloc[
            (self.drone_data["timestamp"] - ts).abs().argsort()[:9]
        ]
        if not closest_drone.empty:
            drones = [
                DroneData(**drone_row) for _, drone_row in closest_drone.iterrows()
            ]
        else:
            drones = None

        return gaze, drones

    def update_data_from_index(self, idx: int):
        """Update the data for the current index and call the callbacks"""
        if idx < 0 or idx >= len(self.timestamps):
            self._logger.warning("Index %d out of range for timestamps", idx)
            return

        ts = self.timestamps[idx]
        gaze, drones = self.get_closest_data(ts)

        if gaze and self.gaze_callback:
            self.gaze_callback([gaze])
        if drones and self.drone_callback:
            self.drone_callback(drones)

    def get_range(self) -> tuple[float, float]:
        """Get the timestamp range of the data for slider limits"""
        return self.timestamps[0], self.timestamps[-1]


class ExperimentDataReplayWindow(QMainWindow):
    """Placeholder for a window that would allow replaying recorded experiment
    data with the visualizer and slider"""

    def __init__(self, parent: QMainWindow | None = None):
        super().__init__(parent)
        self.setWindowTitle("Experiment Data Replay")
        self.setGeometry(150, 150, 1000, 800)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.visualizer = GazeDataCanvas(parent=self)
        self.replay_slider = ReplaySlider(parent=self)

        layout.addWidget(self.visualizer)
        layout.addWidget(self.replay_slider.canvas)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


def main():
    import sys

    app = QApplication(sys.argv)

    window = ExperimentDataReplayWindow()
    replay_folder = DATA_DIR / "experiments/experiment_1/P001/task_1/trial_1"
    replay_data = ReplayData(
        trial_folder=str(replay_folder), gaze_callback=window.visualizer.datas_callback
    )

    window.replay_slider.widget.on_changed(replay_data.update_data_from_index)
    window.replay_slider.widget.valmax = len(replay_data.timestamps) - 1
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
