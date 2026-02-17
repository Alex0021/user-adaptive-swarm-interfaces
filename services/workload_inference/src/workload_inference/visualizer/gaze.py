from collections import deque
from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import PathCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow

from workload_inference.data.data_structures import GazeData

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
