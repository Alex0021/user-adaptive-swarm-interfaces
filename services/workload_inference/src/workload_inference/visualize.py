import contextlib
import logging
import os

# Import torch before PyQt6 to avoid DLL conflicts on Windows
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

os.environ["QT_API"] = "PyQt6"  # Ensure PyQt6 is used for matplotlib backend
import threading
import time
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
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from workload_inference.constants import DATA_DIR
from workload_inference.data_structures import DroneData, GazeData, Listener
from workload_inference.inference import (
    WORKLOAD_COLORS,
    WORKLOAD_LABELS,
    WorkloadInferenceEngine,
)

SPLINE_TRAJECTORY_FILE = DATA_DIR / "spline_trajectory.csv"
DRONE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]


class DroneDataCanvas(FigureCanvas):
    """Matplotlib canvas for top-down drone position visualization"""

    def __init__(
        self,
        parent: QMainWindow | None = None,
        num_drones: int = 9,
        max_history: int = 1000,
        plotting_window: int = 200,
        update_freq: int = 30,
    ):
        """
        Initialize the canvas for drone position visualization.

        Args:
            parent (QMainWindow | None): Parent window for the canvas.
            num_drones (int): Number of drones to track.
            max_history (int): Maximum number of data points to keep in history.
            plotting_window (int): Number of data points to display per drone trail.
            update_freq (int): Frequency of plot updates in Hz.
        """
        self.fig = Figure(figsize=(6, 6), dpi=100)
        super().__init__(self.fig)
        self.parent = parent
        self.num_drones = num_drones
        self.window_size = plotting_window
        self.update_freq = update_freq
        self.data_cb_cnt = 0
        self._timer = QTimer(parent)
        self._timer.timeout.connect(self._update_all)

        # Initialize plot
        self.ax = self.fig.add_subplot(1, 1, 1, adjustable="box")

        # Load spline trajectory for background track
        self._spline_x: NDArray[np.float64] | None = None
        self._spline_z: NDArray[np.float64] | None = None
        self._load_spline_trajectory()

        # Pre-allocated ring buffers per drone (num_drones x window x 2)
        self._buffers = np.full((num_drones, plotting_window, 2), np.nan)
        self._buf_lens = np.zeros(num_drones, dtype=int)  # valid sample count
        self._buf_idx = np.zeros(num_drones, dtype=int)  # write cursor

        # Blit objects: single scatter for all trails, single scatter for positions
        self.trail_scatter: PathCollection | None = None
        self.position_scatter: PathCollection | None = None
        self._sizes_template = np.linspace(2, 20, plotting_window)

        # Pre-compute RGBA color arrays (trail + position) once
        import matplotlib.colors as mcolors

        trail_rgba = np.empty((num_drones, plotting_window, 4))
        pos_rgba = np.empty((num_drones, 4))
        for i in range(num_drones):
            r, g, b = mcolors.to_rgb(DRONE_COLORS[i % len(DRONE_COLORS)])
            trail_rgba[i, :] = (r, g, b, 0.4)
            pos_rgba[i] = (r, g, b, 1.0)
        self._trail_rgba_full = trail_rgba  # (num_drones, window, 4)
        self._pos_rgba = pos_rgba  # (num_drones, 4)

        # Cached concatenated arrays (rebuilt only when total point count changes)
        self._cached_trail_colors: NDArray | None = None
        self._cached_trail_sizes: NDArray | None = None
        self._cached_point_count = -1

        # Blitting state
        self._background = None
        self._blit_ready = False

        self._init_plots()

        # Blitting hooks
        self.mpl_connect("draw_event", self._on_draw)
        self.mpl_connect("resize_event", self._on_resize)
        self._init_blit()

        self._timer.start(1000 // self.update_freq)

    def _load_spline_trajectory(self):
        """Load the spline trajectory CSV for the background track."""
        try:
            spline_df = pd.read_csv(SPLINE_TRAJECTORY_FILE)
            self._spline_x = spline_df["x"].values
            self._spline_z = spline_df["z"].values
        except FileNotFoundError:
            logging.getLogger("DroneDataCanvas").warning(
                "Spline trajectory file not found at '%s'.", SPLINE_TRAJECTORY_FILE
            )

    def _init_plots(self):
        """Initialize plot styling and labels"""
        self.ax.set_title("Drone Positions (Top-Down)")
        self.ax.set_xlabel("Z")
        self.ax.set_ylabel("X")

        # Draw spline trajectory as background
        if self._spline_x is not None and self._spline_z is not None:
            self.ax.plot(
                self._spline_z,
                self._spline_x,
                color="lightgray",
                linewidth=2,
                linestyle="--",
                label="Track",
                zorder=0,
            )
            # Auto-fit axis limits from trajectory with padding
            pad_x = (self._spline_x.max() - self._spline_x.min()) * 0.1
            pad_z = (self._spline_z.max() - self._spline_z.min()) * 0.1
            self.ax.set_ylim(self._spline_x.min() - pad_x, self._spline_x.max() + pad_x)
            # Invert X-axis for clockwise motion from left start
            self.ax.set_xlim(self._spline_z.max() + pad_z, self._spline_z.min() - pad_z)

    def _init_blit(self):
        """Initialize blitting by caching the background"""
        if self.trail_scatter is not None:
            self.trail_scatter.set_animated(True)
        if self.position_scatter is not None:
            self.position_scatter.set_animated(True)

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

        if self.trail_scatter is not None:
            self.ax.draw_artist(self.trail_scatter)
        if self.position_scatter is not None:
            self.ax.draw_artist(self.position_scatter)

        self.blit(self.fig.bbox)

    def _update_all(self):
        """Update all drone plots"""
        self.update_drone_positions()
        self._blit_update()

    def update_drone_positions(self):
        """Update scatter plots for all drone trails and current positions"""
        total_points = int(self._buf_lens.sum())
        if total_points == 0:
            return

        # Build trail offsets from ring buffers (only position data changes each frame)
        trail_offsets = np.empty((total_points, 2))
        current_offsets = np.empty((int((self._buf_lens > 0).sum()), 2))
        offset = 0
        cur_idx = 0
        for drone_id in range(self.num_drones):
            n = self._buf_lens[drone_id]
            if n == 0:
                continue
            buf = self._buffers[drone_id]
            wi = self._buf_idx[drone_id]
            if n < self.window_size:
                # Buffer not yet full: data is [0..n)
                trail_offsets[offset : offset + n] = buf[:n]
            else:
                # Buffer full: read in ring order starting from write cursor
                trail_offsets[offset : offset + self.window_size - wi] = buf[wi:]
                trail_offsets[offset + self.window_size - wi : offset + n] = buf[:wi]
            current_offsets[cur_idx] = trail_offsets[offset + n - 1]
            offset += n
            cur_idx += 1

        # Rebuild cached colors/sizes only when total point count changes
        count_changed = total_points != self._cached_point_count
        if count_changed:
            self._cached_point_count = total_points
            sizes_parts = []
            colors_parts = []
            for drone_id in range(self.num_drones):
                n = self._buf_lens[drone_id]
                if n == 0:
                    continue
                sizes_parts.append(self._sizes_template[-n:])
                colors_parts.append(self._trail_rgba_full[drone_id, -n:])
            self._cached_trail_sizes = np.concatenate(sizes_parts)
            self._cached_trail_colors = np.concatenate(colors_parts)

        # Trail scatter (single artist for all drones)
        if self.trail_scatter is None:
            self.trail_scatter = self.ax.scatter(
                trail_offsets[:, 0],
                trail_offsets[:, 1],
                s=self._cached_trail_sizes,
                c=self._cached_trail_colors,
                zorder=1,
            )
            self.trail_scatter.set_animated(True)
        else:
            self.trail_scatter.set_offsets(trail_offsets)
            if count_changed:
                self.trail_scatter.set_sizes(self._cached_trail_sizes)
                self.trail_scatter.set_facecolors(self._cached_trail_colors)

        # Current position markers (single artist for all drones)
        active_mask = self._buf_lens > 0
        current_colors = self._pos_rgba[active_mask]
        if self.position_scatter is None:
            self.position_scatter = self.ax.scatter(
                current_offsets[:, 0],
                current_offsets[:, 1],
                s=80,
                c=current_colors,
                edgecolors="black",
                linewidths=1,
                marker="o",
                zorder=2,
            )
            self.position_scatter.set_animated(True)
        else:
            self.position_scatter.set_offsets(current_offsets)

    def datas_callback(
        self, datas: Sequence[DroneData], batch_update: bool = False
    ) -> None:
        """Callback to store drone position data (minimal processing)

        Args:
            datas: List of new drone data points to add to the history
            batch_update: Whether to flush the history and only use the given datas
        """
        if batch_update:
            self._buffers[:] = np.nan
            self._buf_lens[:] = 0
            self._buf_idx[:] = 0
        for drone_data in datas:
            self.data_cb_cnt += 1
            drone_id = int(drone_data.id)
            if 0 <= drone_id < self.num_drones:
                wi = self._buf_idx[drone_id]
                self._buffers[drone_id, wi, 0] = float(drone_data.position_z)
                self._buffers[drone_id, wi, 1] = float(drone_data.position_x)
                self._buf_idx[drone_id] = (wi + 1) % self.window_size
                if self._buf_lens[drone_id] < self.window_size:
                    self._buf_lens[drone_id] += 1


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
        self.fig = Figure(figsize=(8, 8), dpi=100)
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
        self.ax_gaze = self.fig.add_subplot(4, 1, (1, 2), aspect=ar, adjustable="box")
        self.ax_validity = self.fig.add_subplot(4, 1, 3, adjustable="box")
        self.ax_pupil = self.fig.add_subplot(4, 1, 4, adjustable="box")

        # Pre-allocated ring buffers (avoids deque -> np.array conversion each frame)
        self._gaze_buf = np.full((plotting_window, 2), np.nan)
        self._validity_buf = np.full((plotting_window, 2), -1, dtype=int)
        self._pupil_buf = np.full((plotting_window, 2), np.nan)
        self._buf_len = 0
        self._buf_idx = 0  # write cursor

        # Pre-allocated padded validity array for imshow (avoids alloc each frame)
        self._validity_padded = np.full((2, plotting_window), -1, dtype=int)

        # TEST data
        d = np.linspace(3.0, 4.0, plotting_window)
        for i in range(plotting_window):
            self._validity_buf[i] = (
                int(i > plotting_window // 2),
                int(i < plotting_window // 2),
            )
            self._pupil_buf[i] = (d[i], np.random.rand() + 3.5)
        # Create a circle trace of gaze points counter clockwise starting from top
        for i in range(plotting_window):
            angle = 2 * np.pi * (i / plotting_window)
            x = (self.screen_width / 2) + (self.screen_width / 4) * np.sin(angle)
            y = (self.screen_height / 2) - (self.screen_height / 4) * np.cos(angle)
            self._gaze_buf[i] = (x, y)
        self._buf_len = plotting_window
        self._buf_idx = 0

        # Blit objects
        self.pupil_hist_lines: list[Line2D] = []
        self.validity_img: AxesImage | None = None
        self.gaze_scatter: PathCollection | None = None
        self.gaze_current: PathCollection | None = None
        self.sizes = np.linspace(5, 50, plotting_window)
        self.colors = np.linspace(0.1, 1.0, plotting_window)

        # Pre-create validity colormap and norm (reused every frame)
        self._validity_cmap = ListedColormap(["white", "red", "green"])
        self._validity_norm = BoundaryNorm(
            [-1.5, -0.5, 0.5, 1.5], self._validity_cmap.N
        )

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
        self.fig.tight_layout(pad=2.0)

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
        if self.gaze_current is not None:
            self.gaze_current.set_animated(True)

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
        if self.gaze_current is not None:
            self.ax_gaze.draw_artist(self.gaze_current)

        self.blit(self.fig.bbox)

    def _update_all(self):
        """Update all plots"""
        self.update_gaze_trace()
        self.update_eye_validity()
        self.update_pupil_diameter()
        self._blit_update()

    def _get_ordered_buf(self, buf: NDArray) -> NDArray:
        """Extract valid data from a ring buffer in chronological order."""
        n = self._buf_len
        if n < self.window_size:
            return buf[:n]
        wi = self._buf_idx
        return np.concatenate((buf[wi:], buf[:wi]))

    def update_pupil_diameter(self):
        """Update line plot for pupil diameter trends"""
        pupil_data = self._get_ordered_buf(self._pupil_buf)
        n = len(pupil_data)
        if n == 0:
            return
        if len(self.pupil_hist_lines) == 0:
            indices = np.arange(-n, 0)
            self.pupil_hist_lines = self.ax_pupil.plot(
                indices, pupil_data[:, 0], label="Left", color="blue"
            )
            self.pupil_hist_lines += self.ax_pupil.plot(
                indices, pupil_data[:, 1], label="Right", color="orange"
            )
            mean_diameter = pupil_data.mean(axis=1)
            self.pupil_hist_lines += self.ax_pupil.plot(
                indices, mean_diameter, label="Mean", linestyle="--", color="black"
            )
            self.ax_pupil.legend()
        else:
            xdata = np.asarray(self.pupil_hist_lines[0].get_xdata())
            if n != xdata.shape[0]:
                indices = np.arange(-n, 0)
                for line in self.pupil_hist_lines:
                    line.set_xdata(indices)

            self.pupil_hist_lines[0].set_ydata(pupil_data[:, 0])
            self.pupil_hist_lines[1].set_ydata(pupil_data[:, 1])
            self.pupil_hist_lines[2].set_ydata(pupil_data.mean(axis=1))

    def update_eye_validity(self):
        """
        Update bar plot for eye validity history
        Using an image mapping to be efficient for plotting
        """
        validity_data = self._get_ordered_buf(self._validity_buf)
        n = len(validity_data)
        # Write into pre-allocated padded array (avoids alloc each frame)
        self._validity_padded[:] = -1
        if n > 0:
            self._validity_padded[:, -n:] = validity_data.T

        if self.validity_img is None:
            self.validity_img = self.ax_validity.imshow(
                self._validity_padded,
                aspect="auto",
                cmap=self._validity_cmap,
                norm=self._validity_norm,
                extent=(-self.window_size, 0, -0.5, 1.5),
            )
        else:
            self.validity_img.set_data(self._validity_padded)

    def update_gaze_trace(self):
        """
        Update scatter plot for gaze position trace
        Only recalculate sizes/colors for NEW points
        """
        gaze_data = self._get_ordered_buf(self._gaze_buf)
        n = len(gaze_data)
        if n == 0:
            return

        # Trail scatter
        if self.gaze_scatter is None:
            self.gaze_scatter = self.ax_gaze.scatter(
                gaze_data[:, 0],
                gaze_data[:, 1],
                s=self.sizes[-n:],
                c=self.colors[-n:],
                cmap="Greys",
                alpha=0.7,
            )
        else:
            if n < self.window_size:
                self.gaze_scatter.set_sizes(self.sizes[-n:])
                self.gaze_scatter.set_array(self.colors[-n:])
            self.gaze_scatter.set_offsets(gaze_data)

        # Current position marker
        current_pos = gaze_data[-1:]
        if self.gaze_current is None:
            self.gaze_current = self.ax_gaze.scatter(
                current_pos[:, 0],
                current_pos[:, 1],
                s=100,
                color="red",
                edgecolors="black",
                linewidths=1,
                marker="o",
                zorder=3,
            )
            self.gaze_current.set_animated(True)
        else:
            self.gaze_current.set_offsets(current_pos)

    def datas_callback(
        self, datas: Sequence[GazeData], batch_update: bool = False
    ) -> None:
        """Callback to only store gaze data (minimal processing)

        Args:
            datas: List of new gaze data points to add to the history
            batch_update: Wether to flush the history and only use the given datas
        """
        if batch_update:
            self._gaze_buf[:] = np.nan
            self._validity_buf[:] = -1
            self._pupil_buf[:] = np.nan
            self._buf_len = 0
            self._buf_idx = 0
        for gaze_data in datas:
            self.data_cb_cnt += 1
            wi = self._buf_idx
            self._gaze_buf[wi, 0] = float(
                gaze_data.left_point_screen_x * self.screen_width
            )
            self._gaze_buf[wi, 1] = float(
                gaze_data.left_point_screen_y * self.screen_height
            )
            self._validity_buf[wi, 0] = int(gaze_data.left_validity)
            self._validity_buf[wi, 1] = int(gaze_data.right_validity)
            self._pupil_buf[wi, 0] = float(gaze_data.left_pupil_diameter)
            self._pupil_buf[wi, 1] = float(gaze_data.right_pupil_diameter)
            self._buf_idx = (wi + 1) % self.window_size
            if self._buf_len < self.window_size:
                self._buf_len += 1


class WorkloadDisplayWidget(QWidget):
    """Widget displaying real-time cognitive workload estimation.

    Layout:
        Row 1: History timeline spanning full width (last 30 s,
               showing both raw and filtered predictions).
        Row 2: Compact class label | probability bars side-by-side.

    Thread safety:
        The engine listener ``_on_workload_data`` is called from the
        inference background thread, so it only stores data into plain
        Python lists.  A QTimer on the **main/GUI thread** periodically
        calls ``_refresh_ui`` to read that data and update Qt widgets /
        matplotlib, which is the only safe way to touch the GUI.
    """

    HISTORY_WINDOW_SEC = 30.0  # fixed time window for the history plot
    UI_REFRESH_MS = 250  # how often the GUI timer fires

    def __init__(
        self,
        parent: QWidget | None = None,
        engine: WorkloadInferenceEngine | None = None,
    ):
        super().__init__(parent)
        self._engine = engine

        # --- Data written by the inference thread (no Qt access here) ---
        import threading

        self._data_lock = threading.Lock()
        self._history_raw: list[int] = []
        self._history_filtered: list[int] = []
        self._history_timestamps: list[float] = []
        self._latest_filtered_class: int = -1
        self._latest_probabilities: np.ndarray = np.zeros(3)
        self._dirty = False  # flag: new data since last UI refresh

        self._init_ui()

        # GUI-thread timer for safe UI updates
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_ui)
        self._refresh_timer.start(self.UI_REFRESH_MS)

        if self._engine is not None:
            self._engine.register_listener(self._on_workload_data)

    def set_engine(self, engine: WorkloadInferenceEngine) -> None:
        """Attach (or replace) the inference engine."""
        self._engine = engine
        engine.register_listener(self._on_workload_data)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # --- Row 1: history timeline (full width) ---
        self._history_fig = Figure(figsize=(6, 1.4), dpi=100)
        self._history_ax = self._history_fig.add_subplot(1, 1, 1)
        self._history_canvas = FigureCanvas(self._history_fig)
        self._history_canvas.setMinimumHeight(90)
        root.addWidget(self._history_canvas, 1)
        self._init_history_plot()
        self._history_fig.tight_layout(pad=0.8)

        # --- Row 2: class label + probability bars ---
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        # Compact workload label
        label_col = QVBoxLayout()
        label_col.setContentsMargins(0, 0, 0, 0)
        self._workload_label = QLabel("N/A")
        self._workload_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._workload_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 4px; "
            "border: 2px solid gray; border-radius: 4px;"
        )
        self._workload_label.setFixedWidth(80)
        label_col.addWidget(self._workload_label)
        self._info_label = QLabel("Inferences: 0")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("font-size: 9px; color: gray;")
        label_col.addWidget(self._info_label)
        bottom_row.addLayout(label_col, 0)

        # Probability bars
        bars_col = QVBoxLayout()
        bars_col.setSpacing(2)
        self._prob_bars: list[QProgressBar] = []
        self._prob_labels: list[QLabel] = []
        self._prob_percents: list[QLabel] = []

        for cls_idx in range(3):
            bar_row = QHBoxLayout()
            bar_row.setSpacing(4)

            label = QLabel(WORKLOAD_LABELS[cls_idx])
            label.setMinimumWidth(36)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            label.setStyleSheet("font-size: 9px;")
            bar_row.addWidget(label, 0)
            self._prob_labels.append(label)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("")
            color = WORKLOAD_COLORS[cls_idx]
            bar.setStyleSheet(
                f"QProgressBar::chunk {{ background-color: {color}; }}"
            )
            bar.setFixedHeight(16)
            bar_row.addWidget(bar, 1)
            self._prob_bars.append(bar)

            percent = QLabel("0%")
            percent.setMinimumWidth(28)
            percent.setAlignment(Qt.AlignmentFlag.AlignRight)
            percent.setStyleSheet("font-size: 9px; font-weight: bold;")
            bar_row.addWidget(percent, 0)
            self._prob_percents.append(percent)

            bars_col.addLayout(bar_row)

        bottom_row.addLayout(bars_col, 1)
        root.addLayout(bottom_row, 0)

        self.setMinimumHeight(140)
        self._add_mockup_data()

    def _init_history_plot(self):
        ax = self._history_ax
        ax.set_title("Prediction history", fontsize=9)
        class_labels = {k: v for k, v in WORKLOAD_LABELS.items() if k >= 0}
        ax.set_ylim(-0.5, len(class_labels) - 0.5)
        ax.set_yticks(list(class_labels.keys()))
        ax.set_yticklabels(class_labels.values(), fontsize=7)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlim(-self.HISTORY_WINDOW_SEC, 0)

        (self._line_raw,) = ax.plot(
            [], [],
            "o", markersize=3, alpha=0.4, label="Raw", color="steelblue",
        )
        (self._line_filtered,) = ax.plot(
            [], [],
            "-", linewidth=2, label="Filtered", color="darkorange",
        )
        ax.legend(fontsize=6, loc="upper left")

    def _add_mockup_data(self):
        now = time.time()
        for i in range(15):
            raw_cls = (i // 5) % 3
            filt_cls = max(0, min(2, raw_cls))
            self._history_raw.append(raw_cls)
            self._history_filtered.append(filt_cls)
            self._history_timestamps.append(now - (15 - i) * 1.0)
        self._latest_filtered_class = 1
        self._latest_probabilities = np.array([0.2, 0.6, 0.2])
        self._dirty = True
        # Immediately paint so the mockup is visible on startup
        self._refresh_ui()

    # ------------------------------------------------------------------
    # Inference thread callback (data only, NO Qt access)
    # ------------------------------------------------------------------

    def _on_workload_data(
        self,
        raw_class: int,
        filtered_class: int,
        probabilities: np.ndarray,
    ) -> None:
        """Called from the inference **background thread**.

        Only touches plain Python/numpy data under a lock.
        The QTimer ``_refresh_ui`` picks it up on the GUI thread.
        """
        now = time.time()
        with self._data_lock:
            self._history_raw.append(raw_class)
            self._history_filtered.append(filtered_class)
            self._history_timestamps.append(now)

            # Trim to the time window
            cutoff = now - self.HISTORY_WINDOW_SEC
            while (
                self._history_timestamps
                and self._history_timestamps[0] < cutoff
            ):
                self._history_timestamps.pop(0)
                self._history_raw.pop(0)
                self._history_filtered.pop(0)

            self._latest_filtered_class = filtered_class
            self._latest_probabilities = probabilities.copy()
            self._dirty = True

    # ------------------------------------------------------------------
    # GUI-thread refresh (called by QTimer, safe to touch widgets)
    # ------------------------------------------------------------------

    def _refresh_ui(self) -> None:
        """Read latest data and update all Qt widgets + matplotlib."""
        with self._data_lock:
            if not self._dirty:
                return
            self._dirty = False
            # Snapshot the data while holding the lock
            timestamps = list(self._history_timestamps)
            raw = list(self._history_raw)
            filtered = list(self._history_filtered)
            filt_class = self._latest_filtered_class
            proba = self._latest_probabilities.copy()

        # -- Update class label --
        label = WORKLOAD_LABELS.get(filt_class, "???")
        color = WORKLOAD_COLORS.get(filt_class, "gray")
        self._workload_label.setText(label)
        self._workload_label.setStyleSheet(
            f"font-size: 18px; font-weight: bold; padding: 4px; "
            f"border: 2px solid {color}; border-radius: 4px; color: {color};"
        )

        # -- Update probability bars --
        for i, (bar, pct_lbl) in enumerate(
            zip(self._prob_bars, self._prob_percents, strict=True)
        ):
            pct = int(proba[i] * 100) if i < len(proba) else 0
            bar.setValue(pct)
            pct_lbl.setText(f"{pct}%")

        # -- Update history plot --
        if timestamps:
            now = timestamps[-1]
            x = [t - now for t in timestamps]
            self._line_raw.set_data(x, raw)
            self._line_filtered.set_data(x, filtered)
            self._history_ax.set_xlim(-self.HISTORY_WINDOW_SEC, 0)
            self._history_canvas.draw_idle()

        self._info_label.setText(f"Inferences: {len(raw)}")


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

    def _resample_df(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        method: str = "nearest",
    ) -> pd.DataFrame:
        """
        Resample a dataframe to the common timestamps.

        When the target sampling rate is higher than the source data rate,
        linear interpolation is used automatically to fill in-between samples.
        Otherwise nearest-neighbor reindexing is used.

        Args:
            df: The dataframe to resample
            timestamp_col: The name of the timestamp column in the dataframe
            method: Resampling method ("nearest" or "interpolate")

        Returns:
            pd.DataFrame: The resampled dataframe aligned to the common timestamps
        """
        if df.empty:
            return df
        if not df[timestamp_col].is_unique:
            df = df.drop_duplicates(timestamp_col)

        # Estimate source data rate from median timestamp delta
        source_dt = df[timestamp_col].diff().median()
        target_dt = 1000.0 / self._sampling_rate
        needs_interpolation = method == "interpolate" or target_dt < source_dt * 0.9

        df = df.set_index(timestamp_col)

        if needs_interpolation:
            src_ts = df.index.values.astype(np.float64)
            result = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    result[col] = np.interp(self.timestamps, src_ts, df[col].values)
                else:
                    # Non-numeric columns: forward-fill via nearest
                    idx = np.searchsorted(src_ts, self.timestamps).clip(
                        0, len(src_ts) - 1
                    )
                    result[col] = df[col].values[idx]
            resampled = pd.DataFrame(result, index=self.timestamps)
            resampled.index.name = timestamp_col
        else:
            resampled = df.reindex(
                self.timestamps,
                method="nearest",
                tolerance=2 * 1000 / self._sampling_rate,
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
    """Window for replaying recorded experiment data with visualizers,
    slider controls, and real-time workload inference."""

    def __init__(
        self,
        parent: QMainWindow | None = None,
        trial_folder: Path | None = None,
        model_path: str | Path | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Experiment Data Replay")
        self.setGeometry(150, 150, 1600, 800)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.gaze_visualizer = GazeDataCanvas(parent=self)
        self.drone_visualizer = DroneDataCanvas(parent=self)

        # Workload inference engine + display
        from workload_inference.inference import InferenceSettings

        self.workload_engine = WorkloadInferenceEngine(
            model_path=model_path,
            settings=InferenceSettings(
                window_size_samples=600,  # 10s at 60 Hz
                inference_interval_samples=60,  # 1s at 60 Hz
                smoothing_predictions=5,
            ),
        )
        self.workload_display = WorkloadDisplayWidget(
            parent=self, engine=self.workload_engine
        )

        if trial_folder:
            self.replay_data = ReplayData(
                trial_folder=trial_folder,
                gaze_callback=self._gaze_replay_callback,
                drone_callback=self.drone_visualizer.datas_callback,
                sampling_rate=60.0,
            )
            self.replay_slider = self.replay_data.slider
        else:
            self.replay_slider = ReplaySlider(parent=self)

        canvas_layout = QHBoxLayout()
        canvas_layout.addWidget(self.gaze_visualizer, 1)
        right_pane_widget = QWidget()
        right_pane_layout = QVBoxLayout()
        right_pane_layout.addWidget(self.drone_visualizer, 2)
        right_pane_layout.addWidget(self.workload_display, 1)
        right_pane_widget.setLayout(right_pane_layout)
        canvas_layout.addWidget(right_pane_widget, 1)

        layout.addWidget(self.replay_slider.canvas)
        layout.addLayout(canvas_layout, 1)

        self.replay_slider.play_btn.on_clicked(self.play)
        self.replay_slider.pause_btn.on_clicked(self.pause)
        self.replay_slider.step_forward_btn.on_clicked(self.step_forward)
        self.replay_slider.step_back_btn.on_clicked(self.step_backward)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _gaze_replay_callback(
        self, datas: Sequence[GazeData], batch_update: bool = False
    ) -> None:
        """Fan-out gaze data to both the visualizer and the workload engine."""
        self.gaze_visualizer.datas_callback(datas, batch_update)
        self.workload_engine.gaze_datas_callback(datas, batch_update)

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
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Replay and visualize experiment data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default experiment without cognitive load estimation
  python -m workload_inference.visualize

  # Specify a custom experiment folder
  python -m workload_inference.visualize /path/to/experiment/folder

  # Use a relative path
  python -m workload_inference.visualize \\
    experiments/experiment_nback/ALH0/FlyingPractice

  # Enable cognitive load estimation with a model
  python -m workload_inference.visualize --model /path/to/model.pt

  # Specify both experiment and model
  python -m workload_inference.visualize \\
    experiments/experiment_nback/ALH0/FlyingPractice \\
    --model models/cognitive_load_model.pt
        """,
    )
    parser.add_argument(
        "trial_folder",
        nargs="?",
        default=None,
        help=(
            "Path to experiment folder (relative or absolute). "
            "If not provided, uses default experiment."
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help=(
            "Path to cognitive load estimation model. "
            "If provided, enables workload inference visualization."
        ),
    )

    args = parser.parse_args()

    # Determine the replay folder
    if args.trial_folder:
        replay_folder = Path(args.trial_folder)
        if not replay_folder.is_absolute():
            # Treat relative paths as relative to DATA_DIR
            replay_folder = DATA_DIR / args.trial_folder
    else:
        # Default experiment
        replay_folder = (
            DATA_DIR / "experiments" / "experiment_nback" / "ALH0" / "FlyingPractice"
        )

    if not replay_folder.exists():
        print(f"Error: Experiment folder not found: {replay_folder}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = ExperimentDataReplayWindow(
        trial_folder=replay_folder, model_path=args.model
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
