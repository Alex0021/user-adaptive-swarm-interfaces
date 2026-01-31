import time
from PyQt6.QtWidgets import QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from collections import deque
from workload_inference.data_structures import GazeData
import threading
from PyQt6.QtCore import pyqtSignal, QObject


class UpdateSignalEmitter(QObject):
    """Emits signals from background thread to update UI"""
    update_signal = pyqtSignal()

class GazeDataCanvas(FigureCanvas):
    """Matplotlib canvas with 3 subplots for gaze visualization"""

    def __init__(self, parent=None, screen_width=1920, screen_height=1200, max_history=1000, plotting_window=100, update_freq=10):
        self.fig = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        super().__init__(self.fig)
        self.parent = parent
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.window_size = plotting_window
        self.update_freq = update_freq
        self.data_cb_cnt = 0

        # Initalize 3 plots
        ar = self.screen_height / self.screen_width
        self.ax_gaze = self.fig.add_subplot(5, 2, (1, 6), aspect=ar, adjustable='box')
        self.ax_left_validity = self.fig.add_subplot(5, 2, 7)
        self.ax_right_validity = self.fig.add_subplot(5, 2, 9)
        self.ax_right_validity.sharex(self.ax_left_validity)
        self.ax_pupil = self.fig.add_subplot(5, 2, (8, 10))

        # Updates
        self._running = False
        self._update_thread = threading.Thread(target=self._update_loops, daemon=True)
        self.update_signal_emitter = UpdateSignalEmitter()
        self.update_signal_emitter.update_signal.connect(self._update_all)

        # Data buffers
        self.gaze_hist: deque[tuple[float, float]] = deque(maxlen=plotting_window)  # (x, y) positions
        self.validity_hist: deque[tuple[int, int]] = deque(maxlen=plotting_window)  # (left_validity, right_validity)
        self.pupil_hist: deque[tuple[float, float]] = deque(maxlen=plotting_window)  # (left_diameter, right_diameter)

        # TEST data
        d = np.linspace(3.0, 4.0, plotting_window)
        for i in range(plotting_window):
            self.validity_hist.append((i > plotting_window // 2, i < plotting_window // 2))
            self.pupil_hist.append((d[i], np.random.rand() + 3.5))
        # Create a circle trace of gaze points counter clockwise starting from top
        for i in range(plotting_window):
            angle = 2 * np.pi * (i / plotting_window)
            x = (self.screen_width / 2) + (self.screen_width / 4) * np.sin(angle)
            y = (self.screen_height / 2) - (self.screen_height / 4) * np.cos(angle)
            self.gaze_hist.append((x, y))

        # Line objects
        self.pupil_hist_lines = None
        self.validity_bars = None
        self.gaze_scatter = None

        self._init_plots()
        self.update_pupil_diameter()
        self.update_eye_validity()
        self.update_gaze_trace()

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
        self.ax_left_validity.set_title("Eye Validity History")
        self.ax_left_validity.set_yticks([0, 1, 2])
        self.ax_left_validity.set_yticklabels(["", "Left", ""])
        self.ax_right_validity.set_xlabel("Sample Index")
        self.ax_right_validity.set_yticks([0, 1, 2])
        self.ax_right_validity.set_yticklabels(["", "Right", ""])
        self.ax_left_validity.set_ylim(-0.5, 2.5)
        self.ax_right_validity.set_ylim(-0.5, 2.5)
        self.ax_right_validity.set_xlim(-self.window_size, 0)
        
        # Pupil diameter plot
        self.ax_pupil.set_title("Pupil Diameter Trend")
        self.ax_pupil.set_xlabel("Sample Index")
        self.ax_pupil.set_ylabel("Diameter (mm)")
        self.ax_pupil.legend(["Left", "Right", "Mean"])
        self.ax_pupil.set_xlim(-self.window_size, 0)

    def _update_all(self):
        """Update all plots"""
        self.update_gaze_trace()
        self.update_eye_validity()
        self.update_pupil_diameter()
        self.draw()

    def start_updates(self):
        """Start the periodic update thread"""
        if not self._running:
            self._running = True
            self._update_thread.start()

    def stop_updates(self):
        """Stop the periodic update thread"""
        if self._running:
            self._running = False
            self._update_thread.join()

    def update_pupil_diameter(self):
        """Update line plot for pupil diameter trends"""
        pupil_data = np.array(self.pupil_hist)
        if self.pupil_hist_lines is None:
            indices = np.arange(0, -len(pupil_data), -1)
            self.pupil_hist_lines = self.ax_pupil.plot(indices, pupil_data[:, 0], label="Left", color='blue')
            self.pupil_hist_lines += self.ax_pupil.plot(indices, pupil_data[:, 1], label="Right", color='orange')
            mean_diameter = np.mean(pupil_data, axis=1)
            self.pupil_hist_lines += self.ax_pupil.plot(indices, mean_diameter, label="Mean", linestyle='--', color='black')
            self.ax_pupil.legend()
        else:
            for i, line in enumerate(self.pupil_hist_lines):
                if i < 2:
                    line.set_ydata([pupil_data[t][i] for t in range(len(pupil_data))])
                else:
                    mean_diameter = np.mean(pupil_data, axis=1)
                    line.set_ydata(mean_diameter)
        self.ax_pupil.relim()
        self.ax_pupil.autoscale_view()

    def update_eye_validity(self):
        """Update bar plot for eye validity history"""
        if self.validity_bars is not None:
            for bar in self.validity_bars:
                bar.remove()
        validity_data = np.array(self.validity_hist)
        left_validity = validity_data[:, 0]
        right_validity = validity_data[:, 1]
        left_colors = ['green' if v == 1 else 'red' for v in left_validity]
        right_colors = ['green' if v == 1 else 'red' for v in right_validity]
        indices = np.arange(0, -len(validity_data), -1)
        self.validity_bars = []
        self.validity_bars += self.ax_left_validity.bar(indices, 2, width=1, label='Left', color=left_colors, alpha=0.6, align='center')
        self.validity_bars += self.ax_right_validity.bar(indices, 2, width=1, label='Right', color=right_colors, alpha=0.6, align='center')


    def update_gaze_trace(self):
        """
        Update scatter plot for gaze position trace
        Use a scatter plot with fading single color and size decrease to indicate recency
        """
        if self.gaze_scatter is not None:
            self.gaze_scatter.remove()
        gaze_data = np.array(self.gaze_hist)
        num_points = len(gaze_data)
        if num_points == 0:
            return
        sizes = np.linspace(10, 100, num_points)[::-1]
        colors = np.linspace(0.1, 1.0, num_points)[::-1]
        self.gaze_scatter = self.ax_gaze.scatter(gaze_data[:, 0], gaze_data[:, 1], s=sizes, c=colors, cmap='Greys', alpha=0.7)
        

    def datas_callback(self, gaze_datas: GazeData):
        """Callback to only store gaze data (minimal processing)"""
        for gaze_data in gaze_datas:
            self.data_cb_cnt += 1
            x = gaze_data.left_point_screen_x * self.screen_width
            y = gaze_data.left_point_screen_y * self.screen_height
            left_validity = gaze_data.left_validity
            right_validity = gaze_data.right_validity
            left_diameter = gaze_data.left_pupil_diameter
            right_diameter = gaze_data.right_pupil_diameter

            self.gaze_hist.append((x, y))
            self.validity_hist.append((left_validity, right_validity))
            self.pupil_hist.append((left_diameter, right_diameter))
    
    def _update_loops(self):
        """Internal method to periodically update plots"""
        while self._running:
            self.update_signal_emitter.update_signal.emit()
            time.sleep(1)  
        

class GazeVisualizerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaze Visualizer")
        self.setGeometry(100, 100, 800, 600)      
        self.canvas = GazeDataCanvas(screen_width=1920, screen_height=1200, plotting_window=200)
        self.setCentralWidget(self.canvas)
    
    def set_update_loop_state(self, running: bool):
        """Start or stop the update loop"""
        if running:
            self.canvas.start_updates()
        else:
            self.canvas.stop_updates()