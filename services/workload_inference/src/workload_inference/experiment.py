from queue import Queue
import threading
import time
from workload_inference.data_structures import GazeData
from workload_inference.constants import DATA_DIR
import yaml
from typing import TextIO
import logging

CONFIG_FILE_NAME = "experiment.yaml"
SAMPLE_CONFIG_FILE_NAME = "sample_experiment.yaml"
DATA_FILE_NAME = "gaze_data.csv"
CSV_HEADER = GazeData.__annotations__.keys()

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
            logger.warning(f"Experiment configuration file '{CONFIG_FILE_NAME}' not found in '{self.base_folder}'."
                           f" Using sample configuration '{SAMPLE_CONFIG_FILE_NAME}'.")
            if not (self.base_folder / SAMPLE_CONFIG_FILE_NAME).exists():
                raise FileNotFoundError(f"Sample experiment configuration file '{SAMPLE_CONFIG_FILE_NAME}' not found"
                                        f" in '{self.base_folder}'. Please create an experiment configuration file.")
            with open(self.base_folder / SAMPLE_CONFIG_FILE_NAME, 'r') as f:
                self.experiment_config = yaml.safe_load(f)
        else:
            with open(self.base_folder / CONFIG_FILE_NAME, 'r') as f:
                self.experiment_config = yaml.safe_load(f)
        self._writer_thread: threading.Thread | None = None
        self._running: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._current_exp_filestream: TextIO | None = None
        self._block_size: int = 100
        self._data_cnt: int = 0
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
            logger.info("Data recording started. Data blocks set to %d.", self._block_size)
    
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
                    line = ",".join(str(gaze_data.__dict__[field]) for field in CSV_HEADER) + "\n"
                    self._current_exp_filestream.write(line)
                self._current_exp_filestream.flush()
                self._data_cnt += self._block_size
            time.sleep(0.01)

        # Write remaining data in the queue
        if self.data_queue.qsize() > 0:
            self._data_cnt += self.data_queue.qsize()
            for _ in range(self.data_queue.qsize()):
                gaze_data = self.data_queue.get()
                line = ",".join(str(getattr(gaze_data, field)) for field in CSV_HEADER) + "\n"
                self._current_exp_filestream.write(line)
            self._current_exp_filestream.flush()
    
    def _initialize_structure(self, overwrite: bool = True) -> None:
        """
        Initialize the experiment data file structure.

        Args:
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        Raises:
            FileExistsError: If the experiment data file already exists and overwrite is False.
        """
        if not "name" in self.experiment_config:
            logger.warning("Experiment name not found in configuration. Using 'anonymous'.")
        if not "name" in self.experiment_config['participant']:
            logger.warning("Participant name not found in configuration. Using 'person'.")
        if not "name" in self.experiment_config['task']:
            logger.warning("Task name not found in configuration. Using 'unknown_task'.")
        exp_name = self.experiment_config.get("name", "anonymous")
        participant_name = self.experiment_config["participant"].get("name", "person")
        task_name = self.experiment_config["task"].get("name", "unknown_task")
         # Make sure folders are snake_case (especially for linux compatibility)
        formatted_str_path = '/'.join([exp_name, participant_name, task_name]).replace(" ", "_").lower()
        exp_folder = self.base_folder / formatted_str_path
        exp_folder.mkdir(parents=True, exist_ok=True)
        if (exp_folder / DATA_FILE_NAME).exists() and not overwrite:
            raise FileExistsError(f"Experiment data file '{DATA_FILE_NAME}' already exists in '{exp_folder}'."
                                  " To bypass, set the 'overwrite' parameter to True.")
        self._current_exp_filestream = open(exp_folder / DATA_FILE_NAME, 'w', encoding='utf-8')
        # Write CSV header
        header = ",".join(CSV_HEADER) + "\n"
        self._current_exp_filestream.write(header)
        self._current_exp_filestream.flush()
        logger.info(f"Data structure ready at '{exp_folder}'.")
    
    
