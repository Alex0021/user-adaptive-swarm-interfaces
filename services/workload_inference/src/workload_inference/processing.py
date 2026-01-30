from collections import deque
from workload_inference.data_structures import GazeData

class DataProcessor:
    """
    Stores N latest data points and process them as needed.
    """

    def __init__(self, maxlen: int = 100, ready_threshold: int = -1):
        self._data_buffer: deque[GazeData] = deque(maxlen=maxlen)
        self._ready_threshold = maxlen if ready_threshold < 0 else ready_threshold

    def datas_callback(self, datas: list[GazeData]) -> None:
        """
        Add new data points to the buffer.

        Args:
            datas (list[GazeData]): The new data points to add.
        """
        for data in datas:
            self._data_buffer.append(data)

    def get_num_samples(self) -> int:
        """
        Get the number of data points currently stored in the buffer.

        Returns:
            int: The number of data points.
        """
        return len(self._data_buffer)
    
    def get_samples(self, range_start: int = 0, range_end: int | None = None) -> list[GazeData]:
        """
        Get data points from the buffer within the specified range.

        Args:
            range_start (int, optional): The starting index of the range. Defaults to 0.
            range_end (int | None, optional): The ending index of the range. Defaults to None.
        Returns:
            list[GazeData]: The list of data points within the specified range.
        """
        return list(self._data_buffer)[range_start:range_end]

    def is_ready(self) -> bool:
        """
        Check if the processor is ready based on the number of stored data points.

        Returns:
            bool: True if ready, False otherwise.
        """
        return len(self._data_buffer) >= self._ready_threshold