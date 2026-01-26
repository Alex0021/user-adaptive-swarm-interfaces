import struct
import threading
import mmap
from typing import Any
import data_structures as dts
import time
import numpy as np
from utilities import ConsoleManager

METADATA_BLOCK_NAME = "TobiiUnityMetadata2"
GAZE_DATA_BLOCK_NAME = "TobiiUnityGazeData2"
GAZE_DATA_BLOCK_CNT = 100

class PyReceiver:

    class Monitor:
        def __init__(self):
            self._last_timestamp: int = 0
            self._data_rate: float = 0.0
            self._data_cnt: int = 0
            self._update_cnt: int = 0
            self._data_cnt_avg: float = 0.0

        def update(self, packets_received: int):
            if self._last_timestamp == 0:
                self.start()
                return
            self._data_cnt += packets_received
            self._update_cnt += 1
            if time.time() - self._last_timestamp >= 1.0:
                self._data_rate = self._data_cnt / (time.time() - self._last_timestamp)
                self._data_cnt_avg = self._data_cnt / self._update_cnt if self._update_cnt > 0 else 0.0
                self._data_cnt = 0
                self._update_cnt = 0
                self._last_timestamp = time.time()

        def start(self):
            self.reset()
            self._last_timestamp = time.time()

        def reset(self):
            self._last_timestamp = 0
            self._data_rate = 0.0
            self._data_cnt = 0
            self._update_cnt = 0
            self._data_cnt_avg = 0.0
        def get_data_rate(self) -> float:
            return self._data_rate

        def get_avg_data_cnt(self) -> float:
            return self._data_cnt_avg

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._running: bool = False
        self._ready: bool = False
        self._metadata_block: mmap.mmap | None = None
        self._gaze_data_block: mmap.mmap | None = None
        self._gaze_data_ptr: int = 0
        self._monitor: PyReceiver.Monitor = PyReceiver.Monitor()
        self._console: ConsoleManager = ConsoleManager()

    def start(self) -> None:
        # Acquire shared memory blocks
        self._metadata_block = self.acquire_shm(METADATA_BLOCK_NAME, sum(dts.sm_metadata.values()))
        self._gaze_data_block = self.acquire_shm(GAZE_DATA_BLOCK_NAME, sum(dts.sm_gaze_data.values()) * GAZE_DATA_BLOCK_CNT, access=mmap.ACCESS_READ)

        self._console.start()
        # Start main thread
        if self._thread is None:
            self._running = True
            self._thread = threading.Thread(target=self._run)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._running = False
            self._thread.join()
            self._thread = None
    
    def _run(self) -> None:
        while self._running:
            # Check the stream_ready flag in metadata
            metadata = self.read_metadata_block()
            self._ready = (metadata["stream_ready"] == 1)

            if not self._ready:
                self._console.update_text("Waiting for stream to be ready...", use_spinner=True)
                time.sleep(0.1)
                continue

            # Here you can add code to read gaze data if needed
            if metadata["active_data_cnt"] == 0:
                self._console.update_text("No new gaze data available...", use_spinner=True)
                time.sleep(0.1)
                continue

            break

        self._monitor.start()

        while self._running:
            metadata = self.read_metadata_block()
            if metadata["active_data_cnt"] > 0:
                gaze_datas = self.read_gaze_data_blocks(metadata["active_data_cnt"])
                self._monitor.update(len(gaze_datas))
                # reset cnt
                metadata["active_data_cnt"] = 0
                self.write_metadata_cnt(metadata)
                self._console.update_text(f"Gaze Data Rate: {self._monitor.get_data_rate():.2f} Hz | Avg Data Count: {self._monitor.get_avg_data_cnt():.2f}", use_spinner=True)
                #self.pretty_print_gaze_data(gaze_datas[-1])  # Print the latest gaze data

        self._monitor.reset()
    
    def acquire_shm(self, block_name: str, block_size: int, access: int = mmap.ACCESS_DEFAULT) -> mmap.mmap:
        """
        Acquire a shared memory block by its name.

        Args:
            block_name (str): The name of the shared memory block.
            block_size (int): The size of the shared memory block.

        Returns:
            mmap.mmap: The acquired shared memory block.
        """
        shm = mmap.mmap(-1, block_size, tagname=block_name, access=access)
        return shm
    
    def read_metadata_block(self) -> dict[str, int]:
        """
        Read the metadata block from shared memory.

        Returns:
            dict[str, int]: A dictionary containing the metadata fields and their values.
        """
        metadata: dict[str, int] = {}
        self._metadata_block.seek(0)
        data = struct.unpack('BBB', self._metadata_block.read(sum(dts.sm_metadata.values())))
        metadata["stream_ready"] = data[0]
        metadata["calibration_ok"] = data[1]
        metadata["active_data_cnt"] = data[2]
        return metadata
    
    def read_gaze_data_blocks(self, count: int = 1) -> list[dict[str, float]]:
        """
        Read gaze data blocks from shared memory.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the gaze data fields and their values.
        """
        gaze_datas: list[dict[str, Any]] = []
        block_size = sum(dts.sm_gaze_data.values())

        for _ in range(count):
            self._gaze_data_block.seek(self._gaze_data_ptr)
            data = struct.unpack('<q10f2B2f', self._gaze_data_block.read(block_size))
            gaze_data: dict[str, Any] = {}
            gaze_data["timestamp"] = data[0]
            gaze_data["left_gaze_point"] = np.array([data[1], data[2], data[3]], dtype=np.float32)
            gaze_data["right_gaze_point"] = np.array([data[4], data[5], data[6]], dtype=np.float32)
            gaze_data["left_point_screen"] = np.array([data[7], data[8]], dtype=np.float32)
            gaze_data["right_point_screen"] = np.array([data[9], data[10]], dtype=np.float32)
            gaze_data["left_validity"] = data[11]
            gaze_data["right_validity"] = data[12]
            gaze_data["left_pupil_diameter"] = np.float32(data[13])
            gaze_data["right_pupil_diameter"] = np.float32(data[14])
            gaze_datas.append(gaze_data)

            self._gaze_data_ptr += block_size
            # Check for circular buffer wrap-around
            if self._gaze_data_ptr >= GAZE_DATA_BLOCK_CNT * block_size:
                self._gaze_data_ptr = 0

        return gaze_datas
    
    def write_metadata_cnt(self, metadata: dict[str, int]) -> None:
        """
        Write the updated active_data_cnt back to the metadata block.
        """
        if not "active_data_cnt" in metadata:
            return
        self._metadata_block.seek(2)  # Offset for active_data_cnt
        self._metadata_block.write(struct.pack('B', metadata["active_data_cnt"]))
        self._metadata_block.flush()

    def pretty_print_gaze_data(self, gaze_data: dict[str, Any]) -> None:
        """
        Pretty print the gaze data.
        """
        print('\r--------------' + ' '*20)
        for key, value in gaze_data.items():
            print(f"  {key}: {value}")
    

if __name__ == "__main__":
    receiver = PyReceiver()
    receiver.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        receiver.stop()