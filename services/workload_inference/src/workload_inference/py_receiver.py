import logging
import mmap
import struct
import threading
import time

import numpy as np
import zmq

import workload_inference.data_structures as dts
from workload_inference.utilities import ConsoleManager


class PyReceiverBase:
    """
    Base class for Gaze Data Receivers.
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = False
        self._ready: bool = False

        self._listeners: list[dts.Listener] = []
        """Listeners for data updates. Each listener is a callable function 
        that takes a list of Dataclass instances as an argument."""

        self._monitor: Monitor = Monitor()
        self._console: ConsoleManager = ConsoleManager()

        self._logger = logging.getLogger("PyReceiverBase")

    def start(self) -> None:
        raise NotImplementedError()

    def stop(self) -> None:
        raise NotImplementedError()

    def register_listener(self, listener: dts.Listener) -> None:
        """
        Register a listener to receive data updates.

        Args:
            listener (Listener): A callable to receive a list of Dataclass instances.
        """
        with self._lock:
            self._listeners.append(listener)

    def clear_listeners(self) -> None:
        """
        Clear all registered listeners.
        """
        with self._lock:
            self._listeners.clear()

    def pretty_print_gaze_data(self, gaze_data: dts.GazeData) -> None:
        """
        Pretty print the gaze data.
        """
        print("\r--------------" + " " * 20)
        for key, value in gaze_data.__dict__.items():
            print(f"  {key}: {value}")

    def is_alive(self) -> bool:
        """
        Check if the receiver was started.

        Returns:
            bool: True if the receiver is currently running, False otherwise.
        """
        return self._running


class SMReceiver(PyReceiverBase):
    """
    Shared Memory Receiver for Unity Data.
    """

    def __init__(
        self,
        mmap_name: str,
        datatype: type[dts.DataclassLike],
        update_rate: int,
        block_count: int = 1,
        listeners: list[dts.Listener] | None = None,
        with_console: bool = False,
    ):
        super().__init__()
        self._data_block = self.acquire_shm(
            mmap_name, datatype.size() * block_count + 8, access=mmap.ACCESS_READ
        )
        self._block_cnt = block_count
        self._datatype = datatype
        self._update_rate = update_rate
        self._data_timestamp: float = 0.0
        self._listeners = listeners if listeners is not None else []
        self._logger.info("SMReceiver initialized.")
        self._with_console = with_console

        self._last_timestamp: float = 0.0

    def start(self) -> None:
        # Acquire shared memory block
        if self._data_block is None:
            raise RuntimeError("Shared memory block is not initialized.")

        if self._with_console:
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
        self._monitor.start()

        target_interval: float = 1.0 / self._update_rate
        while self._running:
            loop_start = time.perf_counter()

            datas: list[dts.DataclassLike] = self.read_data_blocks()
            # Notify listeners
            with self._lock:
                for listener in self._listeners:
                    listener(datas)
            # Update monitor
            self._monitor.update(len(datas))

            if self._with_console:
                self._console.print(
                    f"Data Rate: {self._monitor.get_data_rate():.1f} Hz"
                    f" | Avg Data Count: {self._monitor.get_avg_data_cnt():.1f}"
                    f" | Total: {self._monitor.get_total_packets()}     ",
                    use_spinner=True,
                )

            elapsed = time.perf_counter() - loop_start
            remaining = target_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        self._monitor.reset()

    def acquire_shm(
        self, block_name: str, block_size: int, access: int = mmap.ACCESS_DEFAULT
    ) -> mmap.mmap:
        """
        Acquire a shared memory block by its name.

        Args:
            block_name (str): The name of the shared memory block.
            block_size (int): The size of the shared memory block.

        Returns:
            mmap.mmap: The acquired shared memory block.
        """
        shm = mmap.mmap(-1, block_size, tagname=block_name, access=access)
        self._logger.info("Shared memory block (%s) acquired.", block_name)
        return shm

    def read_data_blocks(self) -> list[dts.DataclassLike]:
        """
        Read the data block(s) from shared memory.

        Returns:
            List[DataclassLike]: A list of instances of the dataclass
            containing the fields and their values.
        """
        assert self._data_block is not None, "Data block is not initialized."
        datas: list[dts.DataclassLike] = []
        self._data_block.seek(0)  # Seek to the beginning to read the timestamp
        self._data_timestamp = struct.unpack("<d", self._data_block.read(8))[
            0
        ]  # Read the timestamp (double, 8 bytes)
        self._data_block.seek(8)  # Seek to the beginning of the data block
        data = self._data_block.read(self._datatype.size() * self._block_cnt)
        for i in range(self._block_cnt):
            block_data = data[
                i * self._datatype.size() : (i + 1) * self._datatype.size()
            ]
            datas.append(self._datatype.from_buffer(block_data))
        return datas


class SMReceiverCircularBuffer(PyReceiverBase):
    """
    Shared Memory Receiver for Unity Data using a circular buffer.
    """

    def __init__(
        self,
        data_mmap_name: str,
        metadata_mmap_name: str,
        datatype: type[dts.DataclassLike],
        buffer_size: int,
        listeners: list[dts.Listener] | None = None,
        with_console: bool = False,
        target_hz: int = 100,
    ):
        super().__init__()
        # Acquire shared memory blocks
        self._metadata_block = self.acquire_shm(
            metadata_mmap_name, dts.Metadata.size(), access=mmap.ACCESS_WRITE
        )
        self._data_block = self.acquire_shm(
            data_mmap_name, datatype.size() * buffer_size, access=mmap.ACCESS_READ
        )
        self._buffer_size = buffer_size
        self._datatype = datatype
        self._listeners = listeners if listeners is not None else []
        self._data_ptr: int = 0
        self._with_console = with_console
        self._target_hz = target_hz
        self._flag_resync = False
        self._logger.info("SMReceiverCircularBuffer initialized.")

    def start(self) -> None:
        if self._metadata_block is None or self._data_block is None:
            raise RuntimeError("Failed to acquire shared memory blocks.")

        if self._with_console:
            self._console.start()
        # Start main thread
        if self._thread is None:
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._running = False
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        target_interval: float = 1.0 / self._target_hz
        while self._running:
            # Check the stream_ready flag in metadata
            metadata = self.read_metadata_block()
            self._ready = metadata.stream_ready == 1

            if not self._ready:
                if self._with_console:
                    self._console.print(
                        "Waiting for stream to be ready...", use_spinner=True
                    )
                time.sleep(0.1)
                continue

            # Here you can add code to read gaze data if needed
            if metadata.active_data_cnt == 0:
                if self._with_console:
                    self._console.print(
                        "No new gaze data available...", use_spinner=True
                    )
                time.sleep(0.1)
                continue

            break

        self._monitor.start()

        while self._running:
            loop_start = time.perf_counter()
            metadata = self.read_metadata_block()
            if metadata.active_data_cnt > 0:
                gaze_datas = self.read_data_blocks(metadata)
                # Notify listeners
                with self._lock:
                    for listener in self._listeners:
                        listener(gaze_datas)
                # Update monitor
                self._monitor.update(len(gaze_datas))
                if self._with_console:
                    self._console.print(
                        f"Gaze Data Rate: {self._monitor.get_data_rate():.1f} Hz"
                        f" | Avg Data Count: {self._monitor.get_avg_data_cnt():.1f}"
                        f" | Total: {self._monitor.get_total_packets()}     ",
                        use_spinner=True,
                    )
                # self.pretty_print_gaze_data(gaze_datas[-1])

            elapsed = time.perf_counter() - loop_start
            remaining = target_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        self._monitor.reset()

    def acquire_shm(
        self, block_name: str, block_size: int, access: int = mmap.ACCESS_DEFAULT
    ) -> mmap.mmap:
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

    def read_metadata_block(self) -> dts.Metadata:
        """
        Read the metadata block from shared memory.

        Returns:
            dts.Metadata: An instance of the Metadata dataclass containing
            the metadata fields and their values.
        """
        assert self._metadata_block is not None, "Metadata block is not initialized."
        self._metadata_block.seek(0)
        data = self._metadata_block.read(dts.Metadata.size())
        return dts.Metadata.from_buffer(data)

    def read_data_blocks(self, metadata: dts.Metadata) -> list[dts.DataclassLike]:
        """
        Read gaze data blocks from shared memory.

        Returns:
            list[DataclassLike]: A list of dataclass instances containing
            the gaze data fields and their values.
        """
        datas: list[dts.DataclassLike] = []
        block_size = self._datatype.size()

        assert self._data_block is not None, "Gaze data block is not initialized."

        # Rewrite data count to 0
        count = metadata.active_data_cnt
        metadata.active_data_cnt = np.uint8(0)
        self.write_metadata_cnt(metadata)

        # Check resync
        if self._flag_resync:
            if metadata.last_offset != self._data_ptr:
                self._logger.warning(
                    "Data pointer mismatch during resync. Expected: %d, Actual: %d",
                    self._data_ptr,
                    metadata.last_offset,
                )
                self._data_ptr = metadata.last_offset

        for _ in range(count):
            self._data_block.seek(self._data_ptr)
            data = self._data_block.read(block_size)
            gaze_data = self._datatype.from_buffer(data)
            datas.append(gaze_data)

            self._data_ptr += block_size
            # Check for circular buffer wrap-around
            if self._data_ptr >= block_size * self._buffer_size:
                # Perform resync next time
                self._flag_resync = True
                self._data_ptr = 0

        return datas

    def write_metadata_cnt(self, metadata: dts.Metadata) -> None:
        """
        Write the updated active_data_cnt back to the metadata block.
        """
        assert self._metadata_block is not None, "Metadata block is not initialized."
        self._metadata_block.seek(2)  # Offset for active_data_cnt
        self._metadata_block.write(metadata.active_data_cnt.tobytes())
        self._metadata_block.flush()


class ZMQReceiver(PyReceiverBase):
    """
    ZeroMQ Receiver for Gaze Data using pub/sub socket architecture.
    """

    SOCKET_SUB_FILTER = ""

    def __init__(
        self, datatype: dts.DataclassLike, address: str = "tcp://localhost:5555"
    ):
        super().__init__()
        self._datatype = datatype
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self.SOCKET_SUB_FILTER)
        self._socket.connect(address)
        self._logger.info("ZMQReceiver initialized.")

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run)
            self._thread.start()
            self._monitor.start()
            self._logger.info(
                "ZMQReceiver started and connected to %s",
                self._socket.getsockopt_string(zmq.LAST_ENDPOINT),
            )

    def stop(self) -> None:
        if self._thread is not None:
            self._running = False
            self._thread.join()
            self._thread = None
            self._logger.info("ZMQReceiver stopped.")

    def _run(self) -> None:
        while self._running:
            try:
                message = self._socket.recv_json(flags=zmq.NOBLOCK)
                if not isinstance(message, dict):
                    self._logger.warning("Received message is not a dictionary.")
                    continue
                gaze_data = self._datatype.from_dict(message)
                # Notify listeners
                with self._lock:
                    for listener in self._listeners:
                        listener([gaze_data])
                # Update monitor
                self._monitor.update(1)
                self._console.print(
                    f"Data Rate: {self._monitor.get_data_rate():.1f} Hz"
                    f" | Avg Data Count: {self._monitor.get_avg_data_cnt():.1f}"
                    f" | Total: {self._monitor.get_total_packets()}     ",
                    use_spinner=True,
                )
                # self.pretty_print_gaze_data(gaze_data)  # Print the latest gaze data
            except zmq.Again:
                time.sleep(0.01)  # No message received, wait a bit


class Monitor:
    def __init__(self):
        self._last_timestamp: float = 0.0
        self._data_rate: float = 0.0
        self._data_cnt: int = 0
        self._update_cnt: int = 0
        self._data_cnt_avg: float = 0.0
        self.total_packets: int = 0

    def update(self, packets_received: int):
        if self._last_timestamp == 0:
            self.start()
            return
        self._data_cnt += packets_received
        self._update_cnt += 1
        self.total_packets += packets_received
        if time.time() - self._last_timestamp >= 1.0:
            self._data_rate = self._data_cnt / (time.time() - self._last_timestamp)
            self._data_cnt_avg = (
                self._data_cnt / self._update_cnt if self._update_cnt > 0 else 0.0
            )
            self._data_cnt = 0
            self._update_cnt = 0
            self._last_timestamp = time.time()

    def start(self):
        self.reset()
        self._last_timestamp = time.time()

    def reset(self):
        self._last_timestamp = 0.0
        self._data_rate = 0.0
        self._data_cnt = 0
        self._update_cnt = 0
        self._data_cnt_avg = 0.0

    def get_data_rate(self) -> float:
        return self._data_rate

    def get_avg_data_cnt(self) -> float:
        return self._data_cnt_avg

    def get_total_packets(self) -> int:
        return self.total_packets
