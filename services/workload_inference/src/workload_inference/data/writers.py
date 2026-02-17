from pathlib import Path
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import logging
import threading
import time
from queue import Queue

from workload_inference.data.data_structures import DataclassLike

class ExperimentDataWriter:
    """Reusable data writer that writes objects to a CSV-like file from a queue.

    Args:
        filepath: Path or string to the output file.
        block_size: Number of entries to batch before flushing to disk.
        queue_size: Maximum queue size for incoming data.
        header: Optional iterable of field names to write as CSV header and to
            use for ordering fields when serializing objects.
        formatter: Optional callable that takes a data object and returns a
            single CSV-formatted string (without trailing newline). If not
            provided, the writer will use `header` with getattr or fall back
            to the object's __dict__ values.
    """

    WAIT_BLOCK_TIMEOUT: float = 0.01
    """Wait delay if block is not yet full."""

    def __init__(
        self,
        filepath: Path | None = None,
        block_size: int = 100,
        queue_size: int = 1000,
        header: Iterable[str] | None = None,
        formatter: Callable[[Any], str] | None = None,
        name: str = "anonymous",
        encoding: str = "utf-8",
    ) -> None:
        self.filepath = filepath
        self._block_size = int(block_size)
        self._queue: Queue = Queue(maxsize=int(queue_size))
        self._header: list[str] | None = list(header) if header is not None else None
        self._formatter = formatter
        self._encoding = encoding
        self._name = name

        self._logger = logging.getLogger(f"[{self._name}DataWriter]")

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._data_cnt = 0

        if self.filepath is not None:
            self.new_file(self.filepath)

    def datas_callback(
        self, datas: Sequence[DataclassLike], batch_update: bool = False
    ) -> None:
        """Callback to push a batch of data objects into the internal queue."""
        if not self._running:
            return
        if self._queue.qsize() + len(datas) > self._queue.maxsize:
            self._logger.warning(
                "Queue overflow attempted: have %d incoming %d max %d",
                self._queue.qsize(),
                len(datas),
                self._queue.maxsize,
            )
            return
        for d in datas:
            self._queue.put(d)

    def new_file(self, filepath: Path) -> None:
        """Set the output file path. Automatically stops the writer
        if it is running and flushes remaining data."""
        if self._running:
            self.stop()

        self.filepath = filepath
        # Ensure parent dir exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._filestream = open(self.filepath, "w", encoding=self._encoding)
        if self._header:
            self._filestream.write(",".join(self._header) + "\n")
            self._filestream.flush()
        self._logger.info("Set output file to '%s'", self.filepath)

    def start(self) -> None:
        """
        Start the internal writer thread.

        Raises:
            ValueError if the output file is not set. Does nothing if already running.
        """
        if self.filepath is None:
            raise ValueError("Filepath is not set. Cannot start writer.")
        with self._lock:
            if self._running:
                return
            self._running = True
            self._data_cnt = 0
            self._thread = threading.Thread(target=self._write_loop, daemon=True)
            self._thread.start()
            self._logger.info("Started writer thread")

    def stop(self) -> None:
        """Stop the writer thread, flush remaining queue and close file."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        # Wait for thread to finish
        if self._thread is not None and self._thread is not threading.current_thread():
            self._logger.info("Waiting for writer thread to finish...")
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                self._logger.warning("Writer thread did not finish in time")
            self._thread = None

        # Write any remaining items in queue
        remaining = self._queue.qsize()
        if remaining:
            self._logger.info("Flushing %d remaining items before close", remaining)
        while not self._queue.empty():
            item = self._queue.get(timeout=self.WAIT_BLOCK_TIMEOUT)
            line = self._format_item(item)
            self._filestream.write(line + "\n")
            self._data_cnt += 1
        self._filestream.flush()
        self._filestream.close()
        self._logger.info("Stopped writer, wrote %d lines", self._data_cnt)

    def _format_item(self, item: Any) -> str:
        """Format a single data object into a CSV line (no newline)."""
        if self._formatter is not None:
            return self._formatter(item)
        if self._header is not None:
            return ",".join(str(getattr(item, f, "")) for f in self._header)
        if hasattr(item, "__dict__"):
            return ",".join(str(v) for v in item.__dict__.values())
        return str(item)

    def _write_loop(self) -> None:
        """Internal thread loop to write data in blocks."""
        if self._filestream is None:
            raise RuntimeError("File stream is not initialized.")

        self._logger.debug("Writer loop running")
        while self._running:
            if self._queue.qsize() >= self._block_size:
                for _ in range(self._block_size):
                    item = self._queue.get(timeout=self.WAIT_BLOCK_TIMEOUT)
                    line = self._format_item(item)
                    self._filestream.write(line + "\n")
                self._filestream.flush()
                self._data_cnt += self._block_size
                self._logger.debug(
                    "Wrote block of %d lines (total %d)",
                    self._block_size,
                    self._data_cnt,
                )
            else:
                time.sleep(self.WAIT_BLOCK_TIMEOUT)

    @property
    def data_count(self) -> int:
        return self._data_cnt
