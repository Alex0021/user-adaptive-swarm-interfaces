import contextlib
import logging
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Iterable, List, Optional, Sequence

from workload_inference.data.data_structures import DataclassLike


class ConsoleManager:
    def __init__(self, interval: float = 0.25, spinner: str = "|/-\\"):
        self._interval = interval

        self._spinner = spinner
        self._text: str = ""
        self._use_spinner: bool = True
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.flush()

    def print(self, text: str, use_spinner: Optional[bool] = None) -> None:
        if text == self._text and use_spinner == self._use_spinner:
            return
        with self._lock:
            self._text = text
            if use_spinner is not None:
                self._use_spinner = use_spinner

    def set_spinner_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._use_spinner = enabled

    def set_spinner(self, spinner: str) -> None:
        with self._lock:
            self._spinner = spinner

    def _run(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            with self._lock:
                prefix = (
                    f"{self._spinner[idx % len(self._spinner)]} "
                    if self._use_spinner
                    else ""
                )
                line = f"\r{prefix}{self._text}"
            sys.stdout.write(line)
            sys.stdout.flush()
            idx += 1
            time.sleep(self._interval)


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
