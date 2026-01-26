import sys
import threading
import time
from typing import Optional

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

    def new_text(self, text: str, use_spinner: Optional[bool] = None) -> None:
        with self._lock:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            self._text = ""
            if use_spinner is not None:
                self._use_spinner = use_spinner

    def update_text(self, text: str, use_spinner: Optional[bool] = None) -> None:
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
                prefix = f"{self._spinner[idx % len(self._spinner)]} " if self._use_spinner else ""
                line = f"\r{prefix}{self._text}"
            sys.stdout.write(line)
            sys.stdout.flush()
            idx += 1
            time.sleep(self._interval)