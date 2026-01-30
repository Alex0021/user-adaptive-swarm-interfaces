import zmq
import time
import threading
from collections import deque
import logging


class EyeTrackerStream:
    """
    Class to handle streaming data from the eye tracker using ZeroMQ.
    """

    class Monitor:
        """
        Used to get some insights on the streaming performance.
        """

        def __init__(self):
            self.total_messages = 0
            self.interval_messages = 0
            self.last_time = 0
            self.queue_count = 0
            self.estimated_frequency = 0.0
            self.avg_queue_length = 0.0

        def start(self):
            self.reset()
            self.last_time = time.time()

        def reset(self):
            self.total_messages = 0
            self.last_time = 0
            self.queue_count = 0
            self.avg_queue_length = 0.0
            self.estimated_frequency = 0.0

        def update(self, msg_count: int, queue_count: int):
            """
            Update the monitor with new message and queue counts.

            :param msg_count (int): Number of new messages sent.
            :param queue_count (int): Number of messages currently in the queue.
            """
            self.total_messages += msg_count
            self.interval_messages += msg_count
            self.queue_count += queue_count
            if time.time() - self.last_time >= 1.0:
                self.estimated_frequency = self.interval_messages / (
                    time.time() - self.last_time
                )
                self.avg_queue_length = (
                    self.queue_count / self.total_messages
                    if self.total_messages > 0
                    else 0.0
                )
                self.last_time = time.time()
                self.interval_messages = 0
                self.queue_count = 0

        def get_frequency(self):
            """
            Return the estimated frequency of messages being sent. (Updated every second)
            """
            return self.estimated_frequency

        def get_avg_queue_cnt(self):
            """
            Return the average queue length of messages.
            """
            return self.avg_queue_length

        def get_total_messages(self):
            """
            Return the total number of messages sent since the monitor started.
            """
            return self.total_messages

    def __init__(self, address: str = "tcp://localhost", port: int = 5555):
        """
        Binds to the given address and port to stream eye tracker data.

        :param address (str): The address to bind the stream to.
        :param port (int): The port to bind the stream to.
        """
        self.address = f"{address}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.address)
        self.running = False
        self.msg_queue: deque[dict[str, float]] = deque(maxlen=100)
        self.thread = None
        self._lock = threading.Lock()
        self.monitor = EyeTrackerStream.Monitor()
        self.logger = logging.getLogger(__name__)

    def start_stream(self):
        """
        Start the eye tracker data stream in a separate thread.
        """
        if not self.running:
            self.logger.info("Request to start eye tracker stream.")
            self.running = True
            self.thread = threading.Thread(target=self._stream_data)
            self.thread.start()
            self.monitor.start()
            self.logger.info("Eye tracker stream started.")

    def stop_stream(self):
        """
        Stop the eye tracker data stream.
        """
        if self.running:
            self.running = False
            self.thread.join()
            self.monitor.reset()
            self.socket.close()
            self.context.term()
            self.logger.info("Eye tracker stream stopped.")

    def _stream_data(self):
        """
        Internal method to simulate streaming data from the eye tracker.
        """
        while self.running:
            while self.msg_queue:
                with self._lock:
                    msg = self.msg_queue.popleft()
                self.socket.send_json(msg)
                self.monitor.update(1, len(self.msg_queue))
            time.sleep(0.01)  # Sleep briefly to avoid busy waiting

    def gaze_data_callback(self, gaze_data: dict[str, float]):
        """
        Callback function to handle incoming gaze data from the eye tracker.
        This function should be connected to the eye tracker's data stream.
        """
        with self._lock:
            self.msg_queue.append(gaze_data)
