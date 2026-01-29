import zmq
import time
import threading
from collections import deque
import logging

class EyeTrackerStream:
    """
    Class to handle streaming data from the eye tracker using ZeroMQ.
    """

    class Metrics:
        """
        Used to get some insights on the streaming performance.
        """
        def __init__(self):
            self.total_messages = 0

        def get_frequency(self):
            pass

        def get_avg_queue_cnt(self):
            pass
            
        def get_total_messages(self):
            return self.total_messages

    def __init__(self, address="tcp://localhost", port=5555):
        self.address = f"{address}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.address)
        self.running = False
        self.msg_queue = deque(maxlen=100)
        self.thread = None
        self._lock = threading.Lock()
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
            self.logger.info("Eye tracker stream started.")
    
    def stop_stream(self):
        """
        Stop the eye tracker data stream.
        """
        if self.running:
            self.running = False
            self.thread.join()
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
            time.sleep(0.01)  # Sleep briefly to avoid busy waiting
    
    def gaze_data_callback(self, gaze_data):
        """
        Callback function to handle incoming gaze data from the eye tracker.
        This function should be connected to the eye tracker's data stream.
        """
        with self._lock:
            self.msg_queue.append(gaze_data)