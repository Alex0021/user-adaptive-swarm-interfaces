import logging
import os
import time

import debugpy
import numpy as np
import zmq
from PyQt6.QtWidgets import QApplication

import workload_inference.data.data_structures as dts
from workload_inference.experiment.app import ExperimentManagerWindow
from workload_inference.experiment.manager import ExperimentManager
from workload_inference.data.py_receiver import (
    SMReceiver,
    SMReceiverCircularBuffer,
    ZMQReceiver,
)
from workload_inference.utils.generator import FakeGazeGenerator


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s::%(message)s",
        handlers=[logging.StreamHandler()],
    )


def main():
    app = QApplication([])
    setup_logging()
    logger = logging.getLogger()
    logger.info("Workload Inference Service Started")

    try:
        experiment_manager = ExperimentManager()
        experiment_manager.initialize_receivers()
        experiment_manager.initialize_data_writers()
        experiment_manager.initialize_listeners()
        experiment_window = ExperimentManagerWindow(experiment_manager)
        experiment_window.show()
        # fake_data_generator = FakeGazeGenerator(
        #     callback=experiment_window._gaze_visualizer.datas_callback,
        #     frequency=60.0,
        #     noise=0.05,
        #     speed=2.5,
        #     pupil_mean=3.5,
        # )
        # fake_data_generator.start()
        experiment_window.attach_listeners()
        experiment_manager.gaze_receiver.start()
        experiment_window.start()

    # receiver = ZMQReceiver()
    # def print_drone_data(datas: list[dts.DroneData]):
    #     for d in datas:
    #         pos = np.array([d.position_x, d.position_y, d.position_z])
    #         print(pos)

    # receiver = SMReceiver(
    #     mmap_name=dts.DRONE_DATA_BLOCK_NAME,
    #     datatype=dts.DroneData,
    #     update_rate=2,
    #     listeners=[print_drone_data],
    #     block_count=dts.DRONE_COUNT,
    # )
    # receiver =  SMReceiverCircularBuffer(
    #     data_mmap_name=dts.GAZE_DATA_BLOCK_NAME,
    #     metadata_mmap_name=dts.METADATA_BLOCK_NAME,
    #     datatype=dts.GazeData,
    #     buffer_size=dts.GAZE_DATA_BLOCK_CNT
    # )
    # receiver.register_listener(experiment_manager.datas_callback)
    # # receiver.register_listener(data_processor.datas_callback)
    # receiver.register_listener(visualizer.canvas.datas_callback)
    # receiver.start()
    # experiment_manager.start_recording()
    except Exception as e:
        logger.error("%s", e)
        return
    try:
        # while True:
        #     print(f"Data Processor Samples: {data_processor.get_num_samples()}")
        #     try:
        #         print(data_processor.get_samples()[-1]) # Print the latest sample
        #     except IndexError as e:
        #         pass
        #     time.sleep(1)
        # visualizer.showMaximized()
        app.exec()
    except KeyboardInterrupt:
        pass
    finally:
        # receiver.stop()
        logger.info("Workload Inference Service Stopped")

    # if os.environ.get("PYTHONDEBUG", "0") == "1":
    #     print("Waiting for debugger to attach...")
    #     debugpy.listen(("0.0.0.0", 5678))
    #     debugpy.wait_for_client()

    # context = zmq.Context()
    # socket = context.socket(zmq.SUB)
    # if os.environ.get("IS_DOCKER", "0") == "1":
    #     socket.connect("tcp://host.docker.internal:5555")
    # else:
    #     socket.connect("tcp://localhost:5555")
    # socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # while True:
    #     data = socket.recv_json()
    #     print(f"Received data: {data}\n")
    #     time.sleep(0.01)


if __name__ == "__main__":
    main()
