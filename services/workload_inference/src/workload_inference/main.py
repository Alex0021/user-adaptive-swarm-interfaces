import time
import debugpy
import os
from workload_inference.py_receiver import SMReceiverCircularBuffer, ZMQReceiver, SMReceiver
import zmq
from workload_inference.experiment import ExperimentManager
from workload_inference.processing import DataProcessor
from workload_inference.visualize import GazeVisualizerWindow
import logging
import workload_inference.data_structures as dts
from PyQt6.QtWidgets import QApplication

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s::%(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

def main():
    app = QApplication([])
    setup_logging()
    logger = logging.getLogger()
    logger.info("Workload Inference Service Started")

    try:
        experiment_manager = ExperimentManager()
        visualizer = GazeVisualizerWindow()
        # receiver = ZMQReceiver()
        def print_nback_data(datas :list[dts.NBackData]):
            for d in datas:
                print(d)
        receiver = SMReceiver(
            mmap_name=dts.NBACK_DATA_BLOCK_NAME,
            datatype=dts.NBackData,
            update_rate=2,
            listeners=[print_nback_data],
            block_count=dts.NBACK_SEQUENCE_LEN,
        )
        # receiver =  SMReceiverCircularBuffer(
        #     data_mmap_name=dts.GAZE_DATA_BLOCK_NAME,
        #     metadata_mmap_name=dts.METADATA_BLOCK_NAME,
        #     datatype=dts.GazeData,
        #     buffer_size=dts.GAZE_DATA_BLOCK_CNT
        # )
        # receiver.register_listener(experiment_manager.datas_callback)
        # # receiver.register_listener(data_processor.datas_callback)
        # receiver.register_listener(visualizer.canvas.datas_callback)
        receiver.start()
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
        receiver.stop()
        experiment_manager.stop_recording()

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