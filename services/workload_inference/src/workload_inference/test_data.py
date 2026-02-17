"""
Test data for workload inference service

This script is used to generate a test experiment under the data/experiments directory.
It will generate a gaze_data.csv and a drone_data.csv file.
"""

import time
import argparse

from workload_inference.utils.constants import DATA_DIR
from workload_inference.utils.generator import FakeGazeGenerator, FakeDroneGenerator
from workload_inference.utils.utilities import ExperimentDataWriter

import logging

def main():
    logging.basicConfig(
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="Generate test data for workload inference service"
    )
    parser.add_argument(
        "-t", "--time",
        type=int,
        default=5, 
        help="Duration of the test data generation in seconds"
    )
    args = parser.parse_args()

    experiment_dir = DATA_DIR / "experiments" / "test_experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    gaze_writer = ExperimentDataWriter(experiment_dir / "gaze_data.csv")
    drone_writer = ExperimentDataWriter(experiment_dir / "drone_data.csv")

    gaze_generator = FakeGazeGenerator(callback=gaze_writer.datas_callback, frequency=60.0)
    drone_generator = FakeDroneGenerator(callback=drone_writer.datas_callback, frequency=30.0)

    gaze_writer.start()
    drone_writer.start()
    gaze_generator.start()
    drone_generator.start()


    print(f"Generating test data in {experiment_dir}... Press Ctrl+C to stop.")
    try:
        start_time = time.time()
        while time.time() - start_time < args.time:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping generators...")
        gaze_generator.stop()
        drone_generator.stop()
        print("Stopping writers...")
        gaze_writer.stop()
        drone_writer.stop()
        print("Done.")

if __name__ == "__main__":
    main()